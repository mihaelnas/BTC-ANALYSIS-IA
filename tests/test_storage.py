"""
Tests for the ParquetWriter class.

Covers:
- Basic buffering, batch-triggered writes, and flush-to-disk
- Time-based file rotation
- Failure handling: a write error never drops buffered snapshots, the
  writer can be retried once the underlying issue is resolved, and
  flush() always closes the file handle even when the write itself fails
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from src.data.storage import ParquetWriter, StorageWriteError

# ─── Helpers ────────────────────────────────────────────────────────────────

def make_snapshot(n_levels: int = 1, timestamp: float = 1_700_000_000.0) -> dict[str, Any]:
    """Build a snapshot dict matching OrderBook.snapshot()'s output shape."""
    snap: dict[str, Any] = {
        "timestamp": timestamp,
        "mid_price": 100.0,
        "spread": 0.1,
        "microprice": 100.02,
    }
    for i in range(1, n_levels + 1):
        snap[f"bid_price_{i}"] = 100.0 - i * 0.1
        snap[f"bid_qty_{i}"] = 1.0
        snap[f"ask_price_{i}"] = 100.1 + i * 0.1
        snap[f"ask_qty_{i}"] = 1.0
    return snap


def fail_write_batch(self: pq.ParquetWriter, batch: Any) -> None:
    """Monkeypatch target simulating a disk-level write failure."""
    raise OSError("No space left on device (simulated)")


# ─── Basic behavior ─────────────────────────────────────────────────────────

class TestParquetWriterBasics:
    def test_buffers_until_flush(self, tmp_path: Path) -> None:
        writer = ParquetWriter(output_dir=tmp_path, n_levels=1, batch_size=10)
        writer.add_snapshot(make_snapshot())

        assert writer.buffer_size == 1
        assert writer.total_rows_written == 0

        writer.flush()

        assert writer.buffer_size == 0
        assert writer.total_rows_written == 1
        assert writer.total_files_created == 1

        files = list(tmp_path.glob("*.parquet"))
        assert len(files) == 1
        table = pq.read_table(files[0])
        assert table.num_rows == 1

    def test_batch_size_triggers_automatic_write(self, tmp_path: Path) -> None:
        writer = ParquetWriter(output_dir=tmp_path, n_levels=1, batch_size=3)
        for _ in range(3):
            writer.add_snapshot(make_snapshot())

        # No explicit flush needed: batch_size was reached on its own.
        assert writer.buffer_size == 0
        assert writer.total_rows_written == 3

    def test_rotation_creates_a_new_file(self, tmp_path: Path) -> None:
        writer = ParquetWriter(
            output_dir=tmp_path, n_levels=1, batch_size=1, rotation_minutes=1
        )

        writer.add_snapshot(make_snapshot(timestamp=1_700_000_000.0))
        assert writer.total_files_created == 1

        # The first file's rotation key is derived from wall-clock time
        # (an existing quirk of _write_batch()'s fallback when no file is
        # open yet, unrelated to this fix); pin it to a known value so the
        # rest of this test is deterministic regardless of when it runs.
        writer._current_rotation_key = writer._rotation_key(1_700_000_000.0)

        writer.add_snapshot(make_snapshot(timestamp=1_700_000_000.0 + 120))  # +2 min

        assert writer.total_files_created == 2
        assert len(list(tmp_path.glob("*.parquet"))) == 2


# ─── Failure handling ───────────────────────────────────────────────────────

class TestParquetWriterFailureHandling:
    def test_write_failure_raises_and_preserves_buffer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writer = ParquetWriter(output_dir=tmp_path, n_levels=1, batch_size=1)
        monkeypatch.setattr(pq.ParquetWriter, "write_batch", fail_write_batch)

        with pytest.raises(StorageWriteError):
            writer.add_snapshot(make_snapshot())

        # The snapshot must still be there: nothing is silently dropped.
        assert writer.buffer_size == 1
        assert writer.total_rows_written == 0
        assert writer.failed_batches == 1

    def test_retry_after_transient_failure_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writer = ParquetWriter(output_dir=tmp_path, n_levels=1, batch_size=1)
        monkeypatch.setattr(pq.ParquetWriter, "write_batch", fail_write_batch)

        with pytest.raises(StorageWriteError):
            writer.add_snapshot(make_snapshot())

        monkeypatch.undo()  # Simulate the disk becoming available again
        writer.flush()

        assert writer.buffer_size == 0
        assert writer.total_rows_written == 1
        assert writer.failed_batches == 1  # the earlier failure is still counted

    def test_rotation_failure_does_not_lose_the_new_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writer = ParquetWriter(
            output_dir=tmp_path, n_levels=1, batch_size=1, rotation_minutes=1
        )

        # Window A: writes successfully and opens the first file.
        writer.add_snapshot(make_snapshot(timestamp=1_700_000_000.0))
        assert writer.total_files_created == 1
        assert writer.total_rows_written == 1

        # Window B: force the rotation's write to fail.
        monkeypatch.setattr(pq.ParquetWriter, "write_batch", fail_write_batch)

        with pytest.raises(StorageWriteError):
            writer.add_snapshot(make_snapshot(timestamp=1_700_000_000.0 + 120))

        # The snapshot from window B must still be buffered -- not lost --
        # even though opening window B's file (a separate step from writing
        # to it) already succeeded before the write itself failed.
        assert writer.buffer_size == 1
        assert writer.total_rows_written == 1  # still just window A's row
        assert writer.total_files_created == 2

        monkeypatch.undo()
        writer.flush()

        assert writer.buffer_size == 0
        assert writer.total_rows_written == 2
        assert writer.total_files_created == 2

    def test_flush_closes_the_writer_and_reraises_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writer = ParquetWriter(output_dir=tmp_path, n_levels=1, batch_size=10)
        writer.add_snapshot(make_snapshot())  # buffered only, batch_size not reached

        monkeypatch.setattr(pq.ParquetWriter, "write_batch", fail_write_batch)

        with pytest.raises(StorageWriteError):
            writer.flush()

        # The underlying file handle is always released, even on failure,
        # so a repeated flush() doesn't leak file descriptors.
        assert writer._current_writer is None
        assert writer.buffer_size == 1
