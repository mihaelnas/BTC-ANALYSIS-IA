"""
Parquet storage writer with batching, rotation, and compression.

Implements a producer-friendly interface:
- add_snapshot(): buffer a single order book snapshot
- flush(): write buffered data to disk
- Automatic file rotation by time window
- Graceful shutdown with signal handling
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

logger = structlog.get_logger(__name__)


def build_schema(n_levels: int = 20) -> pa.Schema:
    """
    Build the PyArrow schema for order book snapshots.

    Args:
        n_levels: Number of bid/ask levels per snapshot.

    Returns:
        A PyArrow Schema with all snapshot columns.
    """
    fields = [
        pa.field("timestamp", pa.float64()),
        pa.field("mid_price", pa.float64()),
        pa.field("spread", pa.float64()),
        pa.field("microprice", pa.float64()),
    ]

    for i in range(1, n_levels + 1):
        fields.append(pa.field(f"bid_price_{i}", pa.float64()))
        fields.append(pa.field(f"bid_qty_{i}", pa.float64()))

    for i in range(1, n_levels + 1):
        fields.append(pa.field(f"ask_price_{i}", pa.float64()))
        fields.append(pa.field(f"ask_qty_{i}", pa.float64()))

    return pa.schema(fields)


class ParquetWriter:
    """
    Batched Parquet writer with time-based file rotation.

    Snapshots are buffered in memory and flushed to disk either:
    - When the buffer reaches `batch_size`
    - When the time rotation window elapses
    - When flush() is called explicitly (e.g., on shutdown)

    Files are named: {output_dir}/ob_YYYYMMDD_HHMM.parquet
    """

    def __init__(
        self,
        output_dir: str | Path,
        n_levels: int = 20,
        batch_size: int = 5000,
        rotation_minutes: int = 60,
        compression: str = "zstd",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._schema = build_schema(n_levels)
        self._n_levels = n_levels
        self._batch_size = batch_size
        self._rotation_minutes = rotation_minutes
        self._compression = compression

        self._buffer: list[dict[str, Any]] = []
        self._current_writer: pq.ParquetWriter | None = None
        self._current_file_path: Path | None = None
        self._current_rotation_key: str | None = None

        self._total_rows_written: int = 0
        self._total_files_created: int = 0

    @property
    def total_rows_written(self) -> int:
        return self._total_rows_written

    @property
    def total_files_created(self) -> int:
        return self._total_files_created

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def _rotation_key(self, timestamp: float | None = None) -> str:
        """
        Generate a rotation key based on the current time.

        The key changes every `rotation_minutes` minutes, triggering file rotation.
        """
        ts = timestamp or time.time()
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        # Round down to the nearest rotation window
        minute_bucket = (dt.minute // self._rotation_minutes) * self._rotation_minutes
        return dt.strftime(f"%Y%m%d_%H{minute_bucket:02d}")

    def _open_writer(self, rotation_key: str) -> None:
        """Open a new Parquet file writer."""
        self._close_writer()

        filename = f"ob_{rotation_key}.parquet"
        self._current_file_path = self._output_dir / filename
        self._current_writer = pq.ParquetWriter(
            str(self._current_file_path),
            schema=self._schema,
            compression=self._compression,
        )
        self._current_rotation_key = rotation_key
        self._total_files_created += 1

        logger.info(
            "parquet_file_opened",
            path=str(self._current_file_path),
            total_files=self._total_files_created,
        )

    def _close_writer(self) -> None:
        """Close the current Parquet writer if open."""
        if self._current_writer is not None:
            self._current_writer.close()
            logger.info(
                "parquet_file_closed",
                path=str(self._current_file_path),
            )
            self._current_writer = None
            self._current_file_path = None
            self._current_rotation_key = None

    def _write_batch(self) -> None:
        """Write the current buffer to the Parquet file."""
        if not self._buffer:
            return

        # Build a RecordBatch from the buffer
        # Transpose list-of-dicts → dict-of-lists
        columns: dict[str, list] = {field.name: [] for field in self._schema}
        for row in self._buffer:
            for col_name in columns:
                columns[col_name].append(row.get(col_name))

        arrays = [pa.array(columns[field.name], type=field.type) for field in self._schema]
        batch = pa.RecordBatch.from_arrays(arrays, schema=self._schema)

        if self._current_writer is None:
            rotation_key = self._rotation_key()
            self._open_writer(rotation_key)

        assert self._current_writer is not None
        self._current_writer.write_batch(batch)
        rows_written = len(self._buffer)
        self._total_rows_written += rows_written
        self._buffer.clear()

        logger.debug(
            "batch_written",
            rows=rows_written,
            total_rows=self._total_rows_written,
        )

    def add_snapshot(self, snapshot: dict[str, Any]) -> None:
        """
        Add a snapshot to the buffer.

        Automatically flushes when:
        - Buffer reaches batch_size
        - File rotation is needed
        """
        # Check if we need to rotate the file
        current_key = self._rotation_key(snapshot.get("timestamp"))
        if self._current_rotation_key and current_key != self._current_rotation_key:
            # Time to rotate: flush current buffer and open new file
            self._write_batch()
            self._open_writer(current_key)

        self._buffer.append(snapshot)

        if len(self._buffer) >= self._batch_size:
            self._write_batch()

    def flush(self) -> None:
        """
        Force-flush any remaining buffered data to disk.

        Call this on graceful shutdown to avoid data loss.
        """
        if self._buffer:
            self._write_batch()
        self._close_writer()

        logger.info(
            "storage_flushed",
            total_rows=self._total_rows_written,
            total_files=self._total_files_created,
        )

    def __enter__(self) -> ParquetWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.flush()
