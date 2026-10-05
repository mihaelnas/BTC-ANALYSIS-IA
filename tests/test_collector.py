"""
Tests for the DepthCollector class.

These tests avoid any real network I/O. The WebSocket connection
(`_producer`) and the storage layer are replaced with lightweight fakes so
the reconnection backoff and write-failure handling can be exercised
directly and quickly, without talking to Binance or touching disk.
"""

from __future__ import annotations

import asyncio
from typing import Any, Self

import pytest

from config.settings import CollectorSettings
from src.data.collector import DepthCollector
from src.data.order_book import OrderBook
from src.data.storage import StorageWriteError

# ─── Helpers ────────────────────────────────────────────────────────────────

def make_update(bids: list[tuple[str, str]], first_id: int, final_id: int) -> dict[str, Any]:
    """A minimal, always-valid depthUpdate event (asks left untouched)."""
    return {
        "e": "depthUpdate",
        "U": first_id,
        "u": final_id,
        "b": [[p, q] for p, q in bids],
        "a": [],
    }


def make_initialized_book() -> OrderBook:
    """An OrderBook ready to accept updates, without any REST call."""
    book = OrderBook()
    book._bids[-100.0] = 1.0
    book._asks[100.1] = 1.0
    book._last_update_id = 100
    book._initialized = True
    return book


class FakeFailingWriter:
    """A writer stand-in whose every write fails, like a full disk."""

    def __init__(self) -> None:
        self.attempts = 0
        self.buffer_size = 0

    def add_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.attempts += 1
        raise StorageWriteError("simulated disk full")


class FlakyWriter:
    """Fails a fixed number of times, then recovers -- a transient hiccup."""

    def __init__(self, fail_first_n: int) -> None:
        self._fail_first_n = fail_first_n
        self.calls = 0
        self.buffer_size = 0

    def add_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.calls += 1
        if self.calls <= self._fail_first_n:
            raise StorageWriteError("transient failure")


# ─── _consumer: write-error handling ────────────────────────────────────────

class TestConsumerWriteErrorHandling:
    async def test_consecutive_write_errors_trigger_self_shutdown(self) -> None:
        settings = CollectorSettings(max_consecutive_write_errors=3)
        collector = DepthCollector(collector_settings=settings)
        collector._order_book = make_initialized_book()
        collector._writer = FakeFailingWriter()
        collector._running = True

        # Five valid, consecutive updates: enough to cross the threshold
        # and verify the collector stops itself rather than retrying
        # forever or silently filling its queue.
        next_id = 101
        for _ in range(5):
            update = make_update(bids=[("100.0", "2.0")], first_id=next_id, final_id=next_id)
            await collector._queue.put(update)
            next_id += 1

        await asyncio.wait_for(collector._consumer(), timeout=5.0)

        assert collector._running is False
        assert collector._consecutive_write_errors >= 3
        assert collector._write_errors == collector._writer.attempts
        assert collector._snapshots_collected == 0

    async def test_successful_write_resets_the_consecutive_counter(self) -> None:
        settings = CollectorSettings(max_consecutive_write_errors=100)
        collector = DepthCollector(collector_settings=settings)
        collector._order_book = make_initialized_book()
        collector._writer = FlakyWriter(fail_first_n=2)
        collector._running = False  # drain only the pre-queued updates below

        next_id = 101
        for _ in range(3):
            update = make_update(bids=[("100.0", "2.0")], first_id=next_id, final_id=next_id)
            await collector._queue.put(update)
            next_id += 1

        await asyncio.wait_for(collector._consumer(), timeout=5.0)

        assert collector._write_errors == 2
        assert collector._consecutive_write_errors == 0  # reset by the 3rd, successful write
        assert collector._snapshots_collected == 1


# ─── _producer: reconnection backoff ────────────────────────────────────────

class TestProducerReconnection:
    async def test_backoff_grows_with_jitter_and_gives_up_after_max_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = CollectorSettings(
            reconnect_delay_base=1.0,
            reconnect_delay_max=4.0,
            reconnect_jitter_ratio=0.5,
            max_reconnect_attempts=3,
        )
        collector = DepthCollector(collector_settings=settings)
        collector._running = True

        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        def always_fails(*args: Any, **kwargs: Any) -> Any:
            raise ConnectionRefusedError("simulated: no network access in tests")

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        monkeypatch.setattr("src.data.collector.ws_connect", always_fails)

        await asyncio.wait_for(collector._producer(), timeout=5.0)

        # It must give up after max_reconnect_attempts rather than retry
        # forever against an endpoint that keeps refusing the connection.
        assert collector._running is False
        assert len(sleeps) == settings.max_reconnect_attempts

        # Backoff must grow (base, then doubling, capped at the ceiling)
        # with jitter adding at most reconnect_jitter_ratio on top.
        expected_delays = [1.0, 2.0, 4.0]
        for observed, expected in zip(sleeps, expected_delays):
            assert expected <= observed <= expected * (1 + settings.reconnect_jitter_ratio)

    async def test_reconnect_count_resets_after_a_successful_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        A connection that succeeds, then later drops, should restart its
        backoff from the base delay -- not continue escalating from where
        a much earlier, unrelated failure left off.
        """
        settings = CollectorSettings(
            reconnect_delay_base=1.0,
            reconnect_delay_max=30.0,
            max_reconnect_attempts=10,
        )
        collector = DepthCollector(collector_settings=settings)
        collector._running = True

        sleeps: list[float] = []
        connect_calls = 0

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            if len(sleeps) >= 2:
                # Stop the test once we have observed a reset.
                collector._running = False

        class FakeConnection:
            """An async context manager mimicking a connection that drops
            immediately after being established, with no messages."""

            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *exc_info: object) -> None:
                return None

            def __aiter__(self) -> FakeConnection:
                return self

            async def __anext__(self) -> Any:
                raise StopAsyncIteration

        def flaky_connect(*args: Any, **kwargs: Any) -> Any:
            nonlocal connect_calls
            connect_calls += 1
            if connect_calls == 1:
                raise ConnectionRefusedError("first attempt fails")
            return FakeConnection()  # later attempts succeed, then close immediately

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        monkeypatch.setattr("src.data.collector.ws_connect", flaky_connect)

        await asyncio.wait_for(collector._producer(), timeout=5.0)

        # First failure -> reconnect_count goes to 1, first backoff sleep.
        # Then the connection succeeds and closes cleanly (no messages),
        # which must reset reconnect_count back to 0 before the loop
        # reconnects and sleeps again -- so both observed delays should be
        # the base delay, not an escalating sequence.
        assert len(sleeps) == 2
        base = settings.reconnect_delay_base
        max_with_jitter = base * (1 + settings.reconnect_jitter_ratio)
        assert base <= sleeps[0] <= max_with_jitter
        assert base <= sleeps[1] <= max_with_jitter
