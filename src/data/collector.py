"""
Asynchronous WebSocket collector for Binance depth stream.

Architecture: Producer-Consumer pattern
- Producer: WebSocket connection → receives depth updates → pushes to asyncio.Queue
- Consumer: Reads from queue → applies updates to local OrderBook → extracts snapshots → writes to Parquet

Features:
- Exponential backoff reconnection
- Graceful shutdown via signal handling
- Structured logging with metrics
- Automatic order book resynchronization on sequence gaps
"""

from __future__ import annotations

import asyncio
import random
import signal
import time
from typing import Any

import orjson
import structlog
import websockets
from websockets.asyncio.client import connect as ws_connect

from config.settings import get_settings, BinanceSettings, CollectorSettings
from src.data.order_book import OrderBook
from src.data.storage import ParquetWriter, StorageWriteError

logger = structlog.get_logger(__name__)


class DepthCollector:
    """
    Real-time order book depth collector for Binance BTCUSDT.

    Connects to the WebSocket depth stream, maintains a local order book,
    and writes snapshots to Parquet files.
    """

    def __init__(
        self,
        binance_settings: BinanceSettings | None = None,
        collector_settings: CollectorSettings | None = None,
        output_dir: str | None = None,
    ) -> None:
        settings = get_settings()
        self._binance = binance_settings or settings.binance
        self._config = collector_settings or settings.collector

        from config.settings import RAW_DATA_DIR
        self._output_dir = output_dir or str(RAW_DATA_DIR)

        self._order_book = OrderBook()
        self._writer: ParquetWriter | None = None
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10000)

        # State tracking
        self._running = False
        self._reconnect_count = 0
        self._snapshots_collected = 0
        self._updates_received = 0
        self._write_errors = 0
        self._consecutive_write_errors = 0
        self._start_time: float = 0.0

        # Buffered updates (received before snapshot is ready)
        self._buffered_updates: list[dict[str, Any]] = []

    @property
    def stats(self) -> dict[str, Any]:
        """Current collector statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "elapsed_seconds": round(elapsed, 1),
            "snapshots_collected": self._snapshots_collected,
            "updates_received": self._updates_received,
            "reconnections": self._reconnect_count,
            "queue_size": self._queue.qsize(),
            "buffer_size": self._writer.buffer_size if self._writer else 0,
            "total_rows_written": self._writer.total_rows_written if self._writer else 0,
            "total_files": self._writer.total_files_created if self._writer else 0,
            "write_errors": self._write_errors,
            "consecutive_write_errors": self._consecutive_write_errors,
        }

    async def _initialize_book(self) -> None:
        """Fetch initial order book snapshot from REST API."""
        logger.info("fetching_initial_snapshot", url=self._binance.snapshot_url)
        await self._order_book.initialize_from_rest(self._binance.snapshot_url)

        # Apply any buffered updates
        applied = 0
        for update in self._buffered_updates:
            if self._order_book.apply_update(update):
                applied += 1
        logger.info(
            "buffered_updates_applied",
            total=len(self._buffered_updates),
            applied=applied,
        )
        self._buffered_updates.clear()

    async def _producer(self) -> None:
        """
        WebSocket producer: connect and push depth updates to the queue.

        Implements exponential backoff for reconnection.
        """
        delay = self._config.reconnect_delay_base

        while self._running:
            try:
                logger.info("ws_connecting", url=self._binance.ws_url)

                async with ws_connect(
                    self._binance.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=2**20,  # 1 MB max message
                ) as ws:
                    logger.info("ws_connected")
                    delay = self._config.reconnect_delay_base  # Reset on success
                    self._reconnect_count = 0

                    async for raw_msg in ws:
                        if not self._running:
                            break

                        try:
                            data = orjson.loads(raw_msg)
                        except orjson.JSONDecodeError:
                            logger.warning("invalid_json", raw=raw_msg[:200])
                            continue

                        # Skip subscription confirmation messages
                        if "e" not in data:
                            continue

                        await self._queue.put(data)
                        self._updates_received += 1

            except websockets.ConnectionClosed as e:
                logger.warning("ws_connection_closed", code=e.code, reason=e.reason)
            except Exception as e:
                logger.error("ws_error", error=str(e), type=type(e).__name__)

            if not self._running:
                break

            # Exponential backoff
            self._reconnect_count += 1
            if self._reconnect_count > self._config.max_reconnect_attempts:
                logger.error("max_reconnect_attempts_reached")
                self._running = False
                break

            # Add jitter so that, in a multi-instance deployment, reconnecting
            # clients don't all hammer the Binance endpoint in lockstep.
            jitter = random.uniform(0, delay * self._config.reconnect_jitter_ratio)
            sleep_for = delay + jitter
            logger.info(
                "ws_reconnecting",
                delay=round(sleep_for, 2),
                attempt=self._reconnect_count,
            )
            await asyncio.sleep(sleep_for)
            delay = min(delay * 2, self._config.reconnect_delay_max)

    async def _consumer(self) -> None:
        """
        Consumer: process depth updates, maintain order book, write snapshots.
        """
        while self._running or not self._queue.empty():
            try:
                update = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # If book not initialized, buffer updates and fetch snapshot
            if not self._order_book.initialized:
                self._buffered_updates.append(update)
                if len(self._buffered_updates) == 1:
                    # Trigger initialization on first update
                    try:
                        await self._initialize_book()
                    except Exception as e:
                        logger.error("snapshot_init_failed", error=str(e))
                        self._buffered_updates.clear()
                continue

            # Apply the update
            success = self._order_book.apply_update(update)

            if not success:
                # Sequence gap → need to resync
                logger.warning("resync_triggered")
                self._order_book = OrderBook()
                self._buffered_updates = [update]
                try:
                    await self._initialize_book()
                except Exception as e:
                    logger.error("resync_failed", error=str(e))
                    self._buffered_updates.clear()
                continue

            # Check for crossed book
            if self._order_book.needs_resync():
                logger.warning("crossed_book_resync")
                self._order_book = OrderBook()
                try:
                    await self._initialize_book()
                except Exception as e:
                    logger.error("resync_failed", error=str(e))
                continue

            # Extract and store snapshot
            snapshot = self._order_book.snapshot(n_levels=self._config.n_levels)
            if snapshot is not None and self._writer is not None:
                try:
                    self._writer.add_snapshot(snapshot)
                except StorageWriteError as e:
                    # Without this, an unhandled exception here would kill
                    # the consumer task silently (asyncio.gather swallows it
                    # via return_exceptions=True): the producer would keep
                    # filling the queue until it saturates, and the whole
                    # pipeline would freeze with no further log output.
                    self._write_errors += 1
                    self._consecutive_write_errors += 1
                    logger.error(
                        "snapshot_write_failed",
                        error=str(e),
                        consecutive_failures=self._consecutive_write_errors,
                        total_failures=self._write_errors,
                    )
                    if (
                        self._consecutive_write_errors
                        >= self._config.max_consecutive_write_errors
                    ):
                        logger.critical(
                            "max_write_errors_reached",
                            consecutive_failures=self._consecutive_write_errors,
                        )
                        self._shutdown()
                else:
                    self._consecutive_write_errors = 0
                    self._snapshots_collected += 1

                    # Periodic stats logging
                    if self._snapshots_collected % 5000 == 0:
                        logger.info("collection_progress", **self.stats)

    async def run(self, duration_seconds: float | None = None) -> None:
        """
        Start the collector.

        Args:
            duration_seconds: Optional maximum collection duration.
                            If None, runs until interrupted.
        """
        self._running = True
        self._start_time = time.time()

        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown)

        self._writer = ParquetWriter(
            output_dir=self._output_dir,
            n_levels=self._config.n_levels,
            batch_size=self._config.parquet_batch_size,
            rotation_minutes=self._config.file_rotation_minutes,
            compression=self._config.compression,
        )

        logger.info(
            "collector_starting",
            ws_url=self._binance.ws_url,
            output_dir=self._output_dir,
            n_levels=self._config.n_levels,
            batch_size=self._config.parquet_batch_size,
            duration=duration_seconds,
        )

        try:
            tasks = [
                asyncio.create_task(self._producer(), name="producer"),
                asyncio.create_task(self._consumer(), name="consumer"),
            ]

            if duration_seconds:
                tasks.append(
                    asyncio.create_task(
                        self._duration_guard(duration_seconds), name="duration_guard"
                    )
                )

            await asyncio.gather(*tasks, return_exceptions=True)

        finally:
            self._running = False
            if self._writer:
                try:
                    self._writer.flush()
                except StorageWriteError as e:
                    logger.critical(
                        "final_flush_failed",
                        error=str(e),
                        pending_rows=self._writer.buffer_size,
                    )
                    raise
                finally:
                    logger.info("collector_stopped", **self.stats)

    async def _duration_guard(self, duration: float) -> None:
        """Automatically stop after the specified duration."""
        await asyncio.sleep(duration)
        logger.info("duration_limit_reached", duration=duration)
        self._shutdown()

    def _shutdown(self) -> None:
        """Initiate graceful shutdown."""
        logger.info("shutdown_requested")
        self._running = False
