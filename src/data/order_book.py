"""
Local Order Book manager.

Maintains an in-memory representation of the BTCUSDT order book by:
1. Fetching an initial REST API snapshot
2. Applying incremental WebSocket depth updates
3. Exposing a .snapshot(n_levels) method for feature extraction

Uses SortedDict for efficient O(log n) price-level lookups and updates.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import orjson
import structlog
from sortedcontainers import SortedDict

logger = structlog.get_logger(__name__)


class OrderBook:
    """
    Local order book reconstructed from Binance depth stream.

    The book is stored as two SortedDicts:
    - bids: price → quantity (sorted descending, best bid first)
    - asks: price → quantity (sorted ascending, best ask first)

    Prices and quantities are stored as floats for computation efficiency.
    """

    def __init__(self) -> None:
        # SortedDict with negated keys for bids → highest price first
        self._bids: SortedDict = SortedDict()
        self._asks: SortedDict = SortedDict()
        self._last_update_id: int = 0
        self._initialized: bool = False
        self._snapshot_timestamp: float = 0.0

    @property
    def initialized(self) -> bool:
        """Whether the order book has been initialized with a snapshot."""
        return self._initialized

    @property
    def last_update_id(self) -> int:
        return self._last_update_id

    @property
    def best_bid(self) -> tuple[float, float] | None:
        """Best bid (price, quantity) or None if empty."""
        if not self._bids:
            return None
        # Bids stored with negated keys, so first key is the highest price
        neg_price = list(self._bids.keys())[0]
        return -neg_price, self._bids[neg_price]

    @property
    def best_ask(self) -> tuple[float, float] | None:
        """Best ask (price, quantity) or None if empty."""
        if not self._asks:
            return None
        price = list(self._asks.keys())[0]
        return price, self._asks[price]

    @property
    def mid_price(self) -> float | None:
        """Mid-price = (best_bid + best_ask) / 2."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return (bid[0] + ask[0]) / 2.0

    @property
    def spread(self) -> float | None:
        """Spread = best_ask - best_bid."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return None
        return ask[0] - bid[0]

    async def initialize_from_rest(self, snapshot_url: str) -> None:
        """
        Fetch initial order book snapshot from Binance REST API.

        Args:
            snapshot_url: Full URL for the depth snapshot endpoint.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(snapshot_url)
            response.raise_for_status()
            data = orjson.loads(response.content)

        self._bids.clear()
        self._asks.clear()

        for price_str, qty_str in data["bids"]:
            price = float(price_str)
            qty = float(qty_str)
            if qty > 0:
                self._bids[-price] = qty  # Negate for descending order

        for price_str, qty_str in data["asks"]:
            price = float(price_str)
            qty = float(qty_str)
            if qty > 0:
                self._asks[price] = qty

        self._last_update_id = data["lastUpdateId"]
        self._initialized = True
        self._snapshot_timestamp = time.time()

        logger.info(
            "order_book_initialized",
            last_update_id=self._last_update_id,
            n_bids=len(self._bids),
            n_asks=len(self._asks),
            best_bid=self.best_bid,
            best_ask=self.best_ask,
        )

    def apply_update(self, update: dict[str, Any]) -> bool:
        """
        Apply a depth stream update to the local order book.

        The update must contain:
        - 'U': first update ID in event
        - 'u': final update ID in event
        - 'b': list of [price, qty] bid updates
        - 'a': list of [price, qty] ask updates

        Returns:
            True if the update was applied successfully, False if out of sequence.
        """
        if not self._initialized:
            logger.warning("update_before_init", update_id=update.get("u"))
            return False

        first_update_id = update["U"]
        final_update_id = update["u"]

        # Drop outdated events (before our snapshot)
        if final_update_id <= self._last_update_id:
            return True  # Silently skip, not an error

        # Check for sequence gap
        if first_update_id > self._last_update_id + 1:
            logger.error(
                "sequence_gap_detected",
                expected=self._last_update_id + 1,
                got=first_update_id,
            )
            return False

        # Apply bid updates
        for price_str, qty_str in update.get("b", []):
            price = float(price_str)
            qty = float(qty_str)
            neg_price = -price
            if qty == 0:
                self._bids.pop(neg_price, None)
            else:
                self._bids[neg_price] = qty

        # Apply ask updates
        for price_str, qty_str in update.get("a", []):
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0:
                self._asks.pop(price, None)
            else:
                self._asks[price] = qty

        self._last_update_id = final_update_id
        return True

    def snapshot(self, n_levels: int = 20) -> dict[str, Any] | None:
        """
        Extract a flat snapshot of the top N bid/ask levels.

        Returns a dictionary with:
        - timestamp: current epoch time (float)
        - mid_price, spread, microprice
        - bid_price_1..N, bid_qty_1..N
        - ask_price_1..N, ask_qty_1..N

        Returns None if the book is not initialized or empty.
        """
        if not self._initialized or not self._bids or not self._asks:
            return None

        now = time.time()
        result: dict[str, Any] = {"timestamp": now}

        # Extract bids (already sorted descending via negated keys)
        bid_keys = list(self._bids.keys())[:n_levels]
        for i, neg_price in enumerate(bid_keys, 1):
            result[f"bid_price_{i}"] = -neg_price
            result[f"bid_qty_{i}"] = self._bids[neg_price]

        # Pad missing bid levels with NaN
        for i in range(len(bid_keys) + 1, n_levels + 1):
            result[f"bid_price_{i}"] = float("nan")
            result[f"bid_qty_{i}"] = float("nan")

        # Extract asks (already sorted ascending)
        ask_keys = list(self._asks.keys())[:n_levels]
        for i, price in enumerate(ask_keys, 1):
            result[f"ask_price_{i}"] = price
            result[f"ask_qty_{i}"] = self._asks[price]

        # Pad missing ask levels with NaN
        for i in range(len(ask_keys) + 1, n_levels + 1):
            result[f"ask_price_{i}"] = float("nan")
            result[f"ask_qty_{i}"] = float("nan")

        # Derived fields
        best_bid_price = result["bid_price_1"]
        best_ask_price = result["ask_price_1"]
        best_bid_qty = result["bid_qty_1"]
        best_ask_qty = result["ask_qty_1"]

        result["mid_price"] = (best_bid_price + best_ask_price) / 2.0
        result["spread"] = best_ask_price - best_bid_price

        # Microprice: volume-weighted mid-price
        total_qty = best_bid_qty + best_ask_qty
        if total_qty > 0:
            result["microprice"] = (
                best_ask_price * best_bid_qty + best_bid_price * best_ask_qty
            ) / total_qty
        else:
            result["microprice"] = result["mid_price"]

        return result

    def needs_resync(self) -> bool:
        """
        Check if the order book might need resynchronization.

        This can happen if:
        - Crossed book (best bid >= best ask)
        - Book is empty on either side
        """
        if not self._bids or not self._asks:
            return True

        bid = self.best_bid
        ask = self.best_ask
        if bid is None or ask is None:
            return True

        # Crossed book detection
        if bid[0] >= ask[0]:
            logger.warning(
                "crossed_book_detected",
                best_bid=bid[0],
                best_ask=ask[0],
            )
            return True

        return False

    def __repr__(self) -> str:
        bid = self.best_bid
        ask = self.best_ask
        bid_str = f"bid={bid[0]:.2f}×{bid[1]:.4f}" if bid else "bid=None"
        ask_str = f"ask={ask[0]:.2f}×{ask[1]:.4f}" if ask else "ask=None"
        return (
            f"OrderBook({bid_str}, {ask_str}"
            f", levels={len(self._bids)}/{len(self._asks)}"
            f", init={self._initialized})"
        )
