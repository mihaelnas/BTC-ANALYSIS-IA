"""
Tests for the OrderBook class.

Covers:
- Initialization from mock REST data
- Update application (normal, out-of-sequence, gap detection)
- Snapshot extraction
- Crossed book detection
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.order_book import OrderBook


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_snapshot_response(
    bids: list[tuple[str, str]],
    asks: list[tuple[str, str]],
    last_update_id: int = 100,
) -> dict:
    """Create a mock REST API snapshot response."""
    return {
        "lastUpdateId": last_update_id,
        "bids": [[p, q] for p, q in bids],
        "asks": [[p, q] for p, q in asks],
    }


def make_update(
    bids: list[tuple[str, str]],
    asks: list[tuple[str, str]],
    first_id: int = 101,
    final_id: int = 101,
) -> dict:
    """Create a mock depth update."""
    return {
        "e": "depthUpdate",
        "U": first_id,
        "u": final_id,
        "b": [[p, q] for p, q in bids],
        "a": [[p, q] for p, q in asks],
    }


@pytest.fixture
def book() -> OrderBook:
    """Create an empty OrderBook."""
    return OrderBook()


@pytest.fixture
def initialized_book() -> OrderBook:
    """Create an OrderBook initialized with mock data."""
    book = OrderBook()
    # Manually initialize without REST call
    book._bids[-100.5] = 1.0  # neg key for descending order
    book._bids[-100.4] = 2.0
    book._bids[-100.3] = 3.0
    book._asks[100.6] = 1.5
    book._asks[100.7] = 2.5
    book._asks[100.8] = 3.5
    book._last_update_id = 100
    book._initialized = True
    return book


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestOrderBookBasics:
    def test_empty_book(self, book: OrderBook) -> None:
        assert not book.initialized
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price is None
        assert book.spread is None

    def test_initialized_book_properties(self, initialized_book: OrderBook) -> None:
        assert initialized_book.initialized
        assert initialized_book.last_update_id == 100

        bid = initialized_book.best_bid
        assert bid is not None
        assert bid[0] == 100.5
        assert bid[1] == 1.0

        ask = initialized_book.best_ask
        assert ask is not None
        assert ask[0] == 100.6
        assert ask[1] == 1.5

    def test_mid_price(self, initialized_book: OrderBook) -> None:
        mid = initialized_book.mid_price
        assert mid is not None
        assert mid == pytest.approx(100.55)

    def test_spread(self, initialized_book: OrderBook) -> None:
        spread = initialized_book.spread
        assert spread is not None
        assert spread == pytest.approx(0.1)


class TestOrderBookUpdates:
    def test_apply_valid_update(self, initialized_book: OrderBook) -> None:
        update = make_update(
            bids=[("100.5", "2.0")],  # Update best bid quantity
            asks=[("100.6", "3.0")],  # Update best ask quantity
            first_id=101,
            final_id=101,
        )
        result = initialized_book.apply_update(update)
        assert result is True
        assert initialized_book.last_update_id == 101
        bid = initialized_book.best_bid
        ask = initialized_book.best_ask
        assert bid is not None
        assert ask is not None
        assert bid[1] == 2.0
        assert ask[1] == 3.0

    def test_skip_old_update(self, initialized_book: OrderBook) -> None:
        """Updates with final_id <= lastUpdateId should be silently skipped."""
        update = make_update(
            bids=[("100.5", "5.0")],
            asks=[],
            first_id=50,
            final_id=99,  # Below lastUpdateId=100
        )
        result = initialized_book.apply_update(update)
        assert result is True  # Skipped, not an error
        bid = initialized_book.best_bid
        assert bid is not None
        assert bid[1] == 1.0  # Unchanged

    def test_detect_sequence_gap(self, initialized_book: OrderBook) -> None:
        """Gap between lastUpdateId and first_id should be detected."""
        update = make_update(
            bids=[],
            asks=[],
            first_id=200,  # Gap: expected 101, got 200
            final_id=200,
        )
        result = initialized_book.apply_update(update)
        assert result is False

    def test_remove_price_level(self, initialized_book: OrderBook) -> None:
        """Quantity=0 means remove the price level."""
        update = make_update(
            bids=[("100.5", "0")],  # Remove best bid
            asks=[],
            first_id=101,
            final_id=101,
        )
        initialized_book.apply_update(update)
        # Best bid should now be 100.4
        assert initialized_book.best_bid is not None
        bid = initialized_book.best_bid
        assert bid is not None
        assert bid[0] == 100.4

    def test_add_new_price_level(self, initialized_book: OrderBook) -> None:
        """New price levels should be inserted in order."""
        update = make_update(
            bids=[("100.55", "0.5")],  # New best bid
            asks=[],
            first_id=101,
            final_id=101,
        )
        initialized_book.apply_update(update)
        assert initialized_book.best_bid is not None
        bid = initialized_book.best_bid
        assert bid is not None
        assert bid[0] == 100.55
        assert bid[1] == 0.5

    def test_update_before_init(self, book: OrderBook) -> None:
        """Updates before initialization should return False."""
        update = make_update(bids=[], asks=[], first_id=1, final_id=1)
        result = book.apply_update(update)
        assert result is False


class TestOrderBookSnapshot:
    def test_snapshot_basic(self, initialized_book: OrderBook) -> None:
        snap = initialized_book.snapshot(n_levels=3)
        assert snap is not None
        assert "timestamp" in snap
        assert "mid_price" in snap
        assert "spread" in snap
        assert "microprice" in snap

        # Check bid levels
        assert snap["bid_price_1"] == 100.5
        assert snap["bid_qty_1"] == 1.0
        assert snap["bid_price_2"] == 100.4
        assert snap["bid_qty_2"] == 2.0

        # Check ask levels
        assert snap["ask_price_1"] == 100.6
        assert snap["ask_qty_1"] == 1.5

    def test_snapshot_padding(self, initialized_book: OrderBook) -> None:
        """If fewer levels than requested, pad with NaN."""
        snap = initialized_book.snapshot(n_levels=5)
        assert snap is not None
        # We have 3 bid levels, so level 4 and 5 should be NaN
        assert np.isnan(snap["bid_price_4"])
        assert np.isnan(snap["bid_qty_4"])

    def test_snapshot_not_initialized(self, book: OrderBook) -> None:
        assert book.snapshot() is None

    def test_microprice_calculation(self, initialized_book: OrderBook) -> None:
        snap = initialized_book.snapshot(n_levels=3)
        assert snap is not None
        # microprice = (P_ask × V_bid + P_bid × V_ask) / (V_bid + V_ask)
        # = (100.6 × 1.0 + 100.5 × 1.5) / (1.0 + 1.5)
        # = (100.6 + 150.75) / 2.5
        # = 251.35 / 2.5
        # = 100.54
        expected = (100.6 * 1.0 + 100.5 * 1.5) / (1.0 + 1.5)
        assert snap["microprice"] == pytest.approx(expected)


class TestCrossedBook:
    def test_not_crossed(self, initialized_book: OrderBook) -> None:
        assert not initialized_book.needs_resync()

    def test_crossed_book_detected(self) -> None:
        book = OrderBook()
        book._bids[-100.6] = 1.0  # bid at 100.6
        book._asks[100.5] = 1.0   # ask at 100.5 — crossed!
        book._initialized = True
        assert book.needs_resync()

    def test_empty_side_needs_resync(self) -> None:
        book = OrderBook()
        book._bids[-100.5] = 1.0
        book._initialized = True
        # No asks → needs resync
        assert book.needs_resync()
