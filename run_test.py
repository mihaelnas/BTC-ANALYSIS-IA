import sys
import os
sys.path.insert(0, '/home/rajo/Developpement/python/PythonIA')
from src.data.order_book import OrderBook
from tests.test_order_book import make_update

book = OrderBook()
book._bids[-100.5] = 1.0
book._bids[-100.4] = 2.0
book._bids[-100.3] = 3.0
book._asks[100.6] = 1.5
book._asks[100.7] = 2.5
book._asks[100.8] = 3.5
book._last_update_id = 100
book._initialized = True

update = make_update(
    bids=[("100.55", "0.5")],
    asks=[],
    first_id=101,
    final_id=101,
)
book.apply_update(update)

print(f"Book best bid: {book.best_bid}")
