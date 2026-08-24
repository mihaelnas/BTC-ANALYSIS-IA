"""
Tests for feature engineering modules.

Covers:
- Microstructure feature computation
- Label generation (fixed and adaptive thresholds)
- Rolling z-score normalization
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.microstructure import (
    compute_mid_price,
    compute_spread,
    compute_microprice,
    compute_obi,
    compute_multi_level_obi,
    compute_ofi,
    compute_cumulative_depth,
    compute_returns,
    compute_all_features,
)
from src.features.labels import (
    compute_future_return,
    classify_fixed_threshold,
    classify_adaptive_threshold,
    generate_labels,
    Label,
)
from src.features.pipeline import rolling_zscore_normalize


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ob_data() -> pd.DataFrame:
    """Create a sample order book DataFrame with 5 levels."""
    np.random.seed(42)
    n = 200

    base_price = 100.0
    data = {"timestamp": np.arange(n, dtype=float)}

    # Generate prices and quantities for 5 levels
    for i in range(1, 6):
        data[f"bid_price_{i}"] = base_price - 0.01 * i + np.random.randn(n) * 0.001
        data[f"bid_qty_{i}"] = np.abs(np.random.randn(n) * 2 + 5)
        data[f"ask_price_{i}"] = base_price + 0.01 * i + np.random.randn(n) * 0.001
        data[f"ask_qty_{i}"] = np.abs(np.random.randn(n) * 2 + 5)

    df = pd.DataFrame(data)

    # Add mid_price and spread
    df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]

    return df


@pytest.fixture
def sample_mid_price() -> pd.Series:
    """Generate a sample mid-price series with trend and noise."""
    np.random.seed(42)
    n = 500
    # Random walk with small drift
    returns = np.random.randn(n) * 0.0001
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, name="mid_price")


# ─── Microstructure Feature Tests ─────────────────────────────────────────────

class TestMidPrice:
    def test_computation(self, sample_ob_data: pd.DataFrame) -> None:
        mid = compute_mid_price(sample_ob_data)
        assert len(mid) == len(sample_ob_data)
        # Mid-price should be between best bid and best ask
        assert (mid >= sample_ob_data["bid_price_1"]).all()
        assert (mid <= sample_ob_data["ask_price_1"]).all()


class TestSpread:
    def test_positive_spread(self, sample_ob_data: pd.DataFrame) -> None:
        spread = compute_spread(sample_ob_data)
        assert (spread > 0).all()  # Spread should always be positive

    def test_spread_matches_prices(self, sample_ob_data: pd.DataFrame) -> None:
        spread = compute_spread(sample_ob_data)
        expected = sample_ob_data["ask_price_1"] - sample_ob_data["bid_price_1"]
        pd.testing.assert_series_equal(spread, expected, check_names=False)


class TestMicroprice:
    def test_between_bid_ask(self, sample_ob_data: pd.DataFrame) -> None:
        microprice = compute_microprice(sample_ob_data)
        assert (microprice >= sample_ob_data["bid_price_1"]).all()
        assert (microprice <= sample_ob_data["ask_price_1"]).all()

    def test_equal_volumes_gives_midprice(self) -> None:
        """When bid and ask volumes are equal, microprice = mid-price."""
        df = pd.DataFrame({
            "bid_price_1": [100.0],
            "ask_price_1": [101.0],
            "bid_qty_1": [10.0],
            "ask_qty_1": [10.0],
        })
        microprice = compute_microprice(df)
        assert microprice.iloc[0] == pytest.approx(100.5)


class TestOBI:
    def test_obi_range(self, sample_ob_data: pd.DataFrame) -> None:
        obi = compute_obi(sample_ob_data, level=1)
        assert (obi >= -1).all()
        assert (obi <= 1).all()

    def test_obi_direction(self) -> None:
        """More bid volume → positive OBI."""
        df = pd.DataFrame({
            "bid_qty_1": [10.0],
            "ask_qty_1": [5.0],
        })
        obi = compute_obi(df, level=1)
        assert obi.iloc[0] > 0

        df2 = pd.DataFrame({
            "bid_qty_1": [5.0],
            "ask_qty_1": [10.0],
        })
        obi2 = compute_obi(df2, level=1)
        assert obi2.iloc[0] < 0

    def test_multi_level_obi(self, sample_ob_data: pd.DataFrame) -> None:
        obi_df = compute_multi_level_obi(sample_ob_data, levels=[1, 3, 5])
        assert "obi_1" in obi_df.columns
        assert "obi_3" in obi_df.columns
        assert "obi_5" in obi_df.columns
        assert "obi_weighted" in obi_df.columns


class TestOFI:
    def test_ofi_shape(self, sample_ob_data: pd.DataFrame) -> None:
        ofi_df = compute_ofi(sample_ob_data, n_levels=3)
        assert len(ofi_df) == len(sample_ob_data)
        assert "ofi_1" in ofi_df.columns
        assert "ofi_total" in ofi_df.columns

    def test_ofi_first_row_nan(self, sample_ob_data: pd.DataFrame) -> None:
        """First row of OFI should be NaN (no previous to diff)."""
        ofi_df = compute_ofi(sample_ob_data, n_levels=1)
        assert pd.isna(ofi_df["ofi_1"].iloc[0])


class TestCumulativeDepth:
    def test_depth_positive(self, sample_ob_data: pd.DataFrame) -> None:
        depth = compute_cumulative_depth(sample_ob_data, n_levels=5)
        assert (depth["depth_bid"] > 0).all()
        assert (depth["depth_ask"] > 0).all()
        assert (depth["depth_total"] > 0).all()

    def test_book_pressure_range(self, sample_ob_data: pd.DataFrame) -> None:
        depth = compute_cumulative_depth(sample_ob_data, n_levels=5)
        assert (depth["book_pressure"] >= 0).all()
        assert (depth["book_pressure"] <= 1).all()


class TestReturns:
    def test_returns_shape(self, sample_ob_data: pd.DataFrame) -> None:
        returns = compute_returns(sample_ob_data, periods=[1, 5])
        assert "return_1" in returns.columns
        assert "return_5" in returns.columns
        assert len(returns) == len(sample_ob_data)

    def test_return_1_first_nan(self, sample_ob_data: pd.DataFrame) -> None:
        returns = compute_returns(sample_ob_data, periods=[1])
        assert pd.isna(returns["return_1"].iloc[0])


class TestAllFeatures:
    def test_compute_all(self, sample_ob_data: pd.DataFrame) -> None:
        features = compute_all_features(
            sample_ob_data,
            n_levels=5,
            obi_levels=[1, 3, 5],
            ofi_levels=3,
        )
        assert len(features) == len(sample_ob_data)
        assert "mid_price" in features.columns
        assert "obi_1" in features.columns
        assert "ofi_1" in features.columns
        assert "depth_bid" in features.columns
        assert "book_pressure" in features.columns


# ─── Label Tests ──────────────────────────────────────────────────────────────

class TestLabels:
    def test_future_return(self, sample_mid_price: pd.Series) -> None:
        ret = compute_future_return(sample_mid_price, horizon=10)
        assert len(ret) == len(sample_mid_price)
        # Last 10 should be NaN
        assert ret.iloc[-10:].isna().all()
        # Others should be finite
        assert ret.iloc[:-10].notna().all()

    def test_fixed_threshold(self, sample_mid_price: pd.Series) -> None:
        ret = compute_future_return(sample_mid_price, horizon=10)
        labels = classify_fixed_threshold(ret, threshold_pct=0.01)
        valid = labels[labels >= 0]
        assert set(valid.unique()).issubset({0, 1, 2})

    def test_adaptive_threshold(self, sample_mid_price: pd.Series) -> None:
        ret = compute_future_return(sample_mid_price, horizon=10)
        labels, threshold = classify_adaptive_threshold(ret, target_neutral_ratio=0.4)
        assert threshold > 0
        valid = labels[labels >= 0]
        # Check approximate neutral ratio
        neutral_ratio = (valid == Label.NEUTRAL).sum() / len(valid)
        assert abs(neutral_ratio - 0.4) < 0.1  # Allow some tolerance

    def test_generate_labels(self, sample_mid_price: pd.Series) -> None:
        result_df, metadata = generate_labels(
            sample_mid_price,
            horizon=10,
            adaptive=True,
            target_neutral_ratio=0.4,
        )
        assert "future_return" in result_df.columns
        assert "label" in result_df.columns
        assert metadata["total_samples"] > 0
        assert "UP" in metadata
        assert "DOWN" in metadata
        assert "NEUTRAL" in metadata


# ─── Normalization Tests ──────────────────────────────────────────────────────

class TestNormalization:
    def test_rolling_zscore(self) -> None:
        np.random.seed(42)
        df = pd.DataFrame({
            "timestamp": range(200),
            "feature_a": np.random.randn(200) * 10 + 50,
            "feature_b": np.random.randn(200) * 5 + 20,
        })

        normalized = rolling_zscore_normalize(df, window=50, exclude_cols=["timestamp"])

        # Timestamp should be unchanged
        pd.testing.assert_series_equal(normalized["timestamp"], df["timestamp"])

        # Normalized features should have approximately zero mean (after warm-up)
        warm_up = normalized.iloc[50:]
        assert abs(warm_up["feature_a"].mean()) < 1.0
        assert abs(warm_up["feature_b"].mean()) < 1.0
