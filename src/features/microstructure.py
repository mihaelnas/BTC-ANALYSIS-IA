"""
Microstructure feature extraction from order book snapshots.

Computes vectorized features from raw LOB snapshots including:
- Order Book Imbalance (OBI) at multiple levels
- Order Flow Imbalance (OFI) — temporal change in volumes
- Microprice, spread, VWAP
- Cumulative depth and book pressure
- Mid-price returns and spread volatility

All computations are vectorized via NumPy/Pandas for performance.
Strict lookahead prevention: all features use only current and past data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def compute_mid_price(df: pd.DataFrame) -> pd.Series:
    """Mid-price = (best_bid + best_ask) / 2."""
    return (df["bid_price_1"] + df["ask_price_1"]) / 2.0


def compute_spread(df: pd.DataFrame) -> pd.Series:
    """Absolute spread = best_ask - best_bid."""
    return df["ask_price_1"] - df["bid_price_1"]


def compute_relative_spread(df: pd.DataFrame) -> pd.Series:
    """Relative spread = spread / mid_price (in basis points ×10000)."""
    mid = compute_mid_price(df)
    spread = compute_spread(df)
    return (spread / mid) * 10000


def compute_microprice(df: pd.DataFrame) -> pd.Series:
    """
    Volume-weighted microprice.

    microprice = (P_ask × V_bid + P_bid × V_ask) / (V_bid + V_ask)

    This adjusts the mid-price toward the side with less volume,
    reflecting where the price is more likely to move.
    """
    total_qty = df["bid_qty_1"] + df["ask_qty_1"]
    microprice = np.where(
        total_qty > 0,
        (df["ask_price_1"] * df["bid_qty_1"] + df["bid_price_1"] * df["ask_qty_1"]) / total_qty,
        (df["bid_price_1"] + df["ask_price_1"]) / 2.0,
    )
    return pd.Series(microprice, index=df.index, name="microprice")


def compute_obi(df: pd.DataFrame, level: int = 1) -> pd.Series:
    """
    Order Book Imbalance at a specific level.

    OBI_k = (V_bid_k - V_ask_k) / (V_bid_k + V_ask_k)

    Range: [-1, 1]
    - Positive: more bid volume → buy pressure
    - Negative: more ask volume → sell pressure
    """
    bid_col = f"bid_qty_{level}"
    ask_col = f"ask_qty_{level}"

    if bid_col not in df.columns or ask_col not in df.columns:
        return pd.Series(np.nan, index=df.index, name=f"obi_{level}")

    total = df[bid_col] + df[ask_col]
    obi = np.where(total > 0, (df[bid_col] - df[ask_col]) / total, 0.0)
    return pd.Series(obi, index=df.index, name=f"obi_{level}")


def compute_multi_level_obi(
    df: pd.DataFrame,
    levels: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute OBI at multiple levels and a weighted aggregate.

    The weighted OBI gives more importance to levels closer to the best price
    using inverse-level weighting: w_k = 1/k.
    """
    if levels is None:
        levels = [1, 3, 5, 10, 20]

    # Filter to available levels
    max_level = max(
        int(c.split("_")[-1])
        for c in df.columns
        if c.startswith("bid_qty_")
    )
    levels = [l for l in levels if l <= max_level]

    result = pd.DataFrame(index=df.index)

    weights = []
    obi_values = []

    for level in levels:
        obi = compute_obi(df, level)
        result[f"obi_{level}"] = obi
        weights.append(1.0 / level)
        obi_values.append(obi.values)

    # Weighted OBI
    if obi_values:
        weights_arr = np.array(weights)
        weights_arr = weights_arr / weights_arr.sum()
        stacked = np.column_stack(obi_values)
        result["obi_weighted"] = stacked @ weights_arr

    return result


def compute_ofi(df: pd.DataFrame, n_levels: int = 5) -> pd.DataFrame:
    """
    Order Flow Imbalance — temporal change in order book volumes.

    For each level k:
        OFI_k = Δ(bid_vol_k) - Δ(ask_vol_k)

    Positive OFI = net buying flow (bids growing, asks shrinking)
    Negative OFI = net selling flow

    Also computes aggregate OFI across multiple levels.
    """
    result = pd.DataFrame(index=df.index)

    total_ofi = pd.Series(0.0, index=df.index)

    for level in range(1, n_levels + 1):
        bid_col = f"bid_qty_{level}"
        ask_col = f"ask_qty_{level}"

        if bid_col not in df.columns or ask_col not in df.columns:
            break

        delta_bid = df[bid_col].diff()
        delta_ask = df[ask_col].diff()
        ofi = delta_bid - delta_ask

        result[f"ofi_{level}"] = ofi
        total_ofi += ofi.fillna(0)

    result["ofi_total"] = total_ofi

    return result


def compute_cumulative_depth(df: pd.DataFrame, n_levels: int = 20) -> pd.DataFrame:
    """
    Cumulative depth (volume) up to level k for both sides.

    depth_bid_k = Σ V_bid_1..k
    depth_ask_k = Σ V_ask_1..k
    """
    result = pd.DataFrame(index=df.index)

    cum_bid = pd.Series(0.0, index=df.index)
    cum_ask = pd.Series(0.0, index=df.index)

    for level in range(1, n_levels + 1):
        bid_col = f"bid_qty_{level}"
        ask_col = f"ask_qty_{level}"

        if bid_col not in df.columns:
            break

        cum_bid = cum_bid + df[bid_col].fillna(0)
        cum_ask = cum_ask + df[ask_col].fillna(0)

    result["depth_bid"] = cum_bid
    result["depth_ask"] = cum_ask
    result["depth_total"] = cum_bid + cum_ask

    # Book pressure: bid depth / total depth
    total = cum_bid + cum_ask
    result["book_pressure"] = np.where(total > 0, cum_bid / total, 0.5)

    return result


def compute_vwap(df: pd.DataFrame, n_levels: int = 10) -> pd.DataFrame:
    """
    Volume-Weighted Average Price for bid and ask sides.

    VWAP_bid = Σ(P_bid_k × V_bid_k) / Σ(V_bid_k)
    """
    result = pd.DataFrame(index=df.index)

    bid_pv_sum = pd.Series(0.0, index=df.index)
    bid_v_sum = pd.Series(0.0, index=df.index)
    ask_pv_sum = pd.Series(0.0, index=df.index)
    ask_v_sum = pd.Series(0.0, index=df.index)

    for level in range(1, n_levels + 1):
        bp = f"bid_price_{level}"
        bq = f"bid_qty_{level}"
        ap = f"ask_price_{level}"
        aq = f"ask_qty_{level}"

        if bp not in df.columns:
            break

        bid_pv_sum += (df[bp] * df[bq]).fillna(0)
        bid_v_sum += df[bq].fillna(0)
        ask_pv_sum += (df[ap] * df[aq]).fillna(0)
        ask_v_sum += df[aq].fillna(0)

    result["vwap_bid"] = np.where(bid_v_sum > 0, bid_pv_sum / bid_v_sum, np.nan)
    result["vwap_ask"] = np.where(ask_v_sum > 0, ask_pv_sum / ask_v_sum, np.nan)

    return result


def compute_returns(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """
    Log returns of mid-price at various lookback periods.

    return_k = log(mid_t / mid_{t-k})
    """
    if periods is None:
        periods = [1, 5, 10, 50]

    result = pd.DataFrame(index=df.index)
    mid = df["mid_price"] if "mid_price" in df.columns else compute_mid_price(df)

    for k in periods:
        result[f"return_{k}"] = np.log(mid / mid.shift(k))

    return result


def compute_spread_volatility(
    df: pd.DataFrame,
    window: int = 100,
) -> pd.Series:
    """
    Rolling standard deviation of the spread.

    Captures microstructure volatility (regime changes in market making).
    """
    spread = df["spread"] if "spread" in df.columns else compute_spread(df)
    res = spread.rolling(window=window, min_periods=10).std()
    return pd.Series(res, name="spread_volatility")


def compute_all_features(
    df: pd.DataFrame,
    n_levels: int = 20,
    obi_levels: list[int] | None = None,
    ofi_levels: int = 5,
    return_periods: list[int] | None = None,
    spread_vol_window: int = 100,
) -> pd.DataFrame:
    """
    Compute all microstructure features from raw order book snapshots.

    Args:
        df: DataFrame with columns: timestamp, bid_price_1..N, bid_qty_1..N,
            ask_price_1..N, ask_qty_1..N, mid_price, spread, microprice
        n_levels: Total number of bid/ask levels in the data
        obi_levels: Levels at which to compute OBI
        ofi_levels: Number of levels for OFI computation
        return_periods: Lookback periods for log returns
        spread_vol_window: Rolling window for spread volatility

    Returns:
        DataFrame with all computed features (same index as input)
    """
    logger.info(
        "computing_features",
        n_rows=len(df),
        n_levels=n_levels,
    )

    features = pd.DataFrame(index=df.index)

    # Preserve timestamp
    features["timestamp"] = df["timestamp"]

    # ─── Price-based features ──────────────────────────────────────────
    features["mid_price"] = (
        df["mid_price"] if "mid_price" in df.columns else compute_mid_price(df)
    )
    features["spread"] = (
        df["spread"] if "spread" in df.columns else compute_spread(df)
    )
    features["relative_spread"] = compute_relative_spread(df)
    features["microprice"] = (
        df["microprice"] if "microprice" in df.columns else compute_microprice(df)
    )

    # Microprice deviation from mid-price (directional signal)
    features["microprice_deviation"] = features["microprice"] - features["mid_price"]

    # ─── Order Book Imbalance ──────────────────────────────────────────
    obi_df = compute_multi_level_obi(df, levels=obi_levels)
    features = pd.concat([features, obi_df], axis=1)

    # ─── Order Flow Imbalance ──────────────────────────────────────────
    ofi_df = compute_ofi(df, n_levels=ofi_levels)
    features = pd.concat([features, ofi_df], axis=1)

    # ─── Depth features ───────────────────────────────────────────────
    depth_df = compute_cumulative_depth(df, n_levels=n_levels)
    features = pd.concat([features, depth_df], axis=1)

    # ─── VWAP ─────────────────────────────────────────────────────────
    vwap_df = compute_vwap(df, n_levels=min(n_levels, 10))
    features = pd.concat([features, vwap_df], axis=1)

    # ─── Returns ──────────────────────────────────────────────────────
    returns_df = compute_returns(df, periods=return_periods)
    features = pd.concat([features, returns_df], axis=1)

    # ─── Spread volatility ────────────────────────────────────────────
    features["spread_volatility"] = compute_spread_volatility(df, window=spread_vol_window)

    logger.info(
        "features_computed",
        n_features=len(features.columns),
        feature_names=list(features.columns),
    )

    return features
