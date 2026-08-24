"""
Label generation for order book prediction.

Creates target labels for supervised learning by classifying future price
movements into UP / DOWN / NEUTRAL categories.

Key principles:
- Labels use FUTURE mid-price data (this is expected and correct)
- Features must NEVER use future data (enforced in pipeline.py)
- Threshold can be fixed or adaptively calibrated
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class Label(IntEnum):
    """Numeric labels for price movement direction."""
    DOWN = 0
    NEUTRAL = 1
    UP = 2


LABEL_NAMES = {Label.DOWN: "DOWN", Label.NEUTRAL: "NEUTRAL", Label.UP: "UP"}


def compute_future_return(
    mid_price: pd.Series,
    horizon: int = 50,
) -> pd.Series:
    """
    Compute the future mid-price return at horizon H.

    future_return_t = (mid_{t+H} - mid_t) / mid_t

    The last H rows will be NaN (no future data available).
    """
    future_price = mid_price.shift(-horizon)
    return ((future_price - mid_price) / mid_price).rename("future_return")


def classify_fixed_threshold(
    future_return: pd.Series,
    threshold_pct: float = 0.01,
) -> pd.Series:
    """
    Classify returns using a fixed percentage threshold.

    Args:
        future_return: Series of future returns (as fractions, not %)
        threshold_pct: Threshold in percentage (0.01 = 0.01%)

    Returns:
        Series of integer labels (0=DOWN, 1=NEUTRAL, 2=UP)
    """
    threshold = threshold_pct / 100.0  # Convert % to fraction

    labels = pd.Series(Label.NEUTRAL, index=future_return.index, dtype=int)
    labels[future_return > threshold] = Label.UP
    labels[future_return < -threshold] = Label.DOWN
    labels[future_return.isna()] = -1  # Mark unlabelable rows

    return labels.rename("label")


def classify_adaptive_threshold(
    future_return: pd.Series,
    target_neutral_ratio: float = 0.40,
) -> tuple[pd.Series, float]:
    """
    Adaptively calibrate the threshold to achieve a target neutral class ratio.

    Uses the empirical distribution of absolute returns to find the threshold
    that yields approximately the desired proportion of NEUTRAL labels.

    Args:
        future_return: Series of future returns
        target_neutral_ratio: Desired proportion of NEUTRAL class (0.0-1.0)

    Returns:
        Tuple of (label Series, calibrated threshold as fraction)
    """
    valid_returns = future_return.dropna()
    abs_returns = valid_returns.abs()

    # Find the threshold at the target_neutral_ratio quantile
    threshold = abs_returns.quantile(target_neutral_ratio)

    logger.info(
        "adaptive_threshold_calibrated",
        threshold_fraction=f"{threshold:.8f}",
        threshold_pct=f"{threshold * 100:.6f}%",
        target_neutral_ratio=target_neutral_ratio,
    )

    labels = pd.Series(Label.NEUTRAL, index=future_return.index, dtype=int)
    labels[future_return > threshold] = Label.UP
    labels[future_return < -threshold] = Label.DOWN
    labels[future_return.isna()] = -1

    return labels.rename("label"), threshold


def generate_labels(
    mid_price: pd.Series,
    horizon: int = 50,
    threshold_pct: float = 0.01,
    adaptive: bool = True,
    target_neutral_ratio: float = 0.40,
) -> tuple[pd.DataFrame, dict]:
    """
    Full label generation pipeline.

    Args:
        mid_price: Series of mid-prices
        horizon: Number of ticks into the future
        threshold_pct: Fixed threshold (used if adaptive=False)
        adaptive: Whether to use adaptive thresholding
        target_neutral_ratio: Target NEUTRAL ratio (if adaptive=True)

    Returns:
        Tuple of:
        - DataFrame with columns: future_return, label
        - Dict with metadata (threshold, class distribution)
    """
    future_return = compute_future_return(mid_price, horizon=horizon)

    if adaptive:
        labels, threshold = classify_adaptive_threshold(
            future_return,
            target_neutral_ratio=target_neutral_ratio,
        )
    else:
        labels = classify_fixed_threshold(future_return, threshold_pct=threshold_pct)
        threshold = threshold_pct / 100.0

    # Compute class distribution (excluding unlabelable rows)
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]

    distribution = {
        LABEL_NAMES[Label(v)]: count
        for v, count in valid_labels.value_counts().sort_index().items()
    }
    total = valid_labels.shape[0]
    ratios = {
        f"{LABEL_NAMES[Label(v)]}_ratio": count / total
        for v, count in valid_labels.value_counts().sort_index().items()
    }

    metadata = {
        "horizon_ticks": horizon,
        "threshold_fraction": threshold,
        "threshold_pct": threshold * 100,
        "adaptive": adaptive,
        "total_samples": total,
        "unlabelable_samples": (~valid_mask).sum(),
        **distribution,
        **ratios,
    }

    logger.info("labels_generated", **metadata)

    result = pd.DataFrame({
        "future_return": future_return,
        "label": labels,
    })

    return result, metadata
