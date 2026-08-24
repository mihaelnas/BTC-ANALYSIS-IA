"""
Feature pipeline orchestration.

End-to-end pipeline:
1. Load raw order book snapshots from Parquet
2. Compute microstructure features
3. Generate labels
4. Apply rolling z-score normalization (no future leakage)
5. Save processed dataset

Designed to be run as a batch process after data collection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import structlog

from config.settings import get_settings, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.features.microstructure import compute_all_features
from src.features.labels import generate_labels

logger = structlog.get_logger(__name__)


def load_raw_snapshots(
    data_dir: str | Path | None = None,
    max_files: int | None = None,
) -> pd.DataFrame:
    """
    Load all raw order book Parquet files from the data directory.

    Files are sorted by name (which is chronological due to naming convention)
    and concatenated into a single DataFrame.

    Args:
        data_dir: Directory containing raw Parquet files (default: data/raw/)
        max_files: Maximum number of files to load (for development/testing)

    Returns:
        Concatenated DataFrame sorted by timestamp
    """
    data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR

    parquet_files = sorted(data_dir.glob("ob_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {data_dir}")

    if max_files:
        parquet_files = parquet_files[:max_files]

    logger.info("loading_raw_data", n_files=len(parquet_files), dir=str(data_dir))

    dfs = []
    for f in parquet_files:
        df = pq.read_table(f).to_pandas()
        dfs.append(df)
        logger.debug("loaded_file", path=f.name, rows=len(df))

    result = pd.concat(dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "raw_data_loaded",
        total_rows=len(result),
        time_span_hours=round(
            (result["timestamp"].iloc[-1] - result["timestamp"].iloc[0]) / 3600, 2
        ),
        columns=len(result.columns),
    )

    return result


def rolling_zscore_normalize(
    features: pd.DataFrame,
    window: int = 5000,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply rolling z-score normalization to features.

    For each feature:
        z_t = (x_t - μ_{t-w:t}) / σ_{t-w:t}

    This uses ONLY past data in the window — no future leakage.

    Args:
        features: DataFrame of features to normalize
        window: Rolling window size
        exclude_cols: Columns to exclude from normalization

    Returns:
        Normalized DataFrame (same shape and index)
    """
    if exclude_cols is None:
        exclude_cols = ["timestamp", "mid_price", "microprice", "label", "future_return"]

    cols_to_normalize = [c for c in features.columns if c not in exclude_cols]

    result = features.copy()

    for col in cols_to_normalize:
        series = features[col]
        rolling_mean = series.rolling(window=window, min_periods=max(10, window // 10)).mean()
        rolling_std = series.rolling(window=window, min_periods=max(10, window // 10)).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        result[col] = (series - rolling_mean) / rolling_std

    n_normalized = len(cols_to_normalize)
    logger.info(
        "normalization_applied",
        n_cols_normalized=n_normalized,
        window=window,
    )

    return result


def run_feature_pipeline(
    raw_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_files: int | None = None,
    normalize: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Execute the full feature engineering pipeline.

    Steps:
    1. Load raw snapshots
    2. Compute microstructure features
    3. Generate labels
    4. Normalize features (rolling z-score)
    5. Drop rows with NaN (warm-up period)
    6. Save to Parquet

    Args:
        raw_data_dir: Directory with raw Parquet files
        output_dir: Directory for processed output
        max_files: Max files to load (for development)
        normalize: Whether to apply z-score normalization

    Returns:
        Tuple of (processed DataFrame, metadata dict)
    """
    settings = get_settings()
    output_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Load ──────────────────────────────────────────────────
    raw_df = load_raw_snapshots(raw_data_dir, max_files=max_files)

    # ─── Step 2: Features ──────────────────────────────────────────────
    features = compute_all_features(
        raw_df,
        n_levels=settings.collector.n_levels,
        obi_levels=settings.features.obi_levels,
        ofi_levels=5,
        return_periods=[1, 5, 10, 50],
        spread_vol_window=100,
    )

    # ─── Step 3: Labels ───────────────────────────────────────────────
    label_df, label_metadata = generate_labels(
        mid_price=features["mid_price"],
        horizon=settings.features.horizon_ticks,
        threshold_pct=settings.features.label_threshold_pct,
        adaptive=settings.features.adaptive_threshold,
        target_neutral_ratio=settings.features.target_neutral_ratio,
    )

    features = pd.concat([features, label_df], axis=1)

    # ─── Step 4: Normalize ─────────────────────────────────────────────
    if normalize:
        features = rolling_zscore_normalize(
            features,
            window=settings.features.rolling_window,
        )

    # ─── Step 5: Clean ─────────────────────────────────────────────────
    # Drop rows with NaN (warm-up period + unlabelable trailing rows)
    rows_before = len(features)
    features = features.dropna(subset=[c for c in features.columns if c != "future_return"])
    features = features[features["label"] >= 0].reset_index(drop=True)
    rows_after = len(features)

    logger.info(
        "rows_cleaned",
        before=rows_before,
        after=rows_after,
        dropped=rows_before - rows_after,
        dropped_pct=round((rows_before - rows_after) / rows_before * 100, 1),
    )

    # ─── Step 6: Save ──────────────────────────────────────────────────
    output_path = output_dir / "features_labeled.parquet"
    features.to_parquet(output_path, compression="zstd", index=False)

    metadata = {
        "raw_rows": len(raw_df),
        "processed_rows": len(features),
        "n_features": len(features.columns) - 3,  # exclude timestamp, label, future_return
        "feature_names": [
            c for c in features.columns if c not in ["timestamp", "label", "future_return"]
        ],
        "output_path": str(output_path),
        "normalized": normalize,
        **label_metadata,
    }

    logger.info("pipeline_complete", **metadata)

    return features, metadata
