"""
Data drift detection for monitoring deployed models.

Implements:
- Kolmogorov-Smirnov test per feature
- Population Stability Index (PSI)
- Summary reporting with alerting thresholds
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DriftResult:
    """Result of a drift test on a single feature."""
    feature: str
    ks_statistic: float
    ks_pvalue: float
    psi: float
    drifted: bool


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index.

    PSI = Σ (p_i - q_i) × ln(p_i / q_i)

    Thresholds:
    - PSI < 0.1: No significant drift
    - 0.1 ≤ PSI < 0.25: Moderate drift
    - PSI ≥ 0.25: Significant drift

    Args:
        reference: Reference (training) distribution
        current: Current (production) distribution
        n_bins: Number of bins for discretization
    """
    # Create bins from reference distribution
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    if len(ref_clean) < n_bins or len(cur_clean) < n_bins:
        return 0.0

    bin_edges = np.percentile(ref_clean, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts = np.histogram(ref_clean, bins=bin_edges)[0]
    cur_counts = np.histogram(cur_clean, bins=bin_edges)[0]

    # Add small epsilon to avoid division by zero
    eps = 1e-6
    ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * n_bins)
    cur_pct = (cur_counts + eps) / (cur_counts.sum() + eps * n_bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

    return float(psi)


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    ks_threshold: float = 0.05,
    psi_threshold: float = 0.1,
) -> list[DriftResult]:
    """
    Run drift detection on all features.

    Args:
        reference_df: Reference (training) data
        current_df: Current (production) data
        feature_cols: Columns to check (default: all numeric)
        ks_threshold: p-value threshold for KS test
        psi_threshold: PSI threshold for drift alert

    Returns:
        List of DriftResult for each feature
    """
    if feature_cols is None:
        feature_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()

    results = []

    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        # Ensure we have numpy ndarrays (avoid pandas ExtensionArray types)
        ref = reference_df[col].dropna().to_numpy()
        cur = current_df[col].dropna().to_numpy()

        if len(ref) < 10 or len(cur) < 10:
            continue

        # KS test
        # KS test may return an object with .statistic and .pvalue or a tuple
        ks_res = stats.ks_2samp(ref, cur)
        if hasattr(ks_res, 'statistic'):
            ks_stat = float(ks_res.statistic)
            ks_pval = float(ks_res.pvalue)
        else:
            # Fallback for older scipy returning a tuple
            ks_stat, ks_pval = map(float, ks_res)

        # PSI
        psi = compute_psi(ref, cur)

        drifted = ks_pval < ks_threshold or psi > psi_threshold

        results.append(DriftResult(
            feature=col,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            psi=psi,
            drifted=drifted,
        ))

        if drifted:
            logger.warning(
                "drift_detected",
                feature=col,
                ks_stat=round(ks_stat, 4),
                ks_pval=round(ks_pval, 6),
                psi=round(psi, 4),
            )

    n_drifted = sum(1 for r in results if r.drifted)
    logger.info(
        "drift_check_complete",
        n_features=len(results),
        n_drifted=n_drifted,
        drift_ratio=round(n_drifted / max(len(results), 1), 2),
    )

    return results


def drift_summary(results: list[DriftResult]) -> pd.DataFrame:
    """Convert drift results to a summary DataFrame."""
    records = [
        {
            "feature": r.feature,
            "ks_statistic": round(r.ks_statistic, 4),
            "ks_pvalue": round(r.ks_pvalue, 6),
            "psi": round(r.psi, 4),
            "drifted": r.drifted,
        }
        for r in results
    ]
    return pd.DataFrame(records).sort_values("psi", ascending=False).reset_index(drop=True)
