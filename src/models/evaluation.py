"""
Evaluation framework with walk-forward validation.

Implements:
- Walk-forward (expanding/sliding window) temporal cross-validation
- Probabilistic metrics: log-loss, multi-class Brier score
- Classification metrics: accuracy, F1-score, confusion matrix
- Calibration analysis: reliability diagrams
- Statistical comparison between models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
import structlog
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report,
)

from src.features.labels import LABEL_NAMES, Label

logger = structlog.get_logger(__name__)


# ─── Protocol for models ──────────────────────────────────────────────────────

class ProbabilisticModel(Protocol):
    """Interface that all models must satisfy for evaluation."""

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...


# ─── Metrics ──────────────────────────────────────────────────────────────────

def multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Multi-class Brier score.

    BS = (1/N) × Σ_i Σ_k (p_ik - y_ik)²

    where y_ik is 1 if sample i belongs to class k, else 0.
    Lower is better. Range: [0, 2] for 3 classes.
    """
    n_classes = y_proba.shape[1]
    y_onehot = np.eye(n_classes)[y_true.astype(int)]
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


@dataclass
class EvaluationResult:
    """Container for evaluation metrics on a single fold/split."""

    # Identification
    fold: int = 0
    train_start: float = 0.0
    train_end: float = 0.0
    test_start: float = 0.0
    test_end: float = 0.0
    n_train: int = 0
    n_test: int = 0

    # Probabilistic metrics
    log_loss_val: float = 0.0
    brier_score: float = 0.0

    # Classification metrics
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0

    # Per-class F1
    f1_per_class: dict[str, float] = field(default_factory=dict)

    # Confusion matrix
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=int))

    # Predictions and labels (for detailed analysis)
    y_true: np.ndarray = field(default_factory=lambda: np.array([]))
    y_proba: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))


def evaluate_predictions(
    y_true: np.ndarray[Any, Any],
    y_proba: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any] | None = None,
    fold: int = 0,
    train_size: int = 0,
    timestamps: tuple[float, float, float, float] = (0, 0, 0, 0),
) -> EvaluationResult:
    """
    Compute all evaluation metrics for a set of predictions.

    Args:
        y_true: True labels (integer encoded)
        y_proba: Predicted probabilities (n_samples × n_classes)
        y_pred: Predicted labels (optional, derived from y_proba if None)
        fold: Fold number for identification
        train_size: Number of training samples
        timestamps: (train_start, train_end, test_start, test_end)

    Returns:
        EvaluationResult with all metrics computed
    """
    if y_pred is None:
        y_pred = np.argmax(y_proba, axis=1)

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Clamp probabilities to avoid log(0)
    y_proba_safe = np.clip(y_proba, 1e-15, 1 - 1e-15)
    y_proba_safe = y_proba_safe / y_proba_safe.sum(axis=1, keepdims=True)

    result = EvaluationResult(
        fold=fold,
        train_start=timestamps[0],
        train_end=timestamps[1],
        test_start=timestamps[2],
        test_end=timestamps[3],
        n_train=train_size,
        n_test=len(y_true),
        log_loss_val=float(log_loss(y_true, y_proba_safe, labels=[0, 1, 2])),
        brier_score=multiclass_brier_score(y_true, y_proba_safe),
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        confusion=confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
        y_true=y_true,
        y_proba=y_proba_safe,
        y_pred=y_pred,
    )

    # Per-class F1
    f1_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    # Ensure f1_per is indexable (f1_score may return a scalar if only one label)
    f1_per = np.atleast_1d(f1_per)
    result.f1_per_class = {
        LABEL_NAMES[Label(i)]: float(f1_per[i]) for i in range(len(f1_per))
    }

    return result


# ─── Walk-Forward Validation ──────────────────────────────────────────────────

def walk_forward_split(
    timestamps: np.ndarray[Any, Any],
    train_window_hours: float = 24.0,
    test_window_hours: float = 4.0,
    step_hours: float = 4.0,
    min_train_samples: int = 10000,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward (temporal) train/test splits.

    Unlike random cross-validation, this respects temporal ordering:
    - Train on [t_start, t_start + train_window]
    - Test on [t_start + train_window, t_start + train_window + test_window]
    - Advance by step_hours

    Args:
        timestamps: Array of Unix timestamps (must be sorted)
        train_window_hours: Size of training window in hours
        test_window_hours: Size of test window in hours
        step_hours: Step size for advancing the window
        min_train_samples: Minimum training samples per fold

    Returns:
        List of (train_indices, test_indices) tuples
    """
    train_window_sec = train_window_hours * 3600
    test_window_sec = test_window_hours * 3600
    step_sec = step_hours * 3600

    t_min = timestamps[0]
    t_max = timestamps[-1]

    splits = []
    t_start = t_min

    while True:
        train_end = t_start + train_window_sec
        test_end = train_end + test_window_sec

        if test_end > t_max:
            break

        train_mask = (timestamps >= t_start) & (timestamps < train_end)
        test_mask = (timestamps >= train_end) & (timestamps < test_end)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) >= min_train_samples and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

        t_start += step_sec

    logger.info(
        "walk_forward_splits",
        n_splits=len(splits),
        train_window_hours=train_window_hours,
        test_window_hours=test_window_hours,
        step_hours=step_hours,
    )

    return splits


# ─── Aggregation & Reporting ──────────────────────────────────────────────────

def aggregate_results(results: list[EvaluationResult]) -> pd.DataFrame:
    """
    Aggregate evaluation results across folds into a summary DataFrame.
    """
    records = []
    for r in results:
        record = {
            "fold": r.fold,
            "n_train": r.n_train,
            "n_test": r.n_test,
            "log_loss": r.log_loss_val,
            "brier_score": r.brier_score,
            "accuracy": r.accuracy,
            "f1_macro": r.f1_macro,
            "f1_weighted": r.f1_weighted,
        }
        for cls_name, f1_val in r.f1_per_class.items():
            record[f"f1_{cls_name}"] = f1_val
        records.append(record)

    df = pd.DataFrame(records)

    # Add summary row
    summary = df.describe().loc["mean"].to_dict()
    summary["fold"] = "MEAN"
    std_row = df.describe().loc["std"].to_dict()
    std_row["fold"] = "STD"

    summary_df = pd.concat([df, pd.DataFrame([summary, std_row])], ignore_index=True)

    return summary_df


def plot_walk_forward_results(
    results: list[EvaluationResult],
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot walk-forward evaluation metrics over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Walk-Forward Validation Results", fontsize=14, fontweight="bold")

    folds = [r.fold for r in results]

    # Log Loss
    ax = axes[0, 0]
    ax.plot(folds, [r.log_loss_val for r in results], "o-", color="#e74c3c", linewidth=2)
    ax.set_title("Log Loss (lower is better)")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Log Loss")
    ax.grid(True, alpha=0.3)

    # Brier Score
    ax = axes[0, 1]
    ax.plot(folds, [r.brier_score for r in results], "s-", color="#3498db", linewidth=2)
    ax.set_title("Brier Score (lower is better)")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Brier Score")
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1, 0]
    ax.plot(folds, [r.accuracy for r in results], "^-", color="#2ecc71", linewidth=2)
    ax.set_title("Accuracy")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)

    # F1 Macro
    ax = axes[1, 1]
    ax.plot(folds, [r.f1_macro for r in results], "D-", color="#9b59b6", linewidth=2)
    ax.set_title("F1 Score (Macro)")
    ax.set_xlabel("Fold")
    ax.set_ylabel("F1 Macro")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("plot_saved", path=str(save_path))

    return fig


def plot_confusion_matrix(
    result: EvaluationResult,
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """Plot a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = [LABEL_NAMES[Label(i)] for i in range(3)]
    cm_normalized = result.confusion.astype(float) / result.confusion.sum(axis=1, keepdims=True)

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix (Fold {result.fold})", fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_calibration(
    result: EvaluationResult,
    n_bins: int = 10,
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """
    Reliability diagram (calibration plot) for each class.

    A well-calibrated model should produce points close to the diagonal.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Calibration Diagrams", fontsize=14, fontweight="bold")

    for cls_idx in range(3):
        ax = axes[cls_idx]
        cls_name = LABEL_NAMES[Label(cls_idx)]

        # Binary: is this class vs not
        y_binary = (result.y_true == cls_idx).astype(int)
        p_class = result.y_proba[:, cls_idx]

        # Bin by predicted probability
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_true_fractions = []

        for i in range(n_bins):
            mask = (p_class >= bin_edges[i]) & (p_class < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_centers.append(p_class[mask].mean())
                bin_true_fractions.append(y_binary[mask].mean())

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(bin_centers, bin_true_fractions, "o-", color="#e74c3c", linewidth=2, label=cls_name)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Class: {cls_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
