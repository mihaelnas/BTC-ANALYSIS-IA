"""
Model performance tracking over time.

Monitors:
- Rolling log-loss and accuracy
- Performance degradation alerts
- Comparison with baseline performance
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import structlog
from sklearn.metrics import log_loss, accuracy_score

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceSnapshot:
    """A single performance measurement at a point in time."""
    timestamp: float
    log_loss_val: float
    accuracy: float
    n_samples: int
    class_distribution: dict[int, int] = field(default_factory=dict)


class PerformanceTracker:
    """
    Tracks model performance over sliding windows.

    Maintains a buffer of recent predictions and computes rolling metrics.
    Raises alerts when performance degrades below baseline thresholds.
    """

    def __init__(
        self,
        window_size: int = 10000,
        baseline_log_loss: float = 1.0986,  # ln(3) — random baseline
        alert_threshold_pct: float = 10.0,
    ) -> None:
        """
        Args:
            window_size: Number of recent predictions to track
            baseline_log_loss: Baseline log-loss to compare against
            alert_threshold_pct: Percentage degradation that triggers an alert
        """
        self._window_size = window_size
        self._baseline_log_loss = baseline_log_loss
        self._alert_threshold_pct = alert_threshold_pct

        self._y_true_buffer: deque[int] = deque(maxlen=window_size)
        self._y_proba_buffer: deque[np.ndarray] = deque(maxlen=window_size)

        self._history: list[PerformanceSnapshot] = []

    def add_prediction(
        self,
        y_true: int,
        y_proba: np.ndarray,
        timestamp: float | None = None,
    ) -> None:
        """Add a single prediction to the tracking buffer."""
        self._y_true_buffer.append(y_true)
        self._y_proba_buffer.append(y_proba)

    def add_batch(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        timestamp: float | None = None,
    ) -> None:
        """Add a batch of predictions."""
        for yt, yp in zip(y_true, y_proba):
            self._y_true_buffer.append(int(yt))
            self._y_proba_buffer.append(yp)

    def compute_current_metrics(self) -> PerformanceSnapshot | None:
        """
        Compute metrics on the current buffer window.

        Returns None if buffer is too small.
        """
        if len(self._y_true_buffer) < 100:
            return None

        y_true = np.array(list(self._y_true_buffer))
        y_proba = np.array(list(self._y_proba_buffer))

        y_pred = np.argmax(y_proba, axis=1)

        snapshot = PerformanceSnapshot(
            timestamp=float(datetime.now().timestamp()),
            log_loss_val=float(log_loss(y_true, np.clip(y_proba, 1e-15, 1 - 1e-15), labels=[0, 1, 2])),
            accuracy=float(accuracy_score(y_true, y_pred)),
            n_samples=int(len(y_true)),
            class_distribution={
                int(c): int(n) for c, n in zip(*np.unique(y_true, return_counts=True))
            },
        )

        self._history.append(snapshot)

        # Check for performance degradation
        if snapshot.log_loss_val > self._baseline_log_loss:
            logger.warning(
                "performance_worse_than_baseline",
                current_log_loss=round(snapshot.log_loss_val, 4),
                baseline_log_loss=round(self._baseline_log_loss, 4),
            )

        return snapshot

    def get_history_df(self) -> pd.DataFrame:
        """Get performance history as a DataFrame."""
        records = [
            {
                "timestamp": s.timestamp,
                "log_loss": s.log_loss_val,
                "accuracy": s.accuracy,
                "n_samples": s.n_samples,
            }
            for s in self._history
        ]
        return pd.DataFrame(records)

    def plot_history(self, save_path: str | None = None) -> Figure:
        """Plot performance over time."""
        df = self.get_history_df()
        if df.empty:
            logger.warning("no_history_to_plot")
            fig, _ = plt.subplots()
            return fig

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Model Performance Over Time", fontsize=14, fontweight="bold")

        # Log Loss
        ax = axes[0]
        ax.plot(range(len(df)), df["log_loss"], "-o", color="#e74c3c", markersize=4)
        ax.axhline(
            self._baseline_log_loss, color="gray", linestyle="--",
            label=f"Random baseline ({self._baseline_log_loss:.4f})"
        )
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Log Loss")
        ax.set_title("Log Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy
        ax = axes[1]
        ax.plot(range(len(df)), df["accuracy"], "-o", color="#2ecc71", markersize=4)
        ax.axhline(1 / 3, color="gray", linestyle="--", label="Random baseline (33%)")
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
