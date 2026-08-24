"""
Baseline models for benchmarking.

These simple models establish the minimum performance threshold.
Any useful model must significantly outperform these baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from src.features.labels import Label

logger = structlog.get_logger(__name__)


class RandomBaseline:
    """
    Uniform random prediction.

    Predicts each class with equal probability (1/3).
    This is the absolute floor — any model should beat this.
    """

    def __init__(self, n_classes: int = 3, seed: int = 42) -> None:
        self.n_classes = n_classes
        self._rng = np.random.RandomState(seed)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return uniform probabilities for each sample."""
        n = len(X)
        return np.full((n, self.n_classes), 1.0 / self.n_classes)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return random class predictions."""
        n = len(X)
        return self._rng.randint(0, self.n_classes, size=n)

    def __repr__(self) -> str:
        return "RandomBaseline()"


class PriorBaseline:
    """
    Prior probability baseline.

    Always predicts the class distribution observed in training data.
    Equivalent to a "most frequent class" classifier for hard predictions.
    """

    def __init__(self) -> None:
        self._class_probs: np.ndarray | None = None
        self._majority_class: int = 1  # Default to NEUTRAL

    def fit(self, y: np.ndarray | pd.Series) -> PriorBaseline:
        """Learn the class distribution from training labels."""
        y = np.asarray(y, dtype=int)
        counts = np.bincount(y, minlength=3)
        self._class_probs = counts / counts.sum()
        self._majority_class = int(np.argmax(counts))

        logger.info(
            "prior_baseline_fit",
            class_probs=self._class_probs.tolist(),
            majority_class=self._majority_class,
        )
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return prior probabilities for each sample."""
        if self._class_probs is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        n = len(X)
        return np.tile(self._class_probs, (n, 1))

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Always predict the majority class."""
        n = len(X)
        return np.full(n, self._majority_class, dtype=int)

    def __repr__(self) -> str:
        probs = self._class_probs.tolist() if self._class_probs is not None else None
        return f"PriorBaseline(probs={probs})"


class MomentumBaseline:
    """
    Momentum (trend continuation) baseline.

    Predicts that the most recent price movement will continue.
    Uses recent log returns as the signal.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        """
        Args:
            threshold: Minimum absolute return to predict a direction.
                       Below this, predicts NEUTRAL.
        """
        self.threshold = threshold

    def predict_from_returns(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """
        Predict direction from recent returns.

        Args:
            returns: Array of recent log returns (one per sample)

        Returns:
            Array of predicted labels
        """
        returns = np.asarray(returns)
        predictions = np.full(len(returns), Label.NEUTRAL, dtype=int)
        predictions[returns > self.threshold] = Label.UP
        predictions[returns < -self.threshold] = Label.DOWN
        return predictions

    def predict_proba_from_returns(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """
        Produce pseudo-probabilities based on return magnitude.

        This is a soft version: the further the return from zero,
        the more confident the prediction.
        """
        returns = np.asarray(returns, dtype=float)
        n = len(returns)
        probs = np.full((n, 3), 1.0 / 3.0)

        for i in range(n):
            r = returns[i]
            if np.isnan(r):
                continue

            # Simple sigmoid-like scaling
            confidence = min(abs(r) * 10000, 0.9)  # Scale and cap

            if r > self.threshold:
                probs[i, Label.UP] = 0.5 + confidence * 0.4
                probs[i, Label.NEUTRAL] = 0.5 - confidence * 0.3
                probs[i, Label.DOWN] = 1.0 - probs[i, Label.UP] - probs[i, Label.NEUTRAL]
            elif r < -self.threshold:
                probs[i, Label.DOWN] = 0.5 + confidence * 0.4
                probs[i, Label.NEUTRAL] = 0.5 - confidence * 0.3
                probs[i, Label.UP] = 1.0 - probs[i, Label.DOWN] - probs[i, Label.NEUTRAL]

            # Clamp to valid probability range
            probs[i] = np.clip(probs[i], 0.01, 0.98)
            probs[i] /= probs[i].sum()

        return probs

    def __repr__(self) -> str:
        return f"MomentumBaseline(threshold={self.threshold})"


class OBIBaseline:
    """
    Order Book Imbalance threshold baseline.

    Uses the weighted OBI directly as a signal:
    - High OBI → predict UP
    - Low OBI → predict DOWN
    - Near zero → predict NEUTRAL

    This tests whether raw OBI alone has predictive power.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold

    def predict_from_obi(self, obi: np.ndarray | pd.Series) -> np.ndarray:
        """Predict from OBI values."""
        obi = np.asarray(obi)
        predictions = np.full(len(obi), Label.NEUTRAL, dtype=int)
        predictions[obi > self.threshold] = Label.UP
        predictions[obi < -self.threshold] = Label.DOWN
        return predictions

    def predict_proba_from_obi(self, obi: np.ndarray | pd.Series) -> np.ndarray:
        """Pseudo-probabilities from OBI."""
        obi = np.asarray(obi, dtype=float)
        n = len(obi)
        probs = np.full((n, 3), 1.0 / 3.0)

        for i in range(n):
            v = obi[i]
            if np.isnan(v):
                continue

            confidence = min(abs(v), 0.9)

            if v > self.threshold:
                probs[i, Label.UP] = 0.4 + confidence * 0.5
                probs[i, Label.NEUTRAL] = 0.4 - confidence * 0.2
                probs[i, Label.DOWN] = 1.0 - probs[i, Label.UP] - probs[i, Label.NEUTRAL]
            elif v < -self.threshold:
                probs[i, Label.DOWN] = 0.4 + confidence * 0.5
                probs[i, Label.NEUTRAL] = 0.4 - confidence * 0.2
                probs[i, Label.UP] = 1.0 - probs[i, Label.DOWN] - probs[i, Label.NEUTRAL]

            probs[i] = np.clip(probs[i], 0.01, 0.98)
            probs[i] /= probs[i].sum()

        return probs

    def __repr__(self) -> str:
        return f"OBIBaseline(threshold={self.threshold})"
