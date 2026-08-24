"""
Tests for model modules.

Covers:
- Baseline models (Random, Prior, Momentum, OBI)
- Evaluation framework (metrics, walk-forward splits)
- Storage (Parquet writer)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import RandomBaseline, PriorBaseline, MomentumBaseline, OBIBaseline
from src.models.evaluation import (
    multiclass_brier_score,
    evaluate_predictions,
    walk_forward_split,
    aggregate_results,
)
from src.data.storage import ParquetWriter, build_schema


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample features and labels."""
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 10)
    y = np.random.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
    return X, y


@pytest.fixture
def sample_timestamps() -> np.ndarray:
    """Generate sequential timestamps spanning 48 hours."""
    n = 500
    start = 1700000000.0
    # 48 hours of data at ~0.1s intervals
    return np.linspace(start, start + 48 * 3600, n)


# ─── Baseline Tests ──────────────────────────────────────────────────────────

class TestRandomBaseline:
    def test_predict_proba_shape(self, sample_data: tuple) -> None:
        X, y = sample_data
        model = RandomBaseline()
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_uniform_probabilities(self, sample_data: tuple) -> None:
        X, y = sample_data
        model = RandomBaseline()
        proba = model.predict_proba(X)
        assert np.allclose(proba, 1 / 3)

    def test_predict_shape(self, sample_data: tuple) -> None:
        X, y = sample_data
        model = RandomBaseline()
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1, 2})


class TestPriorBaseline:
    def test_fit_and_predict(self, sample_data: tuple) -> None:
        X, y = sample_data
        model = PriorBaseline().fit(y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)
        # All rows should have the same probabilities
        assert np.allclose(proba[0], proba[-1])

    def test_majority_class(self, sample_data: tuple) -> None:
        X, y = sample_data
        model = PriorBaseline().fit(y)
        preds = model.predict(X)
        # All predictions should be the majority class
        assert len(np.unique(preds)) == 1

    def test_not_fitted_raises(self, sample_data: tuple) -> None:
        X, y = sample_data
        model = PriorBaseline()
        with pytest.raises(RuntimeError):
            model.predict_proba(X)


class TestMomentumBaseline:
    def test_predict_from_returns(self) -> None:
        returns = np.array([0.001, -0.002, 0.0001, 0.0, -0.001])
        model = MomentumBaseline(threshold=0.0)
        preds = model.predict_from_returns(returns)
        assert preds[0] == 2  # UP
        assert preds[1] == 0  # DOWN
        assert preds[3] == 1  # NEUTRAL (exactly 0)

    def test_proba_shape(self) -> None:
        returns = np.random.randn(100) * 0.001
        model = MomentumBaseline()
        proba = model.predict_proba_from_returns(returns)
        assert proba.shape == (100, 3)
        # Probabilities should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)


class TestOBIBaseline:
    def test_predict_from_obi(self) -> None:
        obi = np.array([0.3, -0.2, 0.05, -0.5, 0.0])
        model = OBIBaseline(threshold=0.1)
        preds = model.predict_from_obi(obi)
        assert preds[0] == 2   # Positive OBI > threshold → UP
        assert preds[1] == 0   # Negative OBI < -threshold → DOWN
        assert preds[2] == 1   # |OBI| < threshold → NEUTRAL


# ─── Evaluation Tests ─────────────────────────────────────────────────────────

class TestBrierScore:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 1, 2])
        y_proba = np.eye(3)
        score = multiclass_brier_score(y_true, y_proba)
        assert score == pytest.approx(0.0)

    def test_worst_predictions(self) -> None:
        y_true = np.array([0, 1, 2])
        # Predict the wrong class with certainty
        y_proba = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        score = multiclass_brier_score(y_true, y_proba)
        assert score == pytest.approx(2.0)

    def test_uniform_predictions(self) -> None:
        y_true = np.array([0, 1, 2])
        y_proba = np.full((3, 3), 1 / 3)
        score = multiclass_brier_score(y_true, y_proba)
        # Expected: (1/3 - 1)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9 = 2/3
        assert score == pytest.approx(2 / 3, abs=0.01)


class TestEvaluatePredictions:
    def test_basic_evaluation(self, sample_data: tuple) -> None:
        X, y = sample_data
        proba = np.random.dirichlet([1, 1, 1], size=len(y))
        result = evaluate_predictions(y, proba)
        assert 0 <= result.accuracy <= 1
        assert result.log_loss_val > 0
        assert result.brier_score > 0
        assert result.confusion.shape == (3, 3)
        assert len(result.f1_per_class) == 3


class TestWalkForwardSplit:
    def test_splits_generated(self, sample_timestamps: np.ndarray) -> None:
        splits = walk_forward_split(
            sample_timestamps,
            train_window_hours=24,
            test_window_hours=4,
            step_hours=4,
            min_train_samples=10,
        )
        assert len(splits) > 0

    def test_no_overlap(self, sample_timestamps: np.ndarray) -> None:
        splits = walk_forward_split(
            sample_timestamps,
            train_window_hours=24,
            test_window_hours=4,
            step_hours=4,
            min_train_samples=10,
        )
        for train_idx, test_idx in splits:
            # Train and test should not overlap
            assert len(set(train_idx) & set(test_idx)) == 0
            # Test should come after train
            assert train_idx[-1] < test_idx[0]

    def test_temporal_ordering(self, sample_timestamps: np.ndarray) -> None:
        splits = walk_forward_split(
            sample_timestamps,
            train_window_hours=24,
            test_window_hours=4,
            step_hours=4,
            min_train_samples=10,
        )
        for train_idx, test_idx in splits:
            assert sample_timestamps[train_idx[-1]] < sample_timestamps[test_idx[0]]


class TestAggregateResults:
    def test_aggregation(self) -> None:
        from src.models.evaluation import EvaluationResult
        results = [
            EvaluationResult(accuracy=0.4, f1_macro=0.35, log_loss_val=1.0),
            EvaluationResult(accuracy=0.5, f1_macro=0.45, log_loss_val=0.9),
        ]
        summary = aggregate_results(results)
        assert len(summary) == 4  # 2 folds + MEAN + STD


# ─── Storage Tests ────────────────────────────────────────────────────────────

class TestParquetWriter:
    def test_schema_generation(self) -> None:
        schema = build_schema(n_levels=5)
        field_names = [f.name for f in schema]
        assert "timestamp" in field_names
        assert "mid_price" in field_names
        assert "bid_price_1" in field_names
        assert "bid_price_5" in field_names
        assert "ask_qty_5" in field_names

    def test_write_and_read(self, tmp_path: Path) -> None:
        writer = ParquetWriter(
            output_dir=tmp_path,
            n_levels=3,
            batch_size=5,
            rotation_minutes=60,
        )

        # Add snapshots
        for i in range(10):
            snap = {"timestamp": float(i), "mid_price": 100.0, "spread": 0.01, "microprice": 100.0}
            for lvl in range(1, 4):
                snap[f"bid_price_{lvl}"] = 100.0 - 0.01 * lvl
                snap[f"bid_qty_{lvl}"] = 1.0
                snap[f"ask_price_{lvl}"] = 100.0 + 0.01 * lvl
                snap[f"ask_qty_{lvl}"] = 1.0
            writer.add_snapshot(snap)

        writer.flush()

        assert writer.total_rows_written == 10
        assert writer.total_files_created >= 1

        # Read back — may be split across multiple files due to time-based rotation
        # (test timestamps are epoch 1970, but batch flush uses current time for rotation key)
        import pyarrow.parquet as pq
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) >= 1

        all_dfs = [pq.read_table(f).to_pandas() for f in parquet_files]
        df = pd.concat(all_dfs, ignore_index=True)
        assert len(df) == 10
        assert "mid_price" in df.columns

    def test_context_manager(self, tmp_path: Path) -> None:
        with ParquetWriter(output_dir=tmp_path, n_levels=3, batch_size=100) as writer:
            snap = {"timestamp": 1.0, "mid_price": 100.0, "spread": 0.01, "microprice": 100.0}
            for lvl in range(1, 4):
                snap[f"bid_price_{lvl}"] = 100.0 - 0.01 * lvl
                snap[f"bid_qty_{lvl}"] = 1.0
                snap[f"ask_price_{lvl}"] = 100.0 + 0.01 * lvl
                snap[f"ask_qty_{lvl}"] = 1.0
            writer.add_snapshot(snap)

        # After context exit, file should be flushed
        assert writer.total_rows_written == 1
