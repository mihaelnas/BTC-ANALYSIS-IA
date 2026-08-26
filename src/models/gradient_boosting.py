"""
Gradient Boosting models for LOB prediction.

Implements LightGBM and XGBoost with:
- Multi-class softmax objective (probabilistic output)
- Optuna hyperparameter optimization
- Walk-forward validation integration
- Feature importance analysis (gain + SHAP)
- Model serialization
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import structlog
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

from config.settings import get_settings, MODELS_DIR
from src.models.evaluation import (
    EvaluationResult,
    evaluate_predictions,
    walk_forward_split,
    aggregate_results,
)

logger = structlog.get_logger(__name__)


class LightGBMModel:
    """
    LightGBM classifier for multi-class LOB prediction.

    Outputs calibrated probabilities for UP/DOWN/NEUTRAL.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        settings = get_settings()

        self._default_params: dict[str, Any] = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "n_estimators": settings.model.lgbm_n_estimators,
            "learning_rate": settings.model.lgbm_learning_rate,
            "max_depth": settings.model.lgbm_max_depth,
            "num_leaves": settings.model.lgbm_num_leaves,
            "min_child_samples": settings.model.lgbm_min_child_samples,
            "subsample": settings.model.lgbm_subsample,
            "colsample_bytree": settings.model.lgbm_colsample_bytree,
            "random_state": settings.model.random_seed,
            "verbose": -1,
            "n_jobs": -1,
        }

        if params:
            self._default_params.update(params)

        self._model: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        early_stopping_rounds: int = 50,
    ) -> LightGBMModel:
        """
        Train the LightGBM model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels
            early_stopping_rounds: Early stopping patience
        """
        if isinstance(X_train, pd.DataFrame):
            self._feature_names = X_train.columns.tolist()

        callbacks = [lgb.log_evaluation(period=0)]  # Suppress per-iteration logs
        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

        self._model = lgb.LGBMClassifier(**self._default_params)

        eval_set = [(X_val, y_val)] if X_val is not None else None

        self._model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        best_iter = (
            self._model.best_iteration_
            if hasattr(self._model, "best_iteration_") and self._model.best_iteration_ > 0
            else self._default_params["n_estimators"]
        )
        logger.info("lgbm_fitted", best_iteration=best_iter, n_features=X_train.shape[1])

        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X)

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            importance_type: "gain" or "split"
        """
        if self._model is None:
            raise RuntimeError("Model not fitted.")

        # Try the sklearn wrapper's attribute first
        try:
            importance = getattr(self._model, "feature_importances_")
            if importance is None:
                raise AttributeError
        except Exception:
            # Fallback: if we have a Booster (loaded from file), use its API
            booster = None
            # lightgbm Booster may be stored under different attributes
            booster = getattr(self._model, "_Booster", None) or getattr(self._model, "booster_", None)
            if booster is None and hasattr(self, "_model"):
                # In case the model was loaded into a separate variable
                booster = getattr(self, "_model", None)

            if booster is None:
                raise RuntimeError("Model does not expose feature importances. Ensure it is fitted or loaded correctly.")

            try:
                importance = booster.feature_importance(importance_type=importance_type)
            except Exception:
                # Last resort: try Booster's get_score
                try:
                    score = booster.get_score(importance_type=importance_type)
                    # Map scores to list with zeros for missing features
                    names = self._feature_names or sorted(score.keys())
                    importance = [score.get(n, 0) for n in names]
                except Exception as exc:
                    raise RuntimeError("Unable to extract feature importances from loaded Booster") from exc

        names = self._feature_names or [f"f_{i}" for i in range(len(importance))]

        df = pd.DataFrame({
            "feature": names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        df["importance_pct"] = df["importance"] / df["importance"].sum() * 100

        return df

    def shap_analysis(
        self,
        X: pd.DataFrame,
        save_path: str | Path | None = None,
        max_display: int = 20,
    ) -> np.ndarray:
        """
        Compute SHAP values and optionally save a summary plot.

        Args:
            X: Feature matrix to analyze
            save_path: Path to save the plot. If None, won't save.
            max_display: Maximum number of features to show in plot.

        Returns:
            SHAP values array
        """
        if self._model is None:
            raise RuntimeError("Model not fitted.")

        logger.info("computing_shap_values", n_samples=len(X))
        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(X)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X,
                plot_type="bar",
                max_display=max_display,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("shap_plot_saved", path=str(save_path))

        return shap_values

    def save(self, path: str | Path | None = None) -> Path:
        """Save the model to disk."""
        if self._model is None:
            raise RuntimeError("Model not fitted.")

        path = Path(path) if path else MODELS_DIR / "lgbm_model.txt"
        path.parent.mkdir(parents=True, exist_ok=True)

        self._model.booster_.save_model(str(path))

        # Save metadata
        meta_path = path.with_suffix(".json")
        meta = {
            "params": self._default_params,
            "feature_names": self._feature_names,
            "n_features": len(self._feature_names),
        }
        # Convert non-serializable types
        meta["params"] = {k: v for k, v in meta["params"].items() if isinstance(v, (int, float, str, bool, list))}
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info("model_saved", path=str(path))
        return path

    def load(self, path: str | Path) -> LightGBMModel:
        """Load a model from disk."""
        path = Path(path)
        booster = lgb.Booster(model_file=str(path))
        self._model = lgb.LGBMClassifier()
        self._model._Booster = booster
        self._model._n_classes = 3

        # Load metadata
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._feature_names = meta.get("feature_names", [])
            self._default_params.update(meta.get("params", {}))

        logger.info("model_loaded", path=str(path))
        return self


def optimize_lgbm_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: np.ndarray,
    n_trials: int = 50,
    train_window_hours: float = 24.0,
    test_window_hours: float = 4.0,
) -> dict[str, Any]:
    """
    Hyperparameter optimization using Optuna with walk-forward validation.

    Args:
        X: Feature matrix
        y: Labels
        timestamps: Timestamps for temporal splitting
        n_trials: Number of Optuna trials
        train_window_hours: Training window
        test_window_hours: Test window

    Returns:
        Best parameters dictionary
    """
    splits = walk_forward_split(
        timestamps,
        train_window_hours=train_window_hours,
        test_window_hours=test_window_hours,
        step_hours=test_window_hours,
    )

    if not splits:
        raise ValueError("No valid walk-forward splits. Need more data.")

    # Use first 3 splits for speed during optimization
    eval_splits = splits[:min(3, len(splits))]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        fold_losses = []
        for train_idx, test_idx in eval_splits:
            model = LightGBMModel(params)
            model.fit(
                X.iloc[train_idx], y.iloc[train_idx],
                X_val=X.iloc[test_idx], y_val=y.iloc[test_idx],
            )
            proba = model.predict_proba(X.iloc[test_idx])
            loss = log_loss(y.iloc[test_idx], proba, labels=[0, 1, 2])
            fold_losses.append(loss)

        return float(np.mean(fold_losses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(
        "optuna_complete",
        best_value=study.best_value,
        best_params=study.best_params,
        n_trials=n_trials,
    )

    return study.best_params


def train_and_evaluate_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: np.ndarray,
    params: dict[str, Any] | None = None,
    train_window_hours: float = 24.0,
    test_window_hours: float = 4.0,
    step_hours: float = 4.0,
) -> tuple[LightGBMModel, list[EvaluationResult], pd.DataFrame]:
    """
    Full train + walk-forward evaluate pipeline for LightGBM.

    Returns:
        Tuple of (final trained model, per-fold results, summary DataFrame)
    """
    splits = walk_forward_split(
        timestamps,
        train_window_hours=train_window_hours,
        test_window_hours=test_window_hours,
        step_hours=step_hours,
    )

    # Auto-adapt windows when no splits are found: progressively shrink windows
    if not splits:
        t_span_hours = (timestamps[-1] - timestamps[0]) / 3600.0
        logger.info("no_splits_attempting_auto_adjust", data_span_hours=t_span_hours,
                    train_window_hours=train_window_hours,
                    test_window_hours=test_window_hours,
                    step_hours=step_hours)

        # If data span is smaller than requested total window, scale down proportionally
        total_requested = float(train_window_hours + test_window_hours)
        if total_requested <= 0:
            raise ValueError("Invalid train/test window configuration")

        # Compute a scaling factor so train+test <= data span (leave small margin)
        scale = min(1.0, max(0.0, (t_span_hours - 0.001) / total_requested))

        # Start with proportional reduction, then iteratively reduce until splits found
        min_train_hours = 0.01  # 36 seconds minimum window to avoid zero
        attempt = 0
        max_attempts = 10
        cur_train = float(train_window_hours) * max(scale, 1e-6)
        cur_test = float(test_window_hours) * max(scale, 1e-6)
        cur_step = float(step_hours) * max(scale, 1e-6)

        while attempt < max_attempts:
            # Ensure reasonable minimums
            if cur_train < min_train_hours or cur_test < 0.01:
                break

            splits = walk_forward_split(
                timestamps,
                train_window_hours=cur_train,
                test_window_hours=cur_test,
                step_hours=max(0.01, cur_step),
            )
            if splits:
                logger.info("auto_adjust_success", attempt=attempt,
                            train_window_hours=cur_train,
                            test_window_hours=cur_test,
                            step_hours=cur_step,
                            n_splits=len(splits))
                break

            # Reduce windows further (halve)
            cur_train /= 2.0
            cur_test /= 2.0
            cur_step = max(0.01, cur_step / 2.0)
            attempt += 1

        if not splits:
            raise ValueError(
                f"No valid walk-forward splits after auto-adjust. Data span={t_span_hours:.3f}h; "
                f"tried down to train={cur_train:.4f}h test={cur_test:.4f}h"
            )

    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        logger.info(
            "training_fold",
            fold=fold_idx,
            n_train=len(train_idx),
            n_test=len(test_idx),
        )

        model = LightGBMModel(params)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            X_val=X.iloc[test_idx], y_val=y.iloc[test_idx],
        )

        proba = model.predict_proba(X.iloc[test_idx])
        preds = model.predict(X.iloc[test_idx])

        result = evaluate_predictions(
            y_true=y.iloc[test_idx].values,
            y_proba=proba,
            y_pred=preds,
            fold=fold_idx,
            train_size=len(train_idx),
            timestamps=(
                timestamps[train_idx[0]],
                timestamps[train_idx[-1]],
                timestamps[test_idx[0]],
                timestamps[test_idx[-1]],
            ),
        )
        results.append(result)

        logger.info(
            "fold_result",
            fold=fold_idx,
            log_loss=round(result.log_loss_val, 4),
            accuracy=round(result.accuracy, 4),
            f1_macro=round(result.f1_macro, 4),
        )

    # Train final model on all data
    final_model = LightGBMModel(params)
    final_model.fit(X, y)

    summary = aggregate_results(results)

    return final_model, results, summary
