
"""
Gradient Boosting models for LOB prediction.

Implements LightGBM with:
- Multi-class probabilistic predictions
- Optuna hyperparameter optimization
- Walk-forward validation integration
- Feature importance analysis
- SHAP analysis
- Model serialization and loading
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

    Class mapping:
        0 = DOWN
        1 = NEUTRAL
        2 = UP

    Predictions use the LightGBM Booster directly.
    This allows the same prediction behavior after training
    and after loading a saved model.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
    ) -> None:

        settings = get_settings()

        self._default_params: dict[str, Any] = {
            "objective": "multiclass",
            "num_class": 3,
            "class_weight": "balanced",
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

        # Direct LightGBM Booster used for prediction and loading.
        self._booster: lgb.Booster | None = None

        self._feature_names: list[str] = []

    @property
    def is_fitted(self) -> bool:
        """Return True if a trained or loaded Booster is available."""
        return self._booster is not None

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
        """

        if isinstance(X_train, pd.DataFrame):
            self._feature_names = X_train.columns.tolist()
        else:
            self._feature_names = [
                f"f_{i}" for i in range(X_train.shape[1])
            ]

        callbacks = [
            lgb.log_evaluation(period=0)
        ]

        eval_set = None

        if X_val is not None and y_val is not None:

            eval_set = [(X_val, y_val)]

            callbacks.append(
                lgb.early_stopping(
                    early_stopping_rounds,
                    verbose=False,
                )
            )

        self._model = lgb.LGBMClassifier(
            **self._default_params
        )

        self._model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        # Keep the actual trained Booster.
        self._booster = self._model.booster_

        best_iter = (
            self._model.best_iteration_
            if self._model.best_iteration_ > 0
            else self._default_params["n_estimators"]
        )

        logger.info(
            "lgbm_fitted",
            best_iteration=best_iter,
            n_features=X_train.shape[1],
        )

        return self

    def predict_proba(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        """
        Return probabilities for DOWN, NEUTRAL and UP.
        """

        if self._booster is None:
            raise RuntimeError(
                "Model not available. Train or load a model first."
            )

        proba = self._booster.predict(X)

        proba = np.asarray(proba)

        if proba.ndim != 2 or proba.shape[1] != 3:
            raise RuntimeError(
                f"Unexpected prediction shape: {proba.shape}. "
                "Expected (n_samples, 3)."
            )

        return proba

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        """Predict class labels."""

        proba = self.predict_proba(X)

        return np.argmax(proba, axis=1)

    def feature_importance(
        self,
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            importance_type: "gain" or "split"
        """

        if self._booster is None:
            raise RuntimeError("Model not available.")

        importance = self._booster.feature_importance(
            importance_type=importance_type
        )

        names = self._feature_names or self._booster.feature_name()

        if len(names) != len(importance):
            names = [
                f"f_{i}" for i in range(len(importance))
            ]

        df = pd.DataFrame({
            "feature": names,
            "importance": importance,
        })

        df = df.sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)

        total = df["importance"].sum()

        if total > 0:
            df["importance_pct"] = (
                df["importance"] / total * 100
            )
        else:
            df["importance_pct"] = 0.0

        return df

    def shap_analysis(
        self,
        X: pd.DataFrame,
        save_path: str | Path | None = None,
        max_display: int = 20,
    ) -> np.ndarray:
        """
        Compute SHAP values and optionally save a summary plot.
        """

        if self._booster is None:
            raise RuntimeError("Model not available.")

        logger.info(
            "computing_shap_values",
            n_samples=len(X),
        )

        explainer = shap.TreeExplainer(self._booster)

        shap_values = explainer.shap_values(X)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.figure(figsize=(12, 8))

            shap.summary_plot(
                shap_values,
                X,
                plot_type="bar",
                max_display=max_display,
                show=False,
            )

            plt.tight_layout()

            plt.savefig(
                save_path,
                dpi=150,
                bbox_inches="tight",
            )

            plt.close()

            logger.info(
                "shap_plot_saved",
                path=str(save_path),
            )

        return shap_values

    def save(
        self,
        path: str | Path | None = None,
    ) -> Path:
        """Save the trained LightGBM Booster and metadata."""

        if self._booster is None:
            raise RuntimeError("Model not available.")

        path = (
            Path(path)
            if path
            else MODELS_DIR / "lgbm_model.txt"
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._booster.save_model(str(path))

        meta_path = path.with_suffix(".json")

        meta = {
            "params": self._default_params,
            "feature_names": self._feature_names,
            "n_features": len(self._feature_names),
            "classes": {
                "0": "DOWN",
                "1": "NEUTRAL",
                "2": "UP",
            },
        }

        meta["params"] = {
            k: v
            for k, v in meta["params"].items()
            if isinstance(
                v,
                (int, float, str, bool, list),
            )
        }

        meta_path.write_text(
            json.dumps(meta, indent=2)
        )

        logger.info(
            "model_saved",
            path=str(path),
        )

        return path

    def load(
        self,
        path: str | Path,
    ) -> LightGBMModel:
        """
        Load a saved LightGBM Booster.

        Important:
        We do not create an unfitted LGBMClassifier.
        Predictions are made directly through the Booster.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        self._booster = lgb.Booster(
            model_file=str(path)
        )

        # Load metadata.
        meta_path = path.with_suffix(".json")

        if meta_path.exists():

            meta = json.loads(
                meta_path.read_text()
            )

            self._feature_names = meta.get(
                "feature_names",
                [],
            )

            self._default_params.update(
                meta.get("params", {})
            )

        else:
            self._feature_names = (
                self._booster.feature_name()
            )

        logger.info(
            "model_loaded",
            path=str(path),
            n_features=self._booster.num_feature(),
        )

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
    Hyperparameter optimization using Optuna
    with walk-forward validation.
    """

    splits = walk_forward_split(
        timestamps,
        train_window_hours=train_window_hours,
        test_window_hours=test_window_hours,
        step_hours=test_window_hours,
    )

    if not splits:
        raise ValueError(
            "No valid walk-forward splits. Need more data."
        )

    eval_splits = splits[:min(3, len(splits))]

    def objective(trial: optuna.Trial) -> float:

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                1000,
                step=100,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                10,
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves",
                15,
                127,
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                20,
                500,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.5,
                1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.5,
                1.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                10.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                10.0,
                log=True,
            ),
        }

        fold_losses = []

        for train_idx, test_idx in eval_splits:

            model = LightGBMModel(params)

            model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
                X_val=X.iloc[test_idx],
                y_val=y.iloc[test_idx],
            )

            proba = model.predict_proba(
                X.iloc[test_idx]
            )

            loss = log_loss(
                y.iloc[test_idx],
                proba,
                labels=[0, 1, 2],
            )

            fold_losses.append(loss)

        return float(np.mean(fold_losses))

    study = optuna.create_study(
        direction="minimize"
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

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
) -> tuple[
    LightGBMModel,
    list[EvaluationResult],
    pd.DataFrame,
]:
    """
    Full training and walk-forward evaluation pipeline.

    Returns:
        Tuple of:
            - Final trained model
            - Per-fold evaluation results
            - Aggregated evaluation summary
    """

    splits = walk_forward_split(
        timestamps,
        train_window_hours=train_window_hours,
        test_window_hours=test_window_hours,
        step_hours=step_hours,
    )

    # Automatically reduce windows if no splits are found.
    if not splits:

        t_span_hours = (
            timestamps[-1] - timestamps[0]
        ) / 3600.0

        logger.info(
            "no_splits_attempting_auto_adjust",
            data_span_hours=t_span_hours,
            train_window_hours=train_window_hours,
            test_window_hours=test_window_hours,
            step_hours=step_hours,
        )

        total_requested = float(
            train_window_hours + test_window_hours
        )

        if total_requested <= 0:
            raise ValueError(
                "Invalid train/test window configuration"
            )

        scale = min(
            1.0,
            max(
                0.0,
                (t_span_hours - 0.001) / total_requested,
            ),
        )

        min_train_hours = 0.01

        attempt = 0
        max_attempts = 10

        cur_train = float(train_window_hours) * max(
            scale,
            1e-6,
        )

        cur_test = float(test_window_hours) * max(
            scale,
            1e-6,
        )

        cur_step = float(step_hours) * max(
            scale,
            1e-6,
        )

        while attempt < max_attempts:

            if cur_train < min_train_hours or cur_test < 0.01:
                break

            splits = walk_forward_split(
                timestamps,
                train_window_hours=cur_train,
                test_window_hours=cur_test,
                step_hours=max(0.01, cur_step),
            )

            if splits:

                logger.info(
                    "auto_adjust_success",
                    attempt=attempt,
                    train_window_hours=cur_train,
                    test_window_hours=cur_test,
                    step_hours=cur_step,
                    n_splits=len(splits),
                )

                break

            cur_train /= 2.0
            cur_test /= 2.0
            cur_step = max(
                0.01,
                cur_step / 2.0,
            )

            attempt += 1

        if not splits:
            raise ValueError(
                f"No valid walk-forward splits after auto-adjust. "
                f"Data span={t_span_hours:.3f}h; "
                f"tried down to train={cur_train:.4f}h "
                f"test={cur_test:.4f}h"
            )

    results = []

    # Walk-forward validation.
    for fold_idx, (train_idx, test_idx) in enumerate(splits):

        logger.info(
            "training_fold",
            fold=fold_idx,
            n_train=len(train_idx),
            n_test=len(test_idx),
        )

        model = LightGBMModel(params)

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            X_val=X.iloc[test_idx],
            y_val=y.iloc[test_idx],
        )

        proba = model.predict_proba(
            X.iloc[test_idx]
        )

        preds = model.predict(
            X.iloc[test_idx]
        )

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

    # Train final model on all available data.
    final_model = LightGBMModel(params)

    final_model.fit(X, y)

    summary = aggregate_results(results)

    return final_model, results, summary
