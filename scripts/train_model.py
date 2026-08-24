#!/usr/bin/env python3
"""
Model training script.

Usage:
    # Run full feature pipeline + train LightGBM
    python scripts/train_model.py

    # Optimize hyperparameters first
    python scripts/train_model.py --optimize --n-trials 50

    # Train with specific parameters
    python scripts/train_model.py --learning-rate 0.03 --n-estimators 800

    # Validation only (no final model save)
    python scripts/train_model.py --validate-only
"""

from __future__ import annotations

from typing import cast

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import structlog

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)

logger = structlog.get_logger("train_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LOB prediction model")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna HP optimization")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't save")
    parser.add_argument("--sample-size", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from config.settings import PROCESSED_DATA_DIR, MODELS_DIR
    from src.features.pipeline import run_feature_pipeline
    from src.models.gradient_boosting import (
        LightGBMModel,
        optimize_lgbm_with_optuna,
        train_and_evaluate_lgbm,
    )
    from src.models.baseline import RandomBaseline, PriorBaseline
    from src.models.evaluation import evaluate_predictions, aggregate_results

    # ─── Load or compute features ──────────────────────────────────────
    features_path = PROCESSED_DATA_DIR / "features_labeled.parquet"

    if features_path.exists():
        logger.info("loading_existing_features", path=str(features_path))
        df = pd.read_parquet(features_path)
    else:
        logger.info("running_feature_pipeline")
        df, _ = run_feature_pipeline()

    if args.sample_size and len(df) > args.sample_size:
        # Take LAST N samples (preserves temporal order)
        df = df.iloc[-args.sample_size:].reset_index(drop=True)
        logger.info("dataset_sampled", size=len(df))

    logger.info("dataset_loaded", rows=len(df), columns=len(df.columns))

    # ─── Prepare X, y ──────────────────────────────────────────────────
    exclude_cols = {"timestamp", "mid_price", "microprice", "label", "future_return"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df.loc[:, feature_cols]
    y = cast(pd.Series, df["label"].astype(int))
    timestamps: np.ndarray = df["timestamp"].to_numpy()

    logger.info("features_prepared", n_features=len(feature_cols), features=feature_cols[:10])

    # ─── Baselines ─────────────────────────────────────────────────────
    logger.info("evaluating_baselines")

    # Random baseline
    random_bl = RandomBaseline()
    random_proba = random_bl.predict_proba(X)
    random_result = evaluate_predictions(np.asarray(y), random_proba, fold=-1)
    logger.info(
        "baseline_random",
        log_loss=round(random_result.log_loss_val, 4),
        accuracy=round(random_result.accuracy, 4),
    )

    # Prior baseline
    prior_bl = PriorBaseline().fit(y)
    prior_proba = prior_bl.predict_proba(X)
    prior_result = evaluate_predictions(np.asarray(y), prior_proba, fold=-1)
    logger.info(
        "baseline_prior",
        log_loss=round(prior_result.log_loss_val, 4),
        accuracy=round(prior_result.accuracy, 4),
    )

    # ─── HP Optimization ──────────────────────────────────────────────
    best_params = {}

    if args.optimize:
        logger.info("starting_optimization", n_trials=args.n_trials)
        best_params = optimize_lgbm_with_optuna(
            X, y, timestamps,
            n_trials=args.n_trials,
        )
        logger.info("optimization_complete", params=best_params)

    # Override with CLI params
    if args.learning_rate:
        best_params["learning_rate"] = args.learning_rate
    if args.n_estimators:
        best_params["n_estimators"] = args.n_estimators
    if args.max_depth:
        best_params["max_depth"] = args.max_depth

    # ─── Train + Evaluate ──────────────────────────────────────────────
    logger.info("training_lgbm", params=best_params or "defaults")

    final_model, results, summary = train_and_evaluate_lgbm(
        X, y, timestamps,
        params=best_params if best_params else None,
    )

    print("\n" + "=" * 80)
    print("WALK-FORWARD EVALUATION RESULTS")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80)

    # Feature importance
    importance = final_model.feature_importance()
    print("\nTOP 15 FEATURES (by gain):")
    print(importance.head(15).to_string(index=False))

    # ─── Save ──────────────────────────────────────────────────────────
    if not args.validate_only:
        model_path = final_model.save()
        logger.info("model_saved", path=str(model_path))

        # Save evaluation summary
        summary_path = MODELS_DIR / "evaluation_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("summary_saved", path=str(summary_path))

        # Save feature importance
        importance_path = MODELS_DIR / "feature_importance.csv"
        importance.to_csv(importance_path, index=False)
        logger.info("importance_saved", path=str(importance_path))

    logger.info("training_complete")


if __name__ == "__main__":
    main()
