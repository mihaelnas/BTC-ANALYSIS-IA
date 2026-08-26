#!/usr/bin/env python3
"""
Model evaluation script.

Usage:
    # Evaluate existing model on processed data
    python scripts/evaluate_model.py

    # Evaluate with custom model path
    python scripts/evaluate_model.py --model-path data/models/lgbm_model.txt

    # Include baselines comparison
    python scripts/evaluate_model.py --with-baselines

    # Save plots
    python scripts/evaluate_model.py --save-plots
"""

from __future__ import annotations

import argparse
import re
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

logger = structlog.get_logger("evaluate_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LOB prediction model")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to saved model (default: data/models/lgbm_model.txt)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to processed features Parquet",
    )
    parser.add_argument("--with-baselines", action="store_true", help="Compare with baselines")
    parser.add_argument("--save-plots", action="store_true", help="Save evaluation plots")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--generate-report", action="store_true", help="Update docs/report.md with results")
    parser.add_argument("--shap", action="store_true", help="Run SHAP analysis (may be slow)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from config.settings import PROCESSED_DATA_DIR, MODELS_DIR
    from src.models.gradient_boosting import LightGBMModel
    from src.models.baseline import RandomBaseline, PriorBaseline, MomentumBaseline, OBIBaseline
    from src.models.evaluation import (
        evaluate_predictions,
        walk_forward_split,
        aggregate_results,
        plot_walk_forward_results,
        plot_confusion_matrix,
        plot_calibration,
    )

    # ─── Load data ─────────────────────────────────────────────────────
    data_path = Path(args.data_path) if args.data_path else PROCESSED_DATA_DIR / "features_labeled.parquet"

    if not data_path.exists():
        logger.error("data_not_found", path=str(data_path))
        print("❌ No processed data found. Run the feature pipeline first:")
        print("   python scripts/train_model.py")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    logger.info("data_loaded", rows=len(df), columns=len(df.columns))

    exclude_cols = {"timestamp", "mid_price", "microprice", "label", "future_return"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df["label"].astype(int)
    timestamps = df["timestamp"].to_numpy()

    # ─── Load model ────────────────────────────────────────────────────
    model_path = Path(args.model_path) if args.model_path else MODELS_DIR / "lgbm_model.txt"

    if not model_path.exists():
        logger.error("model_not_found", path=str(model_path))
        print("❌ No trained model found. Train a model first:")
        print("   python scripts/train_model.py")
        sys.exit(1)

    model = LightGBMModel()
    model.load(model_path)

    # ─── Walk-Forward Evaluation ───────────────────────────────────────
    splits = walk_forward_split(timestamps)

    if not splits:
        logger.warning("insufficient_data_for_walkforward")
        # Fallback: single train/test split (80/20)
        split_idx = int(len(df) * 0.8)
        splits = [(np.arange(split_idx), np.arange(split_idx, len(df)))]

    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # Re-train on this fold's training data
        train_idx_list = train_idx.tolist()
        test_idx_list = test_idx.tolist()
        
        fold_model = LightGBMModel()
        fold_model.fit(X.iloc[train_idx_list], y.iloc[train_idx_list])

        proba = fold_model.predict_proba(X.iloc[test_idx_list])
        preds = fold_model.predict(X.iloc[test_idx_list])

        result = evaluate_predictions(
            y_true=y.iloc[test_idx_list].to_numpy(),
            y_proba=proba,
            y_pred=preds,
            fold=fold_idx,
            train_size=len(train_idx),
            timestamps=(
                float(timestamps[train_idx[0]]),
                float(timestamps[train_idx[-1]]),
                float(timestamps[test_idx[0]]),
                float(timestamps[test_idx[-1]]),
            ),
        )
        results.append(result)

    summary = aggregate_results(results)

    print("\n" + "=" * 80)
    print("WALK-FORWARD EVALUATION — LightGBM")
    print("=" * 80)
    print(summary.to_string(index=False))

    # ─── Baselines comparison ──────────────────────────────────────────
    if args.with_baselines:
        print("\n" + "=" * 80)
        print("BASELINES COMPARISON (on full dataset)")
        print("=" * 80)

        baselines = {
            "Random": RandomBaseline(),
            "Prior": PriorBaseline().fit(y),
        }

        for name, bl in baselines.items():
            proba = bl.predict_proba(X)
            preds = bl.predict(X)
            result = evaluate_predictions(y.to_numpy(), proba, preds, fold=-1)
            print(f"\n{name}:")
            print(f"  Log-loss:  {result.log_loss_val:.4f}")
            print(f"  Brier:     {result.brier_score:.4f}")
            print(f"  Accuracy:  {result.accuracy:.4f}")
            print(f"  F1 Macro:  {result.f1_macro:.4f}")

        # Momentum baseline (needs return_1 column)
        if "return_1" in df.columns:
            mom_bl = MomentumBaseline()
            mom_preds = mom_bl.predict_from_returns(df["return_1"])
            mom_proba = mom_bl.predict_proba_from_returns(df["return_1"])
            result = evaluate_predictions(y.to_numpy(), mom_proba, mom_preds, fold=-1)
            print(f"\nMomentum:")
            print(f"  Log-loss:  {result.log_loss_val:.4f}")
            print(f"  Accuracy:  {result.accuracy:.4f}")
            print(f"  F1 Macro:  {result.f1_macro:.4f}")

        # OBI baseline
        if "obi_weighted" in df.columns:
            obi_bl = OBIBaseline()
            obi_preds = obi_bl.predict_from_obi(df["obi_weighted"])
            obi_proba = obi_bl.predict_proba_from_obi(df["obi_weighted"])
            result = evaluate_predictions(y.to_numpy(), obi_proba, obi_preds, fold=-1)
            print(f"\nOBI:")
            print(f"  Log-loss:  {result.log_loss_val:.4f}")
            print(f"  Accuracy:  {result.accuracy:.4f}")
            print(f"  F1 Macro:  {result.f1_macro:.4f}")

    # ─── Plots ─────────────────────────────────────────────────────────
    if args.save_plots:
        output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_walk_forward_results(results, save_path=output_dir / "walk_forward.png")
        plot_confusion_matrix(results[-1], save_path=output_dir / "confusion_matrix.png")
        plot_calibration(results[-1], save_path=output_dir / "calibration.png")

        print(f"\n📊 Plots saved to {output_dir}")

    # ─── Feature importance ───────────────────────────────────────────
    importance = model.feature_importance()
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE (Top 20)")
    print("=" * 80)
    importance_str = importance.head(20).to_string(index=False)
    print(importance_str)

    # ─── SHAP Analysis ────────────────────────────────────────────────
    if args.shap:
        print("\n" + "=" * 80)
        print("SHAP ANALYSIS")
        print("=" * 80)
        output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Sample data for SHAP to avoid extreme slowness
        X_sample = X.sample(n=min(1000, len(X)), random_state=42)
        model.shap_analysis(X_sample, save_path=output_dir / "shap_summary.png")

    # ─── Generate Report ──────────────────────────────────────────────
    if args.generate_report:
        report_path = project_root / "docs" / "report.md"
        if report_path.exists():
            content = report_path.read_text(encoding="utf-8")
            
            # Build results string
            results_md = (
                f"### Résumé Walk-Forward (LightGBM)\n\n"
                f"```text\n{summary.to_string(index=False)}\n```\n\n"
                f"### Top Features (Gain)\n\n"
                f"```text\n{importance_str}\n```\n"
            )

            # Replace the placeholder in the report
            placeholder = "*Section à compléter après entraînement et évaluation.*"
            if placeholder in content:
                content = content.replace(placeholder, results_md)
                report_path.write_text(content, encoding="utf-8")
                logger.info("report_updated", path=str(report_path))
            else:
                logger.warning("report_placeholder_not_found")
        else:
            logger.warning("report_file_not_found", path=str(report_path))

    logger.info("evaluation_complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Provide user-friendly error messages while logging details
        msg = str(e) or "An unexpected error occurred."

        # Specific helpful hints for common failure modes
        if "feature_importances" in msg or "No feature_importances" in msg:
            print("❌ Erreur : le modèle chargé semble incomplet ou non-fitté.")
            print("   Solution : ré-entrainez un modèle avec `python scripts/train_model.py` ou fournissez un chemin de modèle valide via --model-path.")
        elif "No processed data found" in msg or "data_not_found" in msg:
            print("❌ Données traitées introuvables. Exécutez d'abord le pipeline de features :")
            print("   python scripts/train_model.py")
        else:
            print("❌ Erreur lors de l'évaluation :", msg)

        # Log full exception for debugging purposes
        logger.error("evaluation_failed", error=msg)
        # Exit with non-zero code but without printing raw traceback
        sys.exit(1)
