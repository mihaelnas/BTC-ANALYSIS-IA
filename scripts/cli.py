#!/usr/bin/env python3
"""
Lightweight CLI to orchestrate project pipelines.

Commands:
  collect       Run data collection (wraps scripts/collect_data.py)
  features      Run the feature pipeline (in-process)
  train         Run training (wraps scripts/train_model.py)
  evaluate      Run evaluation (wraps scripts/evaluate_model.py)
  drift-check   Run drift detection on processed data
  serve         Launch the demo web app

This CLI intentionally shells out to the existing scripts for parity.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def run_subscript(script_path: str, args: list[str] | None = None) -> int:
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    return subprocess.call(cmd)


def cmd_collect(ns: argparse.Namespace) -> int:
    args = []
    if ns.duration:
        args += ["--duration", str(ns.duration)]
    if ns.output_dir:
        args += ["--output-dir", ns.output_dir]
    if ns.n_levels:
        args += ["--n-levels", str(ns.n_levels)]
    if ns.dry_run:
        args += ["--dry-run"]
    if ns.verbose:
        args += ["--verbose"]
    return run_subscript("scripts/collect_data.py", args)


def cmd_features(ns: argparse.Namespace) -> int:
    # Run feature pipeline in-process to preserve environment
    from src.features.pipeline import run_feature_pipeline

    print("Running feature pipeline...")
    df, meta = run_feature_pipeline()
    print(f"Processed rows: {len(df)} -> saved to {meta.get('output_path')}")
    return 0


def cmd_train(ns: argparse.Namespace) -> int:
    args = []
    if ns.optimize:
        args += ["--optimize"]
        if ns.n_trials:
            args += ["--n-trials", str(ns.n_trials)]
    if ns.validate_only:
        args += ["--validate-only"]
    if ns.sample_size:
        args += ["--sample-size", str(ns.sample_size)]
    if ns.train_window_hours:
        args += ["--train-window-hours", str(ns.train_window_hours)]
    if ns.test_window_hours:
        args += ["--test-window-hours", str(ns.test_window_hours)]
    if ns.step_hours:
        args += ["--step-hours", str(ns.step_hours)]
    return run_subscript("scripts/train_model.py", args)


def cmd_evaluate(ns: argparse.Namespace) -> int:
    args = []
    if ns.with_baselines:
        args += ["--with-baselines"]
    if ns.save_plots:
        args += ["--save-plots"]
    if ns.output_dir:
        args += ["--output-dir", ns.output_dir]
    return run_subscript("scripts/evaluate_model.py", args)


def cmd_drift_check(ns: argparse.Namespace) -> int:
    from config.settings import PROCESSED_DATA_DIR
    from src.monitoring.drift_detector import detect_drift, drift_summary

    proc_path = Path(ns.data_path) if ns.data_path else PROCESSED_DATA_DIR / "features_labeled.parquet"
    if not proc_path.exists():
        print(f"Processed data not found: {proc_path}")
        return 2

    import pandas as pd

    df = pd.read_parquet(proc_path)
    n = len(df)
    ref_n = int(ns.reference_size) if ns.reference_size else max(1000, n // 5)
    cur_n = int(ns.current_size) if ns.current_size else max(1000, n // 10)

    reference = df.iloc[:ref_n]
    current = df.iloc[-cur_n:]

    print(f"Running drift check: reference={len(reference)} rows, current={len(current)} rows")
    results = detect_drift(reference, current)
    summary = drift_summary(results)

    out = Path(ns.output) if ns.output else Path("data/monitoring")
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    csv_path = out / f"drift_summary_{ts}.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Drift summary written to {csv_path}")
    return 0


def cmd_serve(ns: argparse.Namespace) -> int:
    # Import and launch the app in-process (blocks)
    from src.deployment.app import build_app
    import gradio as gr

    app = build_app()
    app.launch(share=ns.share, server_name=ns.server_name, server_port=ns.server_port, theme=gr.themes.Soft())
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="python scripts/cli.py", description="Project CLI for PythonIA")
    sub = parser.add_subparsers(dest="cmd")

    p_collect = sub.add_parser("collect", help="Collect order book data")
    p_collect.add_argument("--duration", type=float)
    p_collect.add_argument("--output-dir", type=str)
    p_collect.add_argument("--n-levels", type=int)
    p_collect.add_argument("--dry-run", action="store_true")
    p_collect.add_argument("--verbose", action="store_true")

    p_features = sub.add_parser("features", help="Run feature pipeline")

    p_train = sub.add_parser("train", help="Train model (wraps scripts/train_model.py)")
    p_train.add_argument("--optimize", action="store_true")
    p_train.add_argument("--n-trials", type=int)
    p_train.add_argument("--validate-only", action="store_true")
    p_train.add_argument("--sample-size", type=int)
    p_train.add_argument("--train-window-hours", type=float)
    p_train.add_argument("--test-window-hours", type=float)
    p_train.add_argument("--step-hours", type=float)

    p_eval = sub.add_parser("evaluate", help="Evaluate model (wraps scripts/evaluate_model.py)")
    p_eval.add_argument("--with-baselines", action="store_true")
    p_eval.add_argument("--save-plots", action="store_true")
    p_eval.add_argument("--output-dir", type=str)

    p_drift = sub.add_parser("drift-check", help="Run drift detection on processed data")
    p_drift.add_argument("--data-path", type=str)
    p_drift.add_argument("--reference-size", type=int)
    p_drift.add_argument("--current-size", type=int)
    p_drift.add_argument("--output", type=str)

    p_serve = sub.add_parser("serve", help="Launch demo web app")
    p_serve.add_argument("--share", action="store_true")
    p_serve.add_argument("--server-name", type=str, default="0.0.0.0")
    p_serve.add_argument("--server-port", type=int, default=7860)

    ns = parser.parse_args()

    if ns.cmd == "collect":
        raise SystemExit(cmd_collect(ns))
    if ns.cmd == "features":
        raise SystemExit(cmd_features(ns))
    if ns.cmd == "train":
        raise SystemExit(cmd_train(ns))
    if ns.cmd == "evaluate":
        raise SystemExit(cmd_evaluate(ns))
    if ns.cmd == "drift-check":
        raise SystemExit(cmd_drift_check(ns))
    if ns.cmd == "serve":
        raise SystemExit(cmd_serve(ns))

    parser.print_help()


if __name__ == "__main__":
    main()
