#!/usr/bin/env python3
"""
LOB Predictor CLI - Orchestrate all project pipelines.

Quick Start:
  python main.py help              Show interactive help
  python main.py collect           Collect order book data
  python main.py features          Process features
  python main.py train             Train the model
  python main.py evaluate          Evaluate model performance
  python main.py drift-check       Detect data drift
  python main.py serve             Launch web app (Gradio)

For detailed help on any command:
  python main.py <command> --help
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.cli_ui import (
    print_error,
    print_info,
    print_success,
    print_header,
    render_help,
    render_features_summary,
    render_drift_report,
    console,
)
from rich import box
from rich.panel import Panel
from rich.table import Table


def run_subscript(script_path: str, args: list[str] | None = None) -> int:
    """Run a Python script as a subprocess with better error handling."""
    script_full_path = project_root / script_path
    
    if not script_full_path.exists():
        print_error(f"Script not found: {script_full_path}")
        return 1
    
    cmd = [sys.executable, str(script_full_path)]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, cwd=project_root)
        if result.returncode != 0:
            print_error(f"Script '{script_path}' exited with code {result.returncode}")
        return result.returncode
    except Exception as e:
        print_error(f"Failed to run script '{script_path}': {e}")
        return 1


def cmd_help(ns: argparse.Namespace) -> int:
    """Display interactive help for all commands."""
    cmd_name = getattr(ns, "command", None)
    render_help(cmd_name)
    return 0


def cmd_collect(ns: argparse.Namespace) -> int:
    print_header("Order Book Data Ingestion", "Binance Spot BTCUSDT (100ms depth updates)")
    
    table = Table(box=box.SIMPLE_HEAD, border_style="blue")
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Duration", f"{ns.duration or 60} seconds")
    table.add_row("Depth Levels", f"{ns.n_levels or 20} bids/asks")
    table.add_row("Target Directory", str(ns.output_dir or "data/raw"))
    table.add_row("Dry Run Mode", "[bold yellow]YES (No files saved)[/]" if ns.dry_run else "[green]NO (Saving to parquet)[/]")

    console.print(table)
    print_info("Connecting to Binance WebSocket stream...")

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
    
    ret = run_subscript("scripts/collect_data.py", args)
    if ret == 0:
        print_success("Data collection session finished successfully")
    else:
        print_error("Data collection failed")
    return ret


def cmd_features(ns: argparse.Namespace) -> int:
    """Run feature pipeline in-process and display formatted summary."""
    print_header("Feature Engineering Pipeline", "Microstructure signals, labeling & normalization")
    print_info("Computing order book features...")
    try:
        from src.features.pipeline import run_feature_pipeline
        df, meta = run_feature_pipeline()
        
        raw_rows = meta.get("raw_rows", len(df))
        n_features = meta.get("n_features", len(df.columns) - 4)
        class_counts = {
            "DOWN": int(meta.get("DOWN", 0)),
            "NEUTRAL": int(meta.get("NEUTRAL", 0)),
            "UP": int(meta.get("UP", 0)),
        }
        output_path = str(meta.get("output_path", "data/processed/features_labeled.parquet"))
        
        render_features_summary(raw_rows, len(df), n_features, class_counts, output_path)
        return 0
    except Exception as e:
        print_error(f"Feature pipeline failed: {e}")
        return 1


def cmd_train(ns: argparse.Namespace) -> int:
    print_header("LightGBM Model Training", "Walk-forward temporal cross-validation")
    
    table = Table(box=box.SIMPLE_HEAD, border_style="magenta")
    table.add_column("Config", style="bold cyan")
    table.add_column("Setting", style="bold white")

    table.add_row("Hyperparameter Optimization", "[bold yellow]Optuna (Active)[/]" if ns.optimize else "[dim]Default Parameters[/]")
    if ns.optimize and ns.n_trials:
        table.add_row("Optuna Trials", str(ns.n_trials))
    if ns.sample_size:
        table.add_row("Dataset Sample Limit", f"{ns.sample_size:,} rows")
    if ns.validate_only:
        table.add_row("Mode", "[bold yellow]Validation Only[/]")

    console.print(table)
    print_info("Starting training process...")

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
    
    ret = run_subscript("scripts/train_model.py", args)
    if ret == 0:
        print_success("Model training finished successfully")
    else:
        print_error("Model training failed")
    return ret


def cmd_evaluate(ns: argparse.Namespace) -> int:
    print_header("Model Evaluation & Benchmarking", "Evaluation vs Random, Prior, Momentum & OBI baselines")
    
    args = []
    if ns.with_baselines:
        args += ["--with-baselines"]
    if ns.save_plots:
        args += ["--save-plots"]
    if ns.output_dir:
        args += ["--output-dir", ns.output_dir]
    if getattr(ns, "generate_report", False):
        args += ["--generate-report"]
    
    ret = run_subscript("scripts/evaluate_model.py", args)
    if ret == 0:
        print_success("Model evaluation completed")
    else:
        print_error("Model evaluation failed")
    return ret


def cmd_drift_check(ns: argparse.Namespace) -> int:
    print_header("Statistical Feature Drift Detection", "Kolmogorov-Smirnov Test & Population Stability Index (PSI)")
    try:
        from config.settings import PROCESSED_DATA_DIR
        from src.monitoring.drift_detector import detect_drift, drift_summary
        import pandas as pd

        proc_path = Path(ns.data_path) if ns.data_path else PROCESSED_DATA_DIR / "features_labeled.parquet"
        
        if not proc_path.exists():
            print_error(f"Processed data not found at: {proc_path}")
            print_info("Run 'python main.py features' to generate features first")
            return 2

        df = pd.read_parquet(proc_path)
        n = len(df)
        ref_n = int(ns.reference_size) if ns.reference_size else max(1000, n // 5)
        cur_n = int(ns.current_size) if ns.current_size else max(1000, n // 10)

        reference = df.iloc[:ref_n]
        current = df.iloc[-cur_n:]

        print_info(f"Analyzing Reference ({len(reference):,} rows) vs Current ({len(current):,} rows)...")
        results = detect_drift(reference, current)
        summary = drift_summary(results)

        out = Path(ns.output) if ns.output else Path("data/monitoring")
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        csv_path = out / f"drift_summary_{ts}.csv"
        summary.to_csv(csv_path, index=False)
        
        render_drift_report(results, summary, str(csv_path), len(reference), len(current))
        return 0
    except Exception as e:
        print_error(f"Drift detection failed: {e}")
        return 1


def cmd_serve(ns: argparse.Namespace) -> int:
    print_header("Gradio Prediction Web App", "Interactive LOB visualization & model inference")
    try:
        from src.deployment.app import build_app
        import gradio as gr

        app = build_app()
        url = f"http://{ns.server_name}:{ns.server_port}"
        
        console.print(
            Panel(
                f"[bold green]Web Application Ready![/]\n\n"
                f"[bold cyan]Local URL:[/]   [bold white underline]{url}[/]\n"
                f"[bold cyan]Public Link:[/] [white]{'Public link requested (--share)' if ns.share else 'Disabled (Local only)'}[/]",
                border_style="green",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        app.launch(
            share=ns.share,
            server_name=ns.server_name,
            server_port=ns.server_port,
            theme=gr.themes.Soft()
        )
        return 0
    except Exception as e:
        print_error(f"Failed to launch app: {e}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lob-cli",
        description="LOB Predictor - Order Book Microstructure ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    sub = parser.add_subparsers(dest="cmd", help="Command to run")

    # Help command
    p_help = sub.add_parser("help", help="Show help for commands")
    p_help.add_argument("command", nargs="?", help="Specific command to get help for")

    # Collect command
    p_collect = sub.add_parser("collect", help="Collect order book data")
    p_collect.add_argument("--duration", type=float, help="Duration in seconds")
    p_collect.add_argument("--output-dir", type=str, help="Output directory")
    p_collect.add_argument("--n-levels", type=int, help="Order book levels")
    p_collect.add_argument("--dry-run", action="store_true", help="Test without saving")
    p_collect.add_argument("--verbose", action="store_true", help="Verbose output")

    # Features command
    sub.add_parser("features", help="Run feature pipeline")

    # Train command
    p_train = sub.add_parser("train", help="Train model")
    p_train.add_argument("--optimize", action="store_true", help="Hyperparameter optimization")
    p_train.add_argument("--n-trials", type=int, help="Number of trials")
    p_train.add_argument("--validate-only", action="store_true", help="Validation only")
    p_train.add_argument("--sample-size", type=int, help="Sample size")
    p_train.add_argument("--train-window-hours", type=float, help="Training window")
    p_train.add_argument("--test-window-hours", type=float, help="Test window")
    p_train.add_argument("--step-hours", type=float, help="Walk-forward step")

    # Evaluate command
    p_eval = sub.add_parser("evaluate", help="Evaluate model")
    p_eval.add_argument("--with-baselines", action="store_true", help="Compare with baselines")
    p_eval.add_argument("--save-plots", action="store_true", help="Save visualizations")
    p_eval.add_argument("--output-dir", type=str, help="Output directory")
    p_eval.add_argument("--generate-report", action="store_true", help="Update docs/report.md")

    # Drift check command
    p_drift = sub.add_parser("drift-check", help="Detect data drift")
    p_drift.add_argument("--data-path", type=str, help="Path to processed data")
    p_drift.add_argument("--reference-size", type=int, help="Reference window size")
    p_drift.add_argument("--current-size", type=int, help="Current window size")
    p_drift.add_argument("--output", type=str, help="Output directory")

    # Serve command
    p_serve = sub.add_parser("serve", help="Launch web app")
    p_serve.add_argument("--share", action="store_true", help="Share with public link")
    p_serve.add_argument("--server-name", type=str, default="0.0.0.0", help="Server hostname")
    p_serve.add_argument("--server-port", type=int, default=7860, help="Server port")

    ns = parser.parse_args()

    # Handle commands
    try:
        if ns.cmd == "help":
            raise SystemExit(cmd_help(ns))
        elif ns.cmd == "collect":
            raise SystemExit(cmd_collect(ns))
        elif ns.cmd == "features":
            raise SystemExit(cmd_features(ns))
        elif ns.cmd == "train":
            raise SystemExit(cmd_train(ns))
        elif ns.cmd == "evaluate":
            raise SystemExit(cmd_evaluate(ns))
        elif ns.cmd == "drift-check":
            raise SystemExit(cmd_drift_check(ns))
        elif ns.cmd == "serve":
            raise SystemExit(cmd_serve(ns))
        else:
            render_help()
            raise SystemExit(0)
    except KeyboardInterrupt:
        print_error("Operation cancelled by user")
        raise SystemExit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
