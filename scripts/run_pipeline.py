#!/usr/bin/env python3
"""
Orchestrator script for the complete LOB Predictor pipeline.

Usage:
    python scripts/run_pipeline.py --help
"""

import argparse
import subprocess
import sys
from pathlib import Path

import structlog

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)

logger = structlog.get_logger("run_pipeline")


def run_command(cmd: list[str], description: str) -> None:
    logger.info("starting_step", step=description, command=" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        logger.info("step_complete", step=description)
    except subprocess.CalledProcessError as e:
        logger.error("step_failed", step=description, exit_code=e.returncode)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complete LOB Predictor pipeline")
    parser.add_argument("--collect-duration", type=int, default=0,
                        help="Duration in seconds to collect new data (0 to skip collection)")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--generate-report", action="store_true", help="Generate Markdown report")
    parser.add_argument("--shap", action="store_true", help="Run SHAP analysis in evaluation")
    parser.add_argument("--save-plots", action="store_true", help="Save evaluation plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_exe = sys.executable

    logger.info("pipeline_started")

    # 1. Collect Data (Optional)
    if args.collect_duration > 0:
        run_command(
            [python_exe, "scripts/collect_data.py", "--duration", str(args.collect_duration)],
            "data_collection"
        )
    else:
        logger.info("skipping_data_collection", reason="duration=0")

    # 2. Train Model (This also runs the feature pipeline automatically if needed)
    train_cmd = [python_exe, "scripts/train_model.py"]
    if args.optimize:
        train_cmd.extend(["--optimize", "--n-trials", str(args.n_trials)])
    
    run_command(train_cmd, "model_training")

    # 3. Evaluate Model
    eval_cmd = [python_exe, "scripts/evaluate_model.py", "--with-baselines"]
    if args.generate_report:
        eval_cmd.append("--generate-report")
    if args.shap:
        eval_cmd.append("--shap")
    if args.save_plots:
        eval_cmd.append("--save-plots")

    run_command(eval_cmd, "model_evaluation")

    logger.info("pipeline_finished_successfully")


if __name__ == "__main__":
    main()
