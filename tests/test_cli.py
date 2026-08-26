"""
Tests for the CLI interface (scripts/cli.py and main.py).
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.cli import (
    main,
    cmd_help,
    cmd_collect,
    cmd_features,
    cmd_train,
    cmd_evaluate,
    cmd_drift_check,
    cmd_serve,
)
from src.utils.cli_ui import (
    render_help,
    render_features_summary,
    render_drift_report,
)
import argparse


def test_cmd_help():
    ns = argparse.Namespace(command=None)
    res = cmd_help(ns)
    assert res == 0

    ns_collect = argparse.Namespace(command="collect")
    res_collect = cmd_help(ns_collect)
    assert res_collect == 0


def test_main_help():
    with patch.object(sys, "argv", ["main.py", "help"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0


def test_main_subcommand_help():
    with patch.object(sys, "argv", ["main.py", "help", "train"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0


def test_cmd_collect_args():
    with patch("scripts.cli.run_subscript", return_value=0) as mock_run:
        ns = argparse.Namespace(
            duration=5.0,
            output_dir="data/test",
            n_levels=10,
            dry_run=True,
            verbose=True,
        )
        code = cmd_collect(ns)
        assert code == 0
        mock_run.assert_called_once_with(
            "scripts/collect_data.py",
            [
                "--duration",
                "5.0",
                "--output-dir",
                "data/test",
                "--n-levels",
                "10",
                "--dry-run",
                "--verbose",
            ],
        )


def test_cmd_features_success():
    mock_meta = {"output_path": "data/processed/test.parquet", "raw_rows": 100, "n_features": 20, "DOWN": 30, "NEUTRAL": 40, "UP": 30}
    mock_df = MagicMock()
    mock_df.__len__.return_value = 100
    mock_df.columns = ["a", "b", "c", "d"]
    with patch(
        "src.features.pipeline.run_feature_pipeline",
        return_value=(mock_df, mock_meta),
    ):
        ns = argparse.Namespace()
        code = cmd_features(ns)
        assert code == 0


def test_cmd_features_failure():
    with patch(
        "src.features.pipeline.run_feature_pipeline",
        side_effect=RuntimeError("Pipeline error"),
    ):
        ns = argparse.Namespace()
        code = cmd_features(ns)
        assert code == 1


def test_cmd_train_args():
    with patch("scripts.cli.run_subscript", return_value=0) as mock_run:
        ns = argparse.Namespace(
            optimize=True,
            n_trials=10,
            validate_only=False,
            sample_size=1000,
            train_window_hours=12.0,
            test_window_hours=2.0,
            step_hours=1.0,
        )
        code = cmd_train(ns)
        assert code == 0
        mock_run.assert_called_once_with(
            "scripts/train_model.py",
            [
                "--optimize",
                "--n-trials",
                "10",
                "--sample-size",
                "1000",
                "--train-window-hours",
                "12.0",
                "--test-window-hours",
                "2.0",
                "--step-hours",
                "1.0",
            ],
        )


def test_cmd_evaluate_args():
    with patch("scripts.cli.run_subscript", return_value=0) as mock_run:
        ns = argparse.Namespace(
            with_baselines=True,
            save_plots=True,
            output_dir="data/plots",
        )
        code = cmd_evaluate(ns)
        assert code == 0
        mock_run.assert_called_once_with(
            "scripts/evaluate_model.py",
            [
                "--with-baselines",
                "--save-plots",
                "--output-dir",
                "data/plots",
            ],
        )


def test_cmd_drift_check_missing_file(tmp_path):
    missing_file = tmp_path / "non_existent.parquet"
    ns = argparse.Namespace(
        data_path=str(missing_file),
        reference_size=100,
        current_size=50,
        output=str(tmp_path / "monitoring"),
    )
    code = cmd_drift_check(ns)
    assert code == 2


def test_cmd_serve_mock():
    mock_app = MagicMock()
    with patch("src.deployment.app.build_app", return_value=mock_app):
        ns = argparse.Namespace(
            share=False,
            server_name="127.0.0.1",
            server_port=7860,
        )
        code = cmd_serve(ns)
        assert code == 0
        mock_app.launch.assert_called_once()
