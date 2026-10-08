"""
Rich CLI UI helper module for LOB Predictor.
Provides beautiful, readable, and structured terminal output using Rich without emojis.
"""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def print_header(title: str, subtitle: str = "") -> None:
    """Display a prominent, styled panel header."""
    header_text = Text()
    header_text.append(title, style="bold white on blue")
    if subtitle:
        header_text.append(f"\n{subtitle}", style="dim white")
    
    panel = Panel(
        header_text,
        border_style="bright_blue",
        box=box.ROUNDED,
        padding=(0, 2),
    )
    console.print(panel)


def print_info(msg: str) -> None:
    """Print an informational message."""
    console.print(f"[bold blue][INFO][/] [cyan]{msg}[/]")


def print_success(msg: str) -> None:
    """Print a success message."""
    console.print(f"[bold green][OK][/] [bold green]{msg}[/]")


def print_warning(msg: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow][WARNING][/] [yellow]{msg}[/]")


def print_error(msg: str) -> None:
    """Print an error message."""
    console.print(f"[bold red][ERROR][/] [bold red]{msg}[/]")


def render_help(command: str | None = None) -> None:
    """Render interactive, beautiful help guide."""
    commands_info = {
        "collect": {
            "title": "COLLECT — Data Ingestion",
            "desc": "Collect live 100ms order book depth snapshots from Binance (BTCUSDT)",
            "usage": "python main.py collect [OPTIONS]",
            "options": [
                ("--duration SECONDS", "Collection duration in seconds (default: 60)"),
                ("--output-dir PATH", "Target folder for Parquet files"),
                ("--n-levels INT", "Order book depth per snapshot (default: 20)"),
                ("--dry-run", "Test connection without saving to disk"),
                ("--verbose", "Enable detailed debug logs"),
            ],
            "examples": [
                "python main.py collect --duration 300",
                "python main.py collect --dry-run --n-levels 10",
            ],
        },
        "features": {
            "title": "FEATURES — Feature Engineering Pipeline",
            "desc": "Compute 28 microstructure features, adaptive thresholds & z-score normalization",
            "usage": "python main.py features",
            "options": [
                ("(Uses parameters from config/settings.py)", ""),
            ],
            "examples": [
                "python main.py features",
            ],
        },
        "train": {
            "title": "TRAIN — Model Training & Tuning",
            "desc": "Train LightGBM with walk-forward CV, baseline comparisons & Optuna tuning",
            "usage": "python main.py train [OPTIONS]",
            "options": [
                ("--optimize", "Run Optuna hyperparameter optimization"),
                ("--n-trials INT", "Number of Optuna trials (default: 100)"),
                ("--sample-size INT", "Limit dataset size for fast testing"),
                ("--validate-only", "Validate without saving model artifact"),
                ("--train-window-hours N", "Length of training period in hours"),
                ("--test-window-hours N", "Length of testing period in hours"),
            ],
            "examples": [
                "python main.py train",
                "python main.py train --optimize --n-trials 30",
                "python main.py train --sample-size 15000",
            ],
        },
        "evaluate": {
            "title": "EVALUATE — Model Performance Analysis",
            "desc": "Evaluate trained model vs Random, Prior, Momentum & OBI baselines",
            "usage": "python main.py evaluate [OPTIONS]",
            "options": [
                ("--with-baselines", "Include comparison against baseline models"),
                ("--save-plots", "Generate and save confusion matrix & calibration plots"),
                ("--output-dir PATH", "Where to save evaluation plots"),
                ("--generate-report", "Automatically update docs/report.md with results"),
            ],
            "examples": [
                "python main.py evaluate --with-baselines --save-plots",
                "python main.py evaluate --generate-report",
            ],
        },
        "drift-check": {
            "title": "DRIFT-CHECK — Data Drift Monitoring",
            "desc": "Detect statistical feature drift between reference and current windows (KS-test & PSI)",
            "usage": "python main.py drift-check [OPTIONS]",
            "options": [
                ("--data-path PATH", "Path to processed Parquet dataset"),
                ("--reference-size INT", "Reference window sample size"),
                ("--current-size INT", "Current window sample size"),
                ("--output PATH", "Output directory for drift summary CSV"),
            ],
            "examples": [
                "python main.py drift-check",
                "python main.py drift-check --output data/monitoring",
            ],
        },
        "serve": {
            "title": "SERVE — Gradio Web Interface",
            "desc": "Launch interactive web application for live order book predictions",
            "usage": "python main.py serve [OPTIONS]",
            "options": [
                ("--server-port PORT", "Server port number (default: 7860)"),
                ("--server-name HOST", "Server hostname (default: 0.0.0.0)"),
                ("--share", "Create a public Gradio shareable link"),
            ],
            "examples": [
                "python main.py serve",
                "python main.py serve --server-port 8000",
            ],
        },
    }

    if command and command in commands_info:
        info = commands_info[command]
        console.print(Panel(f"[bold gold1]{info['title']}[/]\n[italic white]{info['desc']}[/]", box=box.ROUNDED, border_style="gold1"))
        
        console.print(f"\n[bold cyan]Usage:[/] [green]{info['usage']}[/]\n")
        
        table = Table(title="Options", box=box.SIMPLE_HEAD, border_style="dim")
        table.add_column("Option", style="bold yellow")
        table.add_column("Description", style="white")
        for opt, desc in info["options"]:
            table.add_row(opt, desc)
        console.print(table)
        
        console.print("\n[bold cyan]Examples:[/]")
        for ex in info["examples"]:
            console.print(f"  [dim]$[/] [bold green]{ex}[/]")
    else:
        # Show main guide
        console.print(
            Panel(
                "[bold cyan]LOB Predictor CLI[/] — Order Book Microstructure ML Pipeline\n"
                "[dim]Predict short-term BTCUSDT price movements from order book depth[/]",
                border_style="magenta",
                box=box.DOUBLE,
                padding=(1, 2),
            )
        )

        cmd_table = Table(box=box.ROUNDED, border_style="blue", show_header=True, header_style="bold cyan")
        cmd_table.add_column("Command", style="bold yellow", width=14)
        cmd_table.add_column("Description", style="white")
        cmd_table.add_column("Category", style="dim cyan", width=14)

        cmd_table.add_row("collect", "Collect live order book data (Binance WS)", "Ingestion")
        cmd_table.add_row("features", "Compute 28 microstructure features & labels", "Processing")
        cmd_table.add_row("train", "Train LightGBM & walk-forward CV", "Modeling")
        cmd_table.add_row("evaluate", "Evaluate model vs baselines & generate plots", "Evaluation")
        cmd_table.add_row("drift-check", "Detect statistical data drift (KS & PSI)", "Monitoring")
        cmd_table.add_row("serve", "Launch interactive Gradio web application", "Deployment")

        console.print(cmd_table)
        console.print("\n[bold cyan]Get detailed command help:[/] [yellow]python main.py help <command>[/]")


def render_features_summary(
    raw_rows: int,
    processed_rows: int,
    n_features: int,
    class_counts: dict[str, int],
    output_path: str,
) -> None:
    """Render a clean summary panel after feature engineering."""
    total_labeled = sum(class_counts.values()) or 1
    
    down_n = class_counts.get("DOWN", 0)
    neutral_n = class_counts.get("NEUTRAL", 0)
    up_n = class_counts.get("UP", 0)
    
    down_pct = (down_n / total_labeled) * 100
    neutral_pct = (neutral_n / total_labeled) * 100
    up_pct = (up_n / total_labeled) * 100

    def make_bar(pct: float, color: str, length: int = 15) -> str:
        filled = int((pct / 100) * length)
        return f"[{color}]{'█' * filled}{'░' * (length - filled)}[/]"

    table = Table(box=box.SIMPLE_HEAD, border_style="blue")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Raw Snapshots Loaded", f"{raw_rows:,}")
    table.add_row("Cleaned Rows", f"{processed_rows:,}")
    table.add_row("Microstructure Features", f"{n_features}")
    table.add_row("Parquet Storage Path", f"[dim]{output_path}[/]")

    dist_table = Table(box=box.SIMPLE_HEAD, border_style="dim")
    dist_table.add_column("Class", style="bold")
    dist_table.add_column("Count", justify="right")
    dist_table.add_column("Ratio", justify="right")
    dist_table.add_column("Distribution Visualizer", justify="left")

    dist_table.add_row(
        "[bold red]DOWN[/]",
        f"{down_n:,}",
        f"{down_pct:.1f}%",
        f"{make_bar(down_pct, 'red')} ",
    )
    dist_table.add_row(
        "[bold white]NEUTRAL[/]",
        f"{neutral_n:,}",
        f"{neutral_pct:.1f}%",
        f"{make_bar(neutral_pct, 'white')} ",
    )
    dist_table.add_row(
        "[bold green]UP[/]",
        f"{up_n:,}",
        f"{up_pct:.1f}%",
        f"{make_bar(up_pct, 'green')} ",
    )

    console.print(
        Panel(
            table,
            title="[bold green]Feature Pipeline Completed Successfully[/]",
            border_style="green",
            box=box.ROUNDED,
        )
    )
    console.print(Panel(dist_table, title="[bold cyan]Label Class Distribution (Horizon: 50 Ticks)[/]", border_style="cyan", box=box.ROUNDED))


def render_drift_report(
    results_df: Any,
    summary_df: Any,
    csv_path: str,
    ref_n: int,
    cur_n: int,
) -> None:
    """Render a color-coded Rich table for statistical drift detection."""
    table = Table(
        title=f"Statistical Feature Drift Summary (Ref: {ref_n:,} | Cur: {cur_n:,})",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold magenta",
    )
    
    table.add_column("Feature", style="bold white")
    table.add_column("KS Stat", justify="right", style="cyan")
    table.add_column("KS p-val", justify="right")
    table.add_column("PSI Score", justify="right", style="bold")
    table.add_column("Drift Status", justify="center")

    for _, row in summary_df.iterrows():
        feat = str(row["feature"])
        ks_s = f"{row['ks_statistic']:.4f}"
        ks_p = f"{row['ks_pvalue']:.4e}" if row['ks_pvalue'] < 0.001 else f"{row['ks_pvalue']:.4f}"
        psi = float(row["psi"])
        psi_str = f"{psi:.4f}"

        if psi >= 0.25:
            status = "[bold red]HIGH DRIFT[/]"
            psi_style = "[bold red]" + psi_str + "[/]"
        elif psi >= 0.10:
            status = "[bold yellow]MODERATE[/]"
            psi_style = "[bold yellow]" + psi_str + "[/]"
        else:
            status = "[bold green]STABLE[/]"
            psi_style = "[green]" + psi_str + "[/]"

        table.add_row(feat, ks_s, ks_p, psi_style, status)

    console.print(table)
    
    n_drifted = sum(summary_df["psi"] >= 0.25)
    total = len(summary_df)
    ratio = (n_drifted / total) * 100 if total > 0 else 0
    
    badge_style = "red" if ratio > 30 else ("yellow" if ratio > 10 else "green")
    console.print(
        Panel(
            f"[{badge_style}]Drift Summary:[/] [bold]{n_drifted}/{total}[/] features drifted ([bold]{ratio:.1f}%[/])\n"
            f"[dim]Detailed report saved to: {csv_path}[/]",
            border_style=badge_style,
            box=box.ROUNDED,
        )
    )
