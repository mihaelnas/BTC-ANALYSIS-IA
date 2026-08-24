#!/usr/bin/env python3
"""
Standalone data collection script.

Usage:
    # Collect indefinitely (until Ctrl+C)
    python scripts/collect_data.py

    # Collect for 60 seconds (dry run)
    python scripts/collect_data.py --duration 60

    # Custom output directory
    python scripts/collect_data.py --output-dir /path/to/data

    # Custom number of levels
    python scripts/collect_data.py --n-levels 10
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import structlog


def configure_logging(verbose: bool = False) -> None:
    """Configure structlog for human-readable console output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if verbose else logging.INFO
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect BTCUSDT order book depth data from Binance WebSocket",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Collection duration in seconds (default: indefinite)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for Parquet files (default: data/raw/)",
    )
    parser.add_argument(
        "--n-levels",
        type=int,
        default=20,
        help="Number of order book levels to capture (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Parquet write batch size (default: 5000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, collect for a short duration and print stats",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    logger = structlog.get_logger("collect_data")

    # Handle dry run
    duration = args.duration
    if args.dry_run and duration is None:
        duration = 30.0
        logger.info("dry_run_mode", duration=duration)

    # Import after path setup
    from config.settings import get_settings, CollectorSettings, RAW_DATA_DIR
    from src.data.collector import DepthCollector

    settings = get_settings()

    # Override settings from CLI args
    collector_settings = CollectorSettings(
        n_levels=args.n_levels,
        parquet_batch_size=args.batch_size,
    )

    output_dir = args.output_dir or str(RAW_DATA_DIR)

    logger.info(
        "starting_collection",
        symbol=settings.binance.symbol.upper(),
        ws_url=settings.binance.ws_url,
        output_dir=output_dir,
        n_levels=collector_settings.n_levels,
        duration=duration,
    )

    collector = DepthCollector(
        binance_settings=settings.binance,
        collector_settings=collector_settings,
        output_dir=output_dir,
    )

    try:
        await collector.run(duration_seconds=duration)
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt")
    finally:
        stats = collector.stats
        logger.info("collection_complete", **stats)

        if args.dry_run:
            print("\n" + "=" * 60)
            print("DRY RUN RESULTS")
            print("=" * 60)
            for key, value in stats.items():
                print(f"  {key:.<30} {value}")
            print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
