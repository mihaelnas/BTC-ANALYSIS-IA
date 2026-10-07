"""
Centralized configuration for the LOB Predictor project.

Uses pydantic-settings for type-safe configuration with environment variable overrides.
All project-wide parameters are defined here to ensure consistency across modules.
"""

from __future__ import annotations

from pathlib import Path
from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings


# ─── Project Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
DOCS_DIR = PROJECT_ROOT / "docs"


class LabelClass(str, Enum):
    """Price movement direction classes."""
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class BinanceSettings(BaseSettings):
    """Binance API connection settings."""

    # WebSocket
    ws_base_url: str = "wss://stream.binance.com:9443"
    symbol: str = "btcusdt"
    depth_update_speed: str = "100ms"

    # REST API (for initial snapshot)
    rest_base_url: str = "https://api.binance.com"
    snapshot_limit: int = 1000  # Max depth levels for initial snapshot

    @property
    def ws_url(self) -> str:
        """Full WebSocket URL for the depth stream."""
        return f"{self.ws_base_url}/ws/{self.symbol}@depth@{self.depth_update_speed}"

    @property
    def snapshot_url(self) -> str:
        """REST API URL for order book snapshot."""
        return (
            f"{self.rest_base_url}/api/v3/depth"
            f"?symbol={self.symbol.upper()}&limit={self.snapshot_limit}"
        )

    model_config = {"env_prefix": "BINANCE_"}


class CollectorSettings(BaseSettings):
    """Data collector configuration."""

    # Order book parameters
    n_levels: int = Field(
        default=20,
        description="Number of bid/ask levels to capture per snapshot",
    )
    snapshot_interval_ms: int = Field(
        default=100,
        description="Expected interval between depth updates (milliseconds)",
    )

    # Storage
    parquet_batch_size: int = Field(
        default=5000,
        description="Number of snapshots to accumulate before writing to Parquet",
    )
    file_rotation_minutes: int = Field(
        default=60,
        description="Create a new Parquet file every N minutes",
    )
    compression: str = Field(
        default="zstd",
        description="Parquet compression codec",
    )

    # Reconnection
    reconnect_delay_base: float = Field(
        default=1.0,
        description="Base delay (seconds) for exponential backoff on reconnect",
    )
    reconnect_delay_max: float = Field(
        default=60.0,
        description="Maximum reconnection delay (seconds)",
    )
    max_reconnect_attempts: int = Field(
        default=100,
        description="Maximum consecutive reconnection attempts before giving up",
    )
    reconnect_jitter_ratio: float = Field(
        default=0.1,
        description="Random jitter added to the backoff delay, as a fraction of it",
    )

    # Storage resilience
    max_consecutive_write_errors: int = Field(
        default=20,
        description=(
            "Maximum consecutive snapshot write failures before the collector "
            "stops itself gracefully, to avoid silently losing data forever "
            "(e.g. disk full, permissions revoked)"
        ),
    )

    model_config = {"env_prefix": "COLLECTOR_"}


class FeatureSettings(BaseSettings):
    """Feature engineering configuration."""

    # Prediction horizon
    horizon_ticks: int = Field(
        default=50,
        description="Number of ticks into the future for label (50 ticks × 100ms = 5s)",
    )

    # Label thresholds
    label_threshold_pct: float = Field(
        default=0.01,
        description="Percentage threshold for UP/DOWN classification (0.01 = 0.01%)",
    )
    adaptive_threshold: bool = Field(
        default=True,
        description="If True, calibrate threshold to achieve balanced classes",
    )
    target_neutral_ratio: float = Field(
        default=0.40,
        description="Target ratio of NEUTRAL class when using adaptive threshold",
    )

    # Normalization
    rolling_window: int = Field(
        default=5000,
        description="Window size for rolling z-score normalization",
    )

    # Feature selection
    use_multi_level_obi: bool = Field(
        default=True,
        description="Compute OBI at multiple levels (not just best bid/ask)",
    )
    obi_levels: list[int] = Field(
        default=[1, 3, 5, 10, 20],
        description="Levels at which to compute OBI",
    )

    model_config = {"env_prefix": "FEATURE_"}


class ModelSettings(BaseSettings):
    """Model training configuration."""

    # Walk-forward validation
    train_window_hours: int = Field(
        default=24,
        description="Training window size in hours",
    )
    test_window_hours: int = Field(
        default=4,
        description="Test window size in hours",
    )
    step_hours: int = Field(
        default=4,
        description="Step size for walk-forward (hours)",
    )

    # LightGBM defaults
    lgbm_n_estimators: int = 500
    lgbm_learning_rate: float = 0.05
    lgbm_max_depth: int = 6
    lgbm_num_leaves: int = 31
    lgbm_min_child_samples: int = 100
    lgbm_subsample: float = 0.8
    lgbm_colsample_bytree: float = 0.8

    # Optuna
    optuna_n_trials: int = Field(
        default=50,
        description="Number of Optuna hyperparameter search trials",
    )

    # Random seed
    random_seed: int = 42

    model_config = {"env_prefix": "MODEL_"}


class Settings(BaseSettings):
    """Root settings aggregating all sub-configurations."""

    binance: BinanceSettings = Field(default_factory=BinanceSettings)
    collector: CollectorSettings = Field(default_factory=CollectorSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)


# ─── Global singleton ─────────────────────────────────────────────────────────

_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
