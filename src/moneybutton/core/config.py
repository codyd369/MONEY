"""Typed, validated runtime configuration.

All safety-critical defaults and hard ceilings are encoded here. The
application refuses to start if any invariant from SPEC.md §5.3 is violated.

Usage:
    from moneybutton.core.config import get_settings
    settings = get_settings()  # cached; raises on bad .env
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Hard ceilings (SPEC.md §5.3). These are the maximum values the operator can
# configure via .env; the app will refuse to start if any setting exceeds its
# ceiling. Raising a ceiling requires editing this source file, which is a
# friction we want — a .env typo should never open the floodgates.
# ---------------------------------------------------------------------------
HARD_MAX_POSITION_USD = 100.0
HARD_MAX_OPEN_POSITIONS = 20
HARD_DAILY_LOSS_LIMIT_USD = 200.0
HARD_BANKROLL_USD = 500.0
HARD_KELLY_FRACTION = 0.25

STRATEGY_NAMES: tuple[str, ...] = (
    "calibration",
    "arbitrage",
    "consistency",
    "news",
    "drift",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------- Runtime safety ------------------------------------------------
    dry_run: bool = True
    kalshi_env: Literal["demo", "prod"] = "demo"

    # -------- Bankroll & risk caps -----------------------------------------
    bankroll_usd: float = Field(..., gt=0)
    max_position_usd: float = Field(25.0, gt=0)
    max_open_positions: int = Field(8, gt=0)
    daily_loss_limit_usd: float = Field(50.0, gt=0)
    min_trade_usd: float = Field(5.0, gt=0)
    min_liquidity_usd: float = Field(500.0, ge=0)
    kelly_fraction: float = Field(0.10, gt=0)

    # -------- Per-strategy allocation --------------------------------------
    allocation_calibration: float = Field(0.40, ge=0, le=1)
    allocation_arbitrage: float = Field(0.25, ge=0, le=1)
    allocation_consistency: float = Field(0.15, ge=0, le=1)
    allocation_news: float = Field(0.10, ge=0, le=1)
    allocation_drift: float = Field(0.10, ge=0, le=1)

    # -------- Strategy tuning ---------------------------------------------
    min_edge_bps_calibration: int = 500
    safety_margin_bps_arbitrage: int = 100
    slippage_bps_backtest: int = 25

    # -------- Kalshi -------------------------------------------------------
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = "data/kalshi_private_key.pem"

    # -------- Polymarket ---------------------------------------------------
    polymarket_enabled: bool = True
    polymarket_wallet_private_key: SecretStr = SecretStr("")

    # -------- Sportsbook signals ------------------------------------------
    oddsapi_enabled: bool = True
    oddsapi_key: SecretStr = SecretStr("")

    # -------- News APIs ----------------------------------------------------
    newsapi_key: SecretStr = SecretStr("")
    eventregistry_key: SecretStr = SecretStr("")

    # -------- Anthropic ---------------------------------------------------
    anthropic_api_key: SecretStr = SecretStr("")
    llm_model_general: str = "claude-opus-4-7"
    llm_model_news: str = "claude-haiku-4-5"

    # -------- Content publishing ------------------------------------------
    devto_api_key: SecretStr = SecretStr("")
    auto_publish_devto: bool = False

    # -------- Notifications -----------------------------------------------
    discord_webhook_url: SecretStr = SecretStr("")
    slack_webhook_url: SecretStr = SecretStr("")

    # -------- Scheduling --------------------------------------------------
    timezone: str = "America/New_York"

    # -------- Paths --------------------------------------------------------
    data_dir: str = "data"
    logs_dir: str = "logs"
    sqlite_path: str = "data/moneybutton.db"
    kill_file: str = ".kill"

    # -------- Validators --------------------------------------------------
    @field_validator("bankroll_usd")
    @classmethod
    def _bankroll_ceiling(cls, v: float) -> float:
        if v > HARD_BANKROLL_USD:
            raise ValueError(
                f"BANKROLL_USD={v} exceeds hard ceiling ${HARD_BANKROLL_USD}. "
                f"Edit src/moneybutton/core/config.py (HARD_BANKROLL_USD) to raise it."
            )
        return v

    @field_validator("max_position_usd")
    @classmethod
    def _position_ceiling(cls, v: float) -> float:
        if v > HARD_MAX_POSITION_USD:
            raise ValueError(
                f"MAX_POSITION_USD={v} exceeds hard ceiling ${HARD_MAX_POSITION_USD}."
            )
        return v

    @field_validator("max_open_positions")
    @classmethod
    def _positions_ceiling(cls, v: int) -> int:
        if v > HARD_MAX_OPEN_POSITIONS:
            raise ValueError(
                f"MAX_OPEN_POSITIONS={v} exceeds hard ceiling {HARD_MAX_OPEN_POSITIONS}."
            )
        return v

    @field_validator("daily_loss_limit_usd")
    @classmethod
    def _daily_loss_ceiling(cls, v: float) -> float:
        if v > HARD_DAILY_LOSS_LIMIT_USD:
            raise ValueError(
                f"DAILY_LOSS_LIMIT_USD={v} exceeds hard ceiling ${HARD_DAILY_LOSS_LIMIT_USD}."
            )
        return v

    @field_validator("kelly_fraction")
    @classmethod
    def _kelly_ceiling(cls, v: float) -> float:
        if v > HARD_KELLY_FRACTION:
            raise ValueError(
                f"KELLY_FRACTION={v} exceeds hard ceiling {HARD_KELLY_FRACTION}. "
                f"Fractional Kelly > 0.25 is reckless on an unproven edge."
            )
        return v

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"TIMEZONE={v!r} is not a valid IANA zone") from exc
        return v

    @model_validator(mode="after")
    def _allocations_sum(self) -> "Settings":
        total = (
            self.allocation_calibration
            + self.allocation_arbitrage
            + self.allocation_consistency
            + self.allocation_news
            + self.allocation_drift
        )
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"Per-strategy ALLOCATION_* fractions sum to {total:.4f} > 1.0. "
                f"Adjust in .env so they sum to at most 1.0."
            )
        return self

    @model_validator(mode="after")
    def _min_trade_below_max(self) -> "Settings":
        if self.min_trade_usd > self.max_position_usd:
            raise ValueError(
                f"MIN_TRADE_USD={self.min_trade_usd} > MAX_POSITION_USD="
                f"{self.max_position_usd}; no trade would ever be placeable."
            )
        return self

    @model_validator(mode="after")
    def _live_requires_prod_env(self) -> "Settings":
        # Belt and braces: even if DRY_RUN=false, KALSHI_ENV must be explicit.
        if not self.dry_run and self.kalshi_env not in ("demo", "prod"):
            raise ValueError("KALSHI_ENV must be 'demo' or 'prod' when DRY_RUN=false")
        return self

    # -------- Derived helpers --------------------------------------------
    @property
    def allocations(self) -> dict[str, float]:
        return {
            "calibration": self.allocation_calibration,
            "arbitrage": self.allocation_arbitrage,
            "consistency": self.allocation_consistency,
            "news": self.allocation_news,
            "drift": self.allocation_drift,
        }

    @property
    def kill_file_path(self) -> Path:
        return Path(self.kill_file)

    @property
    def sqlite_db_path(self) -> Path:
        return Path(self.sqlite_path)

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def is_live_enabled(self) -> bool:
        """True only when every live-trading gate is open.

        DRY_RUN must be false AND KALSHI_ENV must be prod. Either tripped
        keeps the system in paper mode regardless of strategy promotion state.
        """
        return (not self.dry_run) and self.kalshi_env == "prod"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached validated Settings. Raises on .env error."""
    return Settings()  # type: ignore[call-arg]


def reset_settings_cache() -> None:
    """Clear the lru_cache. Only tests should call this."""
    get_settings.cache_clear()


if __name__ == "__main__":
    # Smoke test: `python -m moneybutton.core.config` prints loaded settings.
    import json

    s = get_settings()
    redacted = s.model_dump()
    for k, v in list(redacted.items()):
        if isinstance(v, SecretStr):
            redacted[k] = "***" if v.get_secret_value() else "(unset)"
    print(json.dumps(redacted, indent=2, default=str))
