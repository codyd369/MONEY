"""Shared pytest fixtures.

Every test that touches the safety layer needs (a) an isolated temp DB,
(b) an isolated .kill sentinel path, and (c) a Settings instance that points
to (a) and (b) without reading the real .env. Those three fixtures live here
so every test gets them consistently.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def tmp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Isolated temp paths + a clean Settings cache.

    Uses monkeypatch to set every env var the Settings model reads, then
    clears the lru_cache on get_settings so the next call re-reads env.
    """
    db_path = tmp_path / "test.db"
    kill_path = tmp_path / ".kill"
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    env = {
        "DRY_RUN": "true",
        "KALSHI_ENV": "demo",
        "BANKROLL_USD": "500",
        "MAX_POSITION_USD": "25",
        "MAX_OPEN_POSITIONS": "8",
        "DAILY_LOSS_LIMIT_USD": "50",
        "MIN_TRADE_USD": "5",
        "MIN_LIQUIDITY_USD": "500",
        "KELLY_FRACTION": "0.10",
        "ALLOCATION_CALIBRATION": "0.40",
        "ALLOCATION_ARBITRAGE": "0.25",
        "ALLOCATION_CONSISTENCY": "0.15",
        "ALLOCATION_NEWS": "0.10",
        "ALLOCATION_DRIFT": "0.10",
        "SQLITE_PATH": str(db_path),
        "KILL_FILE": str(kill_path),
        "DATA_DIR": str(data_dir),
        "LOGS_DIR": str(logs_dir),
        "TIMEZONE": "America/New_York",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # Also nuke any real .env so pydantic-settings doesn't pick it up.
    monkeypatch.chdir(tmp_path)

    from moneybutton.core.config import reset_settings_cache

    reset_settings_cache()

    yield {
        "db_path": db_path,
        "kill_path": kill_path,
        "data_dir": data_dir,
        "logs_dir": logs_dir,
        "env": env,
    }

    reset_settings_cache()


@pytest.fixture
def tmp_db(tmp_env: dict[str, Any]) -> Path:
    """tmp_env + an initialized DB."""
    from moneybutton.core.db import init_db

    init_db(tmp_env["db_path"])
    return tmp_env["db_path"]


@pytest.fixture
def override_env(monkeypatch: pytest.MonkeyPatch, tmp_env: dict[str, Any]):
    """Helper: `override_env(KEY='...')` to change one env var and reset cache."""
    from moneybutton.core.config import reset_settings_cache

    def _apply(**overrides: str) -> dict[str, Any]:
        for k, v in overrides.items():
            monkeypatch.setenv(k.upper(), v)
        reset_settings_cache()
        return tmp_env

    return _apply
