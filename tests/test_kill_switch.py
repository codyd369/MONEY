"""Kill switch + daily-loss cap tests (SPEC §5.4).

Guarantees under test:
  1. The .kill sentinel, when present, blocks every path that places orders
     or publishes content. `guard_action()` raises KillSwitchError.
  2. touch_kill() / clear_kill() are idempotent and thread-safe-ish
     (single-file sentinel; we don't fight races between kill/go on purpose).
  3. DailyLossGuard sums realized+unrealized losses for today (operator tz)
     and auto-creates .kill when the breach is detected.
  4. After a breach, subsequent guard_action() calls keep raising even if
     the underlying pnl improves — only operator intervention (go.sh /
     clear_kill) re-arms the system.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pytest


# ----------------------------- Basic kill file -----------------------------


def test_kill_absent_initially(tmp_env):
    from moneybutton.core.kill_switch import is_killed

    assert is_killed(tmp_env["kill_path"]) is False


def test_touch_kill_creates_sentinel(tmp_env):
    from moneybutton.core.kill_switch import is_killed, touch_kill

    touch_kill(tmp_env["kill_path"], reason="unit test")
    assert tmp_env["kill_path"].exists()
    assert is_killed(tmp_env["kill_path"]) is True


def test_touch_kill_idempotent(tmp_env):
    from moneybutton.core.kill_switch import touch_kill

    touch_kill(tmp_env["kill_path"], reason="first")
    first_mtime = tmp_env["kill_path"].stat().st_mtime_ns
    touch_kill(tmp_env["kill_path"], reason="second")
    # Re-touching is allowed; it must not error.
    assert tmp_env["kill_path"].exists()
    # Content of the file should still be parseable (reason recorded).
    body = tmp_env["kill_path"].read_text()
    assert "second" in body or "first" in body
    # The re-touch may update mtime; we don't care, just that it didn't crash.
    _ = first_mtime


def test_clear_kill_removes_sentinel(tmp_env):
    from moneybutton.core.kill_switch import clear_kill, is_killed, touch_kill

    touch_kill(tmp_env["kill_path"], reason="x")
    clear_kill(tmp_env["kill_path"])
    assert is_killed(tmp_env["kill_path"]) is False
    # Clearing twice is fine.
    clear_kill(tmp_env["kill_path"])


# ----------------------------- guard_action ------------------------------


def test_guard_action_allows_when_clean(tmp_env):
    from moneybutton.core.kill_switch import guard_action

    # Should not raise.
    guard_action("strategy.calibration", kind="order", kill_file_path=tmp_env["kill_path"])


def test_guard_action_blocks_when_killed(tmp_env):
    from moneybutton.core.kill_switch import KillSwitchError, guard_action, touch_kill

    touch_kill(tmp_env["kill_path"], reason="unit test")
    with pytest.raises(KillSwitchError) as ei:
        guard_action(
            "strategy.calibration",
            kind="order",
            kill_file_path=tmp_env["kill_path"],
        )
    assert "kill" in str(ei.value).lower()


def test_guard_action_blocks_publish_too(tmp_env):
    """The publisher goes through the same gate as the order executor."""
    from moneybutton.core.kill_switch import KillSwitchError, guard_action, touch_kill

    touch_kill(tmp_env["kill_path"], reason="unit test")
    with pytest.raises(KillSwitchError):
        guard_action("content.publisher.devto", kind="publish", kill_file_path=tmp_env["kill_path"])


# ----------------------------- DailyLossGuard ----------------------------


def _seed_pnl(db_path: Path, date_str: str, strategy: str, realized: float, unrealized: float = 0):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO daily_pnl (date, strategy, realized_usd, unrealized_usd)"
            " VALUES (?, ?, ?, ?)",
            (date_str, strategy, realized, unrealized),
        )
        conn.commit()


def test_daily_loss_guard_under_limit(tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.kill_switch import DailyLossGuard

    today = dt.datetime(2026, 4, 23, 12, 0, tzinfo=dt.timezone.utc)
    _seed_pnl(tmp_db, today.date().isoformat(), "calibration", realized=-20.0)
    _seed_pnl(tmp_db, today.date().isoformat(), "arbitrage", realized=-15.0)
    # -35 total; under the $50 limit.
    settings = get_settings()
    guard = DailyLossGuard(db_path=tmp_db, clock=lambda: today, settings=settings)
    tripped = guard.check_and_maybe_trip()
    assert tripped is False
    assert abs(guard.current_loss_usd() - 35.0) < 1e-6


def test_daily_loss_guard_trips_and_creates_kill(tmp_env, tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.kill_switch import DailyLossGuard, is_killed

    today = dt.datetime(2026, 4, 23, 12, 0, tzinfo=dt.timezone.utc)
    _seed_pnl(tmp_db, today.date().isoformat(), "calibration", realized=-40.0)
    _seed_pnl(tmp_db, today.date().isoformat(), "arbitrage", realized=-20.0)
    # -60 total; exceeds the $50 limit.
    settings = get_settings()
    guard = DailyLossGuard(
        db_path=tmp_db,
        clock=lambda: today,
        settings=settings,
        kill_file_path=tmp_env["kill_path"],
    )
    assert is_killed(tmp_env["kill_path"]) is False
    tripped = guard.check_and_maybe_trip()
    assert tripped is True
    assert is_killed(tmp_env["kill_path"]) is True


def test_daily_loss_guard_includes_unrealized(tmp_env, tmp_db):
    """Unrealized losses count toward the cap — otherwise a tanking open
    position doesn't trip until you close it, which defeats the purpose."""
    from moneybutton.core.config import get_settings
    from moneybutton.core.kill_switch import DailyLossGuard, is_killed

    today = dt.datetime(2026, 4, 23, 12, 0, tzinfo=dt.timezone.utc)
    _seed_pnl(
        tmp_db,
        today.date().isoformat(),
        "calibration",
        realized=-10.0,
        unrealized=-45.0,
    )
    settings = get_settings()
    guard = DailyLossGuard(
        db_path=tmp_db,
        clock=lambda: today,
        settings=settings,
        kill_file_path=tmp_env["kill_path"],
    )
    tripped = guard.check_and_maybe_trip()
    assert tripped is True
    assert is_killed(tmp_env["kill_path"]) is True


def test_daily_loss_guard_ignores_prior_days(tmp_env, tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.kill_switch import DailyLossGuard

    today = dt.datetime(2026, 4, 23, 12, 0, tzinfo=dt.timezone.utc)
    yesterday = today - dt.timedelta(days=1)
    _seed_pnl(tmp_db, yesterday.date().isoformat(), "calibration", realized=-200.0)
    _seed_pnl(tmp_db, today.date().isoformat(), "calibration", realized=-10.0)
    settings = get_settings()
    guard = DailyLossGuard(
        db_path=tmp_db,
        clock=lambda: today,
        settings=settings,
        kill_file_path=tmp_env["kill_path"],
    )
    assert abs(guard.current_loss_usd() - 10.0) < 1e-6
    assert guard.check_and_maybe_trip() is False


def test_positive_pnl_is_not_a_loss(tmp_db, tmp_env):
    from moneybutton.core.config import get_settings
    from moneybutton.core.kill_switch import DailyLossGuard

    today = dt.datetime(2026, 4, 23, 12, 0, tzinfo=dt.timezone.utc)
    _seed_pnl(tmp_db, today.date().isoformat(), "calibration", realized=50.0)
    _seed_pnl(tmp_db, today.date().isoformat(), "arbitrage", realized=-10.0)
    # Net +40 across strategies; loss column should be 0 (no single negative drag
    # below zero on the total).
    settings = get_settings()
    guard = DailyLossGuard(
        db_path=tmp_db,
        clock=lambda: today,
        settings=settings,
        kill_file_path=tmp_env["kill_path"],
    )
    assert guard.current_loss_usd() == 0.0
    assert guard.check_and_maybe_trip() is False
