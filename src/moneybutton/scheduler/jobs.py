"""Scheduled jobs (SPEC §14).

Each job is a plain function with no parameters beyond a `feed` (for
strategies that need live Kalshi data). The scheduler constructs jobs
with their dependencies and wraps them in try/except + audit.

Jobs:
  - safety.health_check     every 5 min
  - strategy.calibration    every 15 min  (LIVE + demo paper)
  - data.bankroll_snapshot  hourly
  - reports.weekly_review   Sunday 23:00 local  (TODO: wired in step 24)

The content jobs (content.topics/write/publish) and the other four
strategy scans are added as those steps land.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from moneybutton.core.audit import record as audit_record
from moneybutton.core.config import Settings, get_settings
from moneybutton.core.executor import Executor
from moneybutton.core.kill_switch import DailyLossGuard, is_killed
from moneybutton.core.notifier import notify
from moneybutton.features.common import parse_ts
from moneybutton.strategies.base import Signal

log = logging.getLogger("moneybutton.scheduler")


# ======================== safety.health_check ==========================


def health_check_job(
    *,
    settings: Settings | None = None,
    db_path: str | Path | None = None,
) -> dict:
    """Evaluate daily loss, trip kill switch if breached, log status."""
    settings = settings or get_settings()
    db_path = db_path or settings.sqlite_db_path

    guard = DailyLossGuard(
        db_path=db_path,
        settings=settings,
        kill_file_path=settings.kill_file_path,
    )
    loss_usd = guard.current_loss_usd()
    tripped = guard.check_and_maybe_trip()
    killed = is_killed(settings.kill_file_path)
    status = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "daily_loss_usd": loss_usd,
        "daily_loss_limit_usd": settings.daily_loss_limit_usd,
        "tripped_now": tripped,
        "kill_engaged": killed,
    }
    if tripped:
        notify(
            f"DAILY LOSS CAP BREACHED: loss=${loss_usd:.2f} "
            f">= ${settings.daily_loss_limit_usd:.2f}. Kill switch engaged.",
            level="critical",
            extra=status,
        )
    audit_record(
        actor="safety.health_check",
        action="health_check",
        payload=status,
        outcome="TRIPPED" if tripped else ("KILLED" if killed else "OK"),
    )
    return status


# ====================== strategy.calibration.scan =======================


@dataclass
class MarketSnapshot:
    """The minimum a live-feed needs to supply per market for the calibration
    scanner. Matches the shape the backtest already uses."""

    market: dict
    price_frame: pd.DataFrame
    liquidity_usd: float
    entry_price_cents: int


LiveFeedFn = Callable[[], list[MarketSnapshot]]


def calibration_scan_job(
    *,
    feed: LiveFeedFn,
    scanner_fn: Callable,
    executor: Executor | None = None,
    settings: Settings | None = None,
    db_path: str | Path | None = None,
) -> dict:
    """Scan live markets, produce calibration signals, dispatch to executor.

    `feed` returns the current MarketSnapshots. `scanner_fn` is the closure
    returned by strategies.calibration_strat.make_scanner. Both are injected
    so the same job body is used for live (real feed + real model) and
    paper (synthetic feed + loaded model).
    """
    settings = settings or get_settings()
    db_path = db_path or settings.sqlite_db_path
    executor = executor or Executor(settings=settings, db_path=db_path)

    # Kill-switch is cheap to check at the job boundary too, so we can skip
    # even iterating the feed when engaged.
    if is_killed(settings.kill_file_path):
        return {"outcome": "SKIPPED_KILL_SWITCH", "n_snapshots": 0}

    snapshots = feed()
    n_snapshots = len(snapshots)
    n_signals = 0
    n_placed = 0
    n_simulated = 0
    n_rejected = 0

    for snap in snapshots:
        from moneybutton.backtest.engine import DecisionInput
        ev = DecisionInput(
            market=snap.market,
            as_of_ts=dt.datetime.now(dt.timezone.utc),
            price_frame=snap.price_frame,
        )
        intent = scanner_fn(ev)
        if intent is None:
            continue
        n_signals += 1
        signal = Signal(
            strategy="calibration",
            ticker=snap.market["ticker"],
            side=intent.side,
            edge_bps=intent.edge_bps,
            confidence=intent.confidence,
            suggested_size_usd=intent.size_usd,
            reasoning=intent.reasoning or {},
        )
        report = executor.execute(
            signal,
            entry_price_cents=snap.entry_price_cents,
            liquidity_usd=snap.liquidity_usd,
        )
        if report.outcome == "PLACED":
            n_placed += 1
        elif report.outcome == "SIMULATED":
            n_simulated += 1
        else:
            n_rejected += 1

    summary = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_snapshots": n_snapshots,
        "n_signals": n_signals,
        "n_placed": n_placed,
        "n_simulated": n_simulated,
        "n_rejected": n_rejected,
    }
    audit_record(
        actor="scheduler.calibration_scan",
        action="calibration_scan",
        payload=summary,
        outcome="OK",
    )
    return summary


# ====================== data.bankroll_snapshot ==========================


def bankroll_snapshot_job(
    *,
    settings: Settings | None = None,
    db_path: str | Path | None = None,
) -> dict:
    """Record the current bankroll (from settings or Kalshi portfolio balance)
    to the bankroll table."""
    settings = settings or get_settings()
    db_path = db_path or settings.sqlite_db_path
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    # For paper and dry-run modes we just record the configured value;
    # live mode would pull from Kalshi's /portfolio/balance endpoint.
    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO bankroll (ts, balance_usd, source) VALUES (?, ?, ?)",
            (now, settings.bankroll_usd, "configured"),
        )
        conn.commit()
    return {"ts": now, "balance_usd": settings.bankroll_usd, "source": "configured"}
