"""Orchestrator + paper-trade flow (SPEC §16 step 17).

Simulates one scheduler tick end-to-end without APScheduler or network:
  feed() -> MarketSnapshots -> scanner_fn -> SignalIntent -> Executor.execute
  -> SIMULATED order row in SQLite + audit row.

This proves the wiring is correct and that the kill switch, dedup, sizing
all still fire from the job entry point.
"""

from __future__ import annotations

import datetime as dt
import sqlite3

import numpy as np
import pandas as pd

from moneybutton.core.executor import Executor
from moneybutton.core.kill_switch import touch_kill
from moneybutton.scheduler.jobs import (
    MarketSnapshot,
    bankroll_snapshot_job,
    calibration_scan_job,
    health_check_job,
)


class _StubModel:
    def __init__(self, yes_prob: float, feature_columns: list[str]):
        self.yes_prob = yes_prob
        self.feature_columns = feature_columns

    def predict_yes_prob(self, X):
        return np.array([self.yes_prob] * len(X))


def _snapshot(ticker: str, price_cents: int, liquidity: float = 1000) -> MarketSnapshot:
    ts = "2025-01-10T12:00:00+00:00"
    prices = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "ts": ts,
                "yes_bid_close": price_cents - 1,
                "yes_ask_close": price_cents + 1,
                "last_price_close": price_cents,
                "volume": 20,
                "open_interest": 500,
            }
        ]
    )
    prices["_ts_dt"] = pd.to_datetime(prices["ts"], utc=True)
    market = {
        "ticker": ticker,
        "category": "POLITICS",
        "status": "active",
        "open_time": "2025-01-01T12:00:00+00:00",
        "close_time": "2025-01-20T18:00:00+00:00",
        "volume": 200,
        "open_interest": 500,
    }
    return MarketSnapshot(
        market=market,
        price_frame=prices,
        liquidity_usd=liquidity,
        entry_price_cents=price_cents,
    )


def test_health_check_records_audit(tmp_env, tmp_db):
    status = health_check_job()
    assert status["tripped_now"] is False
    assert status["kill_engaged"] is False
    with sqlite3.connect(tmp_db) as conn:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM audit WHERE actor='safety.health_check'"
        ).fetchone()
    assert n == 1


def test_health_check_trips_on_breach(tmp_env, tmp_db):
    # Seed today with a loss that exceeds DAILY_LOSS_LIMIT_USD (=$50 in fixture).
    with sqlite3.connect(tmp_db) as conn:
        today = dt.datetime.now(dt.timezone.utc).date().isoformat()
        conn.execute(
            "INSERT INTO daily_pnl (date, strategy, realized_usd) VALUES (?, 'calibration', ?)",
            (today, -60.0),
        )
        conn.commit()
    status = health_check_job()
    assert status["tripped_now"] is True
    assert status["kill_engaged"] is True


def test_bankroll_snapshot_writes_row(tmp_env, tmp_db):
    res = bankroll_snapshot_job()
    assert res["balance_usd"] == 500.0
    with sqlite3.connect(tmp_db) as conn:
        rows = conn.execute("SELECT balance_usd, source FROM bankroll").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 500.0


def test_calibration_scan_paper_flow(tmp_env, tmp_db):
    """Scanner sees a snapshot, produces a signal, executor simulates it."""
    from moneybutton.features.pipeline import compute_features
    from moneybutton.strategies.calibration_strat import make_scanner

    snap = _snapshot("TST-PAPER-1", price_cents=40)
    feats = compute_features(snap.market, dt.datetime.now(dt.timezone.utc), snap.price_frame)
    feat_cols = sorted(k for k in feats if k not in ("ticker", "as_of_ts"))
    model = _StubModel(yes_prob=0.55, feature_columns=feat_cols)
    scanner = make_scanner(model=model, min_edge_bps=500)

    executor = Executor(db_path=tmp_db)
    summary = calibration_scan_job(
        feed=lambda: [snap],
        scanner_fn=scanner,
        executor=executor,
    )
    assert summary["n_snapshots"] == 1
    assert summary["n_signals"] == 1
    assert summary["n_simulated"] == 1  # DRY_RUN => SIMULATED
    with sqlite3.connect(tmp_db) as conn:
        rows = conn.execute("SELECT status, ticker FROM orders").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "simulated"
    assert rows[0][1] == "TST-PAPER-1"


def test_calibration_scan_skips_when_killed(tmp_env, tmp_db):
    touch_kill(tmp_env["kill_path"], reason="test")
    summary = calibration_scan_job(
        feed=lambda: [_snapshot("TST-KILL-1", 40)],
        scanner_fn=lambda ev: None,
        executor=Executor(db_path=tmp_db),
    )
    assert summary["outcome"] == "SKIPPED_KILL_SWITCH"
    with sqlite3.connect(tmp_db) as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM orders").fetchone()
    assert n == 0


def test_build_scheduler_registers_expected_jobs(tmp_env, tmp_db):
    from moneybutton.scheduler.scheduler import build_scheduler

    scheduler = build_scheduler(
        calibration_feed_fn=lambda: [],
        calibration_scanner_fn=lambda ev: None,
    )
    job_ids = {j.id for j in scheduler.get_jobs()}
    assert {"safety.health_check", "data.bankroll_snapshot", "strategy.calibration.scan"} <= job_ids


def test_scheduler_wraps_errors_without_crashing(tmp_env, tmp_db):
    """If a job raises, _wrap_safe catches and the outer doesn't propagate."""
    from moneybutton.scheduler.scheduler import _wrap_safe

    def _bad() -> None:
        raise RuntimeError("boom")

    wrapped = _wrap_safe(_bad)
    wrapped()  # should not raise
