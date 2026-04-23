"""Executor path tests (SPEC §12.2).

No network calls: in DRY_RUN mode the executor writes a simulated order
row and never touches Kalshi. These tests verify each gate in isolation.
"""

from __future__ import annotations

import sqlite3

import pytest

from moneybutton.core.config import get_settings
from moneybutton.core.executor import Executor, make_client_order_id
from moneybutton.core.kill_switch import touch_kill
from moneybutton.strategies.base import Signal


def _signal(strategy: str = "calibration", ticker: str = "TST-1", edge_bps: int = 800) -> Signal:
    return Signal(
        strategy=strategy,
        ticker=ticker,
        side="yes",
        edge_bps=edge_bps,
        confidence="med",
        suggested_size_usd=10.0,
        reasoning={"why": "unit test"},
    )


def test_make_client_order_id_is_stable():
    s1 = _signal()
    s2 = _signal()
    assert make_client_order_id(s1) == make_client_order_id(s2)


def test_executor_simulates_in_dry_run(tmp_db):
    ex = Executor(db_path=tmp_db)
    report = ex.execute(_signal(), entry_price_cents=40, liquidity_usd=1000)
    assert report.outcome == "SIMULATED"
    assert report.size_usd > 0
    with sqlite3.connect(tmp_db) as conn:
        rows = conn.execute("SELECT status, client_request_id, count FROM orders").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "simulated"
    assert rows[0][1] == report.client_order_id


def test_executor_blocks_on_kill_switch(tmp_env, tmp_db):
    touch_kill(tmp_env["kill_path"], reason="test")
    ex = Executor(db_path=tmp_db)
    report = ex.execute(_signal(), entry_price_cents=40, liquidity_usd=1000)
    assert report.outcome == "REJECTED_KILL_SWITCH"
    # No order row written.
    with sqlite3.connect(tmp_db) as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM orders").fetchone()
    assert n == 0


def test_executor_dedupes_identical_signals(tmp_db):
    ex = Executor(db_path=tmp_db)
    r1 = ex.execute(_signal(), entry_price_cents=40, liquidity_usd=1000)
    r2 = ex.execute(_signal(), entry_price_cents=40, liquidity_usd=1000)
    assert r1.outcome == "SIMULATED"
    assert r2.outcome == "REJECTED_DUP"
    assert r1.client_order_id == r2.client_order_id
    # Only one row in orders.
    with sqlite3.connect(tmp_db) as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM orders").fetchone()
    assert n == 1


def test_executor_rejects_low_liquidity(tmp_db):
    ex = Executor(db_path=tmp_db)
    # MIN_LIQUIDITY_USD defaults to $500 in the tmp_env fixture.
    report = ex.execute(_signal(), entry_price_cents=40, liquidity_usd=100)
    assert report.outcome == "REJECTED_LIQUIDITY"


def test_executor_rejects_zero_size(tmp_db):
    ex = Executor(db_path=tmp_db)
    # A tiny edge that would produce size below MIN_TRADE_USD on Kelly.
    sig = _signal(edge_bps=50)  # 50 bps edge at p=40c -> sub-dollar size
    report = ex.execute(sig, entry_price_cents=40, liquidity_usd=1000)
    assert report.outcome == "REJECTED_SIZE_ZERO"


def test_executor_rejects_invalid_entry_price(tmp_db):
    ex = Executor(db_path=tmp_db)
    report = ex.execute(_signal(), entry_price_cents=0, liquidity_usd=1000)
    assert report.outcome == "REJECTED_SIZE_ZERO"


def test_simulated_does_not_update_positions(tmp_db):
    """SPEC §12.2: simulated orders record in `orders` but DO NOT open
    positions. This prevents fake exposure from eating the allocation cap."""
    ex = Executor(db_path=tmp_db)
    ex.execute(_signal(), entry_price_cents=40, liquidity_usd=1000)
    with sqlite3.connect(tmp_db) as conn:
        (n_orders,) = conn.execute("SELECT COUNT(*) FROM orders").fetchone()
        (n_positions,) = conn.execute("SELECT COUNT(*) FROM positions").fetchone()
    assert n_orders == 1
    assert n_positions == 0


def test_audit_row_recorded_per_execution(tmp_db):
    ex = Executor(db_path=tmp_db)
    ex.execute(_signal(), entry_price_cents=40, liquidity_usd=1000)
    with sqlite3.connect(tmp_db) as conn:
        rows = conn.execute("SELECT actor, action, outcome FROM audit").fetchall()
    # Executor writes an audit row whose outcome is SIMULATED.
    assert any(r[2] == "SIMULATED" for r in rows)
