"""Strategy promotion-gate tests (SPEC §5.1).

A strategy in SHADOW is only moved to LIVE by check_strategy() when every
threshold is met. This file enumerates all rejection paths and the overfit
warning required by §5.1.
"""

from __future__ import annotations

import pytest


def _make_result(
    sharpe: float = 1.2,
    sortino: float = 1.3,
    max_dd_pct: float = 0.15,
    hit_rate: float = 0.60,
    num_trades: int = 150,
    expectancy_usd: float = 0.15,
    calmar: float = 2.0,
):
    from moneybutton.backtest.metrics import BacktestResult

    return BacktestResult(
        sharpe=sharpe,
        sortino=sortino,
        max_dd_pct=max_dd_pct,
        hit_rate=hit_rate,
        num_trades=num_trades,
        expectancy_usd=expectancy_usd,
        calmar=calmar,
    )


def test_happy_path_promotes():
    from moneybutton.backtest.promotion import PromotionDecision, check_strategy

    decision = check_strategy("calibration", _make_result())
    assert decision.outcome is PromotionDecision.PROMOTED


def test_low_sharpe_rejected():
    from moneybutton.backtest.promotion import PromotionDecision, check_strategy

    decision = check_strategy("calibration", _make_result(sharpe=0.5))
    assert decision.outcome is PromotionDecision.REJECTED_LOW_SHARPE


def test_low_hitrate_rejected_per_strategy():
    from moneybutton.backtest.promotion import PromotionDecision, check_strategy

    # Calibration threshold = 0.52; arbitrage threshold = 0.80.
    # A hit_rate of 0.55 clears calibration but fails arbitrage.
    d1 = check_strategy("calibration", _make_result(hit_rate=0.55))
    d2 = check_strategy("arbitrage", _make_result(hit_rate=0.55))
    assert d1.outcome is PromotionDecision.PROMOTED
    assert d2.outcome is PromotionDecision.REJECTED_LOW_HITRATE


def test_drawdown_over_limit_rejected():
    from moneybutton.backtest.promotion import PromotionDecision, check_strategy

    decision = check_strategy("calibration", _make_result(max_dd_pct=0.30))
    assert decision.outcome is PromotionDecision.REJECTED_HIGH_DD


def test_too_few_trades_rejected():
    from moneybutton.backtest.promotion import PromotionDecision, check_strategy

    decision = check_strategy("calibration", _make_result(num_trades=50))
    assert decision.outcome is PromotionDecision.REJECTED_FEW_TRADES


def test_high_sharpe_triggers_overfit_warning():
    from moneybutton.backtest.promotion import PromotionDecision, check_strategy

    decision = check_strategy("calibration", _make_result(sharpe=2.5))
    assert decision.outcome is PromotionDecision.OVERFIT_WARNING_REQUIRES_OVERRIDE
    assert "overfit" in decision.reason.lower()


def test_overfit_override_promotes():
    from moneybutton.backtest.promotion import PromotionDecision, check_strategy

    decision = check_strategy(
        "calibration", _make_result(sharpe=2.5), override_overfit=True
    )
    assert decision.outcome is PromotionDecision.PROMOTED


def test_unknown_strategy_rejected():
    from moneybutton.backtest.promotion import check_strategy

    with pytest.raises(ValueError):
        check_strategy("not_a_strategy", _make_result())


def test_db_updated_on_promotion(tmp_db):
    """Promotion flips the strategies table state to LIVE and records metrics."""
    import sqlite3

    from moneybutton.backtest.promotion import PromotionDecision, check_strategy

    decision = check_strategy(
        "calibration", _make_result(), persist_to=tmp_db
    )
    assert decision.outcome is PromotionDecision.PROMOTED

    with sqlite3.connect(tmp_db) as conn:
        row = conn.execute(
            "SELECT state, last_backtest_sharpe, last_backtest_hitrate,"
            " last_backtest_max_dd, last_backtest_num_trades, promoted_ts"
            " FROM strategies WHERE name=?",
            ("calibration",),
        ).fetchone()
    assert row is not None
    (state, sharpe, hr, dd, n, promoted_ts) = row
    assert state == "LIVE"
    assert sharpe == pytest.approx(1.2)
    assert hr == pytest.approx(0.60)
    assert dd == pytest.approx(0.15)
    assert n == 150
    assert promoted_ts is not None


def test_db_not_updated_on_rejection(tmp_db):
    """A failing backtest must NOT flip state to LIVE."""
    import sqlite3

    from moneybutton.backtest.promotion import PromotionDecision, check_strategy

    decision = check_strategy(
        "calibration", _make_result(sharpe=0.5), persist_to=tmp_db
    )
    assert decision.outcome is PromotionDecision.REJECTED_LOW_SHARPE

    with sqlite3.connect(tmp_db) as conn:
        (state,) = conn.execute(
            "SELECT state FROM strategies WHERE name=?", ("calibration",)
        ).fetchone()
    assert state == "SHADOW"  # stays in shadow
