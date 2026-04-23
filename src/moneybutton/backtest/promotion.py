"""Promotion gate (SPEC §5.1).

A strategy in SHADOW is only moved to LIVE if every backtest metric clears
its threshold. A Sharpe that's too high triggers an overfit warning that
requires an explicit operator override to unblock promotion.

Thresholds:
    Sharpe           >= 0.8  (required)
    Hit rate         >= strategy-specific (cal 0.52 / arb 0.80 / cons 0.70
                                             / news 0.53 / drift 0.54)
    Max drawdown     <= 0.25
    Num trades       >= 100
    Sharpe > 2.0     => OVERFIT_WARNING_REQUIRES_OVERRIDE

Side effect: when `persist_to` is given, a PROMOTED outcome updates the
strategies table (state -> LIVE, last_backtest_*, promoted_ts). Any rejection
outcome records the metrics but keeps the strategy in its existing state.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from moneybutton.backtest.metrics import BacktestResult
from moneybutton.core.config import STRATEGY_NAMES


class PromotionDecision(str, Enum):
    PROMOTED = "PROMOTED"
    REJECTED_LOW_SHARPE = "REJECTED_LOW_SHARPE"
    REJECTED_LOW_HITRATE = "REJECTED_LOW_HITRATE"
    REJECTED_HIGH_DD = "REJECTED_HIGH_DD"
    REJECTED_FEW_TRADES = "REJECTED_FEW_TRADES"
    OVERFIT_WARNING_REQUIRES_OVERRIDE = "OVERFIT_WARNING_REQUIRES_OVERRIDE"


MIN_SHARPE = 0.8
MAX_DRAWDOWN_PCT = 0.25
MIN_TRADES = 100
OVERFIT_SHARPE_THRESHOLD = 2.0

HITRATE_THRESHOLDS: dict[str, float] = {
    "calibration": 0.52,
    "arbitrage": 0.80,
    "consistency": 0.70,
    "news": 0.53,
    "drift": 0.54,
}


@dataclass(frozen=True)
class PromotionReport:
    outcome: PromotionDecision
    reason: str
    strategy: str
    metrics: BacktestResult


def check_strategy(
    strategy: str,
    result: BacktestResult,
    *,
    override_overfit: bool = False,
    persist_to: str | Path | None = None,
) -> PromotionReport:
    """Evaluate `result` against the promotion thresholds for `strategy`.

    Order of checks matters: structural problems (few trades) first, then
    quality thresholds (Sharpe, hit rate, drawdown), finally the overfit
    warning — so a 2.5-Sharpe + 40-trade run reports REJECTED_FEW_TRADES,
    which is the more actionable finding.
    """
    if strategy not in STRATEGY_NAMES:
        raise ValueError(f"unknown strategy: {strategy!r}")

    threshold = HITRATE_THRESHOLDS[strategy]

    if result.num_trades < MIN_TRADES:
        report = PromotionReport(
            outcome=PromotionDecision.REJECTED_FEW_TRADES,
            reason=f"num_trades={result.num_trades} < {MIN_TRADES}",
            strategy=strategy,
            metrics=result,
        )
    elif result.max_dd_pct > MAX_DRAWDOWN_PCT:
        report = PromotionReport(
            outcome=PromotionDecision.REJECTED_HIGH_DD,
            reason=f"max_dd_pct={result.max_dd_pct:.3f} > {MAX_DRAWDOWN_PCT}",
            strategy=strategy,
            metrics=result,
        )
    elif result.hit_rate < threshold:
        report = PromotionReport(
            outcome=PromotionDecision.REJECTED_LOW_HITRATE,
            reason=(
                f"hit_rate={result.hit_rate:.3f} < per-strategy threshold "
                f"{threshold:.3f}"
            ),
            strategy=strategy,
            metrics=result,
        )
    elif result.sharpe < MIN_SHARPE:
        report = PromotionReport(
            outcome=PromotionDecision.REJECTED_LOW_SHARPE,
            reason=f"sharpe={result.sharpe:.3f} < {MIN_SHARPE}",
            strategy=strategy,
            metrics=result,
        )
    elif result.sharpe > OVERFIT_SHARPE_THRESHOLD and not override_overfit:
        report = PromotionReport(
            outcome=PromotionDecision.OVERFIT_WARNING_REQUIRES_OVERRIDE,
            reason=(
                f"sharpe={result.sharpe:.3f} > {OVERFIT_SHARPE_THRESHOLD} — "
                f"backtest likely overfit; re-run with override_overfit=True "
                f"only after reviewing the report and understanding why the "
                f"Sharpe is this high."
            ),
            strategy=strategy,
            metrics=result,
        )
    else:
        report = PromotionReport(
            outcome=PromotionDecision.PROMOTED,
            reason=(
                f"sharpe={result.sharpe:.3f} hit={result.hit_rate:.3f} "
                f"dd={result.max_dd_pct:.3f} n={result.num_trades}"
            ),
            strategy=strategy,
            metrics=result,
        )

    if persist_to is not None:
        _persist(report, Path(persist_to))
    return report


def _persist(report: PromotionReport, db_path: Path) -> None:
    """Write backtest metrics to the strategies table; flip state only on PROMOTED."""
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    try:
        if report.outcome is PromotionDecision.PROMOTED:
            conn.execute(
                """
                UPDATE strategies SET
                  state = 'LIVE',
                  last_backtest_ts = ?,
                  last_backtest_sharpe = ?,
                  last_backtest_hitrate = ?,
                  last_backtest_max_dd = ?,
                  last_backtest_num_trades = ?,
                  promoted_ts = ?
                WHERE name = ?
                """,
                (
                    now,
                    report.metrics.sharpe,
                    report.metrics.hit_rate,
                    report.metrics.max_dd_pct,
                    report.metrics.num_trades,
                    now,
                    report.strategy,
                ),
            )
        else:
            # Record the backtest metrics but leave state unchanged.
            conn.execute(
                """
                UPDATE strategies SET
                  last_backtest_ts = ?,
                  last_backtest_sharpe = ?,
                  last_backtest_hitrate = ?,
                  last_backtest_max_dd = ?,
                  last_backtest_num_trades = ?
                WHERE name = ?
                """,
                (
                    now,
                    report.metrics.sharpe,
                    report.metrics.hit_rate,
                    report.metrics.max_dd_pct,
                    report.metrics.num_trades,
                    report.strategy,
                ),
            )
        conn.commit()
    finally:
        conn.close()
