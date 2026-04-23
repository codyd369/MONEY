"""Backtest result metrics (SPEC §11.4).

This file defines only the container types + basic metric computation. The
walk-forward engine (backtest/engine.py) produces an equity curve and a
per-trade ledger; the functions here summarize them. Kept dependency-light
so the promotion gate (SPEC §5.1) can be exercised with hand-built results
in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    """Summary metrics for one backtest run.

    `max_dd_pct` is expressed as a fraction (0..1) of allocated capital,
    NOT basis points, NOT percent-of-peak; it is the biggest peak-to-trough
    drop in the equity curve divided by allocated capital.
    """

    sharpe: float
    sortino: float
    max_dd_pct: float
    hit_rate: float
    num_trades: int
    expectancy_usd: float
    calmar: float
    equity_curve: list[float] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def annualized_sharpe(returns: Sequence[float], periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


def annualized_sortino(returns: Sequence[float], periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    downside = [min(0.0, r) for r in returns]
    dvar = sum(d * d for d in downside) / (n - 1)
    dstd = math.sqrt(dvar)
    if dstd == 0:
        return 0.0
    return (mean / dstd) * math.sqrt(periods_per_year)


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Return max drawdown as a positive fraction of peak (0..1)."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            dd = (peak - v) / peak
            worst = max(worst, dd)
    return worst


def calmar(equity_curve: Sequence[float], years: float) -> float:
    if not equity_curve or years <= 0:
        return 0.0
    total_return = equity_curve[-1] / equity_curve[0] - 1.0 if equity_curve[0] else 0.0
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0
    dd = max_drawdown(equity_curve)
    if dd <= 0:
        return 0.0
    return annual_return / dd


def summarize(
    equity_curve: Sequence[float],
    trades: Sequence[dict],
    *,
    years: float,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> BacktestResult:
    """Build a BacktestResult from an equity curve + trade ledger.

    Each trade is expected to have keys: pnl_usd, entry_ts, exit_ts.
    """
    returns: list[float] = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        curr = equity_curve[i]
        if prev == 0:
            returns.append(0.0)
        else:
            returns.append((curr - prev) / prev)

    wins = sum(1 for t in trades if float(t.get("pnl_usd", 0)) > 0)
    n = len(trades)
    hit_rate = (wins / n) if n else 0.0
    expectancy_usd = (sum(float(t.get("pnl_usd", 0)) for t in trades) / n) if n else 0.0

    return BacktestResult(
        sharpe=annualized_sharpe(returns, periods_per_year),
        sortino=annualized_sortino(returns, periods_per_year),
        max_dd_pct=max_drawdown(equity_curve),
        hit_rate=hit_rate,
        num_trades=n,
        expectancy_usd=expectancy_usd,
        calmar=calmar(equity_curve, years),
        equity_curve=list(equity_curve),
        trades=list(trades),
    )
