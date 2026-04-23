"""Backtest engine: holdout simulation for a single strategy (SPEC §11.1).

Scope:
  - Each market produces exactly ONE decision event at its as_of_ts.
  - A signal at that event opens a position; the position is held to
    market resolution. No intra-market exits modeled (simple and honest
    for weekly/daily markets; partial re-entries come in v3).
  - Fills go through backtest/fills.py (bid/ask + slippage).
  - Fees go through backtest/fees.py.
  - The walk-forward variant trains the model periodically; v1 uses a
    single train/val/test split produced by models.calibration.time_split,
    so "backtest" here == "evaluate on test window".
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

from moneybutton.backtest.fees import trade_fee_usd
from moneybutton.backtest.fills import Fill, fill_entry
from moneybutton.backtest.metrics import BacktestResult, summarize
from moneybutton.features.common import parse_ts


@dataclass
class DecisionInput:
    """Everything a strategy sees at one simulation timestamp for one market."""

    market: dict
    as_of_ts: dt.datetime
    price_frame: pd.DataFrame  # already pre-filtered to ticker


@dataclass
class SignalIntent:
    """Backtest-internal signal (richer than live Signal for auditability)."""

    side: str
    size_usd: float
    edge_bps: int
    confidence: str = "med"
    reasoning: dict | None = None


ScannerFn = Callable[[DecisionInput], SignalIntent | None]


@dataclass
class BacktestConfig:
    starting_equity_usd: float = 500.0
    slippage_bps: int = 25
    fee_rate: float = 0.07
    min_edge_bps: int = 0
    # Cap per-trade size to this; the engine will also clamp to
    # starting_equity for safety so a buggy scanner can't blow up the book.
    max_position_usd: float = 25.0


def run_backtest(
    *,
    markets: pd.DataFrame,
    prices: pd.DataFrame,
    scanner: ScannerFn,
    config: BacktestConfig,
    as_of_fn: Callable[[dict], dt.datetime],
    window_start: dt.datetime | None = None,
    window_end: dt.datetime | None = None,
) -> BacktestResult:
    """Iterate markets in as_of_ts order, run scanner, apply fills, score P&L."""
    prices = prices.copy()
    prices["_ts_dt"] = pd.to_datetime(prices["ts"], utc=True, errors="coerce")
    grouped_prices = {t: g.sort_values("_ts_dt") for t, g in prices.groupby("ticker")}

    # Build event list: one DecisionInput per market, sorted by as_of_ts.
    events: list[tuple[pd.Timestamp, DecisionInput]] = []
    for market in markets.to_dict(orient="records"):
        if market.get("result") not in ("yes", "no"):
            continue
        as_of = as_of_fn(market)
        if window_start is not None and as_of < window_start:
            continue
        if window_end is not None and as_of >= window_end:
            continue
        pdf = grouped_prices.get(market["ticker"])
        if pdf is None or pdf.empty:
            continue
        events.append((pd.Timestamp(as_of), DecisionInput(market=market, as_of_ts=as_of, price_frame=pdf)))

    events.sort(key=lambda kv: kv[0])

    equity = config.starting_equity_usd
    equity_curve: list[float] = [equity]
    trades: list[dict] = []

    for _, ev in events:
        intent = scanner(ev)
        if intent is None:
            continue
        if abs(intent.edge_bps) < config.min_edge_bps:
            continue
        size_usd = min(intent.size_usd, config.max_position_usd, equity)
        if size_usd <= 0:
            continue

        fill = fill_entry(
            side=intent.side,
            size_usd=size_usd,
            price_frame=ev.price_frame,
            as_of_ts=pd.Timestamp(ev.as_of_ts),
            slippage_bps=config.slippage_bps,
        )
        if not fill.filled:
            continue

        pnl_usd = _resolve_pnl(
            side=intent.side,
            fill=fill,
            market_result=ev.market["result"],
            fee_rate=config.fee_rate,
        )

        equity += pnl_usd
        equity_curve.append(equity)

        trades.append(
            {
                "ticker": ev.market["ticker"],
                "category": ev.market.get("category"),
                "entry_ts": ev.as_of_ts.isoformat(),
                "exit_ts": _close_ts(ev.market),
                "side": intent.side,
                "edge_bps": intent.edge_bps,
                "confidence": intent.confidence,
                "entry_price_cents": fill.price_cents,
                "contracts": fill.contracts,
                "size_usd": size_usd,
                "resolved_yes": 1 if ev.market["result"] == "yes" else 0,
                "pnl_usd": pnl_usd,
                "equity_after": equity,
            }
        )

    years = 1.0
    if events:
        first, last = events[0][0], events[-1][0]
        years = max(1 / 365.0, (last - first).days / 365.25)

    result = summarize(equity_curve, trades, years=years)
    result.meta = {
        "starting_equity_usd": config.starting_equity_usd,
        "window_start": window_start.isoformat() if window_start else None,
        "window_end": window_end.isoformat() if window_end else None,
        "n_events": len(events),
        "n_trades": len(trades),
        "final_equity_usd": equity,
        "total_return_pct": (equity / config.starting_equity_usd - 1) * 100.0
        if config.starting_equity_usd
        else 0.0,
    }
    return result


def _resolve_pnl(*, side: str, fill: Fill, market_result: str, fee_rate: float) -> float:
    """Close the position at resolution. YES pays $1.00, NO pays $0.00."""
    won_yes = market_result == "yes"
    # Per-contract payout in cents:
    #   yes bet: pays 100 if resolves yes; 0 otherwise
    #   no bet: pays 100 if resolves no; 0 otherwise
    if side == "yes":
        payout_per = 100 if won_yes else 0
    else:
        payout_per = 100 if not won_yes else 0
    gross_pnl_cents = (payout_per - fill.price_cents) * fill.contracts
    # Entry fee. (Simplified: Kalshi also charges on settlement but for this
    # backtest we bundle the whole round-trip into the entry fee. Update if
    # the docs say otherwise at go-live.)
    fee_usd = trade_fee_usd(fill.price_cents, fill.contracts, fee_rate=fee_rate)
    return gross_pnl_cents / 100.0 - fee_usd


def _close_ts(market: dict) -> str | None:
    close = parse_ts(market.get("close_time") or market.get("expiration_time"))
    return close.isoformat() if close else None
