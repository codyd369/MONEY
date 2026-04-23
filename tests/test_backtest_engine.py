"""Backtest engine correctness: fills, fees, and P&L accounting."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from moneybutton.backtest.engine import (
    BacktestConfig,
    DecisionInput,
    SignalIntent,
    run_backtest,
)
from moneybutton.backtest.fees import fee_per_contract_cents, trade_fee_usd
from moneybutton.backtest.fills import fill_entry
from moneybutton.features.common import parse_ts


def _market(ticker: str, result: str, price_series: list[tuple[str, int, int]]) -> tuple[dict, pd.DataFrame]:
    """Helper: build a (market, prices) pair.

    price_series is [(ts_iso, yes_bid, yes_ask), ...].
    """
    rows = []
    for ts_iso, bid, ask in price_series:
        rows.append(
            {
                "ticker": ticker,
                "ts": ts_iso,
                "yes_bid_close": bid,
                "yes_ask_close": ask,
                "last_price_close": (bid + ask) // 2,
                "volume": 10,
                "open_interest": 100,
            }
        )
    prices = pd.DataFrame(rows)
    market = {
        "ticker": ticker,
        "category": "SPORTS",
        "result": result,
        "open_time": price_series[0][0],
        "close_time": price_series[-1][0],
        "volume": 10,
    }
    return market, prices


# ----------------------------- fees --------------------------------------


def test_fee_formula_symmetric_around_50c():
    # Symmetric: 0.07 * 100 * p * (1-p) peaks at 1.75 at p=0.5, rounds up.
    assert fee_per_contract_cents(50) == fee_per_contract_cents(50)
    assert fee_per_contract_cents(49) == fee_per_contract_cents(51)
    assert fee_per_contract_cents(1) == fee_per_contract_cents(99)
    # Peak fee ~= 2 cents at p=0.5.
    assert fee_per_contract_cents(50) == 2
    # Edge prices: tiny fees round up to 1 cent.
    assert fee_per_contract_cents(1) >= 1
    assert fee_per_contract_cents(99) >= 1


def test_fee_invalid_price_returns_zero():
    assert fee_per_contract_cents(0) == 0
    assert fee_per_contract_cents(100) == 0
    assert fee_per_contract_cents(-5) == 0


def test_trade_fee_scales_with_contracts():
    per = fee_per_contract_cents(40)
    assert trade_fee_usd(40, 10) == per * 10 / 100.0
    assert trade_fee_usd(40, 0) == 0.0


# ----------------------------- fills -------------------------------------


def test_fill_entry_yes_side_with_slippage():
    prices = pd.DataFrame(
        [{"ticker": "T", "ts": "2025-01-01T12:00:00Z", "_ts_dt": pd.Timestamp("2025-01-01T12:00:00Z"),
          "yes_bid_close": 39, "yes_ask_close": 41, "last_price_close": 40}]
    )
    f = fill_entry(
        side="yes",
        size_usd=10.0,
        price_frame=prices,
        as_of_ts=pd.Timestamp("2025-01-01T13:00:00Z"),
        slippage_bps=100,  # 100 bps on 41c = 0.41 cents -> rounds to 41c still
    )
    assert f.filled is True
    assert f.price_cents == 41  # slippage is sub-tick, still rounds to 41
    assert f.contracts == 24  # $10 / $0.41 = 24.39 -> floor 24


def test_fill_rejects_when_no_price_before_as_of():
    prices = pd.DataFrame(
        [{"ticker": "T", "ts": "2025-01-02T12:00:00Z", "_ts_dt": pd.Timestamp("2025-01-02T12:00:00Z"),
          "yes_bid_close": 39, "yes_ask_close": 41, "last_price_close": 40}]
    )
    f = fill_entry(
        side="yes",
        size_usd=10.0,
        price_frame=prices,
        as_of_ts=pd.Timestamp("2025-01-01T00:00:00Z"),
        slippage_bps=25,
    )
    assert f.filled is False


def test_fill_rejects_when_size_below_one_contract():
    prices = pd.DataFrame(
        [{"ticker": "T", "ts": "2025-01-01T12:00:00Z", "_ts_dt": pd.Timestamp("2025-01-01T12:00:00Z"),
          "yes_bid_close": 94, "yes_ask_close": 96, "last_price_close": 95}]
    )
    # $0.50 at 96c entry = 0.52 contracts -> floor 0 -> reject
    f = fill_entry(
        side="yes",
        size_usd=0.50,
        price_frame=prices,
        as_of_ts=pd.Timestamp("2025-01-01T13:00:00Z"),
        slippage_bps=25,
    )
    assert f.filled is False
    assert "size < 1 contract" in f.reason


# ----------------------------- engine ------------------------------------


def test_engine_pnl_on_winning_yes_trade():
    """Buy YES at 40c on a market that resolves YES -> +60c per contract minus fees."""
    market, prices = _market(
        "TST-WIN",
        result="yes",
        price_series=[
            ("2025-01-01T12:00:00Z", 39, 41),
            ("2025-01-02T12:00:00Z", 40, 42),
        ],
    )
    markets_df = pd.DataFrame([market])

    def scanner(ev: DecisionInput) -> SignalIntent | None:
        return SignalIntent(side="yes", size_usd=10.0, edge_bps=500, confidence="med")

    result = run_backtest(
        markets=markets_df,
        prices=prices,
        scanner=scanner,
        config=BacktestConfig(starting_equity_usd=100, slippage_bps=0, max_position_usd=25),
        as_of_fn=lambda m: parse_ts("2025-01-02T13:00:00Z"),
    )
    assert result.num_trades == 1
    t = result.trades[0]
    # $10 / 42c = 23 contracts (floor), payout 100c - 42c = 58c gross each
    # minus fees (~2c per contract at 42c).
    assert t["contracts"] == 23
    assert t["entry_price_cents"] == 42
    assert t["resolved_yes"] == 1
    # Gross: 58 * 23 = $13.34. Fees at 42c: 7% * 100 * 0.42 * 0.58 = 1.71 -> ceil 2c/contract
    # Fees: 23 * 0.02 = $0.46. Net: $13.34 - $0.46 = $12.88.
    assert abs(t["pnl_usd"] - 12.88) < 0.01
    assert result.hit_rate == 1.0


def test_engine_pnl_on_losing_yes_trade():
    market, prices = _market(
        "TST-LOSE",
        result="no",
        price_series=[
            ("2025-01-01T12:00:00Z", 59, 61),
            ("2025-01-02T12:00:00Z", 60, 62),
        ],
    )
    markets_df = pd.DataFrame([market])

    def scanner(ev):
        return SignalIntent(side="yes", size_usd=10.0, edge_bps=500)

    result = run_backtest(
        markets=markets_df,
        prices=prices,
        scanner=scanner,
        config=BacktestConfig(starting_equity_usd=100, slippage_bps=0, max_position_usd=25),
        as_of_fn=lambda m: parse_ts("2025-01-02T13:00:00Z"),
    )
    assert result.num_trades == 1
    t = result.trades[0]
    # Bought YES at 62c, resolves NO -> lose 62c per contract plus fees.
    assert t["pnl_usd"] < 0
    assert result.hit_rate == 0.0


def test_engine_no_signal_means_no_trade():
    market, prices = _market(
        "TST-SKIP", "yes", [("2025-01-01T12:00:00Z", 40, 42), ("2025-01-02T12:00:00Z", 41, 43)]
    )
    markets_df = pd.DataFrame([market])
    result = run_backtest(
        markets=markets_df,
        prices=prices,
        scanner=lambda ev: None,
        config=BacktestConfig(starting_equity_usd=100),
        as_of_fn=lambda m: parse_ts("2025-01-02T13:00:00Z"),
    )
    assert result.num_trades == 0
    assert result.equity_curve == [100.0]


def test_engine_respects_window_filters():
    # 2 markets; window_start excludes the earlier one.
    m1, p1 = _market("T1", "yes", [("2025-01-01T12:00:00Z", 40, 42), ("2025-01-02T12:00:00Z", 41, 43)])
    m2, p2 = _market("T2", "yes", [("2025-06-01T12:00:00Z", 40, 42), ("2025-06-02T12:00:00Z", 41, 43)])
    markets_df = pd.DataFrame([m1, m2])
    prices_df = pd.concat([p1, p2], ignore_index=True)

    def as_of_fn(m):
        return parse_ts(m["close_time"]) - dt.timedelta(hours=1)

    scanner = lambda ev: SignalIntent(side="yes", size_usd=5.0, edge_bps=500)  # noqa: E731
    result = run_backtest(
        markets=markets_df,
        prices=prices_df,
        scanner=scanner,
        config=BacktestConfig(),
        as_of_fn=as_of_fn,
        window_start=dt.datetime(2025, 3, 1, tzinfo=dt.timezone.utc),
    )
    # Only T2 is in the window.
    assert result.num_trades == 1
    assert result.trades[0]["ticker"] == "T2"


def test_equity_curve_monotonic_after_wins():
    """Equity grows with consecutive wins."""
    markets: list[dict] = []
    all_prices: list[pd.DataFrame] = []
    for i in range(3):
        ts0 = f"2025-01-{i*3+1:02d}T12:00:00Z"
        ts1 = f"2025-01-{i*3+2:02d}T12:00:00Z"
        m, p = _market(f"T{i}", "yes", [(ts0, 39, 41), (ts1, 40, 42)])
        markets.append(m)
        all_prices.append(p)
    markets_df = pd.DataFrame(markets)
    prices_df = pd.concat(all_prices, ignore_index=True)
    result = run_backtest(
        markets=markets_df,
        prices=prices_df,
        scanner=lambda ev: SignalIntent(side="yes", size_usd=5.0, edge_bps=500),
        config=BacktestConfig(starting_equity_usd=100, slippage_bps=0, max_position_usd=25),
        as_of_fn=lambda m: parse_ts(m["close_time"]) + dt.timedelta(hours=1),
    )
    assert result.num_trades == 3
    # Equity curve is strictly non-decreasing on 3 consecutive wins.
    assert result.equity_curve[-1] > result.equity_curve[0]
