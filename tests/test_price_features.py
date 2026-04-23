"""Regression: price features must not crash on Kalshi 'no-real-quote'
candles where yes_bid=0 and yes_ask=100 (or similar zero/full quotes).
Real fixture from KXDJTVOSTARIFFS — first candle of a market's life
before any orders existed."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from moneybutton.features import price_features


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["_ts_dt"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_price_features_handles_zero_bid_full_ask_quotes():
    """bid=0/ask=100 mid=0.5 is meaningless. Price features must not
    treat that as a real price (otherwise log(0) crashes elsewhere and
    the feature 'yes_price_now' is silently 0.5 garbage)."""
    market = {
        "ticker": "KXDJTVOSTARIFFS",
        "open_time": "2025-09-02T18:00:00Z",
        "close_time": "2025-09-09T18:00:00Z",
    }
    rows = []
    base = dt.datetime(2025, 9, 2, 18, tzinfo=dt.timezone.utc)
    for i in range(5):
        rows.append(
            {
                "ticker": "KXDJTVOSTARIFFS",
                "ts": (base + dt.timedelta(hours=i)).isoformat(),
                "yes_bid_close": 0,    # the 0/100 sentinel
                "yes_ask_close": 100,
                "last_price_close": None,
                "volume": 0,
                "open_interest": 0,
            }
        )
    df = _frame(rows)
    feats = price_features.compute(market, base + dt.timedelta(hours=10), df)
    # All quote-derived features should be None (no real price), not 0.5 garbage.
    assert feats["yes_price_now"] is None
    assert feats["realized_vol_24h"] is None
    assert feats["max_24h"] is None
    assert feats["min_24h"] is None


def test_price_features_handles_mixed_real_and_zero_quotes():
    """A market that starts with zero quotes then gets real trades. The
    real-quote rows should drive features; zero quotes should be ignored."""
    market = {
        "ticker": "KX-MIX",
        "open_time": "2025-09-02T18:00:00Z",
        "close_time": "2025-09-09T18:00:00Z",
    }
    base = dt.datetime(2025, 9, 2, 18, tzinfo=dt.timezone.utc)
    rows = []
    # 3 zero-quote candles
    for i in range(3):
        rows.append({
            "ticker": "KX-MIX",
            "ts": (base + dt.timedelta(hours=i)).isoformat(),
            "yes_bid_close": 0, "yes_ask_close": 100, "last_price_close": None,
            "volume": 0, "open_interest": 0,
        })
    # 5 real-quote candles drifting 50c -> 60c
    for i in range(3, 8):
        bid = 49 + i
        ask = 51 + i
        rows.append({
            "ticker": "KX-MIX",
            "ts": (base + dt.timedelta(hours=i)).isoformat(),
            "yes_bid_close": bid, "yes_ask_close": ask,
            "last_price_close": (bid + ask) // 2,
            "volume": 10, "open_interest": 100,
        })
    df = _frame(rows)
    feats = price_features.compute(market, base + dt.timedelta(hours=10), df)
    # yes_price_now should be the most recent real mid, not None and not 0.5.
    assert feats["yes_price_now"] is not None
    assert 0.55 < feats["yes_price_now"] < 0.65
    # realized_vol must compute without crashing.
    assert feats["realized_vol_24h"] is not None
    assert feats["realized_vol_24h"] >= 0
