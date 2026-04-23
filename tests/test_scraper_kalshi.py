"""Scraper projection + filter tests (SPEC §16 step 10).

Regression coverage for the 2026Q1 Kalshi API drift:
  - `status` rename: settled -> finalized. We now filter locally by
    `result in (yes, no)` so either label works.
  - `category` dropped from the market-level response. We derive a
    partition-friendly pseudo-category from event_ticker / ticker.
"""

from __future__ import annotations

from moneybutton.data.scraper_kalshi import (
    _coerce_market_row,
    _derive_category,
    _filter_resolved,
    _partition_key_for_market,
)


def _finalized_market() -> dict:
    """A real-shape market from the 2026Q1 API: no category, status finalized."""
    return {
        "ticker": "KXMVESPORTSMULTIGAMEEXTENDED-S2026BA15E6925CA-2039BE9ED61",
        "event_ticker": "KXMVESPORTSMULTIGAMEEXTENDED-S2026BA15E6925CA",
        "series_ticker": None,
        "title": "multi-match parlay",
        "category": None,
        "status": "finalized",
        "result": "no",
        "close_time": "2026-04-23T16:14:23Z",
        "expiration_time": "2026-05-06T08:00:00Z",
        "open_time": "2026-04-23T15:43:15Z",
    }


def _classic_market() -> dict:
    """What an older 'settled' market looks like with category populated."""
    return {
        "ticker": "KXFED-25MAR-HIKE",
        "event_ticker": "KXFED-25MAR",
        "series_ticker": "KXFED",
        "category": "Economics",
        "status": "settled",
        "result": "yes",
        "close_time": "2025-03-19T18:00:00Z",
    }


def test_derive_category_from_event_ticker():
    m = _finalized_market()
    assert _derive_category(m) == "MVESPORTSMULTIGAMEEX"  # 20-char cap


def test_derive_category_from_ticker_when_event_missing():
    m = {"ticker": "KXNFL-25NOV17-CHI-DET", "event_ticker": None, "series_ticker": None}
    assert _derive_category(m) == "NFL"


def test_derive_category_fallback_uncategorized():
    assert _derive_category({}) == "UNCATEGORIZED"


def test_coerce_market_row_fills_category_when_null():
    """Post-2026Q1 markets have category=null; we backfill via derive."""
    m = _finalized_market()
    row = _coerce_market_row(m)
    assert row["category"] == "MVESPORTSMULTIGAMEEX"
    assert row["ticker"] == m["ticker"]
    assert row["result"] == "no"


def test_coerce_market_row_keeps_real_category():
    """A market that DOES have category set keeps it unchanged."""
    row = _coerce_market_row(_classic_market())
    assert row["category"] == "Economics"


def test_partition_key_uses_derived_category_on_null():
    key = _partition_key_for_market(_finalized_market())
    assert key.category_or_source == "MVESPORTSMULTIGAMEEX"
    assert key.year_month == "2026-04"


def test_partition_key_uses_classic_category_when_present():
    key = _partition_key_for_market(_classic_market())
    assert key.category_or_source == "ECONOMICS"
    assert key.year_month == "2025-03"


def test_filter_resolved_drops_active_markets():
    active = {"ticker": "A", "status": "open", "result": None}
    resolved_yes = {"ticker": "B", "status": "finalized", "result": "yes"}
    resolved_no = {"ticker": "C", "status": "settled", "result": "no"}
    weird = {"ticker": "D", "status": "finalized", "result": ""}
    kept = _filter_resolved([active, resolved_yes, resolved_no, weird])
    assert {m["ticker"] for m in kept} == {"B", "C"}


def test_filter_resolved_handles_uppercase():
    rows = [
        {"ticker": "X", "result": "YES"},
        {"ticker": "Y", "result": "No"},
        {"ticker": "Z", "result": "pending"},
    ]
    kept = _filter_resolved(rows)
    assert {m["ticker"] for m in kept} == {"X", "Y"}


def test_project_candlesticks_2026q1_schema():
    """Kalshi's 2026Q1 candle schema has *_dollars string fields + _fp
    counts. Fixture is copied from scripts/inspect_candles.py output on
    a real KXDJTVOSTARIFFS market."""
    from moneybutton.data.scraper_kalshi import _project_candlesticks

    resp = {
        "candlesticks": [
            # Candle with quotes but no trades (price={}).
            {
                "end_period_ts": 1756825200,
                "open_interest_fp": "0.00",
                "price": {},
                "volume_fp": "0.00",
                "yes_ask": {
                    "close_dollars": "0.5800",
                    "high_dollars": "1.0000",
                    "low_dollars": "0.5800",
                    "open_dollars": "1.0000",
                },
                "yes_bid": {
                    "close_dollars": "0.5200",
                    "high_dollars": "0.5200",
                    "low_dollars": "0.0000",
                    "open_dollars": "0.4800",
                },
            },
            # Candle with trades + OHLC prices.
            {
                "end_period_ts": 1756828800,
                "open_interest_fp": "50.00",
                "price": {
                    "close_dollars": "0.5800",
                    "high_dollars": "0.5800",
                    "low_dollars": "0.5800",
                    "mean_dollars": "0.5800",
                    "open_dollars": "0.5800",
                },
                "volume_fp": "50.00",
                "yes_ask": {
                    "close_dollars": "0.6000",
                    "high_dollars": "0.9900",
                    "low_dollars": "0.5800",
                    "open_dollars": "0.5800",
                },
                "yes_bid": {
                    "close_dollars": "0.5500",
                    "high_dollars": "0.5500",
                    "low_dollars": "0.0100",
                    "open_dollars": "0.5200",
                },
            },
        ]
    }
    rows = _project_candlesticks(resp, "KXDJTVOSTARIFFS")
    assert len(rows) == 2

    # First candle: no trades -> price fields None, but bid/ask populated.
    r0 = rows[0]
    assert r0["ticker"] == "KXDJTVOSTARIFFS"
    assert r0["ts"].startswith("2025-09-02")
    assert r0["yes_bid_close"] == 52  # "0.5200" -> 52 cents
    assert r0["yes_ask_close"] == 58  # "0.5800" -> 58 cents
    assert r0["yes_bid_open"] == 48
    assert r0["yes_ask_open"] == 100  # 1.0000 -> 100 cents (full $1 ask = no-bid market)
    assert r0["last_price_close"] is None  # empty price dict
    assert r0["volume"] == 0
    assert r0["open_interest"] == 0

    # Second candle: 50 contracts traded, OHLC all at 58 cents.
    r1 = rows[1]
    assert r1["last_price_open"] == 58
    assert r1["last_price_close"] == 58
    assert r1["last_price_high"] == 58
    assert r1["last_price_low"] == 58
    assert r1["volume"] == 50
    assert r1["open_interest"] == 50


def test_project_candlesticks_handles_missing_end_period_ts():
    from moneybutton.data.scraper_kalshi import _project_candlesticks

    resp = {"candlesticks": [{"price": {"close_dollars": "0.50"}}]}  # no ts
    assert _project_candlesticks(resp, "X") == []


def test_project_candlesticks_empty_response():
    from moneybutton.data.scraper_kalshi import _project_candlesticks

    assert _project_candlesticks({}, "X") == []
    assert _project_candlesticks({"candlesticks": []}, "X") == []
    assert _project_candlesticks({"candlesticks": None}, "X") == []
