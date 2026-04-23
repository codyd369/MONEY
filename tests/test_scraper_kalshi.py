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
