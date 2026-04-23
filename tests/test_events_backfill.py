"""Events-first backfill unit tests (no network).

Covers:
  - KalshiClient.list_events now forwards `category=` in the query.
  - Event-to-market category injection: a market whose Kalshi response has
    category=null gets the event's category written into its row.
  - Event prefix-exclude filter: MVE* events are skipped.
"""

from __future__ import annotations

import httpx
import pandas as pd

from moneybutton.kalshi.client import KalshiClient


def _mock_transport_capturing_last():
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["query"] = (
            request.url.query.decode()
            if isinstance(request.url.query, bytes)
            else str(request.url.query)
        )
        return httpx.Response(200, json={"events": [], "cursor": None})

    return httpx.MockTransport(_handler), captured


def test_list_events_forwards_category_param(tmp_env):
    transport, captured = _mock_transport_capturing_last()
    client = httpx.Client(transport=transport)
    with KalshiClient(client=client) as kc:
        kc.list_events(status="settled", category="Politics", limit=100)

    q = captured["query"]
    assert captured["path"] == "/trade-api/v2/events"
    assert "status=settled" in q
    assert "category=Politics" in q
    assert "limit=100" in q


def test_list_events_drops_none_category(tmp_env):
    transport, captured = _mock_transport_capturing_last()
    client = httpx.Client(transport=transport)
    with KalshiClient(client=client) as kc:
        kc.list_events(status="settled", limit=50)
    q = captured["query"]
    assert "status=settled" in q
    assert "limit=50" in q
    assert "category=" not in q  # None must be stripped


def test_event_category_injected_into_market_row():
    """When we fetch a market under an event, we stamp the event's category
    on the market so _partition_key_for_market doesn't fall through to the
    ticker-prefix derive path."""
    from moneybutton.data.scraper_kalshi import (
        _coerce_market_row,
        _partition_key_for_market,
    )

    # Raw market from Kalshi with category=null (the 2026Q1 reality).
    raw_market = {
        "ticker": "KXNEXTIRANLEADER-45JAN01-YES-KHAMENEI",
        "event_ticker": "KXNEXTIRANLEADER-45JAN01",
        "series_ticker": "KXNEXTIRANLEADER",
        "category": None,
        "status": "finalized",
        "result": "no",
        "close_time": "2025-11-20T18:00:00Z",
        "open_time": "2025-01-01T12:00:00Z",
    }
    # Inject event category (what backfill_via_events does before coerce).
    raw_market["category"] = "Elections"

    row = _coerce_market_row(raw_market)
    assert row["category"] == "Elections"
    key = _partition_key_for_market(raw_market)
    assert key.category_or_source == "ELECTIONS"
    assert key.year_month == "2025-11"


def test_event_prefix_exclude_drops_mve():
    from scripts.backfill_via_events import _should_skip_event

    mve_event = {"event_ticker": "KXMVESPORTSMULTIGAMEEXTENDED-S2026ABC"}
    real_event = {"event_ticker": "KXNEXTIRANLEADER-45JAN01"}
    assert _should_skip_event(mve_event, ["KXMVE"]) is True
    assert _should_skip_event(real_event, ["KXMVE"]) is False
    # Empty exclude list accepts everything.
    assert _should_skip_event(mve_event, []) is False
