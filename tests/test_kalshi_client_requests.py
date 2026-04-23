"""Regression tests for kalshi/client request construction.

No network — uses httpx.MockTransport to intercept requests and assert on
the exact URL + headers that would have been sent.
"""

from __future__ import annotations

import httpx
import pytest

from moneybutton.kalshi.client import KalshiClient


def _mock_transport_capturing_last():
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["path"] = request.url.path
        captured["query"] = request.url.query.decode() if isinstance(request.url.query, bytes) else str(request.url.query)
        captured["headers"] = dict(request.headers)
        # Return a tiny valid JSON body so KalshiClient doesn't raise.
        return httpx.Response(200, json={"markets": [], "cursor": None})

    return httpx.MockTransport(_handler), captured


def test_list_markets_drops_none_params(tmp_env):
    """httpx converts None to empty-string in the query string. Kalshi
    rejects `max_close_ts=` with a 400. We must strip Nones before send."""
    transport, captured = _mock_transport_capturing_last()
    client = httpx.Client(transport=transport)
    with KalshiClient(client=client) as kc:
        kc.list_markets(status="settled", limit=200)

    assert captured["method"] == "GET"
    assert captured["path"] == "/trade-api/v2/markets"
    # Only the non-None params should appear in the query string.
    q = captured["query"]
    assert "status=settled" in q
    assert "limit=200" in q
    # Kalshi's "invalid syntax" trap: these must NOT appear as empty strings.
    assert "max_close_ts" not in q
    assert "min_close_ts" not in q
    assert "category" not in q
    assert "cursor" not in q
    assert "event_ticker" not in q


def test_list_markets_keeps_nonNone_params(tmp_env):
    transport, captured = _mock_transport_capturing_last()
    client = httpx.Client(transport=transport)
    with KalshiClient(client=client) as kc:
        kc.list_markets(
            status="settled",
            category="Financials",
            cursor="abc123",
            limit=50,
            min_close_ts=1704067200,
        )
    q = captured["query"]
    assert "status=settled" in q
    assert "category=Financials" in q
    assert "cursor=abc123" in q
    assert "limit=50" in q
    assert "min_close_ts=1704067200" in q
    assert "max_close_ts" not in q  # was None, must be dropped


def test_empty_params_dict_sends_no_query(tmp_env):
    transport, captured = _mock_transport_capturing_last()
    client = httpx.Client(transport=transport)
    with KalshiClient(client=client) as kc:
        kc.exchange_status()
    assert captured["path"] == "/trade-api/v2/exchange/status"
    assert captured["query"] == ""


def test_get_candlesticks_uses_series_path(tmp_env):
    """Kalshi's candlestick endpoint is under /series/{s}/markets/{t}/...
    The short form /markets/{t}/candlesticks returns 404. Pin the path."""
    transport, captured = _mock_transport_capturing_last()
    client = httpx.Client(transport=transport)
    with KalshiClient(client=client) as kc:
        kc.get_candlesticks(
            "KXFED-25MAR-HIKE",
            start_ts=1704067200,
            end_ts=1711843200,
            period_interval=60,
        )

    assert (
        captured["path"]
        == "/trade-api/v2/series/KXFED/markets/KXFED-25MAR-HIKE/candlesticks"
    )
    q = captured["query"]
    assert "start_ts=1704067200" in q
    assert "end_ts=1711843200" in q
    assert "period_interval=60" in q


def test_get_candlesticks_derives_series_from_ticker(tmp_env):
    """Even when the market response doesn't carry series_ticker, the
    client derives it from the ticker prefix. These 2026Q1 markets had
    series_ticker=null in the listing response."""
    transport, captured = _mock_transport_capturing_last()
    client = httpx.Client(transport=transport)
    with KalshiClient(client=client) as kc:
        kc.get_candlesticks(
            "KXNEXTIRANLEADER-45JAN01-HKHO",
            series_ticker=None,
            start_ts=1,
            end_ts=2,
            period_interval=60,
        )
    assert captured["path"] == (
        "/trade-api/v2/series/KXNEXTIRANLEADER/markets/"
        "KXNEXTIRANLEADER-45JAN01-HKHO/candlesticks"
    )


def test_get_candlesticks_explicit_series_overrides(tmp_env):
    """Explicit series_ticker wins over the derived one."""
    transport, captured = _mock_transport_capturing_last()
    client = httpx.Client(transport=transport)
    with KalshiClient(client=client) as kc:
        kc.get_candlesticks(
            "KXFED-25MAR-HIKE",
            series_ticker="EXPLICIT",
            start_ts=1,
            end_ts=2,
        )
    assert "/series/EXPLICIT/markets/" in captured["path"]


def test_backfill_markets_sends_status_settled(tmp_db):
    """Kalshi's /markets `status` query param accepts the legacy vocabulary
    (settled), not the newer response label (finalized). Comma-separated
    values are also rejected with 'invalid status filter'. Pin the exact
    string the backfill sends so a well-meaning future edit can't break
    the scrape again.
    """
    from moneybutton.data.scraper_kalshi import backfill_markets

    transport, captured = _mock_transport_capturing_last()
    client = httpx.Client(transport=transport)
    with KalshiClient(client=client) as kc:
        backfill_markets(client=kc, max_pages=1)

    q = captured["query"]
    assert "status=settled" in q
    # The two things Kalshi rejects:
    assert "status=finalized" not in q
    assert "status=settled%2Cfinalized" not in q
    assert "status=settled,finalized" not in q
