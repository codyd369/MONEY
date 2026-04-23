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
