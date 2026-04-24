"""Tests for the trades endpoint wrapper + projection.

No network: uses httpx.MockTransport to assert the exact URL + query
string Kalshi receives, and verifies that the projection produces
the stable schema we store.
"""

from __future__ import annotations

import httpx
import pandas as pd

from moneybutton.kalshi.client import KalshiClient


def _mock_transport(respond_with: dict):
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["query"] = (
            request.url.query.decode()
            if isinstance(request.url.query, bytes)
            else str(request.url.query)
        )
        return httpx.Response(200, json=respond_with)

    return httpx.MockTransport(_handler), captured


def test_list_trades_hits_correct_path(tmp_env):
    transport, captured = _mock_transport({"trades": [], "cursor": None})
    client = httpx.Client(transport=transport)
    with KalshiClient(client=client) as kc:
        kc.list_trades(ticker="KXBTC-26APR24-T90000", limit=500)
    assert captured["path"] == "/trade-api/v2/markets/trades"
    q = captured["query"]
    assert "ticker=KXBTC-26APR24-T90000" in q
    assert "limit=500" in q


def test_list_trades_drops_none_params(tmp_env):
    """httpx serializes None as empty strings (Kalshi rejects these).
    The None-stripping we added earlier must apply here too."""
    transport, captured = _mock_transport({"trades": [], "cursor": None})
    client = httpx.Client(transport=transport)
    with KalshiClient(client=client) as kc:
        kc.list_trades(ticker="KXBTC-26APR24-T90000", limit=500)
    q = captured["query"]
    assert "cursor" not in q
    assert "min_ts" not in q
    assert "max_ts" not in q


def test_list_trades_passes_ts_filters(tmp_env):
    transport, captured = _mock_transport({"trades": [], "cursor": None})
    client = httpx.Client(transport=transport)
    with KalshiClient(client=client) as kc:
        kc.list_trades(ticker="KX-X", min_ts=1704067200, max_ts=1711843200)
    q = captured["query"]
    assert "min_ts=1704067200" in q
    assert "max_ts=1711843200" in q


def test_project_trade_schema():
    """The stable columnar schema we persist. One raw trade from a real
    Kalshi response, projected."""
    from scripts.backfill_trades import _project_trade

    raw = {
        "trade_id": "t_abc123",
        "ticker": "KXBTC-26APR24-T90000",
        "taker_side": "yes",
        "yes_price": 42,
        "no_price": 58,
        "count": 100,
        "created_time": "2026-04-23T15:00:00.123Z",
    }
    row = _project_trade(raw)
    assert row["trade_id"] == "t_abc123"
    assert row["ticker"] == "KXBTC-26APR24-T90000"
    assert row["taker_side"] == "yes"
    assert row["yes_price_cents"] == 42
    assert row["count"] == 100
    # Timestamp should be normalized to UTC ISO-8601.
    parsed = pd.to_datetime(row["ts"], utc=True, format="ISO8601")
    assert parsed.year == 2026 and parsed.month == 4 and parsed.day == 23


def test_project_trade_handles_missing_fields():
    from scripts.backfill_trades import _project_trade

    row = _project_trade({})
    assert row["trade_id"] == ""
    assert row["ticker"] == ""
    assert row["yes_price_cents"] is None
    assert row["count"] == 0
    # Must produce a valid ts even when created_time is missing.
    assert row["ts"]


def test_trades_dataset_registered_in_store(tmp_env):
    """The store must know about 'trades' as a dataset with a stable
    dedupe key — otherwise write_partition falls through to a no-dedupe
    path and we'd accumulate duplicate trade_ids on reruns."""
    from moneybutton.data.store import _DEDUPE_KEYS

    assert "trades" in _DEDUPE_KEYS
    assert _DEDUPE_KEYS["trades"] == ["trade_id"]


def test_write_and_read_trades_partition(tmp_env):
    from moneybutton.data.store import PartitionKey, read_partition, write_partition

    key = PartitionKey(dataset="trades", category_or_source="CRYPTO", year_month="2026-04")
    rows = pd.DataFrame(
        [
            {"trade_id": "t1", "ticker": "KXBTC-X", "ts": "2026-04-23T15:00:00+00:00",
             "taker_side": "yes", "yes_price_cents": 42, "count": 10},
            {"trade_id": "t2", "ticker": "KXBTC-X", "ts": "2026-04-23T15:00:01+00:00",
             "taker_side": "no", "yes_price_cents": 41, "count": 5},
        ]
    )
    write_partition(key, rows)
    back = read_partition(key)
    assert len(back) == 2
    assert set(back["trade_id"]) == {"t1", "t2"}


def test_trades_dedupe_on_trade_id(tmp_env):
    """Repeated trade_id on rerun must not duplicate rows."""
    from moneybutton.data.store import PartitionKey, read_partition, write_partition

    key = PartitionKey(dataset="trades", category_or_source="CRYPTO", year_month="2026-04")
    first = pd.DataFrame([
        {"trade_id": "t1", "ticker": "KXBTC-X", "ts": "2026-04-23T15:00:00+00:00",
         "taker_side": "yes", "yes_price_cents": 42, "count": 10},
    ])
    second = pd.DataFrame([
        {"trade_id": "t1", "ticker": "KXBTC-X", "ts": "2026-04-23T15:00:00+00:00",
         "taker_side": "yes", "yes_price_cents": 43, "count": 10},  # corrected price
        {"trade_id": "t2", "ticker": "KXBTC-X", "ts": "2026-04-23T15:00:02+00:00",
         "taker_side": "no", "yes_price_cents": 41, "count": 5},
    ])
    write_partition(key, first)
    write_partition(key, second)
    back = read_partition(key)
    assert len(back) == 2  # t1 overwritten, t2 added
    t1 = back[back["trade_id"] == "t1"].iloc[0]
    assert t1["yes_price_cents"] == 43  # new value won
