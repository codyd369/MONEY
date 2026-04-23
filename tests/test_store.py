"""Parquet store roundtrip + idempotent merge tests (SPEC §6.2)."""

from __future__ import annotations

import pandas as pd


def test_write_and_read_partition(tmp_env):
    from moneybutton.data.store import PartitionKey, read_partition, write_partition

    key = PartitionKey(dataset="markets", category_or_source="KXFED", year_month="2025-03")
    rows = pd.DataFrame(
        [
            {"ticker": "KXFED-25MAR-HIKE", "title": "Will the Fed hike in March?", "status": "settled"},
            {"ticker": "KXFED-25MAR-CUT", "title": "Will the Fed cut in March?", "status": "settled"},
        ]
    )
    path = write_partition(key, rows)
    assert path.exists()
    back = read_partition(key)
    assert len(back) == 2
    assert set(back["ticker"]) == {"KXFED-25MAR-HIKE", "KXFED-25MAR-CUT"}


def test_merge_dedupes_on_ticker(tmp_env):
    """Re-writing the same ticker must not create a duplicate row; the new
    row must win (tracks updated market status, etc.)."""
    from moneybutton.data.store import PartitionKey, read_partition, write_partition

    key = PartitionKey(dataset="markets", category_or_source="KXFED", year_month="2025-03")
    write_partition(
        key,
        pd.DataFrame(
            [{"ticker": "KXFED-25MAR-HIKE", "status": "initialized", "result": None}]
        ),
    )
    write_partition(
        key,
        pd.DataFrame(
            [{"ticker": "KXFED-25MAR-HIKE", "status": "settled", "result": "yes"}]
        ),
    )
    df = read_partition(key)
    assert len(df) == 1
    assert df.iloc[0]["status"] == "settled"
    assert df.iloc[0]["result"] == "yes"


def test_prices_dedupe_on_ticker_ts(tmp_env):
    from moneybutton.data.store import PartitionKey, read_partition, write_partition

    key = PartitionKey(dataset="prices", category_or_source="KXFED", year_month="2025-03")
    write_partition(
        key,
        pd.DataFrame(
            [
                {"ticker": "KXFED-25MAR-HIKE", "ts": "2025-03-01T12:00:00Z", "yes_bid": 42},
                {"ticker": "KXFED-25MAR-HIKE", "ts": "2025-03-01T13:00:00Z", "yes_bid": 43},
            ]
        ),
    )
    # Upsert the first row with a corrected value + add a new ts.
    write_partition(
        key,
        pd.DataFrame(
            [
                {"ticker": "KXFED-25MAR-HIKE", "ts": "2025-03-01T12:00:00Z", "yes_bid": 44},
                {"ticker": "KXFED-25MAR-HIKE", "ts": "2025-03-01T14:00:00Z", "yes_bid": 45},
            ]
        ),
    )
    df = read_partition(key).sort_values("ts").reset_index(drop=True)
    assert len(df) == 3
    # The 12:00 row was updated to yes_bid=44 (new row wins).
    assert df.iloc[0]["yes_bid"] == 44


def test_read_dataset_filters(tmp_env):
    from moneybutton.data.store import (
        PartitionKey,
        read_dataset,
        write_partition,
    )

    for cat in ["KXFED", "KXELEC"]:
        for ym in ["2025-01", "2025-02", "2025-03"]:
            write_partition(
                PartitionKey(dataset="markets", category_or_source=cat, year_month=ym),
                pd.DataFrame([{"ticker": f"{cat}-{ym}-A", "status": "settled"}]),
            )
    df = read_dataset("markets", category_or_source="KXFED")
    assert len(df) == 3
    df = read_dataset("markets", year_month_from="2025-02", year_month_to="2025-03")
    assert len(df) == 4  # 2 categories x 2 months
    df = read_dataset(
        "markets",
        category_or_source="KXELEC",
        year_month_from="2025-02",
        year_month_to="2025-02",
    )
    assert len(df) == 1


def test_list_partitions(tmp_env):
    from moneybutton.data.store import PartitionKey, list_partitions, write_partition

    for ym in ["2025-01", "2025-02"]:
        write_partition(
            PartitionKey(dataset="news", category_or_source="rss", year_month=ym),
            pd.DataFrame([{"id": f"n-{ym}", "headline": "x"}]),
        )
    parts = list_partitions("news")
    assert len(parts) == 2
    assert all(p.dataset == "news" for p in parts)


def test_year_month_helpers():
    from moneybutton.data.store import year_month

    assert year_month("2026-04-23T12:00:00Z") == "2026-04"
    assert year_month(1713880000) == "2024-04"
