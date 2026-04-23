"""Kalshi historical backfill scraper (SPEC §7.1, §16 step 10).

First-time scrape is a multi-hour operation on a cold repo. Design goals:
  1. Resumable — cursor + last-completed partition persisted to SQLite
     (scraper_state). A crash/rate-limit/user-Ctrl-C can resume.
  2. Respectful — 1 req/sec default; exponential backoff on 429.
  3. Idempotent — repeated runs converge on the same Parquet output
     (write_partition dedupes by ticker for markets, by (ticker, ts) for
     prices).

CLI:
    python -m moneybutton data backfill --since=2024-01-01 [--category=KXFED]

This file does not wire up the CLI; that happens at build step 26. It does
expose backfill_markets() + backfill_prices() which the CLI will call.
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from moneybutton.core.audit import record as audit_record
from moneybutton.core.config import get_settings
from moneybutton.data.store import PartitionKey, write_partition, year_month
from moneybutton.kalshi.client import KalshiClient

log = logging.getLogger("moneybutton.scraper.kalshi")

DEFAULT_RATE_LIMIT_SLEEP_S = 1.0
DEFAULT_CANDLE_INTERVAL_MIN = 60

MARKET_KEEP_COLUMNS: list[str] = [
    "ticker",
    "event_ticker",
    "series_ticker",
    "title",
    "subtitle",
    "category",
    "status",
    "result",
    "can_close_early",
    "expiration_time",
    "close_time",
    "open_time",
    "expected_expiration_time",
    "previous_yes_bid",
    "previous_yes_ask",
    "last_price",
    "volume",
    "volume_24h",
    "open_interest",
    "liquidity",
    "notional_value",
    "yes_sub_title",
    "no_sub_title",
    "rules_primary",
    "rules_secondary",
    "tick_size",
]


@dataclass
class BackfillProgress:
    """Resumable cursor state. Mirrored to SQLite `scraper_state`."""

    scraper: str
    cursor: str | None
    last_completed_partition: str | None
    updated_ts: str


def _db_path() -> Path:
    return get_settings().sqlite_db_path


def load_progress(scraper: str) -> BackfillProgress:
    with sqlite3.connect(str(_db_path())) as conn:
        row = conn.execute(
            "SELECT cursor, last_completed_partition, updated_ts FROM scraper_state WHERE scraper = ?",
            (scraper,),
        ).fetchone()
    if row is None:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        return BackfillProgress(scraper=scraper, cursor=None, last_completed_partition=None, updated_ts=now)
    return BackfillProgress(scraper=scraper, cursor=row[0], last_completed_partition=row[1], updated_ts=row[2])


def save_progress(p: BackfillProgress) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    with sqlite3.connect(str(_db_path())) as conn:
        conn.execute(
            """
            INSERT INTO scraper_state (scraper, cursor, last_completed_partition, updated_ts)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(scraper) DO UPDATE SET
              cursor = excluded.cursor,
              last_completed_partition = excluded.last_completed_partition,
              updated_ts = excluded.updated_ts
            """,
            (p.scraper, p.cursor, p.last_completed_partition, now),
        )
        conn.commit()


def _coerce_market_row(m: dict) -> dict:
    """Project a raw Kalshi market JSON into the fixed schema we persist.

    Unknown keys are dropped; missing keys become None. This gives the
    Parquet files a stable columnar schema across time even as Kalshi adds
    new fields.
    """
    row = {k: m.get(k) for k in MARKET_KEEP_COLUMNS}
    # Kalshi dropped `category` from the market-level response in 2026Q1.
    # Derive a pseudo-category from the ticker/event_ticker prefix so the
    # Parquet files still partition into useful buckets.
    if not row.get("category"):
        row["category"] = _derive_category(m)
    return row


def _derive_category(m: dict) -> str:
    """Extract a partition-friendly category from ticker or event_ticker.

    Kalshi tickers follow the pattern `KX<SERIES>-<...>` where <SERIES> is
    effectively the category (NFL, FED, MVESPORTS, PRES, etc.). We take
    the first hyphen-separated segment, strip the `KX` prefix, cap length,
    and uppercase. Fallback is 'UNCATEGORIZED' so downstream code is
    schema-stable even on totally malformed tickers.
    """
    for src in (m.get("event_ticker"), m.get("series_ticker"), m.get("ticker")):
        if not src:
            continue
        first = str(src).split("-", 1)[0]
        if first.upper().startswith("KX"):
            first = first[2:]
        first = first.strip().upper()
        if first:
            return first[:20]
    return "UNCATEGORIZED"


def _partition_key_for_market(m: dict) -> PartitionKey:
    category = (m.get("category") or _derive_category(m) or "UNCATEGORIZED").upper()
    ts = m.get("close_time") or m.get("expiration_time") or m.get("open_time")
    ym = year_month(ts) if ts else "unknown"
    return PartitionKey(dataset="markets", category_or_source=category, year_month=ym)


def backfill_markets(
    *,
    client: KalshiClient | None = None,
    category: str | None = None,
    since: dt.date | None = None,
    limit_per_page: int = 200,
    max_pages: int | None = None,
    rate_limit_sleep_s: float = DEFAULT_RATE_LIMIT_SLEEP_S,
    scraper_name: str = "kalshi_markets",
) -> int:
    """Page through settled markets, writing to Parquet by (category, year_month).

    `since` filters locally on close_time / expiration_time — Kalshi's v2
    query-param filters by close_ts are inclusive, but we re-check so small
    API changes don't break the contract.
    Returns the number of market rows written.
    """
    owns_client = client is None
    client = client or KalshiClient()
    written = 0
    pages = 0
    progress = load_progress(scraper_name)
    try:
        cursor = progress.cursor
        while True:
            page = client.list_markets(
                # Kalshi's /markets `status` query accepts the legacy
                # vocabulary (unopened|open|closed|settled); the response
                # uses the newer `finalized` label for resolved markets.
                # They aren't symmetric — querying status=finalized returns
                # 400 "invalid status filter". _filter_resolved below
                # normalizes the response shape regardless.
                status="settled",
                category=category,
                cursor=cursor,
                limit=limit_per_page,
            )
            markets = page.get("markets", []) or []
            markets = _filter_resolved(markets)
            if since is not None and markets:
                markets = _filter_by_since(markets, since)
            if markets:
                by_partition: dict[PartitionKey, list[dict]] = {}
                for m in markets:
                    key = _partition_key_for_market(m)
                    by_partition.setdefault(key, []).append(_coerce_market_row(m))
                for key, rows in by_partition.items():
                    write_partition(key, pd.DataFrame(rows))
                    written += len(rows)
                    log.info("markets: wrote %d rows to %s/%s", len(rows),
                             key.category_or_source, key.year_month)
            cursor = page.get("cursor")
            progress = BackfillProgress(
                scraper=scraper_name,
                cursor=cursor,
                last_completed_partition=progress.last_completed_partition,
                updated_ts=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
            save_progress(progress)
            pages += 1
            if not cursor:
                break
            if max_pages is not None and pages >= max_pages:
                log.info("backfill_markets: max_pages=%d reached, stopping", max_pages)
                break
            time.sleep(rate_limit_sleep_s)
    finally:
        if owns_client:
            client.close()
    audit_record(
        actor="data.scraper_kalshi",
        action="backfill_markets",
        payload={"category": category, "since": str(since) if since else None,
                 "pages": pages, "rows_written": written},
        outcome="OK",
    )
    return written


def _filter_by_since(markets: list[dict], since: dt.date) -> list[dict]:
    kept: list[dict] = []
    since_ts = dt.datetime.combine(since, dt.time.min, tzinfo=dt.timezone.utc)
    for m in markets:
        close = m.get("close_time") or m.get("expiration_time")
        if close is None:
            kept.append(m)
            continue
        try:
            close_dt = dt.datetime.fromisoformat(close.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            kept.append(m)
            continue
        if close_dt >= since_ts:
            kept.append(m)
    return kept


def _filter_resolved(markets: list[dict]) -> list[dict]:
    """Keep only markets whose result is yes/no (fully-resolved)."""
    return [m for m in markets if (m.get("result") or "").lower() in ("yes", "no")]


def backfill_prices_for_tickers(
    tickers: Iterable[str],
    *,
    client: KalshiClient | None = None,
    period_interval: int = DEFAULT_CANDLE_INTERVAL_MIN,
    rate_limit_sleep_s: float = DEFAULT_RATE_LIMIT_SLEEP_S,
) -> int:
    """For each ticker, fetch candlesticks (uses the market's open->close range).

    Writes rows to prices/category={cat}/year_month={ym}/prices.parquet.
    """
    owns_client = client is None
    client = client or KalshiClient()
    written = 0
    try:
        for ticker in tickers:
            try:
                market = client.get_market(ticker).get("market", {})
            except Exception as e:  # noqa: BLE001
                log.warning("failed to fetch market %s: %s", ticker, e)
                continue
            open_ts = market.get("open_time")
            close_ts = market.get("close_time") or market.get("expiration_time")
            category = (market.get("category") or _derive_category(market) or "UNCATEGORIZED").upper()
            series_ticker = market.get("series_ticker")
            if not (open_ts and close_ts):
                continue
            start_epoch = int(dt.datetime.fromisoformat(open_ts.replace("Z", "+00:00")).timestamp())
            end_epoch = int(dt.datetime.fromisoformat(close_ts.replace("Z", "+00:00")).timestamp())
            try:
                resp = client.get_candlesticks(
                    ticker,
                    series_ticker=series_ticker,
                    start_ts=start_epoch,
                    end_ts=end_epoch,
                    period_interval=period_interval,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("failed to fetch candlesticks for %s: %s", ticker, e)
                continue
            rows = _project_candlesticks(resp, ticker)
            if not rows:
                continue
            df = pd.DataFrame(rows)
            # Partition by close month; we accept that a market's price history
            # may span multiple months and we still put it all in the close-month
            # partition. This keeps the "one file per market" invariant.
            ym = year_month(close_ts)
            write_partition(
                PartitionKey(dataset="prices", category_or_source=category, year_month=ym),
                df,
            )
            written += len(df)
            time.sleep(rate_limit_sleep_s)
    finally:
        if owns_client:
            client.close()
    audit_record(
        actor="data.scraper_kalshi",
        action="backfill_prices",
        payload={"rows_written": written},
        outcome="OK",
    )
    return written


def _project_candlesticks(resp: dict, ticker: str) -> list[dict]:
    """Flatten a candlesticks response into per-timestamp rows."""
    out: list[dict] = []
    for c in resp.get("candlesticks", []) or []:
        ts = c.get("end_period_ts") or c.get("ts")
        if ts is None:
            continue
        ts_iso = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).isoformat()
        # Kalshi returns nested price blocks: yes_bid, yes_ask, price (fills).
        yes_bid = c.get("yes_bid", {}) if isinstance(c.get("yes_bid"), dict) else {}
        yes_ask = c.get("yes_ask", {}) if isinstance(c.get("yes_ask"), dict) else {}
        price = c.get("price", {}) if isinstance(c.get("price"), dict) else {}
        out.append(
            {
                "ticker": ticker,
                "ts": ts_iso,
                "yes_bid_open": yes_bid.get("open"),
                "yes_bid_high": yes_bid.get("high"),
                "yes_bid_low": yes_bid.get("low"),
                "yes_bid_close": yes_bid.get("close"),
                "yes_ask_open": yes_ask.get("open"),
                "yes_ask_high": yes_ask.get("high"),
                "yes_ask_low": yes_ask.get("low"),
                "yes_ask_close": yes_ask.get("close"),
                "last_price_open": price.get("open"),
                "last_price_high": price.get("high"),
                "last_price_low": price.get("low"),
                "last_price_close": price.get("close"),
                "volume": c.get("volume"),
                "open_interest": c.get("open_interest"),
            }
        )
    return out
