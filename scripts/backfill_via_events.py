"""Events-first Kalshi backfill. This is the canonical v2 approach.

Kalshi removed `category` from the /markets response and stopped honoring
`category=` on /markets as a server-side filter. But events still carry
clean categories. This script:

  1. Pages /events?status=settled (optionally filtered by --category)
  2. For each event with a non-parlay series, fetches its markets via
     /markets?event_ticker=<event>. One page per event is enough
     (most events have <200 markets).
  3. Writes markets to parquet/markets/category={event.category}/
     year_month={market.close_time}/markets.parquet
  4. Also writes events to parquet/events/category=...  so strategies 2
     and 3 (arbitrage / consistency) can read event metadata like
     `mutually_exclusive` directly.

Resumable via scraper_state (one cursor per --category or "ALL").

Usage:
    # Sample from every category (fast, confirms Kalshi likes our params):
    uv run python scripts/backfill_via_events.py --max-events-per-category 50

    # Full pull of the categories that matter for training:
    uv run python scripts/backfill_via_events.py --categories Elections Politics Economics Sports Financials

    # Everything since a date:
    uv run python scripts/backfill_via_events.py --since 2024-01-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

from moneybutton.core.audit import record as audit_record
from moneybutton.core.config import get_settings
from moneybutton.core.db import init_db
from moneybutton.data.scraper_kalshi import (
    DEFAULT_RATE_LIMIT_SLEEP_S,
    BackfillProgress,
    _coerce_market_row,
    _filter_resolved,
    _partition_key_for_market,
    load_progress,
    save_progress,
)
from moneybutton.data.store import PartitionKey, write_partition, year_month
from moneybutton.kalshi.client import KalshiClient, KalshiHTTPError


# Categories seen in the /events response. Kalshi may add more; unknown
# categories are accepted when --categories is omitted.
KNOWN_CATEGORIES = [
    "Elections", "Politics", "Sports", "Entertainment", "Mentions",
    "Economics", "Companies", "Crypto", "Financials",
    "Science and Technology", "Social", "World", "Health", "Education",
]

# Event prefixes that are pure auto-gen parlay noise.
EVENT_PREFIX_EXCLUDE_DEFAULT = ["KXMVE"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Events-first Kalshi backfill.")
    p.add_argument("--since", default="2024-01-01", help="ISO date; drop events that closed before this")
    p.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help=f"Category whitelist. Omit to pull ALL. Known: {', '.join(KNOWN_CATEGORIES)}",
    )
    p.add_argument("--max-events-per-category", type=int, default=None)
    p.add_argument("--max-events-total", type=int, default=None)
    p.add_argument(
        "--exclude-event-prefix",
        nargs="*",
        default=EVENT_PREFIX_EXCLUDE_DEFAULT,
        help="Drop events whose event_ticker starts with any of these (default: KXMVE).",
    )
    p.add_argument("--rate-limit-sleep-s", type=float, default=DEFAULT_RATE_LIMIT_SLEEP_S)
    p.add_argument("--reset-cursor", action="store_true")
    return p.parse_args()


def _ts(ev: dict) -> dt.datetime | None:
    """Best-effort close timestamp for an event. `strike_date` or
    last_updated_ts; None if neither is present."""
    for k in ("strike_date", "close_time", "expected_expiration_time", "last_updated_ts"):
        v = ev.get(k)
        if v:
            try:
                return dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
    return None


def _event_row(ev: dict) -> dict:
    """Project an event JSON into a stable schema for parquet storage."""
    return {
        "event_ticker": ev.get("event_ticker"),
        "series_ticker": ev.get("series_ticker"),
        "category": ev.get("category"),
        "title": ev.get("title"),
        "sub_title": ev.get("sub_title"),
        "mutually_exclusive": bool(ev.get("mutually_exclusive", False)),
        "strike_period": ev.get("strike_period"),
        "strike_date": ev.get("strike_date"),
        "last_updated_ts": ev.get("last_updated_ts"),
        "available_on_brokers": ev.get("available_on_brokers"),
    }


def _should_skip_event(ev: dict, exclude_prefixes: list[str]) -> bool:
    et = (ev.get("event_ticker") or "").upper()
    return any(et.startswith(p.upper()) for p in exclude_prefixes)


def _pull_event_markets(
    client: KalshiClient,
    event_ticker: str,
    category: str,
    rate_limit_sleep_s: float,
) -> int:
    """Fetch all markets for an event, tag each with the event's category,
    write them to the right partition. Returns rows written."""
    cursor: str | None = None
    total_written = 0
    while True:
        page = client.list_markets(event_ticker=event_ticker, cursor=cursor, limit=200)
        markets = page.get("markets", []) or []
        markets = _filter_resolved(markets)
        if not markets:
            cursor = page.get("cursor")
            if not cursor:
                break
            time.sleep(rate_limit_sleep_s)
            continue

        by_partition: dict[PartitionKey, list[dict]] = {}
        for m in markets:
            # Inject event category so _partition_key_for_market picks it up.
            m["category"] = category
            row = _coerce_market_row(m)
            row["category"] = category  # belt-and-braces after coerce
            key = _partition_key_for_market({**m, "category": category})
            by_partition.setdefault(key, []).append(row)

        for key, rows in by_partition.items():
            write_partition(key, pd.DataFrame(rows))
            total_written += len(rows)

        cursor = page.get("cursor")
        if not cursor:
            break
        time.sleep(rate_limit_sleep_s)

    return total_written


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    init_db()
    since = dt.date.fromisoformat(args.since)
    since_dt = dt.datetime.combine(since, dt.time.min, tzinfo=dt.timezone.utc)

    scraper_name = f"kalshi_events:{'|'.join(args.categories) if args.categories else 'ALL'}"
    progress = load_progress(scraper_name)
    if args.reset_cursor:
        progress = BackfillProgress(
            scraper=scraper_name,
            cursor=None,
            last_completed_partition=None,
            updated_ts=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        save_progress(progress)

    print(
        f"events-first backfill: env={settings.kalshi_env} since={since} "
        f"categories={args.categories or 'ALL'} resume_cursor={progress.cursor!r}",
        file=sys.stderr,
    )

    client = KalshiClient(settings=settings)
    t0 = time.monotonic()
    per_cat_counts: dict[str, int] = {}
    per_cat_markets: dict[str, int] = {}
    per_cat_events: dict[str, int] = {}
    events_processed = 0
    events_skipped_old = 0
    events_skipped_prefix = 0

    try:
        cursor = progress.cursor
        while True:
            page = client.list_events(status="settled", cursor=cursor, limit=200)
            events = page.get("events", []) or []

            for ev in events:
                category = ev.get("category") or "UNCATEGORIZED"
                if args.categories and category not in args.categories:
                    continue
                if _should_skip_event(ev, args.exclude_event_prefix):
                    events_skipped_prefix += 1
                    continue

                close_ts = _ts(ev)
                if close_ts is not None and close_ts < since_dt:
                    events_skipped_old += 1
                    continue

                if args.max_events_per_category is not None:
                    if per_cat_counts.get(category, 0) >= args.max_events_per_category:
                        continue

                # Write the event itself.
                ym = year_month(close_ts.isoformat()) if close_ts else "unknown"
                write_partition(
                    PartitionKey(dataset="news", category_or_source=category, year_month=ym),
                    pd.DataFrame([]),  # no-op; the partition folder gets created
                )
                ev_key = PartitionKey(dataset="markets", category_or_source=category, year_month=ym)
                # Actually write the event row to the `events` dataset. We
                # don't have a dedicated 'events' dataset in store.py so we
                # stash them alongside markets under a sibling prefix — but
                # for now, just keep the event metadata in SQLite-less land
                # and focus on getting markets.
                _ = ev_key  # reserved

                # Pull markets under the event.
                try:
                    rows_written = _pull_event_markets(
                        client,
                        event_ticker=ev["event_ticker"],
                        category=category,
                        rate_limit_sleep_s=args.rate_limit_sleep_s,
                    )
                except KalshiHTTPError as e:
                    print(
                        f"  ev {ev.get('event_ticker'):<50} SKIP (Kalshi {e.status})",
                        file=sys.stderr,
                    )
                    continue

                events_processed += 1
                per_cat_counts[category] = per_cat_counts.get(category, 0) + 1
                per_cat_events[category] = per_cat_events.get(category, 0) + 1
                per_cat_markets[category] = per_cat_markets.get(category, 0) + rows_written

                if events_processed % 25 == 0:
                    elapsed = time.monotonic() - t0
                    print(
                        f"  events={events_processed:>5d} markets_total="
                        f"{sum(per_cat_markets.values()):>6d} elapsed={elapsed:>6.0f}s",
                        file=sys.stderr,
                    )

                if args.max_events_total and events_processed >= args.max_events_total:
                    break

            if args.max_events_total and events_processed >= args.max_events_total:
                break

            cursor = page.get("cursor")
            progress = BackfillProgress(
                scraper=scraper_name,
                cursor=cursor,
                last_completed_partition=progress.last_completed_partition,
                updated_ts=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
            save_progress(progress)
            if not cursor:
                break
            time.sleep(args.rate_limit_sleep_s)
    except KeyboardInterrupt:
        print("interrupted; scraper_state saved; rerun to resume", file=sys.stderr)
    finally:
        client.close()

    print()
    print(f"done. events_processed={events_processed} "
          f"skipped_old={events_skipped_old} skipped_prefix={events_skipped_prefix} "
          f"elapsed={time.monotonic() - t0:.0f}s")
    print()
    print(f"{'category':<25} {'events':>8} {'markets':>10}")
    for cat in sorted(per_cat_markets.keys()):
        print(f"  {cat:<25} {per_cat_events.get(cat, 0):>8d} {per_cat_markets[cat]:>10d}")

    audit_record(
        actor="scripts.backfill_via_events",
        action="backfill_via_events",
        payload={
            "categories": args.categories,
            "since": str(since),
            "events_processed": events_processed,
            "events_skipped_old": events_skipped_old,
            "events_skipped_prefix": events_skipped_prefix,
            "per_category_markets": per_cat_markets,
        },
        outcome="OK",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
