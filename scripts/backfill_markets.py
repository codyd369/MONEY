"""Backfill Kalshi settled markets with progress display and resume.

Usage:
    # Full spec default (since 2024-01-01, all categories):
    uv run python scripts/backfill_markets.py

    # Scope the scrape:
    uv run python scripts/backfill_markets.py --since 2024-01-01
    uv run python scripts/backfill_markets.py --since 2025-01-01 --category Financials
    uv run python scripts/backfill_markets.py --max-pages 5   # smoke test

Resumable — the cursor is persisted to SQLite scraper_state between page
fetches. Ctrl-C and rerun picks up where it left off.

Rate-limited to 1 req/sec by default (polite to Kalshi).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

from moneybutton.core.config import get_settings
from moneybutton.core.db import init_db
from moneybutton.data.scraper_kalshi import (
    DEFAULT_RATE_LIMIT_SLEEP_S,
    _coerce_market_row,
    _filter_by_since,
    _partition_key_for_market,
    load_progress,
    save_progress,
    BackfillProgress,
)
from moneybutton.data.store import write_partition
from moneybutton.kalshi.client import KalshiClient


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill Kalshi settled markets to Parquet.")
    p.add_argument("--since", default="2024-01-01", help="ISO date (YYYY-MM-DD); default 2024-01-01")
    p.add_argument("--category", default=None, help="Kalshi category name (optional)")
    p.add_argument("--limit-per-page", type=int, default=200)
    p.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (for smoke tests)")
    p.add_argument("--rate-limit-sleep-s", type=float, default=DEFAULT_RATE_LIMIT_SLEEP_S)
    p.add_argument("--reset-cursor", action="store_true", help="Clear saved cursor and start over")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    init_db()
    since = dt.date.fromisoformat(args.since)

    scraper_name = f"kalshi_markets:{args.category or 'ALL'}"
    progress = load_progress(scraper_name)
    if args.reset_cursor:
        progress = BackfillProgress(scraper=scraper_name, cursor=None, last_completed_partition=None,
                                    updated_ts=dt.datetime.now(dt.timezone.utc).isoformat())
        save_progress(progress)

    print(
        f"backfill_markets: env={settings.kalshi_env} since={since} "
        f"category={args.category or 'ALL'} resume_cursor={progress.cursor!r}",
        file=sys.stderr,
    )

    client = KalshiClient(settings=settings)
    t0 = time.monotonic()
    pages = 0
    rows_written = 0
    partitions_touched: dict[str, int] = {}

    try:
        cursor = progress.cursor
        while True:
            page = client.list_markets(
                status="settled",
                category=args.category,
                cursor=cursor,
                limit=args.limit_per_page,
            )
            markets = page.get("markets", []) or []
            kept = _filter_by_since(markets, since) if since else markets

            if kept:
                by_partition: dict = {}
                for m in kept:
                    key = _partition_key_for_market(m)
                    by_partition.setdefault(key, []).append(_coerce_market_row(m))
                for key, rows in by_partition.items():
                    import pandas as pd
                    write_partition(key, pd.DataFrame(rows))
                    rows_written += len(rows)
                    part = f"{key.category_or_source}/{key.year_month}"
                    partitions_touched[part] = partitions_touched.get(part, 0) + len(rows)

            cursor = page.get("cursor")
            progress = BackfillProgress(
                scraper=scraper_name,
                cursor=cursor,
                last_completed_partition=progress.last_completed_partition,
                updated_ts=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
            save_progress(progress)
            pages += 1
            elapsed = time.monotonic() - t0

            print(
                f"  page {pages:>4d}  rows_this_page={len(kept):>3d}  "
                f"rows_total={rows_written:>6d}  elapsed={elapsed:>6.0f}s  "
                f"cursor={(cursor or 'END')[:16]}",
                file=sys.stderr,
            )

            if not cursor:
                break
            if args.max_pages is not None and pages >= args.max_pages:
                print(f"backfill_markets: max_pages={args.max_pages} reached, stopping.", file=sys.stderr)
                break

            time.sleep(args.rate_limit_sleep_s)
    except KeyboardInterrupt:
        print("backfill_markets: interrupted; cursor saved; rerun to resume.", file=sys.stderr)
        return 130
    finally:
        client.close()

    print()
    print(f"done. pages={pages} rows_written={rows_written} elapsed={time.monotonic() - t0:.0f}s")
    print("top partitions touched:")
    for part, n in sorted(partitions_touched.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        print(f"  {part:<40} {n:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
