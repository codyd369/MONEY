"""Backfill a curated list of Kalshi categories back to `--since`.

Use this instead of the open-ended backfill_markets.py if you want a
clean dataset of human-curated markets (Politics, Economics, traditional
sports, weather) without wading through the MVE parlay noise.

Usage:
    uv run python scripts/backfill_curated.py --since 2024-01-01
    uv run python scripts/backfill_curated.py --since 2025-01-01 --categories Politics Economics

Caveat: Kalshi may not publish a canonical list of category strings and
the ones that work vary over time. If a given category 400s, the script
skips it and moves on.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

from moneybutton.core.config import get_settings
from moneybutton.core.db import init_db
from moneybutton.data.scraper_kalshi import (
    DEFAULT_RATE_LIMIT_SLEEP_S,
    backfill_markets,
)
from moneybutton.kalshi.client import KalshiClient, KalshiHTTPError

DEFAULT_CATEGORIES: list[str] = [
    "Politics",
    "Economics",
    "Financials",
    "Science and Technology",
    "Sports",
    "Climate and Weather",
    "Crypto",
    "Entertainment",
    "Culture",
    "Companies",
    "World",
    "Health",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill a curated list of categories.")
    p.add_argument("--since", default="2024-01-01")
    p.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    p.add_argument("--max-pages-per-category", type=int, default=None)
    p.add_argument("--rate-limit-sleep-s", type=float, default=DEFAULT_RATE_LIMIT_SLEEP_S)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    init_db()
    since = dt.date.fromisoformat(args.since)

    print(f"env={settings.kalshi_env}  since={since}  categories={args.categories}", file=sys.stderr)

    total = 0
    client = KalshiClient(settings=settings)
    try:
        for cat in args.categories:
            try:
                written = backfill_markets(
                    client=client,
                    category=cat,
                    since=since,
                    max_pages=args.max_pages_per_category,
                    rate_limit_sleep_s=args.rate_limit_sleep_s,
                    scraper_name=f"kalshi_markets:{cat}",
                )
                total += written
                print(f"  {cat:<25} rows_written={written}", file=sys.stderr)
            except KalshiHTTPError as e:
                print(f"  {cat:<25} SKIP (Kalshi {e.status}: {e.body[:120]})", file=sys.stderr)
            except KeyboardInterrupt:
                print("interrupted; moving on", file=sys.stderr)
                return 130
    finally:
        client.close()

    print()
    print(f"done. total rows_written={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
