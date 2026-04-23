"""Backfill news from NewsAPI + EventRegistry + curated RSS feeds.

Writes to data/parquet/news/source=<src>/year_month=<ym>/news.parquet.

Usage:
    # Everything you have keys for (RSS always on; NewsAPI + EventRegistry
    # only if the respective env var is set):
    uv run python scripts/backfill_news.py

    # RSS only, with a wider lookback:
    uv run python scripts/backfill_news.py --rss-only --since 2024-01-01

    # Custom query terms for the paid APIs:
    uv run python scripts/backfill_news.py --keywords "Fed rate" "CPI" "OPEC" "Trump"

RSS feeds are curated for the markets we actually trade on: econ,
politics, finance, crypto, geopolitics. Add more via --extra-rss.

Safe to re-run — write_partition dedupes by id (source-prefixed sha256
of the URL).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

from moneybutton.core.audit import record as audit_record
from moneybutton.core.config import get_settings
from moneybutton.data.scraper_news import (
    EventRegistryScraper,
    NewsAPIScraper,
    RSSScraper,
)


DEFAULT_RSS_FEEDS: list[tuple[str, str]] = [
    # Markets / finance
    ("reuters_business", "https://feeds.reuters.com/reuters/businessNews"),
    ("reuters_markets", "https://feeds.reuters.com/reuters/marketsNews"),
    ("ft_home", "https://www.ft.com/?format=rss"),
    ("bloomberg_markets", "https://feeds.bloomberg.com/markets/news.rss"),
    ("wsj_business", "https://feeds.content.dowjones.io/public/rss/RSSWSJD"),
    ("cnbc_top", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    # Politics / geopolitics
    ("reuters_politics", "https://feeds.reuters.com/Reuters/PoliticsNews"),
    ("ap_top", "https://rsshub.app/apnews/topics/apf-topnews"),
    ("politico_top", "https://rss.politico.com/politics-news.xml"),
    ("axios_politics", "https://api.axios.com/feed/politics/"),
    ("bbc_world", "http://feeds.bbci.co.uk/news/world/rss.xml"),
    # Crypto
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("decrypt", "https://decrypt.co/feed"),
    # Weather / climate (relevant to KXHIGH* / KXSUB* markets)
    ("nhc_hurricanes", "https://www.nhc.noaa.gov/index-at.xml"),
    # Macro / Fed
    ("fed_press", "https://www.federalreserve.gov/feeds/press_all.xml"),
    ("treasury_press", "https://home.treasury.gov/rss/press-releases.xml"),
]


DEFAULT_KEYWORDS: list[str] = [
    "Federal Reserve", "interest rate", "CPI", "unemployment",
    "election", "Trump", "Congress",
    "OPEC", "oil price",
    "Bitcoin", "Ethereum", "crypto",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill news into parquet/news/.")
    p.add_argument("--since", default=None, help="ISO date (default: 7 days ago)")
    p.add_argument("--rss-only", action="store_true", help="Skip NewsAPI and EventRegistry")
    p.add_argument("--skip-rss", action="store_true", help="Skip the RSS pull")
    p.add_argument("--keywords", nargs="+", default=DEFAULT_KEYWORDS)
    p.add_argument("--extra-rss", nargs="+", default=[], help="Extra RSS URLs to add")
    p.add_argument("--rate-limit-sleep-s", type=float, default=0.5)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    since = dt.date.fromisoformat(args.since) if args.since else (dt.date.today() - dt.timedelta(days=7))
    since_dt = dt.datetime.combine(since, dt.time.min, tzinfo=dt.timezone.utc)

    print(f"backfill_news: since={since} rss_only={args.rss_only} skip_rss={args.skip_rss}", flush=True)

    written = {"rss": 0, "newsapi": 0, "eventregistry": 0}

    # ------------------------------ RSS ----------------------------------
    if not args.skip_rss:
        feeds = list(DEFAULT_RSS_FEEDS)
        for i, url in enumerate(args.extra_rss):
            feeds.append((f"extra_{i}", url))
        print(f"  RSS: {len(feeds)} feeds", flush=True)
        rss = RSSScraper(feeds=feeds)
        t0 = time.monotonic()
        try:
            rows_per_slug = rss.fetch_all()
        except Exception as e:  # noqa: BLE001
            print(f"  RSS fetch error: {e}", flush=True)
            rows_per_slug = {}
        # Write each slug's rows to its own partition via fetch_and_store-like
        # path (we already have rows_per_slug, so bypass fetch_and_store).
        from moneybutton.data.scraper_news import _write_rows  # noqa: F401

        total_rss = 0
        for slug, rows in rows_per_slug.items():
            n = _write_rows(f"rss:{slug}", rows)
            total_rss += n
            print(f"    rss:{slug:<25} rows={n}", flush=True)
        written["rss"] = total_rss
        print(f"  RSS total: {total_rss} rows in {time.monotonic() - t0:.0f}s", flush=True)

    # ------------------------------ NewsAPI ------------------------------
    if not args.rss_only and settings.newsapi_key.get_secret_value():
        print(f"  NewsAPI: {len(args.keywords)} queries", flush=True)
        napi = NewsAPIScraper(settings=settings)
        total_napi = 0
        for q in args.keywords:
            try:
                n = napi.fetch_and_store(q=q, since=since_dt)
                total_napi += n
                print(f"    q={q:<25} rows={n}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"    q={q:<25} ERROR {type(e).__name__}: {e}", flush=True)
            time.sleep(args.rate_limit_sleep_s)
        written["newsapi"] = total_napi
    elif not args.rss_only:
        print("  NewsAPI: skipped (NEWSAPI_KEY not set)", flush=True)

    # --------------------------- EventRegistry ----------------------------
    if not args.rss_only and settings.eventregistry_key.get_secret_value():
        print(f"  EventRegistry: {len(args.keywords)} keywords", flush=True)
        er = EventRegistryScraper(settings=settings)
        total_er = 0
        for kw in args.keywords:
            try:
                n = er.fetch_and_store(keyword=kw, since=since_dt)
                total_er += n
                print(f"    kw={kw:<25} rows={n}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"    kw={kw:<25} ERROR {type(e).__name__}: {e}", flush=True)
            time.sleep(args.rate_limit_sleep_s)
        written["eventregistry"] = total_er
    elif not args.rss_only:
        print("  EventRegistry: skipped (EVENTREGISTRY_KEY not set)", flush=True)

    print()
    total = sum(written.values())
    print(f"done. total rows written: {total}")
    print(f"  rss            : {written['rss']}")
    print(f"  newsapi        : {written['newsapi']}")
    print(f"  eventregistry  : {written['eventregistry']}")

    audit_record(
        actor="scripts.backfill_news",
        action="backfill_news",
        payload={"since": str(since), **written},
        outcome="OK",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
