"""Backfill Kalshi candlestick prices for settled markets.

Usage:
    # Most common: top-N by volume across all on-disk settled markets.
    uv run python scripts/backfill_prices.py --top-n 500

    # By category:
    uv run python scripts/backfill_prices.py --top-n 200 --category FINANCIALS

    # Explicit tickers:
    uv run python scripts/backfill_prices.py --tickers KX-FED-25MAR KX-FED-25APR

This is the long tail of the first-time scrape. Each market makes one
candlesticks call (open_time -> close_time, 60m interval), so cost is
roughly N requests. At 1 req/sec, 500 markets = ~10 min; 5000 ≈ 90 min.

Resumable — once a market's prices are on disk, re-running skips it
(write_partition dedupes by ticker+ts anyway, but the script also skips
markets whose Parquet rows are already present to save the API call).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

import pandas as pd

from moneybutton.core.audit import record as audit_record
from moneybutton.core.config import get_settings
from moneybutton.data.scraper_kalshi import (
    DEFAULT_CANDLE_INTERVAL_MIN,
    DEFAULT_RATE_LIMIT_SLEEP_S,
    _project_candlesticks,
)
from moneybutton.data.store import PartitionKey, read_dataset, write_partition, year_month
from moneybutton.kalshi.client import KalshiClient


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill Kalshi candlestick prices for settled markets.")
    p.add_argument("--top-n", type=int, default=None, help="Pick top N settled markets by volume")
    p.add_argument("--tickers", nargs="+", default=None, help="Explicit ticker list")
    p.add_argument("--category", default=None, help="Restrict to one category (uppercased)")
    p.add_argument("--period-interval", type=int, default=DEFAULT_CANDLE_INTERVAL_MIN, help="Candle interval minutes (60 default)")
    p.add_argument("--rate-limit-sleep-s", type=float, default=DEFAULT_RATE_LIMIT_SLEEP_S)
    p.add_argument("--no-skip-existing", action="store_true", help="Refetch even if prices already on disk")
    return p.parse_args()


def _select_tickers(args, markets: pd.DataFrame) -> list[str]:
    if args.tickers:
        return list(args.tickers)
    if args.category:
        markets = markets[markets["category"].str.upper() == args.category.upper()]
    if markets.empty:
        return []
    markets = markets.copy()
    markets["volume"] = pd.to_numeric(markets["volume"], errors="coerce").fillna(0)
    markets = markets.sort_values("volume", ascending=False)
    if args.top_n:
        markets = markets.head(args.top_n)
    return markets["ticker"].tolist()


def _existing_price_tickers() -> set[str]:
    df = read_dataset("prices")
    if df.empty:
        return set()
    return set(df["ticker"].unique())


def main() -> int:
    args = _parse_args()
    settings = get_settings()

    markets = read_dataset("markets")
    if markets.empty:
        print("no markets on disk. Run scripts/backfill_markets.py first.", file=sys.stderr)
        return 1
    markets = markets[markets["result"].isin(["yes", "no"])].copy()

    tickers = _select_tickers(args, markets)
    if not tickers:
        print("no tickers matched selection.", file=sys.stderr)
        return 1

    skip_set: set[str] = set() if args.no_skip_existing else _existing_price_tickers()
    pending = [t for t in tickers if t not in skip_set]
    print(
        f"backfill_prices: {len(tickers)} matched, {len(tickers) - len(pending)} already on disk, "
        f"{len(pending)} to fetch. interval={args.period_interval}m, sleep={args.rate_limit_sleep_s}s",
        file=sys.stderr,
    )

    client = KalshiClient(settings=settings)
    t0 = time.monotonic()
    rows_written = 0
    errors = 0
    try:
        for i, ticker in enumerate(pending, 1):
            market_row = markets[markets["ticker"] == ticker]
            if market_row.empty:
                continue
            m = market_row.iloc[0].to_dict()
            open_ts = m.get("open_time")
            close_ts = m.get("close_time") or m.get("expiration_time")
            category = (m.get("category") or "UNCATEGORIZED").upper()
            series_ticker = m.get("series_ticker")
            if not (open_ts and close_ts):
                continue
            try:
                start_epoch = int(dt.datetime.fromisoformat(open_ts.replace("Z", "+00:00")).timestamp())
                end_epoch = int(dt.datetime.fromisoformat(close_ts.replace("Z", "+00:00")).timestamp())
                resp = client.get_candlesticks(
                    ticker,
                    series_ticker=series_ticker,
                    start_ts=start_epoch,
                    end_ts=end_epoch,
                    period_interval=args.period_interval,
                )
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"  [{i:>5d}/{len(pending)}] {ticker} ERROR {type(e).__name__}: {e}", file=sys.stderr)
                continue

            rows = _project_candlesticks(resp, ticker)
            if rows:
                ym = year_month(close_ts)
                write_partition(
                    PartitionKey(dataset="prices", category_or_source=category, year_month=ym),
                    pd.DataFrame(rows),
                )
                rows_written += len(rows)

            if i % 25 == 0 or i == len(pending):
                elapsed = time.monotonic() - t0
                print(
                    f"  [{i:>5d}/{len(pending)}] rows_written={rows_written:>7d} "
                    f"errors={errors:>3d} elapsed={elapsed:>6.0f}s",
                    file=sys.stderr,
                )
            time.sleep(args.rate_limit_sleep_s)
    except KeyboardInterrupt:
        print("backfill_prices: interrupted; rerun to resume (finished tickers skipped).", file=sys.stderr)
        return 130
    finally:
        client.close()

    audit_record(
        actor="scripts.backfill_prices",
        action="backfill_prices",
        payload={
            "selection": {"top_n": args.top_n, "category": args.category, "tickers": args.tickers},
            "n_pending": len(pending),
            "rows_written": rows_written,
            "errors": errors,
        },
        outcome="OK" if errors == 0 else "PARTIAL",
    )

    print()
    print(
        f"done. tickers_fetched={len(pending) - errors} rows_written={rows_written} "
        f"errors={errors} elapsed={time.monotonic() - t0:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
