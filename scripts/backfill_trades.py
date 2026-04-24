"""Backfill Kalshi trade tape.

Candlesticks bucket trades into OHLCV buckets and throw away the
directional/size information that matters for microstructure features.
This script pulls individual trades for each market.

Per-trade schema (from Kalshi /markets/trades):
  trade_id        str
  ticker          str
  taker_side      'yes' | 'no'     (which side was the aggressor)
  yes_price_cents int 1-99         (trade price, integer cents)
  count           int              (contracts traded)
  ts              str (ISO-8601 UTC)

Written to parquet/trades/category={cat}/year_month={ym}/trades.parquet.
Dedupe key is trade_id.

Usage:
    # Only markets with real trading activity — read from prices dataset.
    uv run python scripts/backfill_trades.py --only-traded --max-per-series 20 --rate-limit-sleep-s 0.3

    # Explicit ticker list:
    uv run python scripts/backfill_trades.py --tickers KXBTC-26APR24-T90000 KXETH-...

Reality check: Kalshi returns trades reverse-chronologically, paginated.
A liquid market can have thousands of trades per day. At 1000/page and
0.3s rate limit, a single heavy market takes ~3-10 sec. For 1000 markets
that's ~1-3 hours. Start with a sample (--sample-n 500) to prove the
feature pipeline before committing to the full pull.
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
from moneybutton.data.scraper_news import _normalize_ts
from moneybutton.data.store import PartitionKey, read_dataset, write_partition, year_month
from moneybutton.kalshi.client import KalshiClient, KalshiHTTPError


def _project_trade(t: dict) -> dict:
    """Stable columnar schema for one trade."""
    price = t.get("yes_price")
    count = t.get("count")
    return {
        "trade_id": str(t.get("trade_id") or ""),
        "ticker": str(t.get("ticker") or ""),
        "ts": _normalize_ts(t.get("created_time")),
        "taker_side": str(t.get("taker_side") or ""),
        "yes_price_cents": int(price) if price is not None else None,
        "count": int(count) if count is not None else 0,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill Kalshi trade tape.")
    p.add_argument("--tickers", nargs="+", default=None, help="Explicit ticker list")
    p.add_argument("--category", default=None, help="Restrict to one category")
    p.add_argument(
        "--only-traded",
        action="store_true",
        help="Select only tickers that have trade rows in the prices dataset (default behavior).",
    )
    p.add_argument("--sample-n", type=int, default=None, help="Random stratified sample size")
    p.add_argument("--max-per-series", type=int, default=None, help="Cap per ticker-prefix")
    p.add_argument("--top-n", type=int, default=None, help="Pick top N by last known volume")
    p.add_argument("--max-trades-per-ticker", type=int, default=5000, help="Pagination cap per market")
    p.add_argument("--since", default=None, help="ISO date; drop trades older than this (local filter)")
    p.add_argument("--rate-limit-sleep-s", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-skip-existing", action="store_true")
    return p.parse_args()


def _existing_trade_tickers() -> set[str]:
    df = read_dataset("trades")
    if df.empty:
        return set()
    return set(df["ticker"].unique())


def _select_tickers(args, markets: pd.DataFrame, prices: pd.DataFrame | None) -> list[str]:
    if args.tickers:
        return list(args.tickers)

    df = markets.copy()
    if args.category:
        df = df[df["category"].str.upper() == args.category.upper()]

    # Default to --only-traded if nothing else narrows the selection. We only
    # want tickers that actually had trades in the candle data (last_price
    # populated), otherwise the /trades endpoint will return empty pages.
    should_filter_traded = args.only_traded or (not args.tickers and not args.top_n)
    if should_filter_traded and prices is not None and not prices.empty:
        traded = set(
            prices[prices["last_price_close"].notna()]["ticker"].unique()
        )
        before = len(df)
        df = df[df["ticker"].isin(traded)]
        print(f"only-traded filter: {before} -> {len(df)} markets", flush=True)

    df["_series"] = df["ticker"].str.split("-", n=1).str[0]
    if args.max_per_series:
        df = df.groupby("_series", group_keys=False).head(args.max_per_series)
        print(f"max-per-series={args.max_per_series}: {len(df)} markets", flush=True)

    if args.sample_n and len(df) > args.sample_n:
        if "category" in df.columns:
            per_cat = df.groupby("category", group_keys=False).apply(
                lambda g: g.sample(
                    n=max(1, int(round(len(g) / len(df) * args.sample_n))),
                    random_state=args.seed,
                )
            )
            df = per_cat.head(args.sample_n).reset_index(drop=True)
        else:
            df = df.sample(n=args.sample_n, random_state=args.seed).reset_index(drop=True)
        print(f"sample-n={args.sample_n}: {len(df)} markets", flush=True)

    if args.top_n:
        df = df.head(args.top_n)

    return df["ticker"].tolist()


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    print(f"backfill_trades: start  env={settings.kalshi_env}", flush=True)

    markets = read_dataset("markets")
    if markets.empty:
        print("no markets on disk. Run backfill_via_events first.", flush=True)
        return 1
    prices = read_dataset("prices")

    tickers = _select_tickers(args, markets, prices)
    if not tickers:
        print("no tickers selected.", flush=True)
        return 1

    skip_set: set[str] = set() if args.no_skip_existing else _existing_trade_tickers()
    pending = [t for t in tickers if t not in skip_set]
    print(
        f"backfill_trades: {len(tickers)} selected, "
        f"{len(tickers) - len(pending)} already have trades, "
        f"{len(pending)} to fetch. sleep={args.rate_limit_sleep_s}s, "
        f"cap={args.max_trades_per_ticker} trades/market",
        flush=True,
    )
    if not pending:
        print("nothing to do. Pass --no-skip-existing to refetch.", flush=True)
        return 0

    since_dt = dt.datetime.fromisoformat(args.since).replace(tzinfo=dt.timezone.utc) if args.since else None
    since_iso = since_dt.isoformat() if since_dt else None

    client = KalshiClient(settings=settings)
    t0 = time.monotonic()
    rows_written = 0
    errors = 0
    try:
        for i, ticker in enumerate(pending, 1):
            market_row = markets[markets["ticker"] == ticker]
            category = (
                (market_row.iloc[0].get("category") or "UNCATEGORIZED").upper()
                if not market_row.empty
                else "UNCATEGORIZED"
            )
            close_ts = market_row.iloc[0].get("close_time") if not market_row.empty else None
            ym = year_month(close_ts) if close_ts else "unknown"
            partition = PartitionKey(dataset="trades", category_or_source=category, year_month=ym)

            # Paginate this ticker's trades.
            cursor: str | None = None
            ticker_rows: list[dict] = []
            ticker_trades_pulled = 0
            while True:
                try:
                    resp = client.list_trades(ticker=ticker, cursor=cursor, limit=1000)
                except KalshiHTTPError as e:
                    errors += 1
                    print(
                        f"  [{i:>5d}/{len(pending)}] {ticker} ERROR Kalshi {e.status}: {e.body[:120]}",
                        flush=True,
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    errors += 1
                    print(
                        f"  [{i:>5d}/{len(pending)}] {ticker} ERROR {type(e).__name__}: {e}",
                        flush=True,
                    )
                    break

                raw = resp.get("trades", []) or []
                if not raw:
                    break

                for t in raw:
                    row = _project_trade(t)
                    if since_iso and row["ts"] < since_iso:
                        # Reverse-chronological; once we pass `since`, stop paginating.
                        cursor = None
                        break
                    ticker_rows.append(row)

                ticker_trades_pulled += len(raw)
                cursor = resp.get("cursor")
                if not cursor or ticker_trades_pulled >= args.max_trades_per_ticker:
                    break
                time.sleep(args.rate_limit_sleep_s)

            if ticker_rows:
                write_partition(partition, pd.DataFrame(ticker_rows))
                rows_written += len(ticker_rows)

            if i % 25 == 0 or i == len(pending):
                elapsed = time.monotonic() - t0
                eta_s = (elapsed / i) * (len(pending) - i) if i > 0 else 0
                eta_str = f"{eta_s / 60:.1f}m" if eta_s < 3600 else f"{eta_s / 3600:.1f}h"
                print(
                    f"  [{i:>5d}/{len(pending)}] rows_written={rows_written:>8d} "
                    f"errors={errors:>3d} elapsed={elapsed:>6.0f}s ETA={eta_str}",
                    flush=True,
                )
            time.sleep(args.rate_limit_sleep_s)
    except KeyboardInterrupt:
        print("backfill_trades: interrupted; rerun to resume.", flush=True)
        return 130
    finally:
        client.close()

    audit_record(
        actor="scripts.backfill_trades",
        action="backfill_trades",
        payload={
            "selection": {
                "category": args.category,
                "sample_n": args.sample_n,
                "max_per_series": args.max_per_series,
                "top_n": args.top_n,
            },
            "n_pending": len(pending),
            "rows_written": rows_written,
            "errors": errors,
        },
        outcome="OK" if errors == 0 else "PARTIAL",
    )

    print()
    print(
        f"done. tickers_fetched={len(pending) - errors} "
        f"rows_written={rows_written} errors={errors} "
        f"elapsed={time.monotonic() - t0:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
