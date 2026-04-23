"""Summarize what's on disk: per-category, per-year_month, volume health.

Usage: uv run python scripts/data_summary.py
"""

from __future__ import annotations

import pandas as pd

from moneybutton.data.store import read_dataset


def main() -> int:
    m = read_dataset("markets")
    if m.empty:
        print("No markets on disk. Run a backfill first.")
        return 1

    m["has_volume"] = pd.to_numeric(m["volume"], errors="coerce").fillna(0) > 0
    m["has_last_price"] = pd.to_numeric(m["last_price"], errors="coerce").notna()
    m["close_ts"] = pd.to_datetime(m["close_time"], utc=True, errors="coerce")
    m["year_month"] = m["close_ts"].dt.strftime("%Y-%m")

    print(f"total markets on disk: {len(m)}")
    print(f"close_time range: {m['close_ts'].min()} .. {m['close_ts'].max()}")
    print(f"result distribution: {dict(m['result'].value_counts())}")
    print()

    print("=" * 70)
    print("by category (derived from event_ticker prefix):")
    print("=" * 70)
    by_cat = m.groupby("category").agg(
        total=("ticker", "count"),
        with_volume=("has_volume", "sum"),
        with_last_price=("has_last_price", "sum"),
        first_close=("close_ts", "min"),
        last_close=("close_ts", "max"),
    ).sort_values("total", ascending=False)
    # Format nicely.
    by_cat["with_volume"] = by_cat["with_volume"].astype(int)
    by_cat["with_last_price"] = by_cat["with_last_price"].astype(int)
    print(by_cat.head(30).to_string())

    print()
    print("=" * 70)
    print("by close-month (top 24):")
    print("=" * 70)
    by_ym = m.groupby("year_month").agg(
        total=("ticker", "count"),
        with_volume=("has_volume", "sum"),
    ).sort_values("year_month")
    by_ym["with_volume"] = by_ym["with_volume"].astype(int)
    print(by_ym.tail(24).to_string())

    print()
    total_volume = int(m["has_volume"].sum())
    total_last_price = int(m["has_last_price"].sum())
    print(
        f"training-quality signal: "
        f"{total_volume} / {len(m)} ({total_volume / len(m):.1%}) had volume>0; "
        f"{total_last_price} / {len(m)} had a non-null last_price"
    )
    if total_volume == 0:
        print()
        print("NOTE: zero markets have volume>0. Either (a) the listing endpoint")
        print("doesn't populate volume (per-market endpoint does), or (b) you")
        print("genuinely pulled the boring auto-gen slice. Either way, price")
        print("candlesticks will tell the real story — run:")
        print("  uv run python scripts/backfill_prices.py --top-n 200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
