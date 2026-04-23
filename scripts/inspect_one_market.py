"""Print one raw market from whatever is on disk. Sanity check for
scraper output — lets you see which fields Kalshi actually populates.

Usage: uv run python scripts/inspect_one_market.py
"""

from __future__ import annotations

import json

from moneybutton.data.store import read_dataset


def main() -> int:
    df = read_dataset("markets")
    if df.empty:
        print("No markets on disk. Run scripts/backfill_markets.py first.")
        return 1
    print(f"markets on disk: {len(df)}")
    print(f"unique categories: {sorted(df['category'].fillna('(null)').unique())}")
    print(f"unique statuses:   {sorted(df['status'].fillna('(null)').unique())}")
    print(f"close_time range:  {df['close_time'].min()}  ..  {df['close_time'].max()}")
    print()
    print("first market (raw):")
    print(json.dumps(df.iloc[0].to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
