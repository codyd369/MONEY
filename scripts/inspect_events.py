"""Probe the /events endpoint to see what categories Kalshi actually ships.

No writes, no Parquet — just a diagnostic. Tells us whether events carry
categories and what the category vocabulary actually looks like.

Usage: uv run python scripts/inspect_events.py
"""

from __future__ import annotations

import collections
import json
import sys

from moneybutton.kalshi.client import KalshiClient


def main() -> int:
    # Pull 5 pages of settled events (1000 events).
    client = KalshiClient()
    categories: collections.Counter = collections.Counter()
    total = 0
    sample: dict | None = None
    try:
        cursor: str | None = None
        for page_i in range(5):
            page = client.list_events(status="settled", limit=200, cursor=cursor)
            events = page.get("events", []) or []
            for ev in events:
                total += 1
                cat = ev.get("category") or "(null)"
                categories[cat] += 1
                if sample is None:
                    sample = ev
            cursor = page.get("cursor")
            if not cursor:
                break
    finally:
        client.close()

    print(f"events scanned: {total}")
    print()
    print("category distribution (top 30):")
    for cat, n in categories.most_common(30):
        print(f"  {cat:<40} {n:>6}")

    if sample is not None:
        print()
        print("first event (raw):")
        print(json.dumps(sample, indent=2, default=str)[:2400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
