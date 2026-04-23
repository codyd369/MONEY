"""Dump the raw candlestick response for one ticker so we can see which
fields Kalshi actually populates (the /candlesticks schema isn't documented
in a way I can rely on; this is the diagnostic).

Usage:
    uv run python scripts/inspect_candles.py                 # auto-picks one
    uv run python scripts/inspect_candles.py KXFED-25MAR-HIKE
"""

from __future__ import annotations

import datetime as dt
import json
import sys

import pandas as pd

from moneybutton.data.store import read_dataset
from moneybutton.kalshi.client import KalshiClient


def _pick_ticker() -> str | None:
    m = read_dataset("markets")
    if m.empty:
        return None
    # Prefer a Politics/Elections market (most likely to have had trades).
    preferred = m[m["category"].isin(["Politics", "Elections", "Economics"])]
    if not preferred.empty:
        return str(preferred.iloc[0]["ticker"])
    return str(m.iloc[0]["ticker"])


def main() -> int:
    ticker = sys.argv[1] if len(sys.argv) > 1 else _pick_ticker()
    if ticker is None:
        print("no markets on disk", file=sys.stderr)
        return 1
    print(f"inspecting {ticker}")

    m = read_dataset("markets")
    row = m[m["ticker"] == ticker]
    if row.empty:
        print(f"ticker {ticker} not in markets dataset", file=sys.stderr)
        return 1
    market = row.iloc[0].to_dict()
    open_ts = market.get("open_time")
    close_ts = market.get("close_time") or market.get("expiration_time")
    if not (open_ts and close_ts):
        print("missing open/close time on this market", file=sys.stderr)
        return 1

    start_epoch = int(dt.datetime.fromisoformat(str(open_ts).replace("Z", "+00:00")).timestamp())
    end_epoch = int(dt.datetime.fromisoformat(str(close_ts).replace("Z", "+00:00")).timestamp())

    # Fetch a short slice (7 days) so we only get a handful of candles to inspect.
    short_end = min(end_epoch, start_epoch + 7 * 24 * 3600)
    client = KalshiClient()
    try:
        resp = client.get_candlesticks(
            ticker,
            start_ts=start_epoch,
            end_ts=short_end,
            period_interval=60,
        )
    finally:
        client.close()

    candles = resp.get("candlesticks", []) or []
    print(f"candle count in first 7 days: {len(candles)}")
    if not candles:
        print("response had zero candles; nothing to inspect")
        print("full resp:", json.dumps(resp, indent=2, default=str))
        return 0

    print()
    print("first 3 raw candles:")
    for c in candles[:3]:
        print(json.dumps(c, indent=2, default=str))

    print()
    print("top-level keys across all candles:")
    keys = set()
    for c in candles:
        keys.update(c.keys())
    for k in sorted(keys):
        # Sample: how often is it populated vs None?
        populated = sum(1 for c in candles if c.get(k) not in (None, {}, []))
        print(f"  {k:<30} populated: {populated}/{len(candles)}")

    # Check whether yes_bid/yes_ask/price sub-dicts have real values.
    for subkey in ("yes_bid", "yes_ask", "price"):
        vals_any = sum(1 for c in candles if isinstance(c.get(subkey), dict) and any(v is not None for v in c[subkey].values()))
        print(f"  sub-dict {subkey!r:<12} has any non-null inner field: {vals_any}/{len(candles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
