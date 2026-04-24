"""Probe one raw trade response to see what fields Kalshi actually sends.

Same diagnostic pattern we used for candles when the projection was
silently dropping the price/volume fields.

Usage: uv run python scripts/inspect_trades.py
"""

from __future__ import annotations

import json
import sys

from moneybutton.data.store import read_dataset
from moneybutton.kalshi.client import KalshiClient


def main() -> int:
    df = read_dataset("trades")
    if df.empty:
        markets = read_dataset("markets")
        if markets.empty:
            print("no markets on disk", file=sys.stderr)
            return 1
        ticker = str(markets.iloc[0]["ticker"])
    else:
        # Pick the ticker with the most trades.
        ticker = str(df.groupby("ticker").size().idxmax())

    print(f"inspecting trades for {ticker}")
    client = KalshiClient()
    try:
        resp = client.list_trades(ticker=ticker, limit=5)
    finally:
        client.close()

    trades = resp.get("trades", []) or []
    print(f"trades returned: {len(trades)}")
    if not trades:
        print("(no trades)")
        return 0

    print()
    print("first 3 raw trades:")
    for t in trades[:3]:
        print(json.dumps(t, indent=2, default=str))

    print()
    print("top-level keys across all trades:")
    keys: set[str] = set()
    for t in trades:
        keys.update(t.keys())
    for k in sorted(keys):
        populated = sum(1 for t in trades if t.get(k) not in (None, "", {}, []))
        sample = next((t.get(k) for t in trades if t.get(k) not in (None, "", {}, [])), None)
        print(f"  {k:<30} populated: {populated}/{len(trades)}  sample: {sample!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
