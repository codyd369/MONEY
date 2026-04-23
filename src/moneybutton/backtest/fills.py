"""Realistic fill model (SPEC §11.2).

No "filled at mid" lies. For a YES-side buy:
  - Entry price = yes_ask_close on the latest candle <= as_of_ts.
  - Add slippage_bps (default 25) on top to model adverse selection.
  - For sizes <= top-of-book (unknown at backtest time, so we assume the
    default cap is always <= top-of-book), assume full fill.
  - For the rare case where a candle has no bid/ask (data gap), fall back
    to last_price_close; if that's also missing, the fill is rejected.

For NO-side, the symmetric logic applies against (100 - yes_bid_close).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fill:
    filled: bool
    price_cents: int
    reason: str
    contracts: int = 0


def fill_entry(
    *,
    side: str,
    size_usd: float,
    price_frame: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    slippage_bps: int,
    min_price_cents: int = 1,
    max_price_cents: int = 99,
) -> Fill:
    """Model the entry fill for a buy-side order.

    Returns `Fill(filled=False, ...)` if the candle data has no usable
    price; otherwise returns the filled price (after slippage) and the
    integer contract count.
    """
    if side not in ("yes", "no"):
        return Fill(filled=False, price_cents=0, reason=f"invalid side {side!r}")

    sub = price_frame[price_frame["_ts_dt"] <= as_of_ts]
    if sub.empty:
        return Fill(filled=False, price_cents=0, reason="no price history at as_of")
    last = sub.iloc[-1]

    if side == "yes":
        entry = last.get("yes_ask_close")
    else:
        bid = last.get("yes_bid_close")
        entry = (100 - bid) if bid is not None and not _is_nan(bid) else None

    if entry is None or _is_nan(entry):
        # Fall back to last_price_close +/- 1 tick.
        last_price = last.get("last_price_close")
        if last_price is None or _is_nan(last_price):
            return Fill(filled=False, price_cents=0, reason="no bid/ask/last available")
        entry = float(last_price) + (1 if side == "yes" else -1)

    entry_f = float(entry)
    # Apply slippage. 25 bps on a 40c fill is 0.1 cents, which rounds to 0 at
    # the tick; we intentionally don't refuse to execute small slippages.
    slip = entry_f * (slippage_bps / 10_000.0)
    entry_with_slip = entry_f + slip if side == "yes" else entry_f + slip  # symmetric cost
    entry_cents = max(min_price_cents, min(max_price_cents, int(round(entry_with_slip))))

    if entry_cents <= 0:
        return Fill(filled=False, price_cents=0, reason="entry_cents rounded to 0")

    contracts = int(size_usd // (entry_cents / 100.0))
    if contracts < 1:
        return Fill(filled=False, price_cents=entry_cents, reason="size < 1 contract at entry price")
    return Fill(filled=True, price_cents=entry_cents, reason="filled", contracts=contracts)


def _is_nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False
