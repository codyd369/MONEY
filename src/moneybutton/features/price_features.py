"""Price/return features (SPEC §8.2)."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import numpy as np
import pandas as pd

from moneybutton.features.common import filter_before


def _nearest_before(df: pd.DataFrame, ts_cutoff: dt.datetime) -> pd.Series | None:
    if df.empty:
        return None
    sub = df[df["_ts_dt"] <= pd.Timestamp(ts_cutoff)]
    if sub.empty:
        return None
    return sub.iloc[-1]


def _mid_from_row(row: pd.Series) -> float | None:
    bid = row.get("yes_bid_close")
    ask = row.get("yes_ask_close")

    def _last_price() -> float | None:
        last = row.get("last_price_close")
        if last is None or (isinstance(last, float) and math.isnan(last)):
            return None
        v = float(last) / 100.0
        return v if 0.0 < v < 1.0 else None

    if bid is None or ask is None or (isinstance(bid, float) and math.isnan(bid)) or (isinstance(ask, float) and math.isnan(ask)):
        return _last_price()

    bid_f = float(bid)
    ask_f = float(ask)
    # Kalshi's "no quote" sentinels: bid=0 (no buyer) or ask=100 (no seller)
    # produce a meaningless 0.5 mid. Fall back to last_price if there's
    # actually been a trade, else None — never a synthetic 0.5.
    if bid_f <= 0 or ask_f >= 100:
        return _last_price()
    mid = (bid_f + ask_f) / 200.0
    if mid <= 0.0 or mid >= 1.0:
        return None
    return mid


def compute(
    market: dict,
    as_of_ts: dt.datetime,
    price_frame: pd.DataFrame,
) -> dict[str, Any]:
    """All price/return features for `market` as of `as_of_ts`.

    `price_frame` contains all rows for this ticker's prices partition; the
    filter_before helper strips out post-`as_of_ts` rows. The feature pipeline
    guarantees this contract.
    """
    filtered = filter_before(price_frame, as_of_ts).sort_values("_ts_dt")
    if filtered.empty:
        return {
            "yes_price_now": None,
            "yes_price_1h_ago": None,
            "yes_price_24h_ago": None,
            "log_returns_1h": None,
            "log_returns_24h": None,
            "realized_vol_24h": None,
            "max_24h": None,
            "min_24h": None,
            "range_24h": None,
        }

    now_row = filtered.iloc[-1]
    now_price = _mid_from_row(now_row)

    def price_at_offset(hours: int) -> float | None:
        cutoff = as_of_ts - dt.timedelta(hours=hours)
        row = _nearest_before(filtered, cutoff)
        return _mid_from_row(row) if row is not None else None

    price_1h = price_at_offset(1)
    price_24h = price_at_offset(24)

    def log_return(a: float | None, b: float | None) -> float | None:
        if a is None or b is None or a <= 0 or b <= 0:
            return None
        return math.log(a / b)

    last_24h = filtered[filtered["_ts_dt"] >= pd.Timestamp(as_of_ts - dt.timedelta(hours=24))]
    if last_24h.empty:
        max_24h = min_24h = range_24h = realized_vol = None
    else:
        mids = last_24h.apply(_mid_from_row, axis=1).dropna().astype(float)
        # Belt-and-braces: a row with bid=0/ask=0 (no quote at all) would still
        # produce mid=0 in some defensive paths. Drop non-positive mids before
        # any log/return arithmetic.
        mids = mids[(mids > 0) & (mids < 1)]
        if mids.empty:
            max_24h = min_24h = range_24h = realized_vol = None
        else:
            max_24h = float(mids.max())
            min_24h = float(mids.min())
            range_24h = max_24h - min_24h
            if len(mids) >= 3:
                returns = np.log(mids).diff().dropna()
                realized_vol = float(returns.std()) if len(returns) > 1 else None
            else:
                realized_vol = None

    return {
        "yes_price_now": now_price,
        "yes_price_1h_ago": price_1h,
        "yes_price_24h_ago": price_24h,
        "log_returns_1h": log_return(now_price, price_1h),
        "log_returns_24h": log_return(now_price, price_24h),
        "realized_vol_24h": realized_vol,
        "max_24h": max_24h,
        "min_24h": min_24h,
        "range_24h": range_24h,
    }
