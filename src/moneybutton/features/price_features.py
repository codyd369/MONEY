"""Price/return features (SPEC §8.2)."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

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
    if bid is None or ask is None or (isinstance(bid, float) and math.isnan(bid)) or (isinstance(ask, float) and math.isnan(ask)):
        last = row.get("last_price_close")
        return float(last) / 100.0 if last is not None and not (isinstance(last, float) and math.isnan(last)) else None
    return (float(bid) + float(ask)) / 200.0


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
        if mids.empty:
            max_24h = min_24h = range_24h = realized_vol = None
        else:
            max_24h = float(mids.max())
            min_24h = float(mids.min())
            range_24h = max_24h - min_24h
            if len(mids) >= 3:
                returns = mids.apply(math.log).diff().dropna()
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
