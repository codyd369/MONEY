"""Orderbook features (SPEC §8.2).

Orderbook snapshots are scarce in historical Kalshi data. When missing, we
return NaN; XGBoost handles NaN natively so the calibration model still
trains on markets without this information.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import pandas as pd

from moneybutton.features.common import filter_before


def compute(
    market: dict,
    as_of_ts: dt.datetime,
    price_frame: pd.DataFrame,
    orderbook_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    # If a proper orderbook frame is supplied, use it. Otherwise fall back
    # to bid/ask from the latest pre-as_of candle.
    if orderbook_frame is not None and not orderbook_frame.empty:
        ob = filter_before(orderbook_frame, as_of_ts).sort_values("_ts_dt")
        if not ob.empty:
            # Placeholder: real implementation would group by (ts, side)
            # and pick top levels. Orderbook schema locked in step 10 of a
            # future build; for now we only emit a stub.
            return {
                "bid_ask_spread_cents": None,
                "mid_price": None,
                "depth_top_5_levels": None,
            }
    filtered = filter_before(price_frame, as_of_ts).sort_values("_ts_dt")
    if filtered.empty:
        return {
            "bid_ask_spread_cents": None,
            "mid_price": None,
            "depth_top_5_levels": None,
        }
    last = filtered.iloc[-1]
    bid = last.get("yes_bid_close")
    ask = last.get("yes_ask_close")
    bid_f = float(bid) if bid is not None and not (isinstance(bid, float) and math.isnan(bid)) else None
    ask_f = float(ask) if ask is not None and not (isinstance(ask, float) and math.isnan(ask)) else None
    if bid_f is None or ask_f is None:
        return {
            "bid_ask_spread_cents": None,
            "mid_price": None,
            "depth_top_5_levels": None,
        }
    return {
        "bid_ask_spread_cents": ask_f - bid_f,
        "mid_price": (bid_f + ask_f) / 2.0 / 100.0,
        "depth_top_5_levels": None,  # not in candles
    }
