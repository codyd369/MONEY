"""Volume / open-interest / flow features (SPEC §8.2)."""

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
) -> dict[str, Any]:
    """`volume` and `open_interest` are sourced from candles summed over 24h;
    the market-level `volume` snapshot is POST-close and would leak, so we
    never use it at feature time."""
    filtered = filter_before(price_frame, as_of_ts)
    if filtered.empty:
        return {
            "volume_24h": None,
            "log_volume_24h": None,
            "open_interest": None,
            "log_oi": None,
            "vol_oi_ratio": None,
            "num_trades_24h": None,
        }

    last_24h = filtered[filtered["_ts_dt"] >= pd.Timestamp(as_of_ts - dt.timedelta(hours=24))]
    volume_24h = float(last_24h["volume"].fillna(0).sum()) if not last_24h.empty else 0.0
    log_volume_24h = math.log1p(volume_24h)
    last_row = filtered.sort_values("_ts_dt").iloc[-1]
    oi = last_row.get("open_interest")
    oi_f = float(oi) if oi is not None and not (isinstance(oi, float) and math.isnan(oi)) else None
    log_oi = math.log1p(oi_f) if oi_f is not None else None
    vol_oi_ratio = (volume_24h / oi_f) if oi_f and oi_f > 0 else None
    num_trades_24h = float(len(last_24h))

    return {
        "volume_24h": volume_24h,
        "log_volume_24h": log_volume_24h,
        "open_interest": oi_f,
        "log_oi": log_oi,
        "vol_oi_ratio": vol_oi_ratio,
        "num_trades_24h": num_trades_24h,
    }
