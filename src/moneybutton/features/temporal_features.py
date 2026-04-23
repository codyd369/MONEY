"""Temporal features (SPEC §8.2)."""

from __future__ import annotations

import datetime as dt
from typing import Any

from moneybutton.features.common import parse_ts

_DTE_BUCKETS = ("0-1d", "1-7d", "7-30d", "30+d")


def _dte_bucket(hours: float) -> str:
    days = hours / 24.0
    if days < 1:
        return "0-1d"
    if days < 7:
        return "1-7d"
    if days < 30:
        return "7-30d"
    return "30+d"


def compute(market: dict, as_of_ts: dt.datetime) -> dict[str, Any]:
    feats: dict[str, Any] = {
        "hour_of_day": as_of_ts.hour,
        "day_of_week": as_of_ts.weekday(),
        "is_weekend": int(as_of_ts.weekday() >= 5),
        "is_market_hours": int(13 <= as_of_ts.hour < 21 and as_of_ts.weekday() < 5),  # 9-5 ET approx in UTC
    }
    close_ts = parse_ts(market.get("close_time") or market.get("expiration_time"))
    if close_ts is not None:
        hours_to_expiry = (close_ts - as_of_ts).total_seconds() / 3600.0
        bucket = _dte_bucket(max(0.0, hours_to_expiry))
    else:
        bucket = "30+d"
    for b in _DTE_BUCKETS:
        feats[f"dte_bucket_{b}"] = int(bucket == b)
    return feats
