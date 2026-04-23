"""Market-level features (SPEC §8.2).

These are time-invariant or slowly-changing properties of the market itself
— category, expiry proximity, days-since-listing, market creator. They are
the same whether you compute at t or t+1h as long as t is before close_time.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

from moneybutton.features.common import parse_ts

_CATEGORIES = ("FINANCIALS", "POLITICS", "SPORTS", "TECH", "WEATHER", "UNCATEGORIZED")


def compute(market: dict, as_of_ts: dt.datetime) -> dict[str, Any]:
    category = (market.get("category") or "UNCATEGORIZED").upper()
    close_ts = parse_ts(market.get("close_time") or market.get("expiration_time"))
    open_ts = parse_ts(market.get("open_time"))

    days_to_expiry = (close_ts - as_of_ts).total_seconds() / 86400.0 if close_ts else None
    log_days_to_expiry = math.log1p(max(0.0, days_to_expiry)) if days_to_expiry is not None else None
    days_since_listing = (as_of_ts - open_ts).total_seconds() / 86400.0 if open_ts else None

    feats: dict[str, Any] = {
        "days_to_expiry": days_to_expiry,
        "log_days_to_expiry": log_days_to_expiry,
        "days_since_listing": days_since_listing,
    }
    # One-hot for known categories; unknown/new categories contribute zero.
    for c in _CATEGORIES:
        feats[f"cat_is_{c}"] = int(category == c)
    return feats
