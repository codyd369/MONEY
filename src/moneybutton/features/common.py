"""Shared helpers for feature modules."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd


def parse_ts(v: Any) -> dt.datetime | None:
    if v is None:
        return None
    if isinstance(v, dt.datetime):
        return v if v.tzinfo is not None else v.replace(tzinfo=dt.timezone.utc)
    if isinstance(v, (int, float)):
        return dt.datetime.fromtimestamp(float(v), tz=dt.timezone.utc)
    if isinstance(v, str):
        s = v.replace("Z", "+00:00")
        try:
            return dt.datetime.fromisoformat(s)
        except ValueError:
            pass
        try:
            return pd.to_datetime(v, utc=True).to_pydatetime()
        except Exception:  # noqa: BLE001
            return None
    return None


def filter_before(price_frame: pd.DataFrame, as_of_ts: dt.datetime) -> pd.DataFrame:
    """Keep only rows whose `ts` is strictly before `as_of_ts`.

    The feature pipeline's no-leakage invariant depends on every feature
    module calling this before reading the price frame.
    """
    if price_frame.empty:
        return price_frame
    if "_ts_dt" not in price_frame.columns:
        price_frame = price_frame.copy()
        price_frame["_ts_dt"] = pd.to_datetime(price_frame["ts"], utc=True, errors="coerce")
    return price_frame[price_frame["_ts_dt"] < pd.Timestamp(as_of_ts)]
