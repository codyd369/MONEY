"""Single source of truth for feature engineering (SPEC §8).

Both training and inference call compute_features(market, as_of_ts, store).
Any drift between train-time and serve-time features is a bug. The module
is intentionally narrow: it composes the per-group modules and enforces
the no-leakage contract via filter_before in each module.

Public API:
    compute_features(market, as_of_ts, prices_df) -> dict
    build_training_frame(markets_df, prices_df, as_of_fn) -> DataFrame

The feature-schema fingerprint is the sorted tuple of output keys; the
model registry uses this to refuse loading a model whose schema doesn't
match the current pipeline.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Callable, Iterable

import pandas as pd

from moneybutton.features import (
    market_features,
    orderbook_features,
    price_features,
    temporal_features,
    volume_features,
)
from moneybutton.features.common import parse_ts


FEATURE_SCHEMA_VERSION = 1


def compute_features(
    market: dict,
    as_of_ts: dt.datetime,
    prices_df: pd.DataFrame,
    orderbook_df: pd.DataFrame | None = None,
) -> dict:
    """One row of features for `market` as of `as_of_ts`.

    `prices_df` must contain rows for this market's ticker; callers usually
    pre-filter by ticker before calling. Post-as_of rows are ignored by
    each module via filter_before, but filtering upstream is a perf win.
    """
    feats: dict = {
        "ticker": market["ticker"],
        "as_of_ts": as_of_ts.isoformat(),
    }
    feats.update(market_features.compute(market, as_of_ts))
    feats.update(price_features.compute(market, as_of_ts, prices_df))
    feats.update(volume_features.compute(market, as_of_ts, prices_df))
    feats.update(orderbook_features.compute(market, as_of_ts, prices_df, orderbook_df))
    feats.update(temporal_features.compute(market, as_of_ts))
    return feats


def feature_schema(sample: dict) -> list[str]:
    """Return the list of feature column names from a sample feature dict.
    Excludes the two identity columns (ticker, as_of_ts)."""
    return sorted(k for k in sample.keys() if k not in ("ticker", "as_of_ts"))


def feature_schema_fingerprint(sample: dict) -> str:
    """sha256 of the sorted feature names — used by the model registry."""
    cols = feature_schema(sample)
    return hashlib.sha256(json.dumps(cols, sort_keys=True).encode()).hexdigest()


def build_training_frame(
    markets_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_fn: Callable[[dict], dt.datetime],
    *,
    label_column: str = "label_resolved",
) -> pd.DataFrame:
    """Materialize a feature frame for every settled market.

    `as_of_fn(market_row_dict) -> datetime` picks the inference-time
    timestamp per market. Typical choices:
      - close_time - 1h  (train on end-of-life snapshots)
      - open_time + 4h   (train on early-snapshot edge)
      - a random point in [open+1h, close-1h)
    """
    if prices_df.empty:
        return pd.DataFrame()
    prices_df = prices_df.copy()
    prices_df["_ts_dt"] = pd.to_datetime(prices_df["ts"], utc=True, errors="coerce")

    rows: list[dict] = []
    grouped = {k: v for k, v in prices_df.groupby("ticker")}
    for market in markets_df.to_dict(orient="records"):
        ticker = market["ticker"]
        pdf = grouped.get(ticker)
        if pdf is None or pdf.empty:
            continue
        as_of = as_of_fn(market)
        feats = compute_features(market, as_of, pdf)
        if market.get("result") is not None:
            feats[label_column] = 1 if market["result"] == "yes" else 0
        rows.append(feats)
    out = pd.DataFrame(rows)
    return out


def default_as_of_before_close(margin: dt.timedelta = dt.timedelta(hours=1)) -> Callable[[dict], dt.datetime]:
    """Return an as_of_fn that picks close_time - margin for each market."""

    def _fn(market: dict) -> dt.datetime:
        close = parse_ts(market.get("close_time") or market.get("expiration_time"))
        if close is None:
            return dt.datetime.now(dt.timezone.utc)
        return close - margin

    return _fn
