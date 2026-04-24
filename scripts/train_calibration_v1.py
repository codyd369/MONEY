"""One-shot: train calibration v1 on whatever Parquet data is on disk.

Used at build-order step 12. The CLI (step 26) will replace this with a
proper `python -m moneybutton models train calibration` command.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from moneybutton.data.store import read_dataset
from moneybutton.features.pipeline import (
    build_training_frame,
    default_as_of_before_close,
    feature_schema,
    feature_schema_fingerprint,
)
from moneybutton.models.calibration import build_train_report, time_split, train
from moneybutton.models.registry import register


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train calibration v1 on on-disk data.")
    p.add_argument(
        "--max-per-series",
        type=int,
        default=100,
        help="Cap any one ticker-series to N markets before training. Prevents "
        "auto-gen series (hourly weather, parlay) from dominating the loss. "
        "Set to 0 to disable.",
    )
    p.add_argument(
        "--min-trade-rows",
        type=int,
        default=5,
        help="Drop markets that had fewer than N candle rows with a real trade. "
        "Markets with no trades tell the model nothing about price dynamics.",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Restrict training set to these event-categories (e.g., Politics Economics Sports).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    markets = read_dataset("markets")
    prices = read_dataset("prices")
    if markets.empty or prices.empty:
        print("No data; run the synthetic generator or data backfill first.")
        return 1
    # Filter to settled markets with a result.
    markets = markets[markets["result"].isin(["yes", "no"])].copy()

    if args.categories:
        markets = markets[markets["category"].isin(args.categories)]
        print(f"filtered to categories {args.categories}: {len(markets)} markets")

    if args.max_per_series and args.max_per_series > 0:
        markets["_series"] = markets["ticker"].str.split("-", n=1).str[0]
        before = len(markets)
        markets = markets.groupby("_series", group_keys=False).head(args.max_per_series)
        after = len(markets)
        markets = markets.drop(columns=["_series"])
        if before != after:
            print(f"max_per_series={args.max_per_series}: {before} -> {after} markets")

    if args.min_trade_rows > 0 and not prices.empty:
        traded = (
            prices[prices["last_price_close"].notna()]
            .groupby("ticker")
            .size()
            .loc[lambda s: s >= args.min_trade_rows]
            .index
        )
        before = len(markets)
        markets = markets[markets["ticker"].isin(traded)]
        after = len(markets)
        print(
            f"min_trade_rows={args.min_trade_rows}: dropped {before - after} markets "
            f"with <{args.min_trade_rows} trade rows; kept {after}"
        )

    if markets.empty:
        print("no markets survived filtering")
        return 1

    print(f"training frame: {len(markets)} markets, {len(prices)} price rows")

    frame = build_training_frame(markets, prices, default_as_of_before_close())
    # Keep only numeric feature columns for training (XGBoost handles NaN but
    # not object dtypes). Coerce, then drop all-null and non-numeric columns.
    id_cols = {"ticker", "as_of_ts", "label_resolved"}
    feature_cols: list[str] = []
    for c in frame.columns:
        if c in id_cols:
            continue
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
        if frame[c].notna().any():
            feature_cols.append(c)
    dropped = [c for c in frame.columns if c not in id_cols and c not in feature_cols]
    if dropped:
        print(f"dropped {len(dropped)} all-null / non-numeric features: {dropped}")

    # Time-based split over the as_of_ts distribution.
    ts = pd.to_datetime(frame["as_of_ts"], utc=True)
    span = ts.max() - ts.min()
    train_end = ts.quantile(0.6).to_pydatetime()
    val_end = ts.quantile(0.8).to_pydatetime()
    test_end = (ts.max() + pd.Timedelta(days=1)).to_pydatetime()
    # Scale embargo to the dataset span so short backtests don't lose the
    # entire val window to a fixed 7-day gap. 5% of span, capped at 7 days,
    # floored at 6 hours (enough to avoid candle-overlap leakage).
    embargo = max(
        pd.Timedelta(hours=6),
        min(pd.Timedelta(days=7), span * 0.05),
    )
    print(f"data span: {span}  embargo: {embargo}")

    split = time_split(
        frame,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        embargo=embargo,
        feature_columns=feature_cols,
    )
    # Guardrail: if val is empty after the split, fall back to no embargo.
    if len(split.X_val) == 0:
        print("val window is empty after embargo; falling back to embargo=0")
        split = time_split(
            frame,
            train_end=train_end,
            val_end=val_end,
            test_end=test_end,
            embargo=pd.Timedelta(0),
            feature_columns=feature_cols,
        )
    print(
        f"split: train {len(split.X_train)} / val {len(split.X_val)} / test {len(split.X_test)}"
    )
    print(f"train window: {split.train_window[0]} -> {split.train_window[1]}")
    print(f"val window:   {split.val_window[0]} -> {split.val_window[1]}")
    print(f"test window:  {split.test_window[0]} -> {split.test_window[1]}")

    if len(split.X_val) == 0 or len(split.X_train) == 0:
        print("ERROR: split produced an empty train or val set; aborting.")
        return 1

    model, metrics = train(split)
    print("metrics:", json.dumps(metrics, indent=2, default=str))

    html = build_train_report(model, split, metrics)

    sample_feats_fp = feature_schema_fingerprint({c: 0 for c in feature_cols})
    entry = register(
        "calibration",
        model,
        feature_columns=feature_cols,
        metadata={
            "metrics": metrics,
            "train_window": [str(x) for x in split.train_window],
            "val_window": [str(x) for x in split.val_window],
            "test_window": [str(x) for x in split.test_window],
            "source": "synthetic_v1",
            "feature_schema_fingerprint": sample_feats_fp,
            "trained_ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        train_report_html=html,
    )
    print(f"registered: {entry.path}")
    print(f"report: {entry.path / 'train_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
