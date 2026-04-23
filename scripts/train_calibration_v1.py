"""One-shot: train calibration v1 on whatever Parquet data is on disk.

Used at build-order step 12. The CLI (step 26) will replace this with a
proper `python -m moneybutton models train calibration` command.
"""

from __future__ import annotations

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


def main() -> int:
    markets = read_dataset("markets")
    prices = read_dataset("prices")
    if markets.empty or prices.empty:
        print("No data; run the synthetic generator or data backfill first.")
        return 1
    # Filter to settled markets with a result.
    markets = markets[markets["result"].isin(["yes", "no"])].copy()

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
    train_end = ts.quantile(0.6).to_pydatetime()
    val_end = ts.quantile(0.8).to_pydatetime()
    test_end = (ts.max() + pd.Timedelta(days=1)).to_pydatetime()

    split = time_split(
        frame,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        feature_columns=feature_cols,
    )
    print(
        f"split: train {len(split.X_train)} / val {len(split.X_val)} / test {len(split.X_test)}"
    )
    print(f"train window: {split.train_window[0]} -> {split.train_window[1]}")
    print(f"val window:   {split.val_window[0]} -> {split.val_window[1]}")
    print(f"test window:  {split.test_window[0]} -> {split.test_window[1]}")

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
