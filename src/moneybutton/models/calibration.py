"""Calibration model: XGBoost + post-hoc isotonic (SPEC §9.1).

For any live market, model.predict_proba returns P(YES | features). The
calibration layer matters: raw XGBoost probabilities are not well calibrated
and our edge calculation (model_prob - market_prob) is only meaningful if
model_prob is an honest probability.

Training protocol:
  - Time-based split with an embargo (default 1 week) to avoid look-ahead
    across the train/val boundary.
  - Fit XGBClassifier on train; fit isotonic regression on val residuals.
  - Report: calibration plot (predicted vs observed in deciles), reliability
    metrics (Brier, log loss), feature importances, per-fold metrics.

Hyperparameters start deliberately conservative (small trees, high
min_child_weight, subsample < 1). Overfit is the enemy; we'd rather leave
edge on the table than promote a fragile model.

CLI (step 26 wires this up):
    python -m moneybutton models train calibration --since=2024-01-01
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier


DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss",
}


@dataclass
class CalibratedClassifier:
    """XGBClassifier with a post-hoc isotonic calibrator."""

    base: XGBClassifier
    calibrator: IsotonicRegression
    feature_columns: list[str] = field(default_factory=list)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].to_numpy()
        raw = self.base.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict(raw)
        calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - calibrated, calibrated])

    def predict_yes_prob(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.predict_proba(X)[:, 1]


@dataclass
class TrainSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_window: tuple[dt.datetime, dt.datetime]
    val_window: tuple[dt.datetime, dt.datetime]
    test_window: tuple[dt.datetime, dt.datetime]


def time_split(
    df: pd.DataFrame,
    *,
    label_col: str = "label_resolved",
    ts_col: str = "as_of_ts",
    train_end: dt.datetime,
    val_end: dt.datetime,
    test_end: dt.datetime,
    embargo: dt.timedelta = dt.timedelta(days=7),
    feature_columns: list[str] | None = None,
) -> TrainSplit:
    """Time-based train/val/test split with an embargo around each boundary.

    Rows whose `ts_col` falls in the embargo windows are DROPPED — they are
    close enough to the boundary that label leakage via overlapping candles
    is possible.
    """
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in {"ticker", ts_col, label_col}]

    ts = pd.to_datetime(df[ts_col], utc=True)
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    test_end_ts = pd.Timestamp(test_end)

    train_mask = ts < (train_end_ts - embargo)
    val_mask = (ts >= train_end_ts) & (ts < (val_end_ts - embargo))
    test_mask = (ts >= val_end_ts) & (ts < test_end_ts)

    def _split(mask: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        sub = df.loc[mask].copy()
        X = sub[feature_columns].copy()
        y = sub[label_col].astype(int).copy()
        return X, y

    X_train, y_train = _split(train_mask)
    X_val, y_val = _split(val_mask)
    X_test, y_test = _split(test_mask)

    return TrainSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_window=(ts.min().to_pydatetime(), train_end_ts.to_pydatetime()),
        val_window=(train_end_ts.to_pydatetime(), val_end_ts.to_pydatetime()),
        test_window=(val_end_ts.to_pydatetime(), test_end_ts.to_pydatetime()),
    )


def train(
    split: TrainSplit,
    *,
    hyperparams: dict[str, Any] | None = None,
) -> tuple[CalibratedClassifier, dict[str, Any]]:
    """Fit XGB + isotonic; return (model, metrics)."""
    params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    base = XGBClassifier(**params)
    # XGBoost handles NaN natively; no imputation.
    base.fit(
        split.X_train,
        split.y_train,
        eval_set=[(split.X_val, split.y_val)],
        verbose=False,
    )

    raw_val = base.predict_proba(split.X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val, split.y_val)

    model = CalibratedClassifier(base=base, calibrator=calibrator, feature_columns=list(split.X_train.columns))

    metrics = _evaluate(model, split)
    return model, metrics


def _evaluate(model: CalibratedClassifier, split: TrainSplit) -> dict[str, Any]:
    def _block(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        if X.empty:
            return {"n": 0}
        p = model.predict_yes_prob(X)
        out = {
            "n": int(len(y)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
        }
        if y.nunique() > 1:
            out["auc"] = float(roc_auc_score(y, p))
        return out

    return {
        "train": _block(split.X_train, split.y_train),
        "val": _block(split.X_val, split.y_val),
        "test": _block(split.X_test, split.y_test),
    }


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[tuple[float, float, int]]:
    """Return [(bucket_mean_pred, bucket_observed_rate, n), ...] for the reliability plot."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out: list[tuple[float, float, int]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(
            (
                float(y_prob[mask].mean()),
                float(y_true[mask].mean()),
                n,
            )
        )
    return out


def build_train_report(
    model: CalibratedClassifier,
    split: TrainSplit,
    metrics: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
) -> str:
    """Render a self-contained HTML report (SPEC §9.1 calibration plot +
    feature importance). Uses matplotlib via Agg (no GUI)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _fig_to_b64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    # Calibration plot on validation.
    val_p = model.predict_yes_prob(split.X_val)
    curve_val = reliability_curve(split.y_val.to_numpy(), val_p)
    fig, ax = plt.subplots(figsize=(5, 5))
    if curve_val:
        xs, ys, ns = zip(*curve_val)
        ax.plot(xs, ys, marker="o", label="validation")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    ax.set_xlabel("predicted P(yes)")
    ax.set_ylabel("observed P(yes)")
    ax.set_title("Calibration (validation)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    img_cal = _fig_to_b64(fig)

    # Feature importance.
    importances = sorted(
        zip(model.feature_columns, model.base.feature_importances_),
        key=lambda kv: kv[1],
        reverse=True,
    )[:25]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(importances))))
    if importances:
        names, values = zip(*importances)
        y = np.arange(len(names))
        ax.barh(y, values)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("gain importance")
        ax.set_title("Top features")
    img_imp = _fig_to_b64(fig)

    rows = ""
    for block in ("train", "val", "test"):
        m = metrics.get(block, {})
        rows += f"<tr><td>{block}</td><td>{m.get('n','-')}</td><td>{m.get('log_loss','-'):.4f}" \
                if "log_loss" in m else f"<tr><td>{block}</td><td>{m.get('n','-')}</td><td>-</td>"
        rows += f"<td>{m.get('brier','-'):.4f}</td>" if "brier" in m else "<td>-</td>"
        rows += f"<td>{m.get('auc','-'):.3f}</td></tr>" if "auc" in m else "<td>-</td></tr>"

    extra_html = ""
    if extra:
        extra_html = "<h2>Extra</h2><pre>" + json.dumps(extra, indent=2, default=str) + "</pre>"

    return f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>calibration model report</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2em auto; color: #222; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
img {{ max-width: 100%; }}
h1, h2 {{ border-bottom: 1px solid #eee; padding-bottom: 4px; }}
</style></head>
<body>
<h1>calibration model report</h1>
<p>Registered: {dt.datetime.now(dt.timezone.utc).isoformat()}</p>
<h2>Metrics</h2>
<table>
<tr><th>split</th><th>n</th><th>log_loss</th><th>brier</th><th>auc</th></tr>
{rows}
</table>
<h2>Calibration</h2>
<img src="data:image/png;base64,{img_cal}" />
<h2>Feature importance (top 25)</h2>
<img src="data:image/png;base64,{img_imp}" />
{extra_html}
</body></html>
"""
