"""Calibration model training on synthetic features.

We generate a small controllable dataset with known signal (two features
correlated with the label), train the calibration model, and assert it
beats the no-information rate. This is a smoke test for the train/val/test
split + isotonic calibration plumbing, not a claim about real edge.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from moneybutton.models.calibration import (
    CalibratedClassifier,
    build_train_report,
    reliability_curve,
    time_split,
    train,
)


def _make_synthetic(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.5, n)
    logits = 0.8 * x1 + 0.4 * x2 + noise
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(0, 1, n) < p).astype(int)

    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    as_of = pd.to_datetime([t0 + pd.Timedelta(days=i * (365 / n)) for i in range(n)])
    return pd.DataFrame(
        {
            "ticker": [f"T-{i}" for i in range(n)],
            "as_of_ts": as_of,
            "x1": x1,
            "x2": x2,
            "x_noise": rng.normal(0, 1, n),
            "label_resolved": y,
        }
    )


def test_split_honors_embargo():
    df = _make_synthetic()
    train_end = dt.datetime(2025, 6, 1, tzinfo=dt.timezone.utc)
    val_end = dt.datetime(2025, 9, 1, tzinfo=dt.timezone.utc)
    test_end = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    split = time_split(df, train_end=train_end, val_end=val_end, test_end=test_end)
    total = len(split.X_train) + len(split.X_val) + len(split.X_test)
    # Some rows are dropped by the embargo; assert we lost fewer than 10% total.
    assert total >= 0.9 * len(df)
    # Train rows all strictly before train_end - embargo.
    assert (pd.to_datetime(df.loc[split.X_train.index, "as_of_ts"], utc=True) <
            pd.Timestamp(train_end) - pd.Timedelta(days=7)).all()


def test_train_beats_random():
    df = _make_synthetic()
    split = time_split(
        df,
        train_end=dt.datetime(2025, 6, 1, tzinfo=dt.timezone.utc),
        val_end=dt.datetime(2025, 9, 1, tzinfo=dt.timezone.utc),
        test_end=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
    )
    model, metrics = train(split)
    assert isinstance(model, CalibratedClassifier)
    # AUC > 0.55 is a low bar on this easy synthetic DGP; failing it means
    # the pipeline is broken, not that the model is bad.
    assert metrics["val"]["auc"] > 0.55
    assert metrics["test"]["auc"] > 0.55


def test_predict_proba_is_calibrated():
    df = _make_synthetic(n=3000)
    split = time_split(
        df,
        train_end=dt.datetime(2025, 6, 1, tzinfo=dt.timezone.utc),
        val_end=dt.datetime(2025, 9, 1, tzinfo=dt.timezone.utc),
        test_end=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
    )
    model, _ = train(split)
    p = model.predict_yes_prob(split.X_val)
    curve = reliability_curve(split.y_val.to_numpy(), p)
    # Sum of |predicted - observed| per bucket, weighted by n, divided by N.
    # Calibrated model has this well below 0.1 on easy synthetic data.
    total_weight = sum(n for _, _, n in curve)
    assert total_weight > 0
    weighted_err = sum(abs(pm - om) * n for pm, om, n in curve) / total_weight
    assert weighted_err < 0.1, f"calibration error {weighted_err:.3f} too high"


def test_build_train_report_returns_html():
    df = _make_synthetic(n=800)
    split = time_split(
        df,
        train_end=dt.datetime(2025, 6, 1, tzinfo=dt.timezone.utc),
        val_end=dt.datetime(2025, 9, 1, tzinfo=dt.timezone.utc),
        test_end=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
    )
    model, metrics = train(split)
    html = build_train_report(model, split, metrics)
    assert html.lstrip().startswith("<!doctype html>")
    assert "Calibration" in html
    assert "Feature importance" in html


def test_registry_roundtrip(tmp_env):
    """Registry writes versioned folders and returns the highest-version
    model whose feature schema fingerprint matches the current pipeline."""
    from moneybutton.models.registry import get_active, list_models, load, register

    df = _make_synthetic(n=500)
    split = time_split(
        df,
        train_end=dt.datetime(2025, 6, 1, tzinfo=dt.timezone.utc),
        val_end=dt.datetime(2025, 9, 1, tzinfo=dt.timezone.utc),
        test_end=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
    )
    model, metrics = train(split)
    entry = register(
        "calibration",
        model,
        feature_columns=model.feature_columns,
        metadata={"val_auc": metrics["val"].get("auc"), "hyperparams": "test"},
    )
    assert entry.version == 1

    # Another registration bumps the version.
    entry2 = register(
        "calibration",
        model,
        feature_columns=model.feature_columns,
        metadata={"hyperparams": "test2"},
    )
    assert entry2.version == 2

    assert {e.version for e in list_models("calibration")} == {1, 2}

    # get_active returns v2 and refuses when the fingerprint doesn't match.
    active = get_active("calibration", current_feature_columns=model.feature_columns)
    assert active is not None and active.version == 2

    mismatched = get_active("calibration", current_feature_columns=["totally_different"])
    assert mismatched is None

    # Load returns the same type we saved.
    loaded = load(active)
    assert isinstance(loaded, CalibratedClassifier)
