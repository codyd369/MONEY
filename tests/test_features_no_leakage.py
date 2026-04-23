"""CRITICAL — features computed at time T may only depend on data with ts<T.

Strategy: compute features at t=T, then perturb every price row with ts>=T
to pure noise, and recompute. The two feature dicts must be identical
(except for fields that are legitimately None because the as-of window has
no data after perturbation).

This is the single most important correctness invariant in the whole
project. A leak here silently inflates every backtest.
"""

from __future__ import annotations

import datetime as dt
import math
import random

import pandas as pd

from moneybutton.features.pipeline import compute_features


def _make_market():
    return {
        "ticker": "SYN-TST-00001",
        "event_ticker": "SYN-TST-1",
        "category": "POLITICS",
        "status": "active",
        "open_time": "2025-03-01T12:00:00+00:00",
        "close_time": "2025-03-15T18:00:00+00:00",
        "volume": 500,
        "open_interest": 200,
        "last_price": 42,
    }


def _make_prices(seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    t0 = dt.datetime(2025, 3, 1, 12, tzinfo=dt.timezone.utc)
    for i in range(24 * 14):  # hourly for 14 days
        ts = t0 + dt.timedelta(hours=i)
        p = 40 + int(rng.gauss(0, 5))
        p = max(5, min(95, p))
        rows.append(
            {
                "ticker": "SYN-TST-00001",
                "ts": ts.isoformat(),
                "yes_bid_close": p - 1,
                "yes_ask_close": p + 1,
                "last_price_close": p,
                "volume": int(rng.uniform(5, 80)),
                "open_interest": 200 + int(rng.gauss(0, 20)),
            }
        )
    return pd.DataFrame(rows)


def _perturb_after(prices: pd.DataFrame, as_of_ts: dt.datetime, seed: int) -> pd.DataFrame:
    """Replace every row at or after `as_of_ts` with noise. If the feature
    pipeline is leakage-free, features at `as_of_ts` should be unchanged."""
    rng = random.Random(seed)
    out = prices.copy()
    ts_dt = pd.to_datetime(out["ts"], utc=True)
    mask = ts_dt >= pd.Timestamp(as_of_ts)
    n = int(mask.sum())
    if n == 0:
        return out
    for col in ("yes_bid_close", "yes_ask_close", "last_price_close"):
        out.loc[mask, col] = [rng.randint(1, 99) for _ in range(n)]
    out.loc[mask, "volume"] = [rng.randint(0, 10_000) for _ in range(n)]
    out.loc[mask, "open_interest"] = [rng.randint(0, 10_000) for _ in range(n)]
    return out


def _equal_features(a: dict, b: dict) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if va is None and vb is None:
            continue
        if isinstance(va, float) and isinstance(vb, float):
            if math.isnan(va) and math.isnan(vb):
                continue
            if math.isclose(va, vb, rel_tol=1e-9, abs_tol=1e-9):
                continue
            return False
        if va == vb:
            continue
        return False
    return True


def test_no_leakage_mid_market():
    market = _make_market()
    prices = _make_prices(seed=1)
    as_of = dt.datetime(2025, 3, 8, 12, tzinfo=dt.timezone.utc)  # mid-life

    baseline = compute_features(market, as_of, prices)
    perturbed = compute_features(market, as_of, _perturb_after(prices, as_of, seed=999))
    assert _equal_features(baseline, perturbed), (
        "Features at t=T changed when data at ts>=T was randomized. This is a leak."
    )


def test_no_leakage_at_open():
    market = _make_market()
    prices = _make_prices(seed=2)
    as_of = dt.datetime(2025, 3, 2, 0, tzinfo=dt.timezone.utc)  # 12h after open

    baseline = compute_features(market, as_of, prices)
    perturbed = compute_features(market, as_of, _perturb_after(prices, as_of, seed=888))
    assert _equal_features(baseline, perturbed)


def test_no_leakage_near_close():
    market = _make_market()
    prices = _make_prices(seed=3)
    as_of = dt.datetime(2025, 3, 15, 17, tzinfo=dt.timezone.utc)  # 1h before close

    baseline = compute_features(market, as_of, prices)
    perturbed = compute_features(market, as_of, _perturb_after(prices, as_of, seed=777))
    assert _equal_features(baseline, perturbed)


def test_leak_detection_works():
    """Self-test: deliberately introduce a post-as_of change in a BEFORE row
    (violating the setup assumption) and show that features DO change. This
    proves _equal_features is sensitive, so a real leak wouldn't pass silently.
    """
    market = _make_market()
    prices = _make_prices(seed=4)
    as_of = dt.datetime(2025, 3, 8, 12, tzinfo=dt.timezone.utc)

    baseline = compute_features(market, as_of, prices)
    # Perturb an earlier row — this SHOULD change features.
    tampered = prices.copy()
    early_mask = pd.to_datetime(tampered["ts"], utc=True) < pd.Timestamp(as_of)
    idx = tampered[early_mask].index[-1]  # last pre-as_of row
    tampered.loc[idx, "last_price_close"] = 99
    tampered.loc[idx, "yes_bid_close"] = 98
    tampered.loc[idx, "yes_ask_close"] = 100
    changed = compute_features(market, as_of, tampered)
    assert not _equal_features(baseline, changed), (
        "_equal_features failed to detect a pre-as_of perturbation — the "
        "no-leakage tests above are vacuous."
    )


def test_feature_schema_fingerprint_stable():
    """The sorted tuple of feature names must be stable run-to-run. The model
    registry uses this fingerprint to refuse loading a model whose features
    don't match the current pipeline output."""
    from moneybutton.features.pipeline import feature_schema_fingerprint

    market = _make_market()
    prices = _make_prices(seed=5)
    as_of = dt.datetime(2025, 3, 8, 12, tzinfo=dt.timezone.utc)

    feats = compute_features(market, as_of, prices)
    fp_a = feature_schema_fingerprint(feats)
    fp_b = feature_schema_fingerprint(feats)
    assert fp_a == fp_b
    assert len(fp_a) == 64  # sha256 hex digest
