"""Drift strategy tests (SPEC §10.6)."""

from __future__ import annotations

from moneybutton.strategies.drift_strat import (
    PairedPriceSnapshot,
    detect_calibration_drift,
    detect_cross_platform_drift,
)


def _snap(
    kalshi_before: float = 0.50,
    kalshi_now: float = 0.50,
    other_before: float = 0.50,
    other_now: float = 0.50,
    conf: float = 0.9,
) -> PairedPriceSnapshot:
    return PairedPriceSnapshot(
        kalshi_ticker="KX-TEST",
        other_platform="polymarket",
        other_id="pm-test",
        kalshi_price_1h_ago=kalshi_before,
        kalshi_price_now=kalshi_now,
        other_price_1h_ago=other_before,
        other_price_now=other_now,
        mapping_confidence=conf,
    )


def test_matching_moves_emit_nothing():
    snaps = [_snap(kalshi_before=0.50, kalshi_now=0.60, other_before=0.50, other_now=0.60)]
    assert detect_cross_platform_drift(snaps, min_divergence_bps=100) == []


def test_kalshi_jumps_while_other_flat_emits_no_on_kalshi():
    # Kalshi 0.50 -> 0.65 (+1500bps); Other flat (0.50 -> 0.50).
    snaps = [_snap(kalshi_before=0.50, kalshi_now=0.65, other_before=0.50, other_now=0.50)]
    opps = detect_cross_platform_drift(snaps, min_divergence_bps=500)
    assert len(opps) == 1
    o = opps[0]
    assert o.side == "no"
    assert o.divergence_bps == 1500
    assert o.edge_hint_bps > 0


def test_other_jumps_while_kalshi_flat_emits_yes_on_kalshi():
    snaps = [_snap(kalshi_before=0.50, kalshi_now=0.50, other_before=0.50, other_now=0.65)]
    opps = detect_cross_platform_drift(snaps, min_divergence_bps=500)
    assert len(opps) == 1
    assert opps[0].side == "yes"
    assert opps[0].divergence_bps == -1500


def test_below_threshold_skipped():
    snaps = [_snap(kalshi_before=0.50, kalshi_now=0.52, other_before=0.50, other_now=0.50)]
    assert detect_cross_platform_drift(snaps, min_divergence_bps=500) == []


def test_low_mapping_confidence_skipped():
    snaps = [_snap(kalshi_before=0.50, kalshi_now=0.60, other_before=0.50, other_now=0.40, conf=0.70)]
    assert detect_cross_platform_drift(snaps, min_divergence_bps=100, min_mapping_confidence=0.85) == []


def test_preexisting_gap_skipped():
    # Pair was already far apart before the window; this isn't drift.
    snaps = [_snap(kalshi_before=0.40, kalshi_now=0.55, other_before=0.60, other_now=0.65)]
    assert detect_cross_platform_drift(snaps, min_divergence_bps=100) == []


def test_ranked_by_absolute_divergence():
    big = _snap(kalshi_before=0.50, kalshi_now=0.75, other_before=0.50, other_now=0.50)
    small = _snap(kalshi_before=0.50, kalshi_now=0.58, other_before=0.50, other_now=0.50)
    opps = detect_cross_platform_drift([small, big], min_divergence_bps=500)
    assert len(opps) == 2
    assert abs(opps[0].divergence_bps) >= abs(opps[1].divergence_bps)


# ---------------------- calibration drift alert -----------------------


def test_calibration_drift_pauses_on_degradation():
    import random

    rng = random.Random(1)
    # Baseline Brier ~0.15; recent Brier much worse.
    # Generate predictions close to truth to get ~0.15 baseline... actually we just
    # compare against `baseline_brier` passed in. Build recent with high error.
    n = 200
    y_true = [rng.randint(0, 1) for _ in range(n)]
    y_pred = [abs(y - 0.8) for y in y_true]  # pred systematically off
    alert = detect_calibration_drift(
        baseline_brier=0.15, recent_y_true=y_true, recent_y_pred=y_pred
    )
    assert alert is not None
    assert alert.recent_brier > alert.baseline_brier
    assert alert.recommend_pause is True


def test_calibration_drift_no_alert_when_stable():
    """Brier on alternating 0/1 with constant 0.5 predictions is 0.25, equal
    to our baseline — so no pause recommendation."""
    n = 200
    y_true = [i % 2 for i in range(n)]
    y_pred = [0.5] * n
    alert = detect_calibration_drift(
        baseline_brier=0.25,
        recent_y_true=y_true,
        recent_y_pred=y_pred,
    )
    assert alert is not None
    assert abs(alert.recent_brier - 0.25) < 1e-9
    assert alert.recommend_pause is False


def test_calibration_drift_requires_min_n():
    alert = detect_calibration_drift(
        baseline_brier=0.15,
        recent_y_true=[0, 1, 0],
        recent_y_pred=[0.5, 0.5, 0.5],
    )
    assert alert is None
