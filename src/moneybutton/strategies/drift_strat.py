"""Drift strategy (SPEC §10.6).

Cross-platform drift: when two semantically-paired markets (Kalshi +
Polymarket, or Kalshi + implied sportsbook probability) move apart from
each other over a short window. The hypothesis is that one venue leads
and the other lags; we bet the laggard catches up.

This is weaker than arbitrage (no guaranteed payoff) so we set the bar
high: mapping_confidence >= 0.85 (lower than arb's 0.95 because we're
only exploiting a probabilistic drift, not claiming a free lunch) AND
divergence magnitude >= min_divergence_bps.

The complementary "calibration drift" detector monitors the calibration
model's recent accuracy versus history; if recent Brier score is
materially worse than the training Brier, pause the strategy (risk
signal, not a trade signal). That detector is included here as
detect_calibration_drift and is meant to feed the scheduler's safety
job rather than the executor.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable


@dataclass
class PairedPriceSnapshot:
    kalshi_ticker: str
    other_platform: str
    other_id: str
    kalshi_price_1h_ago: float
    kalshi_price_now: float
    other_price_1h_ago: float
    other_price_now: float
    mapping_confidence: float


@dataclass
class DriftOpportunity:
    snapshot: PairedPriceSnapshot
    bet_on: str  # which venue to bet on catching up: 'kalshi' or 'other'
    side: str  # 'yes' | 'no'
    divergence_bps: int
    edge_hint_bps: int  # rough size of expected re-convergence
    ts: dt.datetime


def detect_cross_platform_drift(
    pairs: Iterable[PairedPriceSnapshot],
    *,
    min_divergence_bps: int,
    min_mapping_confidence: float = 0.85,
) -> list[DriftOpportunity]:
    """For each pair, compute the 1h move on each venue. If one venue moved
    while the other didn't (and they were roughly in sync before), emit an
    opportunity to bet the laggard catches up."""
    out: list[DriftOpportunity] = []
    now = dt.datetime.now(dt.timezone.utc)
    for snap in pairs:
        if snap.mapping_confidence < min_mapping_confidence:
            continue
        k_move = snap.kalshi_price_now - snap.kalshi_price_1h_ago
        o_move = snap.other_price_now - snap.other_price_1h_ago
        divergence = k_move - o_move  # positive => kalshi moved up vs other
        divergence_bps = int(round(divergence * 10_000))
        if abs(divergence_bps) < min_divergence_bps:
            continue

        # Pre-drift prices should be roughly equal; otherwise we're not
        # really identifying drift but just a persistent pricing gap.
        gap_before = snap.kalshi_price_1h_ago - snap.other_price_1h_ago
        if abs(gap_before) > 0.05:  # 500 bps pre-existing gap
            continue

        # Bet the laggard catches up.
        if divergence_bps > 0:
            # Kalshi jumped up relative to Other. Bet Other_yes catches up
            # (i.e., buy YES on the other venue) OR bet Kalshi corrects
            # down (buy NO on kalshi). Because we want a Kalshi signal, go
            # NO on kalshi.
            bet_on, side = "kalshi", "no"
        else:
            bet_on, side = "kalshi", "yes"

        edge_hint_bps = int(abs(divergence_bps) * 0.5)  # assume half re-converges
        out.append(
            DriftOpportunity(
                snapshot=snap,
                bet_on=bet_on,
                side=side,
                divergence_bps=divergence_bps,
                edge_hint_bps=edge_hint_bps,
                ts=now,
            )
        )
    out.sort(key=lambda o: abs(o.divergence_bps), reverse=True)
    return out


# -------------------- Calibration-drift risk signal --------------------


@dataclass
class CalibrationDriftAlert:
    strategy: str
    baseline_brier: float
    recent_brier: float
    recent_n: int
    relative_degradation: float  # (recent - baseline) / baseline
    recommend_pause: bool


def detect_calibration_drift(
    *,
    baseline_brier: float,
    recent_y_true: list[int],
    recent_y_pred: list[float],
    threshold_rel_degradation: float = 0.25,
    min_n: int = 50,
    strategy: str = "calibration",
) -> CalibrationDriftAlert | None:
    """Compare recent Brier score to training baseline. Recommend PAUSING
    the strategy when recent Brier is ≥ threshold worse than baseline.

    Not a trade signal — feeds the scheduler safety job."""
    if len(recent_y_true) < min_n or len(recent_y_pred) != len(recent_y_true):
        return None
    n = len(recent_y_true)
    recent_brier = sum((p - y) ** 2 for p, y in zip(recent_y_pred, recent_y_true)) / n
    if baseline_brier <= 0:
        return None
    rel = (recent_brier - baseline_brier) / baseline_brier
    return CalibrationDriftAlert(
        strategy=strategy,
        baseline_brier=baseline_brier,
        recent_brier=recent_brier,
        recent_n=n,
        relative_degradation=rel,
        recommend_pause=rel >= threshold_rel_degradation,
    )
