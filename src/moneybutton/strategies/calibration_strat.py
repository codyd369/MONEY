"""Calibration strategy (SPEC §10.2).

Compare the calibration model's predicted P(YES) against the current market
mid-price. When |model_prob - market_prob| exceeds MIN_EDGE_BPS_CALIBRATION
and confidence is high, emit a Signal for the side the model favors.

For live use: loads the active model via models.registry.get_active, scans
every active Kalshi market with sufficient liquidity, returns a list of
Signals ranked by |edge_bps|.

For backtest: provides a scanner_fn() that wraps compute_features +
model.predict_yes_prob to be plugged straight into backtest/engine.run_backtest.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from moneybutton.backtest.engine import DecisionInput, SignalIntent
from moneybutton.core.config import Settings, get_settings
from moneybutton.features.common import parse_ts
from moneybutton.features.pipeline import compute_features, feature_schema
from moneybutton.models.calibration import CalibratedClassifier
from moneybutton.models.registry import ModelEntry, get_active, load
from moneybutton.strategies.base import Signal, Strategy


@dataclass
class CalibrationStrategy(Strategy):
    """Live scanner that reads the active model from the registry."""

    name: str = "calibration"
    settings: Settings = None  # type: ignore[assignment]
    model_entry: ModelEntry | None = None
    _model: CalibratedClassifier | None = None

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()

    def _ensure_model(self, feature_columns: list[str]) -> CalibratedClassifier | None:
        if self._model is not None:
            return self._model
        entry = self.model_entry or get_active("calibration", current_feature_columns=feature_columns)
        if entry is None:
            return None
        self.model_entry = entry
        self._model = load(entry)
        return self._model

    def _mid_from_market_snapshot(self, market: dict) -> float | None:
        # Live path: live feed stores yes_bid/ask on the market object.
        last_price = market.get("last_price")
        if last_price is not None:
            return float(last_price) / 100.0
        yb = market.get("yes_bid")
        ya = market.get("yes_ask")
        if yb is not None and ya is not None:
            return (float(yb) + float(ya)) / 200.0
        return None

    def scan(self, now: dt.datetime) -> list[Signal]:  # pragma: no cover - wired in live run
        # A real implementation pulls current markets from live_feed and
        # prices from Parquet. Deferred to the live_feed build; for now the
        # backtest-facing API below is what runs end-to-end.
        return []

    def backtest(self, start: dt.date, end: dt.date):  # pragma: no cover - use run_backtest directly
        raise NotImplementedError(
            "Use backtest.engine.run_backtest with make_scanner() for v1; a "
            "self-contained .backtest() wrapper lands at step 15."
        )


def make_scanner(
    *,
    model: CalibratedClassifier,
    min_edge_bps: int,
    high_confidence_edge_bps: int | None = None,
    min_volume: float = 0.0,
) -> Callable[[DecisionInput], SignalIntent | None]:
    """Return a scanner_fn suitable for backtest.engine.run_backtest.

    - min_edge_bps: absolute edge threshold to produce a signal at all.
    - high_confidence_edge_bps: when |edge| >= this, emit confidence='high'.
    - min_volume: 24h volume gate; skip thin markets.
    """
    hi = high_confidence_edge_bps if high_confidence_edge_bps is not None else (min_edge_bps * 2)
    fc = model.feature_columns

    def _scanner(ev: DecisionInput) -> SignalIntent | None:
        market = ev.market
        feats = compute_features(market, ev.as_of_ts, ev.price_frame)
        # Pull market price from the latest pre-as_of candle.
        now_price_decimal = feats.get("yes_price_now")
        if now_price_decimal is None:
            return None
        if feats.get("volume_24h") is None or feats["volume_24h"] < min_volume:
            return None

        # Build the exact feature-order vector the model was trained on.
        X = pd.DataFrame([{c: feats.get(c) for c in fc}])
        model_prob = float(model.predict_yes_prob(X)[0])
        edge = model_prob - now_price_decimal
        edge_bps = int(round(edge * 10_000))
        if abs(edge_bps) < min_edge_bps:
            return None

        side = "yes" if edge_bps > 0 else "no"
        # Flip entry to the NO side: market_prob_no = 1 - market_prob_yes.
        # edge_bps we pass to the signal is the magnitude on the chosen side.
        if side == "no":
            edge_bps_on_side = int(round((now_price_decimal - model_prob) * 10_000))
        else:
            edge_bps_on_side = edge_bps

        confidence = "high" if abs(edge_bps) >= hi else "med"

        return SignalIntent(
            side=side,
            size_usd=25.0,  # engine will clamp to max_position_usd
            edge_bps=edge_bps_on_side,
            confidence=confidence,
            reasoning={
                "model_prob": model_prob,
                "market_prob": now_price_decimal,
                "edge_bps": edge_bps,
                "ts": ev.as_of_ts.isoformat(),
            },
        )

    return _scanner
