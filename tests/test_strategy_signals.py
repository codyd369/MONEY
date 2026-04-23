"""Signal / strategy integration tests.

Covers:
  1. Signal dataclass construction — defaults are sane.
  2. The calibration strategy's make_scanner produces well-formed signals
     only above the edge threshold and flips sides correctly.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from moneybutton.backtest.engine import DecisionInput
from moneybutton.features.common import parse_ts
from moneybutton.strategies.base import Signal


class _StubModel:
    """A model stand-in that returns a configured prob for every X."""

    def __init__(self, yes_prob: float, feature_columns: list[str]):
        self.yes_prob = yes_prob
        self.feature_columns = feature_columns

    def predict_yes_prob(self, X) -> np.ndarray:
        return np.array([self.yes_prob] * len(X))


def _decision_event(market_price_cents: int = 40) -> DecisionInput:
    market = {
        "ticker": "T-1",
        "category": "POLITICS",
        "status": "settled",
        "result": "yes",
        "open_time": "2025-01-01T12:00:00Z",
        "close_time": "2025-01-10T18:00:00Z",
        "volume": 100,
    }
    prices = pd.DataFrame(
        [
            {
                "ticker": "T-1",
                "ts": "2025-01-09T12:00:00Z",
                "yes_bid_close": market_price_cents - 1,
                "yes_ask_close": market_price_cents + 1,
                "last_price_close": market_price_cents,
                "volume": 50,
                "open_interest": 500,
            }
        ]
    )
    prices["_ts_dt"] = pd.to_datetime(prices["ts"], utc=True)
    as_of = parse_ts("2025-01-09T13:00:00Z")
    return DecisionInput(market=market, as_of_ts=as_of, price_frame=prices)


def test_signal_dataclass_defaults():
    s = Signal(
        strategy="calibration",
        ticker="KX-1",
        side="yes",
        edge_bps=500,
        confidence="med",
        suggested_size_usd=10.0,
    )
    assert s.reasoning == {}
    assert s.expires_at is None


def test_calibration_scanner_emits_yes_signal_on_positive_edge():
    from moneybutton.strategies.calibration_strat import make_scanner

    ev = _decision_event(market_price_cents=40)  # market_prob = 0.40
    # Model predicts 0.55 -> edge = +1500 bps -> YES signal.
    from moneybutton.features.pipeline import compute_features

    feat_cols = sorted(k for k in compute_features(ev.market, ev.as_of_ts, ev.price_frame).keys()
                       if k not in ("ticker", "as_of_ts"))
    model = _StubModel(yes_prob=0.55, feature_columns=feat_cols)
    scanner = make_scanner(model=model, min_edge_bps=500)
    intent = scanner(ev)
    assert intent is not None
    assert intent.side == "yes"
    assert intent.edge_bps >= 1400
    assert intent.confidence in ("med", "high")


def test_calibration_scanner_emits_no_signal_when_below_threshold():
    from moneybutton.strategies.calibration_strat import make_scanner

    ev = _decision_event(market_price_cents=40)  # market = 0.40
    from moneybutton.features.pipeline import compute_features

    feat_cols = sorted(k for k in compute_features(ev.market, ev.as_of_ts, ev.price_frame).keys()
                       if k not in ("ticker", "as_of_ts"))
    model = _StubModel(yes_prob=0.43, feature_columns=feat_cols)  # edge ~ +300 bps
    scanner = make_scanner(model=model, min_edge_bps=500)
    assert scanner(ev) is None


def test_calibration_scanner_flips_to_no_side_on_negative_edge():
    from moneybutton.strategies.calibration_strat import make_scanner

    ev = _decision_event(market_price_cents=60)  # market = 0.60
    from moneybutton.features.pipeline import compute_features

    feat_cols = sorted(k for k in compute_features(ev.market, ev.as_of_ts, ev.price_frame).keys()
                       if k not in ("ticker", "as_of_ts"))
    model = _StubModel(yes_prob=0.30, feature_columns=feat_cols)  # edge -3000 bps
    scanner = make_scanner(model=model, min_edge_bps=500)
    intent = scanner(ev)
    assert intent is not None
    assert intent.side == "no"
    # Edge reported on the NO side = market_p - model_p = 0.30 -> +3000 bps
    assert intent.edge_bps >= 2900


def test_calibration_scanner_skips_low_volume():
    from moneybutton.strategies.calibration_strat import make_scanner

    ev = _decision_event(market_price_cents=40)
    from moneybutton.features.pipeline import compute_features

    feat_cols = sorted(k for k in compute_features(ev.market, ev.as_of_ts, ev.price_frame).keys()
                       if k not in ("ticker", "as_of_ts"))
    model = _StubModel(yes_prob=0.55, feature_columns=feat_cols)
    scanner = make_scanner(model=model, min_edge_bps=500, min_volume=1_000_000)
    assert scanner(ev) is None
