"""Regression: compute_features must produce a whole-second ISO string for
as_of_ts. Sub-second precision from default_as_of_after_open (which does
open + life/2 for short markets) used to leak into the stored string and
break pandas' ISO format inference when mixed with whole-second rows.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from moneybutton.features.pipeline import (
    compute_features,
    default_as_of_after_open,
)


def _prices_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "ticker": "KX-SHORT-MKT",
                "ts": "2026-04-23T01:00:00+00:00",
                "yes_bid_close": 42,
                "yes_ask_close": 44,
                "last_price_close": 43,
                "volume": 10,
                "open_interest": 100,
            }
        ]
    )
    df["_ts_dt"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_compute_features_as_of_ts_is_whole_seconds():
    """A market with life=3h gets as_of = open + 1.5h (sub-second-safe,
    but microsecond arithmetic may sneak in). compute_features must
    strip microseconds before storing the ISO string."""
    market = {
        "ticker": "KX-SHORT-MKT",
        "category": "Crypto",
        "open_time": "2026-04-23T00:00:00+00:00",
        "close_time": "2026-04-23T03:00:00+00:00",
    }
    # Force a sub-second datetime to simulate the bug path.
    as_of = dt.datetime(2026, 4, 23, 1, 34, 46, 500000, tzinfo=dt.timezone.utc)
    feats = compute_features(market, as_of, _prices_frame())
    # No microseconds in the stored ISO string.
    assert "." not in feats["as_of_ts"]
    # Parses cleanly.
    parsed = pd.to_datetime(feats["as_of_ts"], utc=True, format="ISO8601")
    assert parsed.second == 46
    assert parsed.microsecond == 0


def test_default_as_of_after_open_short_market():
    """For a 4h-life market (life < 2 * margin=4h), the fn returns
    open + life/2 = open + 2h. Must survive the compute_features round
    trip — specifically the stored ISO string must parse."""
    market = {
        "ticker": "KX-SHORT",
        "open_time": "2026-04-23T00:00:00+00:00",
        "close_time": "2026-04-23T04:00:00+00:00",
    }
    as_of = default_as_of_after_open()(market)
    feats = compute_features(market, as_of, _prices_frame())
    parsed = pd.to_datetime(feats["as_of_ts"], utc=True, format="ISO8601")
    # halfway through a 4h market = 2h after open
    assert parsed.hour == 2


def test_mixed_whole_and_subsecond_inputs_parse():
    """The whole point: a column with mixed '...+00:00' and '.XXXXXX+00:00'
    strings must parse with format='ISO8601'. This is the operator's
    exact crash reproduced as a unit test."""
    s = pd.Series([
        "2026-04-23T01:34:46+00:00",
        "2026-04-23T01:34:46.500000+00:00",
    ])
    parsed = pd.to_datetime(s, utc=True, format="ISO8601")
    assert len(parsed) == 2
    assert parsed[0].second == 46
    assert parsed[1].microsecond == 500000
