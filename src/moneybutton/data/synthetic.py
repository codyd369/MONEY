"""Synthetic Kalshi-shaped data generator.

WHEN TO USE THIS:
  - You haven't run data/scraper_kalshi.py backfill_markets yet, AND you want
    to exercise the feature pipeline / model / backtest end-to-end.
  - The calibration model trained on synthetic data IS NOT predictive of
    real markets. Synthetic data is for plumbing validation only.

Data-generating process (DGP):
  - N markets across four categories (FINANCIALS, POLITICS, SPORTS, TECH).
  - Each market has a latent "true" yes-probability that depends on a few
    observable features (category bias, volume-z, days-to-expiry bucket) plus
    Gaussian noise in logit space.
  - Market resolves as Bernoulli(true_p).
  - 60m candlestick price walk starts at 0.50 and random-walks toward the
    true_p with realistic volatility, so the model can find *some* edge
    but not much — which matches the low end of real Kalshi markets.

Importantly the "edge" is baked into a small set of features so any honest
backtest on synthetic will show a positive-edge-but-noisy result, which
is what real Kalshi will look like at best.

API:
    generate_dataset(out_dir=None, n_markets=800, seed=42) -> summary dict

Output matches SPEC §6.2 partition scheme so downstream features/ and
backtest/ code can consume it without branching on data source.
"""

from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from moneybutton.data.store import PartitionKey, write_partition, year_month


CATEGORIES = ("FINANCIALS", "POLITICS", "SPORTS", "TECH")
_CATEGORY_BIAS = {
    "FINANCIALS": -0.20,
    "POLITICS": 0.05,
    "SPORTS": 0.00,
    "TECH": 0.15,
}


@dataclass
class GenParams:
    n_markets: int = 800
    start_date: dt.date = dt.date(2024, 1, 1)
    end_date: dt.date = dt.date(2026, 1, 1)
    candle_interval_min: int = 60
    max_candles_per_market: int = 48  # cap so the dataset stays small
    seed: int = 42


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _sample_market(rng: random.Random, idx: int, params: GenParams) -> dict:
    category = rng.choice(CATEGORIES)
    # Random open date in [start, end - 30d); duration 1..90 days.
    span_days = (params.end_date - params.start_date).days - 30
    open_offset = rng.randrange(0, span_days)
    open_date = params.start_date + dt.timedelta(days=open_offset)
    duration_days = rng.randint(1, 90)
    close_date = open_date + dt.timedelta(days=duration_days)
    open_ts = dt.datetime(open_date.year, open_date.month, open_date.day, tzinfo=dt.timezone.utc)
    close_ts = dt.datetime(close_date.year, close_date.month, close_date.day, 18, tzinfo=dt.timezone.utc)

    volume = int(abs(rng.lognormvariate(6.5, 1.2)))
    open_interest = int(volume * rng.uniform(0.2, 0.8))
    # Latent logit: category bias + volume-z + small noise.
    vol_z = (math.log(max(volume, 1)) - 6.5) / 1.2
    # Z-score of days-to-expiry; very long markets slightly bias toward NO.
    dte_z = (duration_days - 30) / 20.0
    logit_true = _CATEGORY_BIAS[category] + 0.18 * vol_z - 0.12 * dte_z + rng.gauss(0, 0.6)
    true_p = max(0.02, min(0.98, _sigmoid(logit_true)))

    result = "yes" if rng.random() < true_p else "no"

    return {
        "ticker": f"SYN-{category[:3]}-{idx:05d}",
        "event_ticker": f"SYN-{category[:3]}-{idx // 10:04d}",
        "series_ticker": f"SYN{category[:3]}",
        "title": f"Synthetic {category.lower()} market #{idx}",
        "subtitle": None,
        "category": category,
        "status": "settled",
        "result": result,
        "can_close_early": False,
        "expiration_time": close_ts.isoformat(),
        "close_time": close_ts.isoformat(),
        "open_time": open_ts.isoformat(),
        "expected_expiration_time": close_ts.isoformat(),
        "previous_yes_bid": None,
        "previous_yes_ask": None,
        "last_price": None,  # filled after candle walk
        "volume": volume,
        "volume_24h": max(1, int(volume / max(1, duration_days))),
        "open_interest": open_interest,
        "liquidity": volume,
        "notional_value": 1.0,
        "yes_sub_title": "YES",
        "no_sub_title": "NO",
        "rules_primary": "synthetic",
        "rules_secondary": None,
        "tick_size": 1,
        # Private columns used by the generator only:
        "_true_p": true_p,
        "_open_ts": open_ts,
        "_close_ts": close_ts,
        "_duration_days": duration_days,
    }


def _simulate_candles(market: dict, params: GenParams, rng: random.Random) -> list[dict]:
    """Random walk from 0.50 toward true_p with realistic volatility."""
    open_ts: dt.datetime = market["_open_ts"]
    close_ts: dt.datetime = market["_close_ts"]
    true_p: float = market["_true_p"]
    total_minutes = int((close_ts - open_ts).total_seconds() // 60)
    n_candles = min(params.max_candles_per_market, max(2, total_minutes // params.candle_interval_min))
    step_min = max(params.candle_interval_min, total_minutes // n_candles)

    p = 0.5
    rows: list[dict] = []
    for i in range(n_candles):
        pull = (true_p - p) * 0.08
        noise = rng.gauss(0, 0.04)
        p = max(0.02, min(0.98, p + pull + noise))
        ts = open_ts + dt.timedelta(minutes=step_min * (i + 1))
        bid_c = max(1, int(round(p * 100 - 1)))
        ask_c = min(99, int(round(p * 100 + 1)))
        last_c = int(round(p * 100))
        rows.append(
            {
                "ticker": market["ticker"],
                "ts": ts.isoformat(),
                "yes_bid_open": bid_c,
                "yes_bid_high": bid_c + 1,
                "yes_bid_low": max(1, bid_c - 1),
                "yes_bid_close": bid_c,
                "yes_ask_open": ask_c,
                "yes_ask_high": min(99, ask_c + 1),
                "yes_ask_low": ask_c,
                "yes_ask_close": ask_c,
                "last_price_open": last_c,
                "last_price_high": min(99, last_c + 1),
                "last_price_low": max(1, last_c - 1),
                "last_price_close": last_c,
                "volume": int(rng.uniform(5, 80)),
                "open_interest": market["open_interest"],
            }
        )
    return rows


def generate_dataset(params: GenParams | None = None) -> dict:
    """Write n_markets synthetic markets + their candles to the Parquet store."""
    p = params or GenParams()
    rng = random.Random(p.seed)
    np.random.seed(p.seed)

    markets: list[dict] = [_sample_market(rng, i, p) for i in range(p.n_markets)]
    market_rows_by_key: dict[PartitionKey, list[dict]] = {}
    price_rows_by_key: dict[PartitionKey, list[dict]] = {}

    for m in markets:
        public_cols = {k: v for k, v in m.items() if not k.startswith("_")}
        ym = year_month(m["close_time"])
        mkey = PartitionKey("markets", m["category"], ym)
        market_rows_by_key.setdefault(mkey, []).append(public_cols)

        candles = _simulate_candles(m, p, rng)
        # Fill the market's last_price from the final candle so downstream
        # features that read last_price see something sane.
        if candles:
            public_cols["last_price"] = candles[-1]["last_price_close"]
        pkey = PartitionKey("prices", m["category"], ym)
        price_rows_by_key.setdefault(pkey, []).extend(candles)

    written_markets = 0
    written_prices = 0
    for key, rows in market_rows_by_key.items():
        write_partition(key, pd.DataFrame(rows))
        written_markets += len(rows)
    for key, rows in price_rows_by_key.items():
        write_partition(key, pd.DataFrame(rows))
        written_prices += len(rows)

    return {
        "markets_written": written_markets,
        "prices_written": written_prices,
        "market_partitions": len(market_rows_by_key),
        "price_partitions": len(price_rows_by_key),
        "seed": p.seed,
    }


if __name__ == "__main__":
    summary = generate_dataset()
    print(summary)
