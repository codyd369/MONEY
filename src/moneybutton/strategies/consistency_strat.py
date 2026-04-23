"""Consistency strategy (SPEC §10.4).

Structural edge: exploit mutually-exclusive-exhaustive (MEE) markets whose
YES prices don't sum to $1.00. Example: a Kalshi event with N candidates
("Who wins the 2026 election?") where exactly one resolves YES.

Math:
  Let P_i = yes_ask_i / 100 for each market in the cluster.
  If sum(P_i) > 1.0 + margin: market overvalues the set. Buying NO on each
    costs sum(1 - P_i) = N - sum(P_i), pays out (N-1) $1 on resolution
    (since exactly one resolves YES). Break-even condition:
      sum(1 - P_i) <= N - 1  <=>  sum(P_i) >= 1
    so an overpriced sum is a profitable NO-basket. Profit per $1 staked
    = (N - 1 - sum(1 - P_i)) / sum(1 - P_i).
  If sum(P_i) < 1.0 - margin: market undervalues the set. Buying YES on
    each costs sum(P_i), pays out exactly $1 (one will resolve YES).
    Profit = (1 - sum(P_i)) / sum(P_i).

Spec caveats:
  - Requires simultaneous execution on N markets. Without that, a moving
    quote on any leg destroys the edge.
  - Fees are paid on each leg and can eat small imbalances.
  - LLM market-mapping / cluster discovery is deferred; v1 clusters by
    shared Kalshi event_ticker — the platform groups MEE markets that way.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable


@dataclass
class MEECluster:
    """Mutually-exclusive-exhaustive cluster of Kalshi markets.

    Markets must share an event_ticker and the event's semantics must be
    "exactly one resolves YES".
    """

    event_ticker: str
    markets: list[dict]
    # Each market dict carries at minimum: ticker, yes_bid_cents, yes_ask_cents,
    # top_of_book_yes_usd, top_of_book_no_usd.


@dataclass
class ConsistencyOpportunity:
    cluster_event: str
    direction: str  # 'basket_no' when sum > 1, 'basket_yes' when sum < 1
    sum_of_yes_asks: float
    imbalance_bps: int  # (sum - 1.0) * 10000, signed
    legs: list[tuple[str, str, int, int]]  # (ticker, side, entry_cents, top_of_book_usd)
    cost_usd_per_dollar: float
    profit_usd_per_dollar: float
    max_usd: float
    ts: dt.datetime


def detect_mee_inconsistency(
    cluster: MEECluster,
    *,
    min_imbalance_bps: int,
    max_size_usd: float,
    fee_rate: float = 0.07,
) -> ConsistencyOpportunity | None:
    """Return an opportunity when the cluster deviates from 1.0 sum by
    at least `min_imbalance_bps`. Accounts for fees naively (per-leg)."""
    if len(cluster.markets) < 2:
        return None

    # Normalize prices to fractions 0..1.
    yes_asks = [m["yes_ask_cents"] / 100.0 for m in cluster.markets]
    yes_bids = [m["yes_bid_cents"] / 100.0 for m in cluster.markets]
    no_asks = [1.0 - yb for yb in yes_bids]  # NO ask ~= 1 - YES bid for 1-tick spreads
    sum_yes_asks = sum(yes_asks)
    imbalance_bps = int(round((sum_yes_asks - 1.0) * 10_000))

    if abs(imbalance_bps) < min_imbalance_bps:
        return None

    n = len(cluster.markets)
    now = dt.datetime.now(dt.timezone.utc)

    if sum_yes_asks < 1.0:
        # Buy YES on each leg; pay sum_yes_asks for $1 guaranteed payout.
        cost = sum_yes_asks
        fees = fee_rate * sum(p * (1 - p) for p in yes_asks)  # per-contract fee * 1 contract each, as fraction
        profit_per_dollar = (1.0 - cost - fees) / cost if cost > 0 else 0.0
        legs = [
            (m["ticker"], "yes", m["yes_ask_cents"], float(m.get("top_of_book_yes_usd", 0.0)))
            for m in cluster.markets
        ]
        avail = min((leg[3] / (leg[2] / 100.0) for leg in legs if leg[2] > 0), default=0.0)
        direction = "basket_yes"
    else:
        # Buy NO on each leg; pay sum_no_asks for (N-1) $1 guaranteed payout.
        cost = sum(no_asks)
        fees = fee_rate * sum(p * (1 - p) for p in no_asks)
        # Pay `cost` per (N-1) payout. Normalize to per-$1-payout for the API.
        cost_per_dollar = cost / max(1, (n - 1))
        profit_per_dollar = (1.0 - cost_per_dollar - fees / max(1, (n - 1))) / cost_per_dollar if cost_per_dollar > 0 else 0.0
        legs = [
            (
                m["ticker"],
                "no",
                int(round((1 - m["yes_bid_cents"] / 100.0) * 100)),
                float(m.get("top_of_book_no_usd", 0.0)),
            )
            for m in cluster.markets
        ]
        cost = cost_per_dollar
        avail = min((leg[3] / (leg[2] / 100.0) for leg in legs if leg[2] > 0), default=0.0)
        direction = "basket_no"

    if profit_per_dollar <= 0:
        return None

    deployed = min(avail, max_size_usd) if avail > 0 else 0.0
    if deployed <= 0:
        return None

    return ConsistencyOpportunity(
        cluster_event=cluster.event_ticker,
        direction=direction,
        sum_of_yes_asks=sum_yes_asks,
        imbalance_bps=imbalance_bps,
        legs=legs,
        cost_usd_per_dollar=cost,
        profit_usd_per_dollar=profit_per_dollar,
        max_usd=deployed,
        ts=now,
    )


def group_by_event(markets: Iterable[dict]) -> list[MEECluster]:
    """Group markets by event_ticker.

    Kalshi's native grouping is the right shape for MEE detection. The spec
    notes that LLM-based relationship discovery (§10.4) is a later upgrade.
    """
    buckets: dict[str, list[dict]] = {}
    for m in markets:
        key = m.get("event_ticker")
        if not key:
            continue
        buckets.setdefault(key, []).append(m)
    return [MEECluster(event_ticker=k, markets=v) for k, v in buckets.items() if len(v) >= 2]
