"""Arbitrage strategy (SPEC §10.3).

Core idea:
    For a pair of semantically equivalent binary markets A and B that
    resolve on the same event:
        YES_A_price + NO_B_price < $1.00 - SAFETY_MARGIN
    means a risk-free (modulo execution risk) profit exists: buy YES on A
    and NO on B, pay < $1, collect exactly $1 on resolution.

    In practice the pair is Kalshi_market_X vs Polymarket_market_Y
    where our market_mapping layer has confirmed they resolve identically.

Honest caveats baked into the code:
  1. SPEC §7.2: Polymarket execution requires EVM wallet signing which is
     NOT implemented. Without it we can emit SIGNALS (route the Kalshi
     leg to the executor, and alert the operator to execute Polymarket
     manually) OR operate in signals-only mode.
  2. Pair equivalence is brittle: a market-mapping confidence < 0.95 is
     a red flag; we bake that threshold into find_arbs() and refuse to
     emit signals below it.
  3. Slippage + fees on both platforms eat most thin arbs. The spec
     default SAFETY_MARGIN_BPS_ARBITRAGE = 100 bps (1 cent) is the bar.

Backtest note (v1): arbitrage requires historical Polymarket price
alignment that I don't have on disk. The "backtest" here is a unit-test
demonstration that find_arbs() + the safety margin math are correct.
A true historical arbitrage backtest requires a paired Kalshi/Polymarket
dataset, which we'll build once both platforms are scraping (step 22).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable


@dataclass
class PairedMarket:
    """Kalshi market + semantically equivalent other-platform market."""

    kalshi_ticker: str
    other_platform: str  # 'polymarket' | 'sportsbook_consensus'
    other_id: str
    # Prices as fractions 0..1.
    kalshi_yes_ask: float
    kalshi_no_ask: float  # = 1 - yes_bid typically
    other_yes_ask: float
    other_no_ask: float
    # Top-of-book size in $ on each side (min across legs).
    kalshi_yes_size_usd: float
    kalshi_no_size_usd: float
    other_yes_size_usd: float
    other_no_size_usd: float
    # Confidence from market_mapping: 0..1.
    mapping_confidence: float
    # Polarity — True means "kalshi_yes <-> other_yes"; False means inverted
    # ("kalshi_yes <-> other_no").
    aligned_polarity: bool = True


@dataclass
class ArbOpportunity:
    pair: PairedMarket
    side_on_kalshi: str  # 'yes' | 'no' — which side we buy on Kalshi
    cost_per_dollar: float  # total cost for a $1 payout (< 1.0 if arb exists)
    edge_bps: int  # (1.0 - cost) * 10000
    max_usd: float  # how much we can deploy given top-of-book on both legs
    ts: dt.datetime


def _arb_cost_aligned(pair: PairedMarket) -> tuple[float, float, float]:
    """Return (cost_buy_yes_side, cost_buy_no_side, min_size_usd)."""
    # Buy YES on Kalshi + NO on Other (aligned polarity):
    cost_yes_side = pair.kalshi_yes_ask + pair.other_no_ask
    max_size_yes = min(pair.kalshi_yes_size_usd, pair.other_no_size_usd)
    # Buy NO on Kalshi + YES on Other:
    cost_no_side = pair.kalshi_no_ask + pair.other_yes_ask
    max_size_no = min(pair.kalshi_no_size_usd, pair.other_yes_size_usd)
    return cost_yes_side, cost_no_side, min(max_size_yes, max_size_no)


def _arb_cost_inverted(pair: PairedMarket) -> tuple[float, float, float]:
    """Polarity is flipped: kalshi_yes maps to other_no.
    Buy YES on Kalshi + YES on Other (both map to YES of the underlying event
    under inverted mapping this is contradiction — skip)."""
    cost_yes_side = pair.kalshi_yes_ask + pair.other_yes_ask
    max_size_yes = min(pair.kalshi_yes_size_usd, pair.other_yes_size_usd)
    cost_no_side = pair.kalshi_no_ask + pair.other_no_ask
    max_size_no = min(pair.kalshi_no_size_usd, pair.other_no_size_usd)
    return cost_yes_side, cost_no_side, min(max_size_yes, max_size_no)


def find_arbs(
    pairs: Iterable[PairedMarket],
    *,
    safety_margin_bps: int,
    min_mapping_confidence: float = 0.95,
    max_size_usd: float = 100.0,
) -> list[ArbOpportunity]:
    """Emit ArbOpportunities whose total cost per $1 payout is below
    (1 - safety_margin) with confident mappings."""
    margin = safety_margin_bps / 10_000.0
    cost_ceiling = 1.0 - margin

    out: list[ArbOpportunity] = []
    now = dt.datetime.now(dt.timezone.utc)
    for pair in pairs:
        if pair.mapping_confidence < min_mapping_confidence:
            continue
        if pair.aligned_polarity:
            cost_yes_side, cost_no_side, avail = _arb_cost_aligned(pair)
        else:
            cost_yes_side, cost_no_side, avail = _arb_cost_inverted(pair)

        if avail <= 0:
            continue

        # Pick the cheaper side.
        if cost_yes_side <= cost_no_side:
            cost, side = cost_yes_side, "yes"
        else:
            cost, side = cost_no_side, "no"
        if cost >= cost_ceiling:
            continue

        edge_bps = int(round((1.0 - cost) * 10_000))
        deployed = min(avail, max_size_usd)
        out.append(
            ArbOpportunity(
                pair=pair,
                side_on_kalshi=side,
                cost_per_dollar=cost,
                edge_bps=edge_bps,
                max_usd=deployed,
                ts=now,
            )
        )
    # Rank by edge, biggest first.
    out.sort(key=lambda o: o.edge_bps, reverse=True)
    return out
