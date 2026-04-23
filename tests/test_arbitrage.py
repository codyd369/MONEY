"""Arbitrage strategy correctness tests (SPEC §10.3).

No network: we construct PairedMarkets by hand and verify find_arbs()
emits opportunities exactly when the combined cost < (1 - safety_margin).
"""

from __future__ import annotations

import pytest

from moneybutton.strategies.arbitrage_strat import (
    ArbOpportunity,
    PairedMarket,
    find_arbs,
)


def _pair(
    kalshi_yes_ask: float = 0.45,
    kalshi_no_ask: float = 0.55,
    other_yes_ask: float = 0.60,
    other_no_ask: float = 0.40,
    mapping_confidence: float = 1.0,
    aligned: bool = True,
    size: float = 100.0,
) -> PairedMarket:
    return PairedMarket(
        kalshi_ticker="KX-TEST",
        other_platform="polymarket",
        other_id="pm-test",
        kalshi_yes_ask=kalshi_yes_ask,
        kalshi_no_ask=kalshi_no_ask,
        other_yes_ask=other_yes_ask,
        other_no_ask=other_no_ask,
        kalshi_yes_size_usd=size,
        kalshi_no_size_usd=size,
        other_yes_size_usd=size,
        other_no_size_usd=size,
        mapping_confidence=mapping_confidence,
        aligned_polarity=aligned,
    )


def test_arb_detected_when_costs_sum_below_dollar():
    # Kalshi yes 45c + Polymarket no 40c = 85c total for $1 payout -> 15c edge.
    pair = _pair(kalshi_yes_ask=0.45, other_no_ask=0.40)
    arbs = find_arbs([pair], safety_margin_bps=100)
    assert len(arbs) == 1
    a = arbs[0]
    assert a.side_on_kalshi == "yes"
    assert abs(a.cost_per_dollar - 0.85) < 1e-9
    assert a.edge_bps == 1500
    assert a.max_usd == 100.0


def test_no_arb_when_costs_sum_above_safety_margin():
    # Kalshi yes 55c + Polymarket no 44c = 99c total -> 1c edge < 1c safety margin.
    pair = _pair(kalshi_yes_ask=0.55, other_no_ask=0.44)
    arbs = find_arbs([pair], safety_margin_bps=100)
    assert arbs == []


def test_arb_picks_cheaper_side():
    # Buy yes side: 50c + 50c = 1.00 (no arb)
    # Buy no side: 45c + 45c = 0.90 (arb)
    pair = _pair(
        kalshi_yes_ask=0.50, kalshi_no_ask=0.45,
        other_yes_ask=0.45, other_no_ask=0.50,
    )
    arbs = find_arbs([pair], safety_margin_bps=100)
    assert len(arbs) == 1
    assert arbs[0].side_on_kalshi == "no"
    assert abs(arbs[0].cost_per_dollar - 0.90) < 1e-9


def test_mapping_confidence_gate():
    pair = _pair(kalshi_yes_ask=0.40, other_no_ask=0.40, mapping_confidence=0.80)
    arbs = find_arbs([pair], safety_margin_bps=100, min_mapping_confidence=0.95)
    assert arbs == []
    # Even a huge arb is refused when the mapping is unsure.


def test_ranked_by_edge_descending():
    cheap = _pair(kalshi_yes_ask=0.20, other_no_ask=0.10)  # 70 bps edge... wait let me compute
    medium = _pair(kalshi_yes_ask=0.40, other_no_ask=0.35)
    # cheap: 30c -> 70c edge = 7000 bps
    # medium: 75c -> 25c edge = 2500 bps
    arbs = find_arbs([medium, cheap], safety_margin_bps=100)
    assert len(arbs) == 2
    assert arbs[0].edge_bps > arbs[1].edge_bps
    assert arbs[0].pair is cheap


def test_size_capped_by_min_top_of_book():
    pair = _pair(kalshi_yes_ask=0.40, other_no_ask=0.30, size=50.0)
    arbs = find_arbs([pair], safety_margin_bps=100, max_size_usd=200)
    assert len(arbs) == 1
    assert arbs[0].max_usd == 50.0  # bounded by top-of-book, not by max_size


def test_max_size_cap():
    pair = _pair(kalshi_yes_ask=0.40, other_no_ask=0.30, size=500.0)
    arbs = find_arbs([pair], safety_margin_bps=100, max_size_usd=100)
    assert arbs[0].max_usd == 100.0  # bounded by max_size_usd


def test_zero_size_pair_emits_nothing():
    pair = _pair(kalshi_yes_ask=0.10, other_no_ask=0.10, size=0.0)
    assert find_arbs([pair], safety_margin_bps=100) == []
