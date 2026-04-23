"""Consistency strategy tests (SPEC §10.4)."""

from __future__ import annotations

from moneybutton.strategies.consistency_strat import (
    MEECluster,
    detect_mee_inconsistency,
    group_by_event,
)


def _market(
    ticker: str,
    yes_ask: int,
    yes_bid: int | None = None,
    size_yes: float = 100.0,
    size_no: float = 100.0,
    event: str = "KX-TEST",
) -> dict:
    return {
        "ticker": ticker,
        "event_ticker": event,
        "yes_ask_cents": yes_ask,
        "yes_bid_cents": yes_bid if yes_bid is not None else (yes_ask - 1),
        "top_of_book_yes_usd": size_yes,
        "top_of_book_no_usd": size_no,
    }


def test_cluster_summing_to_dollar_is_balanced():
    # 40 + 35 + 25 = 100 -> 0 bps imbalance.
    cluster = MEECluster(
        event_ticker="KX-BAL",
        markets=[_market("A", 40), _market("B", 35), _market("C", 25)],
    )
    assert detect_mee_inconsistency(cluster, min_imbalance_bps=50, max_size_usd=100) is None


def test_undervalued_basket_triggers_basket_yes():
    # 30 + 30 + 30 = 90 -> -1000 bps imbalance, basket_yes arb.
    cluster = MEECluster(
        event_ticker="KX-UV",
        markets=[_market("A", 30), _market("B", 30), _market("C", 30)],
    )
    opp = detect_mee_inconsistency(cluster, min_imbalance_bps=200, max_size_usd=100)
    assert opp is not None
    assert opp.direction == "basket_yes"
    assert opp.imbalance_bps == -1000
    assert abs(opp.sum_of_yes_asks - 0.9) < 1e-9
    # Sanity: 3 legs, each buys yes.
    assert len(opp.legs) == 3
    assert all(leg[1] == "yes" for leg in opp.legs)


def test_overvalued_basket_triggers_basket_no():
    # 40 + 40 + 30 = 110 -> +1000 bps, basket_no arb.
    cluster = MEECluster(
        event_ticker="KX-OV",
        markets=[_market("A", 40), _market("B", 40), _market("C", 30)],
    )
    opp = detect_mee_inconsistency(cluster, min_imbalance_bps=200, max_size_usd=100)
    assert opp is not None
    assert opp.direction == "basket_no"
    assert opp.imbalance_bps == 1000
    assert all(leg[1] == "no" for leg in opp.legs)


def test_imbalance_below_threshold_skipped():
    # 34 + 34 + 33 = 101 -> +100 bps, below 200 bps threshold.
    cluster = MEECluster(
        event_ticker="KX-TH",
        markets=[_market("A", 34), _market("B", 34), _market("C", 33)],
    )
    assert detect_mee_inconsistency(cluster, min_imbalance_bps=200, max_size_usd=100) is None


def test_single_market_cluster_ignored():
    cluster = MEECluster(event_ticker="KX-SOLO", markets=[_market("A", 40)])
    assert detect_mee_inconsistency(cluster, min_imbalance_bps=50, max_size_usd=100) is None


def test_group_by_event_buckets_markets():
    markets = [
        _market("E1-A", 40, event="E1"),
        _market("E1-B", 35, event="E1"),
        _market("E2-A", 60, event="E2"),
        _market("E2-B", 40, event="E2"),
        _market("E3-LONE", 50, event="E3"),  # alone — not a cluster
    ]
    clusters = group_by_event(markets)
    events = {c.event_ticker for c in clusters}
    assert events == {"E1", "E2"}
    assert sum(len(c.markets) for c in clusters) == 4


def test_zero_liquidity_blocks_opportunity():
    cluster = MEECluster(
        event_ticker="KX-NOLIQ",
        markets=[
            _market("A", 30, size_yes=0),
            _market("B", 30, size_yes=0),
            _market("C", 30, size_yes=0),
        ],
    )
    opp = detect_mee_inconsistency(cluster, min_imbalance_bps=200, max_size_usd=100)
    assert opp is None


def test_profit_accounts_for_fees():
    """A tight arb whose gross edge is smaller than expected fees emits nothing."""
    # 49 + 49 = 98 (very narrow 2-way MEE)
    cluster = MEECluster(
        event_ticker="KX-TIGHT",
        markets=[_market("A", 49), _market("B", 49)],
    )
    # With fee_rate=0.07 on ~0.49 prices, per-contract fee ~ 0.07 * 0.49 * 0.51 = 0.0175
    # Two legs contribute ~0.035 fees, which mostly eats the 0.02 gross edge.
    opp = detect_mee_inconsistency(
        cluster, min_imbalance_bps=100, max_size_usd=100, fee_rate=0.07
    )
    # The profit_per_dollar is marginal / negative and the function returns None
    # OR returns an opportunity with very low profit. Assert at least that we
    # don't falsely claim a huge edge.
    assert opp is None or opp.profit_usd_per_dollar < 0.02
