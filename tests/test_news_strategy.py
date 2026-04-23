"""News strategy tests (SPEC §10.5)."""

from __future__ import annotations

import datetime as dt

from moneybutton.strategies.news_strat import (
    NewsItem,
    RelevanceScore,
    scan_market_for_news_edge,
)


def _market():
    return {
        "ticker": "KX-FED-HIKE",
        "title": "Will the Fed hike in March?",
        "rules_primary": "Resolves yes if ...",
    }


def _news_item(minutes_ago: int = 5, id_: str = "n-1", headline: str = "Fed holds steady"):
    return NewsItem(
        id=id_,
        source="rss:reuters",
        headline=headline,
        body="body",
        url="https://example.com/1",
        ts=dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes_ago),
    )


def test_material_confident_fresh_produces_signal():
    market = _market()

    def relevance(item, mkt):
        return RelevanceScore(material=True, direction="no", confidence=0.9, reasoning="held steady")

    signals = scan_market_for_news_edge(
        market=market,
        recent_news=[_news_item(minutes_ago=2)],
        relevance_fn=relevance,
    )
    assert len(signals) == 1
    s = signals[0]
    assert s.side == "no"
    assert s.confidence_level == "high"
    assert s.edge_hint_bps > 0


def test_stale_news_skipped():
    def rel(item, mkt):
        return RelevanceScore(material=True, direction="yes", confidence=0.95, reasoning="x")

    signals = scan_market_for_news_edge(
        market=_market(),
        recent_news=[_news_item(minutes_ago=120)],  # > 30 min default
        relevance_fn=rel,
    )
    assert signals == []


def test_low_confidence_skipped():
    def rel(item, mkt):
        return RelevanceScore(material=True, direction="yes", confidence=0.4, reasoning="maybe")

    signals = scan_market_for_news_edge(
        market=_market(),
        recent_news=[_news_item()],
        relevance_fn=rel,
    )
    assert signals == []


def test_unclear_direction_skipped():
    def rel(item, mkt):
        return RelevanceScore(material=True, direction="unclear", confidence=0.9, reasoning="ambiguous")

    signals = scan_market_for_news_edge(
        market=_market(),
        recent_news=[_news_item()],
        relevance_fn=rel,
    )
    assert signals == []


def test_immaterial_skipped():
    def rel(item, mkt):
        return RelevanceScore(material=False, direction="yes", confidence=0.99, reasoning="off-topic")

    signals = scan_market_for_news_edge(
        market=_market(),
        recent_news=[_news_item()],
        relevance_fn=rel,
    )
    assert signals == []


def test_edge_hint_scales_with_freshness():
    def rel(item, mkt):
        return RelevanceScore(material=True, direction="yes", confidence=0.9, reasoning="x")

    fresh_signal = scan_market_for_news_edge(
        market=_market(),
        recent_news=[_news_item(minutes_ago=1)],
        relevance_fn=rel,
    )[0]
    stale_signal = scan_market_for_news_edge(
        market=_market(),
        recent_news=[_news_item(minutes_ago=25)],
        relevance_fn=rel,
    )[0]
    assert fresh_signal.edge_hint_bps > stale_signal.edge_hint_bps


def test_confidence_level_tiers():
    def rel_for(c):
        def _r(item, mkt):
            return RelevanceScore(material=True, direction="yes", confidence=c, reasoning="x")
        return _r

    for c, expected in [(0.72, "med"), (0.9, "high"), (0.65, "low")]:
        signals = scan_market_for_news_edge(
            market=_market(),
            recent_news=[_news_item(minutes_ago=2)],
            relevance_fn=rel_for(c),
            min_confidence=0.5,  # allow 0.65 through for the 'low' tier
        )
        if expected == "low":
            assert signals == [] or signals[0].confidence_level == "low"
        else:
            assert signals[0].confidence_level == expected
