"""News strategy (SPEC §10.5).

Core loop:
  1. Fetch recent news items (last N minutes) from data/parquet/news/.
  2. For each active Kalshi market, score the most recent items for
     relevance via the LLM (brain/prompts/news_relevance.md).
  3. If a high-confidence "material" item exists that points in a clear
     direction, emit a Signal with confidence scaled by the LLM's confidence
     and the staleness of the news.

Guardrails:
  - Only news <= max_age_minutes counts (default 30). Old news is already
    priced in.
  - LLM confidence must clear min_confidence (default 0.7).
  - If the LLM returns "material=false" or direction="unclear", skip.

Backtest: running this properly requires historical news + historical
prices. v1 provides the scanner; a news backtest will be possible once
the scraper runs against real sources and accumulates a history.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable

log = logging.getLogger("moneybutton.news")


@dataclass
class NewsItem:
    id: str
    source: str
    headline: str
    body: str
    url: str
    ts: dt.datetime


@dataclass
class RelevanceScore:
    """What news_relevance.md returns, validated."""

    material: bool
    direction: str  # 'yes' | 'no' | 'unclear'
    confidence: float
    reasoning: str


RelevanceFn = Callable[[NewsItem, dict], RelevanceScore]


@dataclass
class NewsSignal:
    market_ticker: str
    news_id: str
    side: str  # 'yes' | 'no'
    confidence_level: str  # 'low' | 'med' | 'high'
    edge_hint_bps: int  # rough size of expected re-price
    relevance: RelevanceScore
    ts: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


def scan_market_for_news_edge(
    *,
    market: dict,
    recent_news: Iterable[NewsItem],
    relevance_fn: RelevanceFn,
    now: dt.datetime | None = None,
    max_age_minutes: int = 30,
    min_confidence: float = 0.7,
    base_edge_hint_bps: int = 300,
) -> list[NewsSignal]:
    """Return one NewsSignal per material + confident + fresh news item."""
    now = now or dt.datetime.now(dt.timezone.utc)
    max_age = dt.timedelta(minutes=max_age_minutes)

    signals: list[NewsSignal] = []
    for item in recent_news:
        if (now - item.ts) > max_age or item.ts > now:
            continue
        score = relevance_fn(item, market)
        if not score.material or score.direction not in ("yes", "no"):
            continue
        if score.confidence < min_confidence:
            continue
        # Freshness scales edge-hint: newer => bigger edge_hint.
        age_min = max(0.0, (now - item.ts).total_seconds() / 60.0)
        freshness = max(0.1, 1.0 - age_min / max_age_minutes)
        edge_hint = int(round(base_edge_hint_bps * freshness * score.confidence))
        confidence_level = (
            "high" if score.confidence >= 0.85 and age_min <= 10 else ("med" if score.confidence >= 0.7 else "low")
        )
        signals.append(
            NewsSignal(
                market_ticker=market["ticker"],
                news_id=item.id,
                side=score.direction,
                confidence_level=confidence_level,
                edge_hint_bps=edge_hint,
                relevance=score,
            )
        )
    return signals
