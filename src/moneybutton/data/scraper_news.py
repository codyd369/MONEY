"""News scrapers: NewsAPI, EventRegistry, and RSS (SPEC §10.5).

Each scraper normalizes raw feed items into our standard schema:
  - id (source-prefixed, stable): e.g. newsapi:<url-hash>
  - source: 'newsapi' | 'eventregistry' | 'rss:<feed-slug>'
  - headline, body, url, ts_iso
  - raw: the original payload, kept for LLM context

Scrapers persist to data/parquet/news/source={src}/year_month={ym}/...
Caller is responsible for:
  - Respecting rate limits (NewsAPI free: 100 req/day; EventRegistry free:
    2000 events/day).
  - Passing `since_ts` to resume from the last successful fetch.

Note: this module makes outbound HTTP calls; they are skipped on import
and only happen when a scraper method is called explicitly.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import httpx

from moneybutton.core.config import Settings, get_settings
from moneybutton.data.store import PartitionKey, write_partition, year_month

log = logging.getLogger("moneybutton.scraper.news")

NEWS_SCHEMA_COLS = ["id", "source", "headline", "body", "url", "ts_iso", "raw_json"]


def _sanitize_source(s: str) -> str:
    """Make a source name safe for a filesystem path (Windows forbids ':')."""
    return (
        s.replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("?", "_")
        .replace("*", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def _hash_id(source: str, url_or_id: str) -> str:
    h = hashlib.sha256(url_or_id.encode("utf-8")).hexdigest()[:24]
    # The id still uses ':' as a separator (fine for SQLite text columns); only
    # the partition path is sanitized.
    return f"{source}:{h}"


def _write_rows(source: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    import pandas as pd

    safe_source = _sanitize_source(source)
    frame = pd.DataFrame(rows, columns=NEWS_SCHEMA_COLS)
    # Bucket by year_month of ts_iso for partitioning.
    written = 0
    for ym, group in frame.groupby(frame["ts_iso"].map(year_month)):
        write_partition(
            PartitionKey(dataset="news", category_or_source=safe_source, year_month=ym),
            group,
        )
        written += len(group)
    return written


# ============================== NewsAPI ================================


@dataclass
class NewsAPIScraper:
    settings: Settings = field(default_factory=get_settings)
    client: httpx.Client | None = None
    base_url: str = "https://newsapi.org/v2"

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = httpx.Client(timeout=30.0)

    def _key(self) -> str:
        return self.settings.newsapi_key.get_secret_value()

    def fetch_headlines(
        self,
        *,
        q: str | None = None,
        sources: str | None = None,
        since: dt.datetime | None = None,
        page_size: int = 50,
    ) -> list[dict]:
        params: dict = {"apiKey": self._key(), "pageSize": page_size}
        if q:
            params["q"] = q
        if sources:
            params["sources"] = sources
        if since:
            params["from"] = since.isoformat()
        assert self.client is not None
        r = self.client.get(f"{self.base_url}/everything", params=params)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        rows = []
        for a in articles:
            url = a.get("url", "")
            if not url:
                continue
            rows.append(
                {
                    "id": _hash_id("newsapi", url),
                    "source": "newsapi",
                    "headline": a.get("title") or "",
                    "body": a.get("description") or "",
                    "url": url,
                    "ts_iso": a.get("publishedAt") or dt.datetime.now(dt.timezone.utc).isoformat(),
                    "raw_json": str(a),
                }
            )
        return rows

    def fetch_and_store(self, *, q: str | None = None, since: dt.datetime | None = None) -> int:
        rows = self.fetch_headlines(q=q, since=since)
        return _write_rows("newsapi", rows)


# ============================= EventRegistry ===========================


@dataclass
class EventRegistryScraper:
    settings: Settings = field(default_factory=get_settings)
    client: httpx.Client | None = None
    base_url: str = "https://eventregistry.org/api/v1"

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = httpx.Client(timeout=30.0)

    def _key(self) -> str:
        return self.settings.eventregistry_key.get_secret_value()

    def fetch_events(self, *, keyword: str, since: dt.datetime | None = None, count: int = 50) -> list[dict]:
        body: dict = {
            "apiKey": self._key(),
            "keyword": keyword,
            "eventsCount": count,
            "resultType": "events",
        }
        if since:
            body["dateStart"] = since.date().isoformat()
        assert self.client is not None
        r = self.client.post(f"{self.base_url}/event/getEvents", json=body)
        r.raise_for_status()
        data = r.json()
        events = data.get("events", {}).get("results", [])
        rows = []
        for e in events:
            rows.append(
                {
                    "id": _hash_id("eventregistry", str(e.get("uri", ""))),
                    "source": "eventregistry",
                    "headline": e.get("title", {}).get("eng") or e.get("title", "") or "",
                    "body": e.get("summary", {}).get("eng") or e.get("summary", "") or "",
                    "url": e.get("uri", ""),
                    "ts_iso": e.get("eventDate") or dt.datetime.now(dt.timezone.utc).isoformat(),
                    "raw_json": str(e),
                }
            )
        return rows

    def fetch_and_store(self, *, keyword: str, since: dt.datetime | None = None) -> int:
        rows = self.fetch_events(keyword=keyword, since=since)
        return _write_rows("eventregistry", rows)


# ================================ RSS ==================================


@dataclass
class RSSScraper:
    feeds: list[tuple[str, str]] = field(default_factory=list)  # (slug, url)
    client: httpx.Client | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            # Many news sites 302/308/301 their feed URLs; httpx defaults to
            # NOT following redirects (unlike requests). Turn that on.
            # Some sites also 403 a bare-bones client; send a plausible UA.
            self.client = httpx.Client(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; moneybutton-rss-reader/0.1; "
                        "+https://github.com/codyd369/money)"
                    ),
                    "Accept": "application/rss+xml, application/atom+xml, text/xml, */*",
                },
            )

    def fetch_all(self) -> dict[str, list[dict]]:
        import feedparser

        out: dict[str, list[dict]] = {}
        for slug, url in self.feeds:
            assert self.client is not None
            try:
                r = self.client.get(url)
                r.raise_for_status()
                parsed = feedparser.parse(r.content)
            except Exception as e:  # noqa: BLE001
                log.warning("RSS fetch failed for %s: %s", slug, e)
                continue
            rows = []
            for entry in parsed.entries:
                link = entry.get("link") or ""
                rows.append(
                    {
                        "id": _hash_id(f"rss:{slug}", link),
                        "source": f"rss:{slug}",
                        "headline": entry.get("title") or "",
                        "body": entry.get("summary") or "",
                        "url": link,
                        "ts_iso": (
                            entry.get("published") or entry.get("updated") or dt.datetime.now(dt.timezone.utc).isoformat()
                        ),
                        "raw_json": str(dict(entry)),
                    }
                )
            out[slug] = rows
        return out

    def fetch_and_store(self) -> int:
        total = 0
        for slug, rows in self.fetch_all().items():
            total += _write_rows(f"rss:{slug}", rows)
        return total
