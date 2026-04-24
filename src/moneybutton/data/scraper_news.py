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


def _normalize_ts(raw: str | None) -> str:
    """Accept any publisher date format, return ISO-8601 UTC string.

    RSS / NewsAPI / EventRegistry all use different conventions (RFC 2822
    with named TZ, ISO-8601 with Z, epoch, etc). This helper tries pandas
    first, then dateutil with a named-TZ hint, and falls back to 'now'
    as an absolute last resort so a bad timestamp doesn't poison a write.
    """
    if not raw:
        return dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        import pandas as _pd

        return _pd.to_datetime(raw, utc=True).isoformat()
    except Exception:  # noqa: BLE001
        pass
    try:
        from dateutil import parser as _dp

        _TZ = {
            "EST": -5 * 3600, "EDT": -4 * 3600,
            "CST": -6 * 3600, "CDT": -5 * 3600,
            "MST": -7 * 3600, "MDT": -6 * 3600,
            "PST": -8 * 3600, "PDT": -7 * 3600,
            "UTC": 0, "GMT": 0, "UT": 0, "Z": 0,
            "BST": 3600, "CET": 3600, "CEST": 2 * 3600,
        }
        parsed = _dp.parse(str(raw), tzinfos=_TZ)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc).isoformat()
    except Exception:  # noqa: BLE001
        return dt.datetime.now(dt.timezone.utc).isoformat()


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


def _text_or_dict_first(v) -> str:
    """EventRegistry sometimes returns {'eng': '...', 'fra': '...'} and
    sometimes a plain string. Coerce to a string either way (prefer 'eng',
    fall back to any value, finally empty string). Never returns a dict."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        # Prefer English; else take the first available text value.
        val = v.get("eng")
        if val:
            return str(val)
        for candidate in v.values():
            if candidate:
                return str(candidate)
        return ""
    return str(v)


# ============================== NewsAPI ================================


# NewsAPI Developer (free) plan only serves articles from the last ~30 days
# and rejects older queries with 426 Upgrade Required. Clamp accordingly.
NEWSAPI_MAX_LOOKBACK_DAYS = 28


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
            # Free tier rejects `from` older than ~30 days with 426.
            cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=NEWSAPI_MAX_LOOKBACK_DAYS)
            effective_since = max(since, cutoff)
            params["from"] = effective_since.isoformat()
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
                    "headline": str(a.get("title") or ""),
                    "body": str(a.get("description") or ""),
                    "url": url,
                    "ts_iso": _normalize_ts(a.get("publishedAt")),
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
                    # EventRegistry returns title/summary either as a string
                    # or as a {"eng": "...", "fra": "..."} dict. Coerce.
                    "headline": _text_or_dict_first(e.get("title")),
                    "body": _text_or_dict_first(e.get("summary")),
                    "url": str(e.get("uri") or ""),
                    "ts_iso": _normalize_ts(e.get("eventDate")),
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
                        "ts_iso": _normalize_ts(
                            entry.get("published") or entry.get("updated")
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
