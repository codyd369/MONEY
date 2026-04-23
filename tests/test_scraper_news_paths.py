"""Regression: news writer must produce Windows-safe partition paths.

Bug: sources like 'rss:bloomberg_markets' produced 'source=rss:bloomberg_markets'
as a directory name. Windows forbids ':' in filenames (reserves it for drive
letters), so mkdir threw OSError [WinError 123].

Fix: _sanitize_source replaces ':' (and other forbidden Windows chars) with
'_' before constructing the partition path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from moneybutton.data import scraper_news


def test_sanitize_source_removes_colon():
    assert scraper_news._sanitize_source("rss:bloomberg") == "rss_bloomberg"
    assert scraper_news._sanitize_source("eventregistry") == "eventregistry"
    assert scraper_news._sanitize_source("rss:coindesk") == "rss_coindesk"


def test_sanitize_source_removes_other_windows_forbidden():
    assert scraper_news._sanitize_source('bad/name') == "bad_name"
    assert scraper_news._sanitize_source('a\\b') == "a_b"
    assert scraper_news._sanitize_source('q?x') == "q_x"
    assert scraper_news._sanitize_source('a<b>c') == "a_b_c"
    assert scraper_news._sanitize_source('p|ipe') == "p_ipe"


def test_write_rows_with_colon_in_source_succeeds(tmp_env):
    """End-to-end: a source containing ':' (as RSS scraper produces) must
    write to disk without an OSError. Before the fix this was the exact
    crash the operator hit on Windows."""
    rows = [
        {
            "id": "rss:bloomberg:abc123",
            "source": "rss:bloomberg_markets",
            "headline": "Test headline",
            "body": "Test body",
            "url": "https://example.com/1",
            "ts_iso": "2025-09-02T18:00:00+00:00",
            "raw_json": "{}",
        }
    ]
    n = scraper_news._write_rows("rss:bloomberg_markets", rows)
    assert n == 1

    # The partition folder should be named with sanitized source.
    from moneybutton.core.config import get_settings

    data_dir = Path(get_settings().data_dir)
    partitions = list((data_dir / "parquet" / "news").glob("source=*"))
    assert len(partitions) == 1
    folder_name = partitions[0].name
    assert ":" not in folder_name
    assert folder_name == "source=rss_bloomberg_markets"


def test_empty_rows_returns_zero_no_crash(tmp_env):
    assert scraper_news._write_rows("rss:any", []) == 0


# ------------------------ timestamp normalization -------------------------


def test_normalize_ts_handles_rfc2822_with_named_tz():
    """RSS feeds publish dates with named TZ like 'EST' that pandas can't
    handle. _normalize_ts must accept them."""
    out = scraper_news._normalize_ts("Thu, 23 Apr 2026 16:16:17 EST")
    # Must be a valid ISO-8601 UTC string (pandas can parse it back).
    import pandas as pd

    parsed = pd.to_datetime(out, utc=True)
    # 16:16 EST = 21:16 UTC.
    assert parsed.hour == 21
    assert parsed.day == 23
    assert parsed.month == 4
    assert parsed.year == 2026


def test_normalize_ts_handles_iso_with_z():
    out = scraper_news._normalize_ts("2026-04-23T16:16:17Z")
    import pandas as pd

    parsed = pd.to_datetime(out, utc=True)
    assert parsed.hour == 16


def test_normalize_ts_handles_already_normalized_iso_offset():
    out = scraper_news._normalize_ts("2026-04-23T16:16:17+00:00")
    import pandas as pd

    assert pd.to_datetime(out, utc=True).hour == 16


def test_normalize_ts_handles_empty_and_garbage():
    # Empty falls back to "now" — valid ISO string either way.
    for bad in (None, "", "  ", "not a date at all"):
        out = scraper_news._normalize_ts(bad)
        import pandas as pd

        # Must parse without error.
        pd.to_datetime(out, utc=True)


def test_year_month_handles_named_tz():
    """Regression: operator hit ValueError 'un-recognized timezone EST'
    on a real RSS publishedAt string. year_month must absorb it."""
    from moneybutton.data.store import year_month

    # The exact string from the operator's trace.
    assert year_month("Thu, 23 Apr 2026 16:16:17 EST") == "2026-04"


def test_year_month_handles_garbage_without_raising():
    """Any totally unparseable string falls back to the current UTC month
    so a single bad timestamp never aborts a batch write."""
    import datetime as _dt

    from moneybutton.data.store import year_month

    out = year_month("complete nonsense")
    # Must be a valid YYYY-MM string.
    assert len(out) == 7
    assert out[4] == "-"
    # Should be today's UTC month (within reason — test may straddle a month
    # boundary, so we accept either the current or next month).
    now = _dt.datetime.now(_dt.timezone.utc)
    acceptable = {
        now.strftime("%Y-%m"),
        (now + _dt.timedelta(days=1)).strftime("%Y-%m"),
        (now - _dt.timedelta(days=1)).strftime("%Y-%m"),
    }
    assert out in acceptable
