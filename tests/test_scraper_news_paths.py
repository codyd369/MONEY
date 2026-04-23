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
