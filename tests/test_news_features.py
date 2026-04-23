"""News-feature tests — reads news_relevance SQLite table and emits
per-market aggregates. Must not leak across the as_of boundary."""

from __future__ import annotations

import datetime as dt
import sqlite3

import pytest

from moneybutton.features import news_features


def _seed_relevance(db_path, rows: list[dict]) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS news_relevance (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          scored_ts TEXT NOT NULL,
          news_id TEXT NOT NULL,
          ticker TEXT NOT NULL,
          material INTEGER NOT NULL,
          direction TEXT,
          confidence REAL,
          reasoning TEXT,
          model TEXT
        );
        """
    )
    for r in rows:
        conn.execute(
            """INSERT INTO news_relevance
               (scored_ts, news_id, ticker, material, direction, confidence, reasoning, model)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["scored_ts"],
                r["news_id"],
                r["ticker"],
                int(r["material"]),
                r.get("direction"),
                r.get("confidence"),
                r.get("reasoning"),
                r.get("model", "gemini/gemini-2.0-flash-exp"),
            ),
        )
    conn.commit()
    conn.close()


def test_no_table_returns_all_none(tmp_env, tmp_db):
    # fresh DB has no news_relevance table; init_db only seeds the core schema.
    import sqlite3

    conn = sqlite3.connect(str(tmp_db))
    conn.execute("DROP TABLE IF EXISTS news_relevance")
    conn.commit()
    conn.close()

    feats = news_features.compute(
        {"ticker": "KX-TEST"},
        dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc),
        db_path=tmp_db,
    )
    assert all(v is None for v in feats.values())


def test_counts_within_windows(tmp_env, tmp_db):
    now = dt.datetime(2025, 6, 1, 12, tzinfo=dt.timezone.utc)

    def iso(hours_ago: int) -> str:
        return (now - dt.timedelta(hours=hours_ago)).isoformat()

    _seed_relevance(
        tmp_db,
        [
            # Within 24h:
            {"scored_ts": iso(1), "news_id": "n1", "ticker": "KX-X", "material": 1,
             "direction": "yes", "confidence": 0.9},
            {"scored_ts": iso(3), "news_id": "n2", "ticker": "KX-X", "material": 1,
             "direction": "no", "confidence": 0.7},
            {"scored_ts": iso(5), "news_id": "n3", "ticker": "KX-X", "material": 0,
             "direction": "unclear", "confidence": 0.4},
            # Within 168h but outside 24h:
            {"scored_ts": iso(48), "news_id": "n4", "ticker": "KX-X", "material": 1,
             "direction": "yes", "confidence": 0.8},
            # Outside 168h (should NOT be counted):
            {"scored_ts": iso(200), "news_id": "n5", "ticker": "KX-X", "material": 1,
             "direction": "yes", "confidence": 0.95},
            # Different ticker (should NOT be counted):
            {"scored_ts": iso(2), "news_id": "n6", "ticker": "KX-OTHER", "material": 1,
             "direction": "yes", "confidence": 0.9},
        ],
    )

    feats = news_features.compute({"ticker": "KX-X"}, now, db_path=tmp_db)
    assert feats["news_count_24h"] == 3
    assert feats["news_count_168h"] == 4
    assert feats["news_material_count_24h"] == 2
    assert feats["news_material_count_168h"] == 3
    assert abs(feats["news_material_fraction_24h"] - 2 / 3) < 1e-9
    # 24h bias: 1 yes - 1 no = 0 (unclear doesn't count)
    assert feats["news_bias_yes_24h"] == 0
    # 168h bias: 2 yes - 1 no = 1
    assert feats["news_bias_yes_168h"] == 1
    assert feats["news_max_confidence_24h"] == pytest.approx(0.9)
    assert feats["news_mean_confidence_24h"] == pytest.approx((0.9 + 0.7 + 0.4) / 3)


def test_no_leakage_future_scores_excluded(tmp_env, tmp_db):
    as_of = dt.datetime(2025, 6, 1, 12, tzinfo=dt.timezone.utc)
    _seed_relevance(
        tmp_db,
        [
            # Scored 1h before as_of -> counted.
            {"scored_ts": (as_of - dt.timedelta(hours=1)).isoformat(),
             "news_id": "past", "ticker": "KX-X", "material": 1,
             "direction": "yes", "confidence": 0.9},
            # Scored AT as_of -> excluded (strict <).
            {"scored_ts": as_of.isoformat(),
             "news_id": "present", "ticker": "KX-X", "material": 1,
             "direction": "yes", "confidence": 0.9},
            # Scored 1h AFTER as_of -> excluded (leakage guard).
            {"scored_ts": (as_of + dt.timedelta(hours=1)).isoformat(),
             "news_id": "future", "ticker": "KX-X", "material": 1,
             "direction": "yes", "confidence": 0.9},
        ],
    )
    feats = news_features.compute({"ticker": "KX-X"}, as_of, db_path=tmp_db)
    assert feats["news_count_24h"] == 1  # only "past"


def test_ticker_without_news_emits_zeros(tmp_env, tmp_db):
    _seed_relevance(
        tmp_db,
        [
            {"scored_ts": "2025-06-01T10:00:00+00:00", "news_id": "n1", "ticker": "KX-A",
             "material": 1, "direction": "yes", "confidence": 0.9},
        ],
    )
    feats = news_features.compute(
        {"ticker": "KX-B"},  # different ticker; table exists but no rows
        dt.datetime(2025, 6, 1, 12, tzinfo=dt.timezone.utc),
        db_path=tmp_db,
    )
    assert feats["news_count_24h"] == 0
    assert feats["news_material_count_24h"] == 0
    assert feats["news_max_confidence_24h"] is None  # MAX over 0 rows -> None


def test_market_missing_ticker():
    feats = news_features.compute({}, dt.datetime.now(dt.timezone.utc), db_path=None)
    assert all(v is None for v in feats.values())
