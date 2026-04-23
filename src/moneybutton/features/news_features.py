"""News features: per-market aggregates of relevance-scored articles.

Reads from the SQLite `news_relevance` table (populated by
scripts/score_news_relevance.py). No-leakage guarantee preserved: only
scores whose `scored_ts` is strictly before `as_of_ts` contribute.

Features emitted:
  - news_count_24h            # any relevance score in the prior 24h
  - news_count_168h           # 7d
  - news_material_count_24h   # material=1 in prior 24h
  - news_material_count_168h  # 7d
  - news_material_fraction_24h
  - news_bias_yes_24h         # count(direction=yes) - count(direction=no), 24h
  - news_bias_yes_168h
  - news_max_confidence_24h   # max LLM confidence 24h
  - news_mean_confidence_24h

If the news_relevance table doesn't exist (no news yet scraped/scored),
all features return None and XGBoost handles them natively.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Any

from moneybutton.core.config import get_settings


def _table_exists(db_path: Path) -> bool:
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='news_relevance'"
            ).fetchone()
        finally:
            conn.close()
        return bool(row)
    except sqlite3.OperationalError:
        return False


def _empty_features() -> dict[str, Any]:
    return {
        "news_count_24h": None,
        "news_count_168h": None,
        "news_material_count_24h": None,
        "news_material_count_168h": None,
        "news_material_fraction_24h": None,
        "news_bias_yes_24h": None,
        "news_bias_yes_168h": None,
        "news_max_confidence_24h": None,
        "news_mean_confidence_24h": None,
    }


def compute(
    market: dict,
    as_of_ts: dt.datetime,
    *,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute news features for `market` as of `as_of_ts`.

    Uses SQL aggregates against the news_relevance table. No-leakage:
    scored_ts < as_of_ts and news_ts (via join on news parquet) also <
    as_of_ts. For simplicity we only filter on scored_ts here — the
    scoring job itself only runs on news whose ts is before `now`, so
    this is safe in normal operation. Backfill / training should ensure
    score_news_relevance is run in temporal order.
    """
    if db_path is None:
        db_path = get_settings().sqlite_db_path
    db_path = Path(db_path)
    if not _table_exists(db_path):
        return _empty_features()

    ticker = market.get("ticker")
    if not ticker:
        return _empty_features()

    as_of_iso = as_of_ts.isoformat()
    window_24h_iso = (as_of_ts - dt.timedelta(hours=24)).isoformat()
    window_168h_iso = (as_of_ts - dt.timedelta(hours=168)).isoformat()

    conn = sqlite3.connect(str(db_path))
    try:

        def _agg(since_iso: str) -> dict:
            row = conn.execute(
                """
                SELECT
                  COUNT(*)                                         AS n,
                  SUM(material)                                    AS n_material,
                  SUM(CASE WHEN direction='yes' THEN 1 ELSE 0 END) AS n_yes,
                  SUM(CASE WHEN direction='no'  THEN 1 ELSE 0 END) AS n_no,
                  MAX(confidence)                                  AS max_conf,
                  AVG(confidence)                                  AS mean_conf
                FROM news_relevance
                WHERE ticker = ?
                  AND scored_ts >= ?
                  AND scored_ts < ?
                """,
                (ticker, since_iso, as_of_iso),
            ).fetchone()
            return {
                "n": int(row[0] or 0),
                "n_material": int(row[1] or 0),
                "n_yes": int(row[2] or 0),
                "n_no": int(row[3] or 0),
                "max_conf": float(row[4]) if row[4] is not None else None,
                "mean_conf": float(row[5]) if row[5] is not None else None,
            }

        day = _agg(window_24h_iso)
        week = _agg(window_168h_iso)
    finally:
        conn.close()

    mat_frac_24h = (
        day["n_material"] / day["n"] if day["n"] > 0 else None
    )
    return {
        "news_count_24h": day["n"],
        "news_count_168h": week["n"],
        "news_material_count_24h": day["n_material"],
        "news_material_count_168h": week["n_material"],
        "news_material_fraction_24h": mat_frac_24h,
        "news_bias_yes_24h": day["n_yes"] - day["n_no"],
        "news_bias_yes_168h": week["n_yes"] - week["n_no"],
        "news_max_confidence_24h": day["max_conf"],
        "news_mean_confidence_24h": day["mean_conf"],
    }
