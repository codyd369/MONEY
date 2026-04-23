"""Structured audit log (SPEC §6.1 `audit` table).

Every consequential action — signal generated, order placed, kill switch
tripped, strategy promoted, article published — writes an audit row with
the full reasoning blob. The dashboard and the weekly_review job read from
this table, so keep the payload strictly JSON-serializable.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any

from moneybutton.core.config import get_settings


def _default_db_path() -> Path:
    return get_settings().sqlite_db_path


def record(
    actor: str,
    action: str,
    *,
    payload: dict[str, Any] | None = None,
    reasoning: str | dict[str, Any] | None = None,
    outcome: str | None = None,
    db_path: str | Path | None = None,
    ts: dt.datetime | None = None,
) -> int:
    """Insert an audit row. Returns the new row id.

    `payload` is JSON-encoded. `reasoning` can be either a string (stored as
    is) or a dict (JSON-encoded). `outcome` is a short human-readable status
    like 'PLACED', 'REJECTED', 'SKIPPED'.
    """
    path = Path(db_path) if db_path is not None else _default_db_path()
    when = (ts or dt.datetime.now(dt.timezone.utc)).isoformat()
    payload_json = json.dumps(payload, default=str) if payload is not None else None
    if isinstance(reasoning, dict):
        reasoning_text = json.dumps(reasoning, default=str)
    else:
        reasoning_text = reasoning

    conn = sqlite3.connect(str(path), timeout=30.0)
    try:
        cur = conn.execute(
            "INSERT INTO audit (ts, actor, action, payload_json, reasoning, outcome)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (when, actor, action, payload_json, reasoning_text, outcome),
        )
        conn.commit()
        return int(cur.lastrowid or 0)
    finally:
        conn.close()


def tail(
    limit: int = 20,
    actor_prefix: str | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return the last `limit` audit rows, newest first. Used by `status` CLI."""
    path = Path(db_path) if db_path is not None else _default_db_path()
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        if actor_prefix:
            rows = conn.execute(
                "SELECT * FROM audit WHERE actor LIKE ? ORDER BY id DESC LIMIT ?",
                (actor_prefix + "%", limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]
