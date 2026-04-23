"""SQLite operational store.

Schema mirrors SPEC.md §6.1. Migrations are idempotent: `init_db()` runs all
CREATE TABLE IF NOT EXISTS statements plus any additive ALTERs, and records
the applied version in a schema_version row. Destructive migrations are
intentionally not supported — operational data is easier to rebuild from
scraped parquet than to migrate in place.

The `connect()` context manager is the one correct way to open the DB:
- Enables foreign keys.
- Sets WAL for concurrent reads.
- Commits on clean exit, rolls back on exception.
- Returns sqlite3.Row so callers can access columns by name.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from moneybutton.core.config import STRATEGY_NAMES, get_settings

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER PRIMARY KEY,
  applied_ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bankroll (
  ts TEXT PRIMARY KEY,
  balance_usd REAL NOT NULL,
  source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategies (
  name TEXT PRIMARY KEY,
  state TEXT NOT NULL CHECK (state IN ('SHADOW', 'LIVE', 'DISABLED')),
  allocation_pct REAL NOT NULL,
  last_backtest_ts TEXT,
  last_backtest_sharpe REAL,
  last_backtest_hitrate REAL,
  last_backtest_max_dd REAL,
  last_backtest_num_trades INTEGER,
  promoted_ts TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  strategy TEXT NOT NULL,
  ticker TEXT NOT NULL,
  side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
  edge_bps INTEGER NOT NULL,
  confidence TEXT CHECK (confidence IN ('low', 'med', 'high')),
  reasoning TEXT,
  acted_on INTEGER NOT NULL DEFAULT 0,
  client_request_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_strategy_ts ON signals(strategy, ts);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);

CREATE TABLE IF NOT EXISTS orders (
  client_request_id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  strategy TEXT NOT NULL,
  ticker TEXT NOT NULL,
  side TEXT,
  action TEXT CHECK (action IN ('buy', 'sell')),
  count INTEGER,
  limit_price_cents INTEGER,
  kalshi_order_id TEXT,
  status TEXT,
  fill_price_cents INTEGER,
  notional_usd REAL,
  signal_id INTEGER REFERENCES signals(id)
);
CREATE INDEX IF NOT EXISTS idx_orders_strategy_ts ON orders(strategy, ts);
CREATE INDEX IF NOT EXISTS idx_orders_ticker ON orders(ticker);

CREATE TABLE IF NOT EXISTS positions (
  ticker TEXT NOT NULL,
  strategy TEXT NOT NULL,
  side TEXT NOT NULL,
  count INTEGER NOT NULL,
  avg_entry_cents INTEGER NOT NULL,
  opened_ts TEXT NOT NULL,
  unrealized_usd REAL DEFAULT 0,
  realized_usd REAL DEFAULT 0,
  closed_ts TEXT,
  PRIMARY KEY (ticker, strategy)
);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy);

CREATE TABLE IF NOT EXISTS daily_pnl (
  date TEXT NOT NULL,
  strategy TEXT NOT NULL,
  realized_usd REAL DEFAULT 0,
  unrealized_usd REAL DEFAULT 0,
  PRIMARY KEY (date, strategy)
);

CREATE TABLE IF NOT EXISTS articles (
  slug TEXT PRIMARY KEY,
  ts_drafted TEXT NOT NULL,
  ts_published TEXT,
  topic_tool_name TEXT,
  title TEXT,
  body_md TEXT,
  affiliate_links_json TEXT,
  platforms_published_json TEXT,
  outbox_paths_json TEXT
);

CREATE TABLE IF NOT EXISTS affiliate_links (
  tool_slug TEXT PRIMARY KEY,
  url TEXT NOT NULL,
  program TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  actor TEXT,
  action TEXT,
  payload_json TEXT,
  reasoning TEXT,
  outcome TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit(actor);

-- scraper_state: resumable cursor for long-running historical backfills.
CREATE TABLE IF NOT EXISTS scraper_state (
  scraper TEXT PRIMARY KEY,
  cursor TEXT,
  last_completed_partition TEXT,
  updated_ts TEXT NOT NULL
);
"""


@contextlib.contextmanager
def connect(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the operational SQLite DB with safe defaults.

    - WAL + synchronous=NORMAL for good concurrency and durability.
    - Foreign keys enforced.
    - Row factory set so callers can access columns by name.
    """
    path = Path(db_path) if db_path is not None else get_settings().sqlite_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")
    finally:
        conn.close()


def init_db(db_path: str | Path | None = None) -> None:
    """Create tables if missing. Safe to call on every startup."""
    path = Path(db_path) if db_path is not None else get_settings().sqlite_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Separate connection: we want executescript autocommit semantics.
    conn = sqlite3.connect(str(path), timeout=30.0)
    try:
        conn.executescript(SCHEMA_SQL)
        cur = conn.execute("SELECT MAX(version) FROM schema_version")
        row = cur.fetchone()
        current = row[0] if row and row[0] is not None else 0
        if current < SCHEMA_VERSION:
            conn.execute(
                "INSERT INTO schema_version (version, applied_ts) VALUES (?, datetime('now'))",
                (SCHEMA_VERSION,),
            )
        conn.commit()
        _seed_strategy_rows(conn)
    finally:
        conn.close()


def _seed_strategy_rows(conn: sqlite3.Connection) -> None:
    """Ensure a row exists in `strategies` for every named strategy.

    Every strategy starts in SHADOW state with its configured allocation.
    Re-running is a no-op for rows that already exist.
    """
    settings = get_settings()
    for name in STRATEGY_NAMES:
        allocation = settings.allocations[name]
        conn.execute(
            """
            INSERT OR IGNORE INTO strategies (name, state, allocation_pct, notes)
            VALUES (?, 'SHADOW', ?, 'seeded on init_db')
            """,
            (name, allocation),
        )
    conn.commit()


def table_names(db_path: str | Path | None = None) -> list[str]:
    """Utility used by tests and the status CLI."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [r["name"] for r in rows]


if __name__ == "__main__":
    # `python -m moneybutton.core.db` initializes the DB and prints the schema.
    init_db()
    print(f"Initialized DB at {get_settings().sqlite_db_path}")
    for name in table_names():
        print(f"  - {name}")
