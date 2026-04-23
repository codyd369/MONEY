"""Quick operational status: config, DB, data, models, recent audit.

Usage:
    uv run python scripts/status.py

Safe to run anytime — no writes, no network.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from moneybutton.core.config import get_settings
from moneybutton.data.store import list_partitions, read_dataset
from moneybutton.models.registry import list_models


def _h(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> int:
    s = get_settings()

    _h("runtime config")
    print(f"  DRY_RUN        = {s.dry_run}")
    print(f"  KALSHI_ENV     = {s.kalshi_env}")
    print(f"  BANKROLL_USD   = {s.bankroll_usd}")
    print(f"  MAX_POSITION   = {s.max_position_usd}")
    print(f"  DAILY_LOSS_LIM = {s.daily_loss_limit_usd}")
    print(f"  KELLY_FRACTION = {s.kelly_fraction}")
    print(f"  timezone       = {s.timezone}")
    print(f"  kill engaged   = {s.kill_file_path.exists()}")
    print(f"  LLM general    = {s.llm_model_general}")
    print(f"  LLM news       = {s.llm_model_news}")
    print(f"  has Anthropic  = {bool(s.anthropic_api_key.get_secret_value())}")
    print(f"  has Gemini     = {bool(s.gemini_api_key.get_secret_value())}")
    print(f"  has Groq       = {bool(s.groq_api_key.get_secret_value())}")
    print(f"  has OpenRouter = {bool(s.openrouter_api_key.get_secret_value())}")
    print(f"  Ollama base    = {s.ollama_base_url}")
    print(f"  has Kalshi id  = {bool(s.kalshi_api_key_id)}")
    print(f"  has Kalshi PEM = {Path(s.kalshi_private_key_path).exists()}")
    print(f"  has Polymarket = {bool(s.polymarket_wallet_private_key.get_secret_value())}")
    print(f"  has NewsAPI    = {bool(s.newsapi_key.get_secret_value())}")
    print(f"  has EventReg   = {bool(s.eventregistry_key.get_secret_value())}")
    print(f"  has OddsAPI    = {bool(s.oddsapi_key.get_secret_value())}")

    db_path = Path(s.sqlite_path)
    _h("sqlite")
    if not db_path.exists():
        print(f"  DB not found at {db_path}. Run: uv run python -m moneybutton.core.db")
    else:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT name, state, allocation_pct, last_backtest_sharpe, "
                "last_backtest_hitrate, last_backtest_max_dd, last_backtest_num_trades "
                "FROM strategies ORDER BY name"
            ).fetchall()
            (n_orders,) = conn.execute("SELECT COUNT(*) FROM orders").fetchone()
            (n_positions,) = conn.execute("SELECT COUNT(*) FROM positions WHERE closed_ts IS NULL").fetchone()
            (n_audit,) = conn.execute("SELECT COUNT(*) FROM audit").fetchone()
        print(f"  {'name':<15} {'state':<10} {'alloc':>6} {'Sharpe':>8} {'hit':>6} {'DD':>6} {'n':>6}")
        for name, state, al, sh, hr, dd, nt in rows:
            sh_s = f"{sh:.3f}" if sh is not None else "-"
            hr_s = f"{hr:.3f}" if hr is not None else "-"
            dd_s = f"{dd:.3f}" if dd is not None else "-"
            nt_s = str(nt) if nt is not None else "-"
            print(f"  {name:<15} {state:<10} {al:>6.2f} {sh_s:>8} {hr_s:>6} {dd_s:>6} {nt_s:>6}")
        print(f"  orders rows: {n_orders}")
        print(f"  open positions: {n_positions}")
        print(f"  audit rows: {n_audit}")

    _h("parquet")
    for dataset in ("markets", "prices", "orderbook", "news", "features"):
        parts = list_partitions(dataset)
        df = read_dataset(dataset) if parts else None
        n_rows = len(df) if df is not None and not df.empty else 0
        print(f"  {dataset:<10} partitions={len(parts):>4d}  rows={n_rows}")

    _h("models")
    models = list_models()
    if not models:
        print("  none. Run: uv run python scripts/train_calibration_v1.py")
    for entry in models:
        print(f"  {entry.path.name}  fp={entry.feature_schema.get('fingerprint', '?')[:12]}...")

    _h("last 10 audit rows")
    if db_path.exists():
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT ts, actor, action, outcome FROM audit ORDER BY id DESC LIMIT 10"
            ).fetchall()
        for ts, actor, action, outcome in rows:
            print(f"  {ts}  {actor:<35}  {action:<20}  {outcome or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
