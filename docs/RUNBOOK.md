# Operator runbook

Short reference for running this repo locally in PyCharm. Full operator
manual is deferred to SPEC §16 step 27 (README rewrite).

## First-time setup

1. Install `uv` (see https://astral.sh/uv).
2. `uv sync --all-extras` — creates `.venv/` with all deps (~2 GB, 2-5 min).
3. In PyCharm: `File → Settings → Project → Python Interpreter → Add → Existing` → point at `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows).
4. Copy `.env.example` → `.env` and fill in the keys (see the table in the README / operator handoff).
5. Save your Kalshi RSA private key as `data/kalshi_private_key.pem`.

## Safety defaults

- `DRY_RUN=true` — no real orders placed. Flip to `false` only after a strategy is promoted AND you've typed `I ACCEPT THE RISK` in the CLI (CLI lands at step 26).
- `KALSHI_ENV=demo` — routes to `demo-api.kalshi.co`. Use this until you've run the full paper flow for at least a week.
- `./kill.sh` creates `.kill`, blocking all order placement and publishing.
- `./go.sh` removes `.kill` after a 5-second countdown.

## Sanity checks (run these any time)

```
uv run pytest                   # 128 tests should pass
uv run python scripts/status.py # config, DB, data, models, recent audit
```

## Initial data pipeline (first run, real Kalshi)

Sandbox couldn't reach Kalshi; you're running these from your PC.

```
uv run python -m moneybutton.core.db              # init SQLite (idempotent)

uv run python scripts/backfill_markets.py --since 2024-01-01
# Full spec. Resumable. Multi-hour on first run (hundreds of pages).
# Ctrl-C is safe; rerun to resume via saved cursor.

uv run python scripts/backfill_prices.py --top-n 1000
# Candles for the top 1000 settled markets by volume.
# ~1 req/sec, so ~17 min. Skips tickers already on disk.
```

## Train + backtest the calibration model

```
uv run python scripts/train_calibration_v1.py
uv run python scripts/walk_forward_calibration.py
uv run python scripts/backtest_all_strategies.py
```

`data/reports/backtest/calibration/*.html` has the reliability plot + equity curve + drawdown.

## Where things are

| path | what |
|---|---|
| `data/moneybutton.db` | SQLite: strategies, orders, positions, audit |
| `data/parquet/` | Scraped Kalshi data (markets/prices/orderbook/news/features) |
| `data/models/calibration_v*/` | Registered models + metadata + train report |
| `data/reports/backtest/<strategy>/` | HTML reports per backtest run |
| `logs/moneybutton.log` | Rotating notifier log (TimedRotatingFileHandler, 14d) |
| `.kill` | Kill-switch sentinel. Present = all order/publish paths refused. |

## Where "money would leak" (red flags to watch)

1. `scripts/status.py` shows `kill engaged = True` unexpectedly → check last 10 audit rows.
2. `DRY_RUN=false` AND `KALSHI_ENV=prod` AND any strategy in LIVE → live trading IS armed.
3. `data/moneybutton.db` grew by more than ~50MB/day → audit churn; investigate which actor.
4. `data/parquet/features/` grows while no feature-pipeline change was intended → silent schema drift.

## Common PyCharm run configurations

Add these as Run configurations (Run → Edit Configurations → `+` → Python):

| Name | Script path | Parameters |
|---|---|---|
| status | `scripts/status.py` | |
| backfill markets | `scripts/backfill_markets.py` | `--since 2024-01-01` |
| backfill prices | `scripts/backfill_prices.py` | `--top-n 1000` |
| train calibration | `scripts/train_calibration_v1.py` | |
| walk-forward | `scripts/walk_forward_calibration.py` | |
| all strategies | `scripts/backtest_all_strategies.py` | |
| test suite | (set module name to `pytest`) | |

Set the working directory to the repo root and the interpreter to `.venv/`.

## Next steps (SPEC §16 not yet built)

- **Step 23**: content engine (topics/write/publish with dev.to outbox)
- **Step 24**: Streamlit dashboard
- **Step 25**: weekly_review job (wires `brain/prompts/weekly_review.md`)
- **Step 26**: CLI (`python -m moneybutton live|paper|backtest|train|status|data|content|report`)
- **Step 27**: full README rewrite
