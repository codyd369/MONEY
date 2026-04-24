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

## News scraping + relevance scoring

```
# Free RSS feeds always. Uses NEWSAPI_KEY / EVENTREGISTRY_KEY if set in .env.
# Default --since is 7 days ago; override for backfill.
uv run python scripts/backfill_news.py --since 2025-01-01

# Score news relevance against markets using the LLM (settings.llm_model_news,
# default anthropic/claude-haiku-4-5; set to gemini/gemini-2.0-flash-exp for
# free). Keyword pre-filter drops ~99% of pairs before any LLM call.
uv run python scripts/score_news_relevance.py --max-pairs 2000

# --max-pairs caps total LLM calls per run for cost control. The score table
# dedupes on (news_id, ticker) so rerunning is additive.
```

News features (news_count_24h, news_material_count_24h, news_bias_yes_24h,
news_max_confidence_24h, and 168h variants) are automatically added to
the training frame by `features/pipeline.py`. If no news is on disk, the
features are all None and XGBoost handles them natively.

### Picking an LLM provider for news scoring

Observed rate limits on free tiers (2026):

| Provider | Model | Free RPM | Free TPD | Notes |
|---|---|---:|---:|---|
| Groq | `llama-3.1-8b-instant` | 30 | 500k | Best free option for bulk scoring |
| Groq | `llama-3.3-70b-versatile` | 30 | 100k | Stronger but burns TPD fast |
| Gemini | `gemini-2.5-flash-lite` | 10-15 | ~250k | Strict on some accounts |
| Gemini | `gemini-2.5-flash` | 5-10 | ~250k | Often tighter than docs suggest |
| Anthropic | `claude-haiku-4-5` | — | — | Paid, ~$0.25/M in / $1.25/M out, unthrottled |

With the trimmed news-relevance prompt (~500 tokens/call), approximate
pairs-scorable-per-day per model:

  * groq/llama-3.1-8b-instant   1000 pairs/day
  * groq/llama-3.3-70b-versatile  200 pairs/day
  * anthropic/claude-haiku-4-5  unlimited, ~$0.10 per 500 pairs

For bulk scoring (thousands of pairs to build training features), use
8B instant. For live news-relevance on tens of markets per day, 70B
versatile or Haiku is fine.

To switch, update `.env`:

```
LLM_MODEL_NEWS=groq/llama-3.1-8b-instant
GROQ_API_KEY=gsk_...   # from console.groq.com
```

`score_news_relevance.py` auto-picks a safe per-call sleep based on the
model string. Override with `--rate-limit-sleep-s N` if your free tier
is higher than the defaults.

## Train + backtest the calibration model

```
uv run python scripts/train_calibration_v1.py
# --max-per-series 100 (default) caps any single ticker-series so auto-gen
#   series (hourly weather, parlay) don't dominate the loss function.
# --min-trade-rows 5 drops markets that never actually traded.
# --categories Politics Economics Sports restricts the training set.

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
