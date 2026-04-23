# The Money Button v2 — Edge Engine Build Spec

> **How to use this file:** Save this as `SPEC.md` in an empty directory. Open Claude Code in that directory (`claude` in your terminal). Then paste this single instruction:
>
> > Read `SPEC.md` end-to-end before writing any code. Ask me every question in the "Questions you must ask before coding" section. Once I answer, build the project according to the spec, in the order given in the "Build order" section. Implement the safety layer first with tests, then the data layer, then ONE strategy end-to-end (calibration), backtest it, show me results, and only then proceed to the other four strategies. Confirm with me before installing dependencies and before any network call that touches a real account or costs money.

---

## 1. What you are building

A locally-hosted, single-developer quantitative trading system for Kalshi prediction markets, with five independent edge strategies that share infrastructure, plus a secondary AI-dev-tools content/affiliate engine.

The core insight driving this design: a generic LLM scoring "is this market mispriced?" is not an edge — the LLM is downstream of the same news the market is pricing on and is poorly calibrated relative to a market with skin in the game. **Real edge in markets like Kalshi comes from four sources, none of which is "ask Claude what it thinks":**

1. **Statistical edge** — being better-calibrated than the market through models trained on historical resolutions, applied selectively to low-liquidity markets where the crowd is small or unsophisticated.
2. **Structural edge** — exploiting market mechanics: cross-platform price differences, conditional-probability violations between related markets, end-of-life price drift patterns.
3. **Information edge** — processing relevant information faster than the market through news-reaction systems with an embedding-based mapping layer.
4. **Behavioral edge** — exploiting documented biases (long-shot, round-number, recency).

This system implements all four through five strategies. They share a safety core, an executor, a data layer, and a backtesting framework. Each strategy must pass a backtest gate before it gets to trade live capital. The LLM is used for narrative summaries, news relevance scoring, and research — never as the primary signal generator.

The content engine (AI dev tools reviews, dev.to auto-publish, drafts to outbox for other platforms) is preserved as a secondary revenue stream because (a) it is genuinely the higher-EV channel for a solo operator with $500 of trading capital, and (b) it doubles as audience-building for any future productization of the trading system.

### Operator profile (assume this in every design decision)
- Solo developer, US-based (Georgia).
- Comfortable running Node/Python locally; not running a VPS, Docker swarm, k8s, or any cloud infrastructure.
- Total at-risk trading capital: **≤ $500**, allocated across strategies.
- Already has: a Kalshi or Polymarket account; an Anthropic API key with credits.
- Does not yet have: paid news APIs, X API access, sportsbook accounts, affiliate program memberships, a domain, hosting, or an existing audience. Design for this — bootstrap each as needed but never block.

### Honesty contract — surface this to the operator before any trading goes live
1. **No strategy is a money printer.** Realistic per-strategy edge is 1–5% per identified trade, with hit rates of 52–60%. Variance dominates expected value over any timeframe shorter than ~6 months of consistent trading. Drawdowns of 20–30% are normal even with real edge.
2. **Backtest performance is an upper bound, not an expectation.** Live performance is typically 30–60% worse than backtest due to slippage, fee surprises, market regime changes, and overfit features the backtest didn't catch. Any backtest showing >2.0 Sharpe is almost certainly overfit and the bot must surface this warning.
3. **The system improves with iteration.** v1 of any strategy is the worst version. The infrastructure (data pipeline, backtest engine, executor with safety) compounds value across strategies and across time even when individual strategies underperform.
4. **Auto-publishing AI content has narrow legitimate channels in 2026.** Only dev.to gets real auto-publish (their API + audience tolerates AI-assisted content if disclosed). Medium has no posting API. Substack has no posting API. Reddit auto-posts get banned. X requires paid tier. LinkedIn requires app review. Everything except dev.to routes to a markdown outbox folder for human review.
5. **Disclosure footers are non-negotiable.** Every published article has an AI-assistance disclosure and (when applicable) an FTC affiliate disclosure. Configurable text, but cannot be removed without editing source.

Do not silently weaken any of these in code. If the operator pushes back, surface the risk in chat but build it the way the spec says.

---

## 2. The five strategies (one-paragraph each + edge source)

**Strategy 1 — Calibration model.** XGBoost classifier trained on Kalshi's historical resolved markets, predicting `P(YES | market features)`. Features include market category, time-to-expiry, current and historical price, volume profile, order book depth, market-creator, and temporal features. For any live market, compare model probability to market price; trade when divergence exceeds threshold AND model confidence is high. **Edge source:** statistical — small, sophisticated traders are not present in low-liquidity Kalshi markets, so crowd calibration is weak there.

**Strategy 2 — Cross-market arbitrage.** Maintain a mapping between Kalshi markets and equivalent markets on Polymarket and (optionally) sportsbooks. When implied probabilities diverge by more than transaction costs + safety margin, take both sides. **Edge source:** structural — pure execution play, no forecasting required. Highest-confidence signals when they exist; just rare and small per-trade.

**Strategy 3 — Conditional consistency.** Identify related Kalshi markets (e.g., "X wins primary" + "X wins general") and detect when their prices violate basic probability laws (P(A∩B) > min(P(A), P(B)), mutually-exclusive markets summing to >1.05, etc.). Construct positions that profit from convergence. **Edge source:** structural — pure math, no model risk.

**Strategy 4 — News reaction.** Ingest news feeds (RSS + NewsAPI free tier + EventRegistry as fallback), compute sentence-transformer embeddings, find Kalshi markets with high cosine similarity to new headlines, score relevance with a small LLM call, trade if material AND the bot is faster than the market. **Edge source:** information + latency. **This is the lowest-priority strategy to deploy live** — pros with paid X firehoses are competing for the same edge. Build it for the infrastructure, run it in shadow mode, deploy live only if backtest is convincing.

**Strategy 5 — End-of-life drift.** Statistical analysis of historical Kalshi market price trajectories in the final hours before resolution, segmented by category and current-price bucket. Identify exploitable patterns (drift toward 50, drift toward extremes, mean reversion). When live markets enter exploitable buckets, take small positions. **Edge source:** behavioral — market microstructure patterns persist because they're driven by predictable trader behavior near resolution.

---

## 3. Tech stack (do not deviate without asking)

- **Language:** Python 3.11+
- **Package manager:** `uv` (fall back to `pip + venv` and tell operator if unavailable)
- **Data:**
  - **SQLite** (`sqlite3` stdlib) for operational state — orders, positions, audit, configs. Row-oriented, transactional.
  - **Parquet** (via `pyarrow`) for analytical data — historical prices, features, backtest outputs. Columnar, fast for ML.
  - **Pandas** for in-memory manipulation.
- **ML:** `xgboost`, `scikit-learn`, `sentence-transformers` (CPU model only — `all-MiniLM-L6-v2` is fine and runs on a laptop).
- **HTTP:** `httpx`
- **Crypto (Kalshi auth):** `cryptography` for RSA-PSS signing.
- **CLI:** `click`
- **Dashboard:** `streamlit`
- **Scheduling:** `APScheduler` in-process. **No system cron.** Whole system runs as one process started by `python -m moneybutton run`.
- **LLM:** `anthropic` SDK. Used for: news relevance scoring, content writing, weekly self-review, market-mapping suggestions for strategies 2 and 3. Use the [Anthropic API docs](https://docs.claude.com/en/api/overview) for current model strings — confirm with operator at first run.

**Forbidden additions in v1:** Docker, Postgres, Redis, Celery, FastAPI, React, Next.js, any cloud SDK, any GPU dependency, PyTorch (sentence-transformers uses it but only in CPU mode — fine), TensorFlow. Push back if you feel the urge to add infrastructure.

---

## 4. File tree (target end-state)

```
money-button/
├── README.md                           # Operating manual (write last)
├── SPEC.md                             # This file
├── pyproject.toml
├── .env.example                        # Every env var documented
├── .gitignore                          # .env, *.db, *.pem, .kill, logs/, data/{parquet,models,outbox}/
├── kill.sh                             # touch .kill
├── go.sh                               # rm .kill with 5-second countdown
├── data/
│   ├── moneybutton.db                  # Operational SQLite
│   ├── kalshi_private_key.pem          # Operator-supplied
│   ├── parquet/
│   │   ├── markets/                    # Resolved-market snapshots, partitioned by category/year-month
│   │   ├── prices/                     # Per-market price-time series
│   │   ├── orderbook/                  # Snapshots when available
│   │   ├── news/                       # Ingested headlines + embeddings
│   │   └── features/                   # Materialized feature tables for training
│   ├── models/
│   │   └── calibration_v{N}/           # Versioned model artifacts: model.joblib, metadata.json, train_report.html
│   ├── outbox/                         # Content drafts for non-API platforms
│   └── reports/
│       └── backtest/                   # HTML backtest reports per strategy/run
├── logs/
│   └── moneybutton.log                 # Rotated daily, 14-day retention
├── src/moneybutton/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── core/
│   │   ├── config.py                   # Pydantic settings with validation + ceilings
│   │   ├── db.py                       # SQLite schema + idempotent migrations
│   │   ├── kill_switch.py              # .kill file + daily-loss-cap + position-cap
│   │   ├── notifier.py                 # Discord/Slack webhook + log fallback
│   │   ├── audit.py                    # Every action logged with reasoning
│   │   └── money.py                    # Bankroll, fractional Kelly, per-strategy allocation
│   ├── brain/                          # LLM helpers (NOT primary signal source)
│   │   ├── client.py                   # Anthropic wrapper: retries, structured JSON, caching
│   │   └── prompts/
│   │       ├── news_relevance.md
│   │       ├── content_topic.md
│   │       ├── content_write.md
│   │       ├── market_mapping.md       # For strategies 2 & 3
│   │       └── weekly_review.md
│   ├── kalshi/
│   │   ├── auth.py                     # RSA-PSS request signing
│   │   ├── client.py                   # /trade-api/v2 wrapper
│   │   ├── executor.py                 # Strategy-agnostic; takes Signal, applies safety, places order
│   │   └── positions.py                # Position tracking, exit rules
│   ├── data/
│   │   ├── store.py                    # Parquet read/write helpers, partition management
│   │   ├── scraper_kalshi.py           # Historical resolved markets + price histories
│   │   ├── scraper_polymarket.py       # For arbitrage strategy
│   │   ├── scraper_sportsbook.py       # Optional, for sports arb
│   │   ├── scraper_news.py             # NewsAPI / RSS / EventRegistry
│   │   └── live_feed.py                # Real-time data ingestion during operation
│   ├── features/
│   │   ├── pipeline.py                 # Deterministic feature engineering, single source of truth
│   │   ├── market_features.py          # category, expiry, listing age
│   │   ├── price_features.py           # volatility, momentum, range
│   │   ├── volume_features.py          # liquidity, OI, flow
│   │   ├── orderbook_features.py       # spread, depth (when available)
│   │   ├── temporal_features.py        # hour, day, days-to-expiry buckets
│   │   └── relationships.py            # Detects related markets for strategies 2 & 3
│   ├── models/
│   │   ├── calibration.py              # XGBoost: train, calibrate (isotonic), save, load, predict
│   │   ├── drift.py                    # End-of-life drift statistical model
│   │   ├── news_relevance.py           # sentence-transformers wrapper + LLM scorer
│   │   └── registry.py                 # Versioning, metadata, promotion state
│   ├── strategies/
│   │   ├── base.py                     # Strategy ABC: scan(now) -> list[Signal]; backtest_score
│   │   ├── calibration_strat.py        # Strategy 1
│   │   ├── arbitrage_strat.py          # Strategy 2
│   │   ├── consistency_strat.py        # Strategy 3
│   │   ├── news_strat.py               # Strategy 4
│   │   ├── drift_strat.py              # Strategy 5
│   │   └── orchestrator.py             # Runs all warm strategies, ranks signals, allocates capital
│   ├── backtest/
│   │   ├── engine.py                   # Walk-forward simulation
│   │   ├── fills.py                    # Realistic fill model: bid/ask, slippage, partial fills
│   │   ├── fees.py                     # Kalshi fee schedule
│   │   ├── metrics.py                  # Sharpe, Sortino, max DD, hit rate, expectancy, calmar
│   │   ├── reports.py                  # HTML report generator
│   │   └── promotion.py                # Decides whether a backtested strategy is "warm" enough to trade live
│   ├── content/
│   │   ├── topics.py
│   │   ├── research.py
│   │   ├── writer.py
│   │   ├── affiliate.py
│   │   └── publisher/
│   │       ├── devto.py
│   │       └── outbox.py
│   ├── dashboard/
│   │   └── app.py                      # Streamlit: bankroll, per-strategy P&L, model performance, kill switch
│   └── scheduler/
│       ├── runner.py                   # APScheduler bootstrap
│       └── jobs.py                     # Job definitions
└── tests/
    ├── test_kill_switch.py             # WRITE FIRST
    ├── test_money.py                   # Sizing math + per-strategy allocation
    ├── test_kalshi_auth.py             # Signature against demo
    ├── test_features_no_leakage.py     # CRITICAL — verify no future data leaks into past features
    ├── test_backtest_engine.py         # Verify walk-forward correctness, fills, fees
    ├── test_calibration_model.py       # Train on synthetic data; verify calibration
    ├── test_promotion_gate.py          # Verify a bad backtest fails to promote
    └── test_strategy_signals.py        # Each strategy produces well-formed Signal objects
```

---

## 5. Safety architecture (BUILD FIRST, with tests, before any other code)

This is unchanged from v1 in spirit — same kill switch, same daily loss cap, same audit log. Two additions for v2:

### 5.1 Strategy promotion gate
A strategy starts in `SHADOW` mode: signals are generated and logged to audit, but no orders are placed. To be promoted to `LIVE`, the strategy must pass `backtest.promotion.check_strategy()` which requires:

| Metric | Required for promotion |
|---|---|
| Out-of-sample Sharpe (>= 6 months walk-forward) | ≥ 0.8 |
| Hit rate | ≥ strategy-specific threshold (calibration: 52%, arbitrage: 80%, consistency: 70%, news: 53%, drift: 54%) |
| Max drawdown in backtest | ≤ 25% of allocated capital |
| Number of trades in backtest | ≥ 100 |
| Sharpe > 2.0 | Triggers OVERFIT WARNING, requires explicit operator override to promote |

Promotion state lives in SQLite, can be flipped to SHADOW at any time via CLI. Re-promotion requires a fresh backtest.

### 5.2 Per-strategy capital allocation
Configured in `.env` and validated to sum to ≤ 100%:

```
ALLOCATION_CALIBRATION=0.40
ALLOCATION_ARBITRAGE=0.25
ALLOCATION_CONSISTENCY=0.15
ALLOCATION_NEWS=0.10
ALLOCATION_DRIFT=0.10
```

Orchestrator enforces: a strategy cannot open new positions if its current notional exposure (sum of open position sizes attributed to it) exceeds `BANKROLL_USD * ALLOCATION_X`. Tracked per-strategy in the `positions` table.

### 5.3 Hard limits (same as v1, restated)
| Variable | Default | Hard ceiling | Purpose |
|---|---|---|---|
| `DRY_RUN` | `true` | n/a | No real orders, no real publishing. |
| `KALSHI_ENV` | `demo` | n/a | Paper trading until explicitly flipped. |
| `MAX_POSITION_USD` | `25` | `100` | Per-trade cap. |
| `MAX_OPEN_POSITIONS` | `8` | `20` | Across all strategies. |
| `DAILY_LOSS_LIMIT_USD` | `50` | `200` | Auto-creates `.kill` if breached. |
| `MIN_TRADE_USD` | `5` | n/a | Below this, skip (fees eat it). |
| `MIN_LIQUIDITY_USD` | `500` | n/a | Skip illiquid markets. |
| `BANKROLL_USD` | required | $500 unless overridden | App refuses to start if missing. |
| `KELLY_FRACTION` | `0.10` | `0.25` | 1/10 Kelly is paranoid; correct for an unproven edge. |

### 5.4 Tests written first
- `test_kill_switch.py` — `.kill` blocks executor and publisher; daily-loss cap auto-creates `.kill`; both tested with mocked clock and DB.
- `test_money.py` — per-strategy allocation enforced; sizing never exceeds caps; never goes below `MIN_TRADE_USD`.
- `test_promotion_gate.py` — a backtest with Sharpe 0.5 fails to promote; a backtest with 50 trades fails (under min trades); a backtest with Sharpe 2.5 triggers overfit warning.

---

## 6. Data model

### 6.1 Operational state — SQLite
```sql
CREATE TABLE IF NOT EXISTS bankroll (
  ts TEXT PRIMARY KEY,
  balance_usd REAL NOT NULL,
  source TEXT NOT NULL                  -- 'init', 'trade_close', 'deposit', 'manual_adjust'
);

CREATE TABLE IF NOT EXISTS strategies (
  name TEXT PRIMARY KEY,                -- 'calibration', 'arbitrage', 'consistency', 'news', 'drift'
  state TEXT NOT NULL,                  -- 'SHADOW', 'LIVE', 'DISABLED'
  allocation_pct REAL NOT NULL,
  last_backtest_ts TEXT,
  last_backtest_sharpe REAL,
  last_backtest_hitrate REAL,
  last_backtest_max_dd REAL,
  promoted_ts TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  strategy TEXT NOT NULL,
  ticker TEXT NOT NULL,
  side TEXT NOT NULL,                   -- 'yes' | 'no'
  edge_bps INTEGER NOT NULL,
  confidence TEXT,                      -- 'low' | 'med' | 'high'
  reasoning TEXT,                       -- structured JSON: features, model output, etc.
  acted_on INTEGER NOT NULL DEFAULT 0,  -- bool
  client_request_id TEXT
);

CREATE TABLE IF NOT EXISTS orders (
  client_request_id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  strategy TEXT NOT NULL,
  ticker TEXT NOT NULL,
  side TEXT,
  action TEXT,                          -- 'buy' | 'sell'
  count INTEGER,
  limit_price_cents INTEGER,
  kalshi_order_id TEXT,
  status TEXT,
  fill_price_cents INTEGER,
  notional_usd REAL,
  signal_id INTEGER REFERENCES signals(id)
);

CREATE TABLE IF NOT EXISTS positions (
  ticker TEXT,
  strategy TEXT,
  side TEXT,
  count INTEGER,
  avg_entry_cents INTEGER,
  opened_ts TEXT,
  unrealized_usd REAL,
  realized_usd REAL DEFAULT 0,
  closed_ts TEXT,
  PRIMARY KEY (ticker, strategy)        -- Two strategies can independently hold the same ticker
);

CREATE TABLE IF NOT EXISTS daily_pnl (
  date TEXT,
  strategy TEXT,
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
  actor TEXT,                           -- 'strategy.calibration', 'safety.kill_switch', 'content.writer'
  action TEXT,
  payload_json TEXT,
  reasoning TEXT,                       -- full reasoning blob (LLM or model output)
  outcome TEXT
);
```

### 6.2 Analytical store — Parquet
Partitioned hierarchically, written via `pyarrow.parquet`, read via `pandas.read_parquet`:

```
data/parquet/
├── markets/category={cat}/year_month={ym}/markets.parquet
│   columns: ticker, event_ticker, title, subtitle, status, result, expiration_ts, created_ts, ...
├── prices/category={cat}/year_month={ym}/prices.parquet
│   columns: ticker, ts, yes_bid, yes_ask, last_price, volume, open_interest
├── orderbook/category={cat}/year_month={ym}/orderbook.parquet
│   columns: ticker, ts, side, price, size, level
├── news/source={src}/year_month={ym}/news.parquet
│   columns: id, ts, source, headline, body, url, embedding (list[float])
└── features/strategy={strat}/year_month={ym}/features.parquet
    columns: depend on strategy; always include ticker, ts, label_resolved
```

### 6.3 Models on disk
```
data/models/calibration_v{N}/
├── model.joblib              # Pickled XGBoost + isotonic calibrator
├── metadata.json             # train_window, val_window, feature_columns, hyperparams, git_sha
├── train_report.html         # Calibration plot, feature importance, validation metrics
└── feature_schema.json       # Exact feature names + dtypes for inference-time validation
```

`models/registry.py` provides `get_active_calibration_model()` which returns the highest version that's been promoted, and refuses to load any model whose `feature_schema.json` doesn't match the current `features.pipeline` output.

---

## 7. Data layer

### 7.1 Kalshi historical scraper (`data/scraper_kalshi.py`)
First-time backfill is a multi-hour operation. Build it to be resumable.

- Endpoint: `GET /trade-api/v2/markets?status=settled&limit=1000` (paginate via cursor).
- For each settled market, also fetch `GET /trade-api/v2/markets/{ticker}/candlesticks` for the price-time series (interval determined by market lifetime — 1h candles for short markets, daily for long).
- Write to Parquet partitioned by category and year-month.
- Maintain a `scraper_state` table in SQLite tracking the cursor and last-completed partition. On crash, resume from there.
- Respect Kalshi rate limits: implement exponential backoff on 429s; default to 1 request/second to be conservative.
- CLI: `python -m moneybutton data backfill --since=2024-01-01 [--category=KXFED]`.

### 7.2 Polymarket scraper (`data/scraper_polymarket.py`)
For strategy 2 (arbitrage). Use Polymarket's Gamma API for market metadata and CLOB API for prices. Skip if operator confirmed they don't have / can't access Polymarket.

### 7.3 Sportsbook scraper (`data/scraper_sportsbook.py`)
Optional. The-odds-api.com has a free tier (500 req/month). If operator opts in, scrape moneylines for sports events and map to Kalshi sports markets. **Warning to surface in chat:** sportsbooks ban successful arbers; this strategy will degrade or get accounts limited if it actually works.

### 7.4 News scraper (`data/scraper_news.py`)
Three pluggable sources, configured by env:
- **RSS** (always on, free): top 10 finance/politics/tech feeds, configurable list in `data/rss_sources.txt`.
- **NewsAPI** (free tier: 100 req/day, requires `NEWSAPI_KEY`): broader coverage.
- **EventRegistry** (free tier: 2000 events/day, requires `EVENTREGISTRY_KEY`): event-clustered news.

For each ingested headline: store raw + compute embedding via `sentence-transformers/all-MiniLM-L6-v2` (CPU, free, ~100ms/item), store in `parquet/news/`.

### 7.5 Live feed (`data/live_feed.py`)
During operation, the scheduler runs:
- Every 15 min: snapshot active Kalshi markets + prices to Parquet (so backtests can use the same data later).
- Every 5 min during US business hours: pull news increments.
- Every 1 min during sports events: poll relevant Kalshi sports markets at higher frequency.

---

## 8. Feature engineering (`features/pipeline.py`)

**This module is the single source of truth for feature definitions.** Both training and inference call the same functions. Any feature drift between train and serve is a bug.

### 8.1 No-leakage requirement
A feature computed for time `t` may use only data with `ts < t`. This is enforced by the API:

```python
def compute_features(market: Market, as_of_ts: datetime, store: ParquetStore) -> dict:
    """Compute features for `market` as if we were standing at `as_of_ts`.
    Implementation MUST filter all underlying data to `ts < as_of_ts` before computing.
    """
```

`test_features_no_leakage.py` verifies this by computing features at `t=T` and again at `t=T+1d`, then perturbing all data in `[T, T+1d]` to noise, and asserting that the features at `t=T` are unchanged.

### 8.2 Feature groups (each in its own file under `features/`)
- **market_features**: one-hot category, log_days_to_expiry, days_since_listing, market_creator_one_hot
- **price_features**: yes_price_now, yes_price_1h_ago, yes_price_24h_ago, log_returns_1h, log_returns_24h, realized_vol_24h, max_24h, min_24h, range_24h
- **volume_features**: volume_24h, log_volume_24h, open_interest, log_oi, vol_oi_ratio, num_trades_24h
- **orderbook_features**: bid_ask_spread_cents, mid_price, depth_top_5_levels (when orderbook data available; else NaN, model handles via XGBoost native NaN support)
- **temporal_features**: hour_of_day, day_of_week, is_weekend, is_market_hours, days_to_expiry_bucket (0-1d, 1-7d, 7-30d, 30+d)
- **relationships** (used by strategies 2 & 3, not by calibration): finds related markets via shared event_ticker, LLM-suggested mappings stored in `data/parquet/market_mappings.parquet`

---

## 9. Models

### 9.1 Calibration (`models/calibration.py`)
- **Algorithm:** XGBoost classifier (`objective=binary:logistic`).
- **Calibration:** post-hoc isotonic regression on validation set. Without this, XGBoost probabilities are not well-calibrated and the edge calculation is wrong.
- **Hyperparameters:** start with `n_estimators=500, max_depth=5, learning_rate=0.05, min_child_weight=10, subsample=0.8, colsample_bytree=0.8`. These are deliberately conservative — overfitting is the enemy.
- **Train/val/test split:** time-based, with embargo. E.g., train on markets resolved before 2025-07-01, validate on 2025-07-01 to 2025-10-01 (with 1-week embargo: skip markets straddling the boundary), test on 2025-10-01 to 2026-01-01.
- **CLI:** `python -m moneybutton models train calibration` saves a new versioned model to `data/models/calibration_v{N}/`.
- **Calibration plot** in train_report.html: predicted probability vs observed frequency in 10 buckets. Bot must surface this; a model with terrible calibration is worthless even with high accuracy.

### 9.2 Drift (`models/drift.py`)
- **Algorithm:** non-parametric. For each historical market, compute price trajectory in last 24h before resolution. Bucket by `(category, current_price_bucket, hours_to_resolution_bucket)`. For each bucket, store: `mean_drift_cents`, `resolution_rate_yes`, `n_observations`.
- **Inference:** given a live market in bucket `B`, look up the bucket; if `n_observations >= 30` and the resolution rate differs from the current price by more than threshold, signal.
- Saved to `data/models/drift_v{N}/buckets.parquet`.

### 9.3 News relevance (`models/news_relevance.py`)
- Two-stage:
  1. **Embedding similarity** (fast, free): for each new headline, compute cosine similarity to all active Kalshi market descriptions. Top-K candidates (default K=10).
  2. **LLM relevance scoring** (slow, costs money): for each candidate above similarity threshold, ask the LLM via `news_relevance.md` prompt to score `{material: bool, direction: 'yes'|'no'|'unclear', confidence: 0-1, reasoning: str}`.
- Result is the news-strategy signal.

### 9.4 Model registry (`models/registry.py`)
- `register(name, version, metadata)` — writes to `data/models/{name}_v{N}/`.
- `get_active(name)` — returns the highest-versioned model that's been promoted; refuses to load if `feature_schema.json` mismatches current pipeline output.
- `list_models(name)` — for the dashboard.

---

## 10. Strategies (`strategies/`)

### 10.1 Base interface (`strategies/base.py`)
```python
@dataclass
class Signal:
    strategy: str
    ticker: str
    side: Literal['yes', 'no']
    edge_bps: int                # positive = bullish on `side`
    confidence: Literal['low', 'med', 'high']
    suggested_size_usd: float
    reasoning: dict              # JSON-serializable, audited
    expires_at: datetime         # signal stale after this

class Strategy(ABC):
    name: str
    @abstractmethod
    def scan(self, now: datetime) -> list[Signal]: ...
    @abstractmethod
    def backtest(self, start: date, end: date) -> BacktestResult: ...
```

### 10.2 Calibration strategy (`strategies/calibration_strat.py`)
- For each active Kalshi market with sufficient liquidity: compute features via `features.pipeline.compute_features`, predict via active calibration model, compute `edge_bps = (model_prob - market_prob) * 10000`.
- If `abs(edge_bps) >= MIN_EDGE_BPS_CALIBRATION` (default 500) and model confidence high (model agreement across N=5 ensemble models trained with different seeds): signal.

### 10.3 Arbitrage strategy (`strategies/arbitrage_strat.py`)
- Iterate the `market_mappings` table (Kalshi ↔ Polymarket and Kalshi ↔ sportsbook).
- For each mapping: fetch live prices both sides simultaneously.
- If `implied_prob_A - implied_prob_B > (fees_A + fees_B + SAFETY_MARGIN_BPS)`: signal pair (long the cheap side, short the expensive side).
- **Special handling:** signals come in pairs and must execute together or not at all. Executor needs an `atomic_pair` mode.

### 10.4 Consistency strategy (`strategies/consistency_strat.py`)
- Use `features.relationships.find_related_markets()` to identify related-market clusters.
- For each cluster, evaluate constraints (e.g., for mutually-exclusive: `sum(yes_prices) ≤ 1.05`).
- When violated by more than threshold: construct a position structure that profits from convergence.

### 10.5 News strategy (`strategies/news_strat.py`)
- Subscribe to new news rows in Parquet (poll new files every 5 min).
- For each: `models.news_relevance.score()` → if `material AND confidence > 0.7`: signal.
- Default `state = SHADOW` until backtest demonstrates value.

### 10.6 Drift strategy (`strategies/drift_strat.py`)
- For each active market with `hours_to_resolution < 24`: look up `(category, price_bucket, hours_bucket)` in drift model.
- If exploitable pattern: signal small size.

### 10.7 Orchestrator (`strategies/orchestrator.py`)
- Every 15 min: run `scan(now)` on every strategy in `LIVE` or `SHADOW` state.
- Collect all signals. Deduplicate by `(ticker, side)`: if two strategies signal the same direction, keep the higher-confidence one and attribute equally to both for tracking.
- Apply per-strategy capital allocation: skip signals that would exceed strategy's allocation.
- Apply global limits: `MAX_OPEN_POSITIONS`, `DAILY_LOSS_LIMIT_USD`.
- Hand surviving signals to executor (which applies the kill switch and idempotency).
- All signals (acted-on or not) get a row in `signals` table.

---

## 11. Backtesting (`backtest/`)

This is where strategies prove they have edge before touching real money. Build it carefully — a wrong backtest is worse than no backtest.

### 11.1 Engine (`backtest/engine.py`)
- **Walk-forward simulation:** train on `[T0, T1]`, evaluate on `[T1+embargo, T2]`, advance `T1 += step`, repeat.
- **For each evaluation window:** simulate the strategy as if running live, using only data with `ts < current_simulation_ts`. The features pipeline's no-leakage guarantee makes this safe.
- **Output:** per-trade ledger (timestamp, ticker, side, size, entry price, exit price, exit reason, P&L) + equity curve.

### 11.2 Fills (`backtest/fills.py`)
**No "filled at mid price" lies.** Default fill model:
- Buy at `yes_ask` for YES side (or `no_ask` for NO).
- For sizes ≤ top-of-book size: assume full fill.
- For sizes > top-of-book: walk the book; if depth insufficient, partial fill at average price across consumed levels.
- Add `slippage_bps` (default 25 bps) on top to model adverse selection.

### 11.3 Fees (`backtest/fees.py`)
- Verify Kalshi's current fee schedule (it has changed; was 0% on many markets, may not be now).
- Apply per-trade.

### 11.4 Metrics (`backtest/metrics.py`)
- Sharpe (annualized, risk-free rate = 0)
- Sortino
- Max drawdown (% of allocated capital)
- Hit rate (% trades profitable)
- Expectancy (avg P&L per trade)
- Calmar ratio (annual return / max DD)
- Number of trades

### 11.5 Promotion (`backtest/promotion.py`)
- `check_strategy(strategy_name, backtest_result) -> PromotionDecision`
- Returns one of: `PROMOTED`, `REJECTED_LOW_SHARPE`, `REJECTED_LOW_HITRATE`, `REJECTED_HIGH_DD`, `REJECTED_FEW_TRADES`, `OVERFIT_WARNING_REQUIRES_OVERRIDE`.
- Updates `strategies` SQLite table on `PROMOTED`.

### 11.6 Reports (`backtest/reports.py`)
HTML report per backtest run, saved to `data/reports/backtest/{strategy}/{run_ts}.html`:
- Equity curve plot
- Drawdown plot
- Per-trade scatter (entry price vs P&L)
- Calibration plot (for calibration strategy)
- Summary metrics table
- Promotion decision

---

## 12. Content engine (secondary, condensed)

Same niche (AI dev tools), same structure as v1, simplified here:

- **Topics** (`content/topics.py`): daily, scan HN/PH/GH-trending/dev.to, dedupe vs `articles` table, LLM picks one.
- **Research** (`content/research.py`): fetch tool homepage/pricing/README, web-search for 3 third-party reviews ≤ 90 days old, summarize into structured fact sheet (no fabrication: missing facts → `null`, not invented).
- **Writer** (`content/writer.py`): voice = pragmatic senior engineer, 700–1100 words, mandatory sections (TL;DR, what it does, who cares, comparison, pricing reality, verdict, disclosure footer).
- **Affiliate** (`content/affiliate.py`): inject links only when `affiliate_links` table has entry for the tool slug. Until operator signs up for programs, articles publish without links and earn $0 — correct behavior.
- **Publishers**: `devto.py` real auto-publish (default `published: false` so operator reviews on dev.to before flipping to published unless `AUTO_PUBLISH_DEVTO=true`); `outbox.py` writes platform-specific markdown drafts for Medium, Substack, LinkedIn, Reddit, X-thread.
- **Disclosure footer** non-removable without code change. AI-assistance + (when applicable) FTC affiliate disclosure.

---

## 13. CLI surface

```
# Lifecycle
python -m moneybutton run                        # Start scheduler
python -m moneybutton dashboard                  # Streamlit
python -m moneybutton status                     # Bankroll, per-strategy P&L, open positions, last 5 actions
python -m moneybutton kill                       # touch .kill
python -m moneybutton resume                     # rm .kill (5-second countdown)
python -m moneybutton paper                      # Force DRY_RUN=true, KALSHI_ENV=demo for this session
python -m moneybutton live                       # Requires typing 'I ACCEPT THE RISK'

# Data
python -m moneybutton data backfill --since=2024-01-01 [--category=KXFED]
python -m moneybutton data refresh-news
python -m moneybutton data scrape-polymarket
python -m moneybutton data show-stats            # How much data we have, partitions, sizes

# Models
python -m moneybutton models train calibration [--since=...] [--until=...]
python -m moneybutton models train drift
python -m moneybutton models list
python -m moneybutton models report calibration_v3   # Open HTML report

# Strategies
python -m moneybutton strategies list                # Show name, state, allocation, last backtest
python -m moneybutton strategies backtest calibration --since=2025-01-01
python -m moneybutton strategies promote calibration --version=3
python -m moneybutton strategies shadow news         # Demote to shadow mode
python -m moneybutton strategies disable arbitrage

# Content
python -m moneybutton content addlink {slug} {url} --program {name}
python -m moneybutton content publish-now {slug}
python -m moneybutton content review-week
```

---

## 14. Scheduler jobs

| Job | Interval | Module |
|---|---|---|
| `safety.health_check` | 1 min | DB writeable, .env loaded, kill switch state |
| `data.live_snapshot_kalshi` | 15 min | Snapshot active markets + prices to Parquet |
| `data.live_snapshot_polymarket` | 15 min | Same for Polymarket (if enabled) |
| `data.refresh_news` | 5 min (business hrs) | Pull RSS + NewsAPI increments |
| `strategies.scan_all` | 15 min | Orchestrator runs every warm/shadow strategy, collects signals |
| `kalshi.update_positions` | 1 hour | Refresh positions, P&L, exit checks |
| `models.retrain_calibration` | 1 week (Sun 02:00) | Backfill new data, retrain, backtest, surface to operator |
| `content.discover_topic` | 1 day (08:00 local) | Pick a tool to cover |
| `content.research_and_draft` | 1 day (09:00 local) | Generate article |
| `content.publish` | 1 day (10:00 local) | dev.to + outbox |
| `analytics.weekly_review` | 1 week (Sun 18:00) | Brain reviews own performance, writes summary |

All jobs wrapped in `safe_job()`: catches exceptions → logs traceback to audit → notifies → does NOT crash the scheduler.

---

## 15. Out of scope for v2

Do not build without explicit follow-up:
- Stock trading (Alpaca/IBKR).
- Auto-posting to Reddit, X, LinkedIn, Hacker News.
- Multi-user / multi-tenant / cloud deployment.
- DFS or sportsbook *trading* (sportsbook *scraping* for arbitrage prices is in scope).
- Crypto trading.
- Email scraping or cold-outreach automation.
- Any GPU-dependent model.
- Reinforcement learning. (Tempting but a v3 problem.)

---

## 16. Build order — DO NOT DEVIATE

After each numbered step, **commit and tell the operator what just shipped** before moving on. Do not batch.

1. Project scaffold: `pyproject.toml`, `.env.example`, `.gitignore`, `kill.sh`, `go.sh`, empty module dirs with `__init__.py`.
2. `core/config.py` with full validation. Smoke-test by printing settings.
3. `core/db.py` with full schema. Verify via `sqlite3 data/moneybutton.db .schema`.
4. **Tests written:** `test_kill_switch.py`, `test_money.py`, `test_promotion_gate.py`. They will fail.
5. `core/kill_switch.py`, `core/money.py`, `core/audit.py`, `core/notifier.py` — make tests pass.
6. `brain/client.py` with Anthropic SDK + retry + structured JSON output. Smoke test.
7. `kalshi/auth.py` + `test_kalshi_auth.py` against demo environment.
8. `kalshi/client.py` — read-only methods first.
9. `data/store.py` — Parquet read/write helpers.
10. `data/scraper_kalshi.py` — backfill historical resolved markets and price histories. **Run a small test backfill (1 month, 1 category) and verify before doing full backfill.**
11. `features/pipeline.py` and the per-group feature modules. **Write `test_features_no_leakage.py` and make it pass before continuing.**
12. `models/calibration.py` + `models/registry.py`. Train a v1 model on the small backfill. Save with metadata.
13. `backtest/engine.py`, `backtest/fills.py`, `backtest/fees.py`, `backtest/metrics.py`, `backtest/reports.py`, `backtest/promotion.py`. Write `test_backtest_engine.py`.
14. `strategies/base.py` + `strategies/calibration_strat.py`. Run end-to-end backtest. **Show operator the report.** Discuss whether to continue with full backfill.
15. **Full historical backfill** (could take hours). Retrain calibration model on full data. Re-backtest.
16. `kalshi/executor.py`, `kalshi/positions.py` — DRY_RUN-only path first; verify audit rows.
17. `strategies/orchestrator.py` — wire calibration through to executor in DRY_RUN against demo env.
18. `data/scraper_polymarket.py` (if operator confirmed Polymarket access).
19. `strategies/arbitrage_strat.py` + backtest.
20. `features/relationships.py` + `strategies/consistency_strat.py` + backtest.
21. `models/drift.py` + `strategies/drift_strat.py` + backtest.
22. `data/scraper_news.py` + `models/news_relevance.py` + `strategies/news_strat.py` + backtest. Default to SHADOW.
23. `content/*` — full content engine.
24. `dashboard/app.py`.
25. `scheduler/runner.py` + `scheduler/jobs.py`. Run for 30 min in paper mode. Verify audit log.
26. `cli.py`.
27. `README.md`.

**Promotion to live trading happens manually**, per-strategy, only after the operator reviews each strategy's backtest report. Never auto-promote.

---

## 17. Things to verify against live docs (your training cutoff is not authoritative)

Before writing code that touches these, fetch current docs and confirm:
- **Anthropic API:** model strings, SDK version, rate limits — <https://docs.claude.com/en/api/overview>
- **Kalshi API:** auth scheme, current `/markets` query params, settled-market endpoint, candlestick endpoint, fee schedule, rate limits — <https://trading-api.readme.io>
- **Polymarket APIs:** Gamma + CLOB current schema — <https://docs.polymarket.com>
- **dev.to API:** article POST schema and rate limits — <https://developers.forem.com/api>
- **NewsAPI:** free tier limits — <https://newsapi.org/docs>
- **EventRegistry:** free tier limits — <https://eventregistry.org/documentation>
- **The-odds-api** (if operator opts in for sportsbook scraping): <https://the-odds-api.com/>

If any have changed in a way that breaks the spec, **stop and tell the operator** rather than papering over it.

---

## 18. Questions you must ask before coding

Ask the operator all of these in chat **before writing any code**. Group them logically.

**Account access**
1. **Confirm: Kalshi or Polymarket?** Which platform do you actually have an account on? If only one, the other strategy (arbitrage) is impaired — confirm whether to build it anyway with the assumption you'll create the second account.
2. **Sportsbook scraping?** Want to opt into the-odds-api (free tier) for sports arbitrage signals? You don't need a sportsbook account just for *signals*; you'd only need one to *execute* the arbitrage. We can build signal-only first.

**LLM**
3. **Confirm Anthropic model name.** Spec defaults to `claude-sonnet-4-5` for general tasks. For news-relevance scoring (called frequently), do you want `claude-haiku-4-5` for cost?

**Bankroll & risk**
4. **Bankroll confirm.** Spec assumes `BANKROLL_USD=500`. Confirm exact value.
5. **Per-strategy allocation confirm.** Defaults: 40/25/15/10/10 (calibration/arbitrage/consistency/news/drift). Adjust?
6. **Initial caps confirm:** `MAX_POSITION_USD=25`, `DAILY_LOSS_LIMIT_USD=50`, `KELLY_FRACTION=0.10`. Adjust?

**Data sources**
7. **News API access.** Will you sign up for NewsAPI (free, 100 req/day) and/or EventRegistry (free, 2000 events/day)? RSS-only is workable but weaker. The bot will still work in shadow mode for the news strategy without these.
8. **Historical backfill window.** Default: scrape Kalshi resolved markets back to 2024-01-01 (~2 years). More = better models, but longer first-time scrape. Want longer or shorter?

**Content engine**
9. **dev.to account?** If yes, provide API key in `.env` later. If no, I'll default `AUTO_PUBLISH_DEVTO=false` and route everything to the outbox until you set one up.
10. **Discord/Slack webhook URL** for notifications, or "log to file only is fine."
11. **Local timezone** for content jobs (default 08:00, 09:00, 10:00 local).
12. **Voice sample** for the writer — paste 2–3 paragraphs of writing in a voice you'd like imitated. If none, default = "pragmatic senior engineer."

**Workflow**
13. **First strategy to validate end-to-end.** Spec says calibration first (step 14). Confirm, or pick another?
14. **Stop-and-review checkpoints.** I'm planning to pause and show you results after: step 14 (first backtest), step 15 (full-data backtest), step 17 (paper trading), each subsequent strategy. Want more or fewer checkpoints?

---

## 19. Final reminders before you start

- Build the safety layer first, with tests, before anything else.
- The features pipeline is the single source of truth — same code path for train and serve. No leakage.
- Default to dry-run / paper / outbox-only / shadow-strategy.
- A strategy gets to trade live capital ONLY after passing the promotion gate AND being explicitly promoted by the operator via CLI.
- Backtest realism is non-negotiable — bid/ask fills, fees, slippage, partial fills.
- Every action gets an audit row with full reasoning.
- Ask § 18's questions, then commit and explain after each § 16 step.
- The goal of v2 is **a system the operator trusts to find and exploit real edge**, not a system that has made money. Trust → small live capital → measurement → iteration → scale. That sequence is the only one that works.
