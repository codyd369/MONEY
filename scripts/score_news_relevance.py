"""Score news articles against markets using the LLM (Gemini Flash by
default, which is free-tier on AI Studio).

Writes one row per (news_id, ticker) pair to a new `news_relevance` table
in SQLite. Keyword pre-filter drops 99%+ of pairs before any LLM call.

Flow:
  1. Load news rows and markets.
  2. For each news item, extract keywords from its headline+body.
  3. For each market, extract keywords from its title+rules_primary.
  4. A pair passes the pre-filter iff any news keyword overlaps with any
     market keyword. Cheap set intersection.
  5. For each surviving pair, call brain/client.complete() with the
     news_relevance.md prompt and parse the JSON response.
  6. Persist (news_id, ticker, material, direction, confidence, reasoning).

Tokens per LLM call: ~400 in / 60 out. At Gemini Flash free tier (1M
tokens/day), ~2000 pairs can be scored per day without paying. At
Haiku pricing (~$0.25/M in) ~2000 pairs = a few cents.

Usage:
    uv run python scripts/score_news_relevance.py --since 2024-01-01 --max-pairs 500
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

from moneybutton.brain.client import BrainClient, render_prompt
from moneybutton.core.audit import record as audit_record
from moneybutton.core.config import get_settings
from moneybutton.core.db import init_db
from moneybutton.data.store import read_dataset


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS news_relevance (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scored_ts TEXT NOT NULL,
  news_id TEXT NOT NULL,
  ticker TEXT NOT NULL,
  material INTEGER NOT NULL,
  direction TEXT,
  confidence REAL,
  reasoning TEXT,
  model TEXT,
  UNIQUE(news_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_news_relevance_ticker ON news_relevance(ticker);
CREATE INDEX IF NOT EXISTS idx_news_relevance_news ON news_relevance(news_id);
"""


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "is", "are", "will", "be", "was", "were", "has",
    "have", "had", "do", "does", "did", "from", "as", "that", "this",
    "these", "those", "it", "its", "but", "not", "who", "what", "when",
    "where", "why", "how", "which", "can", "could", "should", "would",
    "may", "might", "new", "york",
}
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]{2,}")


def _keywords(text: str | None) -> set[str]:
    if not text:
        return set()
    toks = {w.lower() for w in _WORD_RE.findall(text)}
    return {t for t in toks if t not in _STOPWORDS and len(t) >= 4}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score news relevance to markets via LLM.")
    p.add_argument("--since", default=None, help="ISO date; only score news from since onward (default: 7d ago)")
    p.add_argument("--news-since", default=None, help="Alias of --since for clarity")
    p.add_argument("--model", default=None, help="LiteLLM model id; defaults to settings.llm_model_news")
    p.add_argument("--max-pairs", type=int, default=1000, help="Cap LLM calls per run (token budget safety)")
    p.add_argument(
        "--min-keyword-overlap",
        type=int,
        default=3,
        help=(
            "Require at least N shared keywords between news and market. "
            "2 is very loose (millions of pairs on a wide scrape). 3 is a "
            "practical default that still catches obvious matches."
        ),
    )
    p.add_argument(
        "--only-open-markets",
        action="store_true",
        help="Skip markets whose result is already settled (yes/no). For live scoring.",
    )
    p.add_argument(
        "--rate-limit-sleep-s",
        type=float,
        default=None,
        help="Sleep between LLM calls. Default: model-aware (Gemini Flash free "
        "tier is 5-15 RPM => 4-13s; Anthropic / Groq paid => 0.2s).",
    )
    p.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=20,
        help="Abort the run after this many consecutive LLM errors (likely a "
        "quota or wrong-model issue; bailing keeps the run from spinning).",
    )
    return p.parse_args()


def _default_sleep_for_model(model: str) -> float:
    """Pick a per-call sleep that matches the model's likely free-tier RPM.

    Gemini AI Studio free tier RPM as of 2026 (observed, varies by account):
      gemini-2.5-flash       :  5-10  RPM
      gemini-2.5-flash-lite  : 10-15  RPM
      gemini-2.0-flash       : 15     RPM
    Groq free tier:
      llama-3.3-70b          : 30     RPM  (much looser — preferred free pick)
      llama-3.1-8b           : 30     RPM
    Anthropic / Ollama: no per-call throttle needed.
    """
    m = model.lower()
    if "groq" in m:
        return 2.0  # 30 RPM; leave margin
    if "gemini" in m:
        if "lite" in m:
            return 6.5  # ~9 RPM; observed free tier is stricter than docs
        return 13.0  # 5 RPM safe
    return 0.2


def _ensure_schema(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def _already_scored_pairs(db_path: Path) -> set[tuple[str, str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT news_id, ticker FROM news_relevance").fetchall()
    finally:
        conn.close()
    return {(r[0], r[1]) for r in rows}


def _insert_score(
    db_path: Path,
    *,
    news_id: str,
    ticker: str,
    material: bool,
    direction: str | None,
    confidence: float | None,
    reasoning: str | None,
    model: str,
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO news_relevance
              (scored_ts, news_id, ticker, material, direction, confidence, reasoning, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dt.datetime.now(dt.timezone.utc).isoformat(),
                news_id,
                ticker,
                1 if material else 0,
                direction,
                confidence,
                reasoning,
                model,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    init_db()
    db_path = Path(settings.sqlite_path)
    _ensure_schema(db_path)

    model = args.model or settings.llm_model_news
    sleep_s = args.rate_limit_sleep_s if args.rate_limit_sleep_s is not None else _default_sleep_for_model(model)
    print(f"score_news_relevance: start  model={model}  sleep={sleep_s:.1f}s/call", flush=True)
    if "gemini" in model.lower() and sleep_s < 12 and "lite" not in model.lower():
        print(
            "  NOTE: gemini-2.5-flash free tier is often 5 RPM. Consider "
            "--model gemini/gemini-2.5-flash-lite (15 RPM, looser quota) "
            "or pass --rate-limit-sleep-s 13 to stay under the strict tier.",
            flush=True,
        )

    # Load news.
    news = read_dataset("news")
    if news.empty:
        print("no news on disk. Run scripts/backfill_news.py first.", flush=True)
        return 1
    since_arg = args.news_since or args.since
    if since_arg:
        since_dt = pd.Timestamp(since_arg, tz="UTC")
    else:
        since_dt = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    news["ts"] = pd.to_datetime(news["ts_iso"], utc=True, errors="coerce")
    news = news[news["ts"] >= since_dt].copy()
    print(f"news rows in scope (>= {since_dt.date()}): {len(news)}", flush=True)

    # Load markets (prefer still-open ones — live relevance is what matters;
    # for training data, all resolved markets in the time window are OK).
    markets = read_dataset("markets")
    if markets.empty:
        print("no markets on disk. Run backfill_via_events first.", flush=True)
        return 1
    if args.only_open_markets:
        markets = markets[~markets["result"].isin(["yes", "no"])].copy()
        print(f"markets in scope (open only): {len(markets)}", flush=True)
    # Build per-market keyword sets.
    markets["_kws"] = (markets["title"].fillna("") + " " + markets["rules_primary"].fillna("")).map(_keywords)
    print(f"markets in scope: {len(markets)}", flush=True)

    # Build candidate pairs via keyword overlap.
    news["_kws"] = (news["headline"].fillna("") + " " + news["body"].fillna("")).map(_keywords)

    pairs: list[tuple[pd.Series, pd.Series, int]] = []
    mkw_list = list(zip(markets["ticker"], markets["_kws"]))
    for _, n in news.iterrows():
        if not n["_kws"]:
            continue
        for ticker, mkws in mkw_list:
            if not mkws:
                continue
            shared = n["_kws"] & mkws
            if len(shared) >= args.min_keyword_overlap:
                pairs.append((n, ticker, len(shared)))
    # Rank pairs by overlap size descending.
    pairs.sort(key=lambda t: -t[2])
    print(f"candidate pairs (after keyword prefilter): {len(pairs)}", flush=True)
    if not pairs:
        print("nothing to score.", flush=True)
        return 0
    if len(pairs) > 100_000:
        print(
            f"  WARNING: {len(pairs):,} pairs is a lot. Consider raising "
            f"--min-keyword-overlap (currently {args.min_keyword_overlap}), "
            f"narrowing --news-since, or --only-open-markets.",
            flush=True,
        )

    # Skip already-scored pairs.
    already = _already_scored_pairs(db_path)
    new_pairs = [(n, t, o) for (n, t, o) in pairs if (n["id"], t) not in already]
    print(f"  new pairs (not already scored): {len(new_pairs)}", flush=True)
    if args.max_pairs and len(new_pairs) > args.max_pairs:
        print(f"  capping at --max-pairs {args.max_pairs}", flush=True)
        new_pairs = new_pairs[: args.max_pairs]

    # LLM scoring.
    brain = BrainClient(settings=settings)
    t0 = time.monotonic()
    scored = 0
    errors = 0
    consecutive_errors = 0
    for i, (news_row, ticker, overlap) in enumerate(new_pairs, 1):
        market = markets[markets["ticker"] == ticker].iloc[0]
        try:
            system = "You score whether a news headline is MATERIAL to a specific prediction market."
            # Trim context aggressively: for a yes/no relevance call, the model
            # only needs the headline and the market's title + one sentence of
            # rules. 1500 chars each was ~800 context tokens and burned
            # through Groq's 100k TPD free cap in ~190 calls.
            user_msg = render_prompt(
                "news_relevance",
                headline=str(news_row.get("headline") or "")[:300],
                body=str(news_row.get("body") or "")[:400],
                market_title=str(market.get("title") or "")[:200],
                market_rules=str(market.get("rules_primary") or "")[:400],
                as_of_ts=str(news_row.get("ts_iso") or ""),
            )
            resp = brain.complete(
                system=system,
                messages=[{"role": "user", "content": user_msg}],
                model=model,
                max_tokens=200,
                json_mode=True,
                use_cache=True,
            )
        except Exception as e:  # noqa: BLE001
            errors += 1
            consecutive_errors += 1
            if errors <= 5 or consecutive_errors == args.max_consecutive_errors:
                # Log the first 5 errors and the one that triggers abort.
                msg = str(e).replace("\n", " ")[:300]
                print(f"  [{i}/{len(new_pairs)}] LLM error {type(e).__name__}: {msg}", flush=True)
            if consecutive_errors >= args.max_consecutive_errors:
                print(
                    f"  ABORT: {consecutive_errors} consecutive LLM errors. "
                    f"Likely a quota or wrong-model issue. {scored} pairs persisted "
                    f"so far. Switch model or wait, then rerun (already-scored "
                    f"pairs are skipped).",
                    flush=True,
                )
                break
            time.sleep(sleep_s)
            continue
        else:
            consecutive_errors = 0

        parsed = resp.json_ or {}
        _insert_score(
            db_path,
            news_id=str(news_row["id"]),
            ticker=ticker,
            material=bool(parsed.get("material", False)),
            direction=parsed.get("direction"),
            confidence=float(parsed.get("confidence") or 0.0),
            reasoning=str(parsed.get("reasoning") or "")[:500],
            model=model,
        )
        scored += 1
        if i % 25 == 0 or i == len(new_pairs):
            elapsed = time.monotonic() - t0
            eta = (elapsed / i) * (len(new_pairs) - i) if i > 0 else 0
            print(
                f"  [{i:>5d}/{len(new_pairs)}] scored={scored} errors={errors} "
                f"elapsed={elapsed:.0f}s ETA={eta / 60:.1f}m",
                flush=True,
            )
        time.sleep(sleep_s)

    print()
    print(f"done. scored={scored} errors={errors} elapsed={time.monotonic() - t0:.0f}s")

    audit_record(
        actor="scripts.score_news_relevance",
        action="score_news_relevance",
        payload={
            "model": model,
            "candidate_pairs": len(pairs),
            "new_pairs": len(new_pairs),
            "scored": scored,
            "errors": errors,
        },
        outcome="OK" if errors == 0 else "PARTIAL",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
