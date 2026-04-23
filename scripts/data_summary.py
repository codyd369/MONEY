"""Summarize what's on disk: markets, prices, news, features, models.

Usage: uv run python scripts/data_summary.py
"""

from __future__ import annotations

import pandas as pd

from moneybutton.data.store import read_dataset


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _markets_summary() -> pd.DataFrame | None:
    m = read_dataset("markets")
    if m.empty:
        print("MARKETS: none on disk. Run scripts/backfill_via_events.py or scripts/backfill_markets.py first.")
        return None

    m["has_listing_volume"] = pd.to_numeric(m["volume"], errors="coerce").fillna(0) > 0
    m["close_ts"] = pd.to_datetime(m["close_time"], utc=True, errors="coerce")
    m["year_month"] = m["close_ts"].dt.strftime("%Y-%m")

    print(f"MARKETS on disk: {len(m)}")
    print(f"close_time range: {m['close_ts'].min()} .. {m['close_ts'].max()}")
    print(f"result distribution: {dict(m['result'].value_counts())}")
    return m


def _markets_by_category(m: pd.DataFrame) -> None:
    _section("markets by category")
    by_cat = m.groupby("category").agg(
        total=("ticker", "count"),
        first_close=("close_ts", "min"),
        last_close=("close_ts", "max"),
    ).sort_values("total", ascending=False)
    print(by_cat.head(30).to_string())


def _markets_by_month(m: pd.DataFrame) -> None:
    _section("markets by close-month (tail 24)")
    by_ym = m.groupby("year_month").agg(total=("ticker", "count"))
    print(by_ym.tail(24).to_string())


def _prices_summary(markets: pd.DataFrame | None) -> None:
    _section("prices (candlesticks)")
    p = read_dataset("prices")
    if p.empty:
        print("PRICES: none on disk. Run scripts/backfill_prices.py to fetch candlesticks.")
        return
    print(f"total price rows: {len(p):,}")
    unique_tickers_with_prices = int(p["ticker"].nunique())
    print(f"unique tickers with any candle data: {unique_tickers_with_prices:,}")

    # Per-ticker candle counts.
    per_ticker = p.groupby("ticker").size().sort_values(ascending=False)
    print(f"candles per ticker: min={per_ticker.min()}, median={int(per_ticker.median())}, "
          f"mean={per_ticker.mean():.0f}, max={per_ticker.max()}")

    # Real trading signal: rows where last_price_close is populated.
    traded = p[p["last_price_close"].notna()]
    tickers_with_trades = int(traded["ticker"].nunique()) if len(traded) else 0
    print(f"rows with last_price_close populated: {len(traded):,} "
          f"({(len(traded) / len(p) * 100):.1f}% of price rows)")
    print(f"tickers with ANY trade rows: {tickers_with_trades:,} / {unique_tickers_with_prices:,}")

    # Non-zero candle volume (we get this from the per-market endpoint).
    if "volume" in p.columns:
        p_vol = pd.to_numeric(p["volume"], errors="coerce").fillna(0)
        nonzero = int((p_vol > 0).sum())
        print(f"candle rows with volume>0: {nonzero:,} ({nonzero / len(p) * 100:.1f}%)")

    # Coverage against markets table, if available.
    if markets is not None and not markets.empty:
        m_tickers = set(markets["ticker"].unique())
        p_tickers = set(p["ticker"].unique())
        covered = m_tickers & p_tickers
        print(
            f"coverage: {len(covered)} / {len(m_tickers)} markets have prices "
            f"({len(covered) / len(m_tickers) * 100:.1f}%)"
        )
        if covered and "category" in markets.columns:
            cat_of = dict(zip(markets["ticker"], markets["category"]))
            per_cat_cov: dict[str, tuple[int, int]] = {}
            for t in m_tickers:
                c = cat_of.get(t, "UNKNOWN")
                tot, cov = per_cat_cov.get(c, (0, 0))
                per_cat_cov[c] = (tot + 1, cov + (1 if t in covered else 0))
            print()
            print("coverage by category:")
            print(f"{'category':<25} {'markets':>8} {'with_prices':>12}")
            for c, (tot, cov) in sorted(per_cat_cov.items(), key=lambda kv: -kv[1][1]):
                print(f"  {c:<25} {tot:>8d} {cov:>12d}")


def _news_summary() -> None:
    _section("news")
    n = read_dataset("news")
    if n.empty:
        print("NEWS: none on disk. News strategy (§10.5) needs data/scraper_news.py run.")
        return
    print(f"news rows: {len(n):,}")
    by_source = n.groupby("source").size().sort_values(ascending=False)
    print(by_source.to_string())


def _features_summary() -> None:
    _section("features")
    f = read_dataset("features")
    if f.empty:
        print("FEATURES: none on disk (computed lazily by the feature pipeline).")
        return
    print(f"feature rows: {len(f):,}")


def _models_summary() -> None:
    from moneybutton.models.registry import list_models

    _section("models")
    models = list_models()
    if not models:
        print("none. Run: uv run python scripts/train_calibration_v1.py")
        return
    for entry in models:
        fp = entry.feature_schema.get("fingerprint", "?")
        n_cols = entry.metadata.get("feature_columns_count", "?")
        print(f"  {entry.path.name:<30} fp={fp[:12]}...  cols={n_cols}")


def main() -> int:
    markets = _markets_summary()
    if markets is not None:
        _markets_by_category(markets)
        _markets_by_month(markets)
    _prices_summary(markets)
    _news_summary()
    _features_summary()
    _models_summary()

    print()
    print("=" * 70)
    print("hints")
    print("=" * 70)
    if markets is None:
        print("  run scripts/backfill_via_events.py first")
    else:
        p = read_dataset("prices")
        if p.empty:
            print("  run scripts/backfill_prices.py to pull candlesticks")
        else:
            # A "healthy enough to train" heuristic.
            traded_tickers = int(p[p["last_price_close"].notna()]["ticker"].nunique()) if len(p) else 0
            if traded_tickers < 50:
                print(f"  only {traded_tickers} tickers have any real trade data — probably")
                print(f"  too thin to train meaningfully. Expand the event backfill:")
                print(f"    uv run python scripts/backfill_via_events.py "
                      f"--since 2024-01-01 --categories Elections Politics Economics Sports")
                print(f"  then rerun scripts/backfill_prices.py")
            else:
                print(f"  {traded_tickers} tickers have trade rows — try:")
                print("    uv run python scripts/train_calibration_v1.py")
                print("    uv run python scripts/walk_forward_calibration.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
