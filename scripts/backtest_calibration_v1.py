"""End-to-end backtest of the calibration strategy on whatever Parquet data
is on disk. Loads calibration_v1 from the registry, evaluates on the test
window, writes an HTML report + prints a text summary.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from moneybutton.backtest.engine import BacktestConfig, run_backtest
from moneybutton.backtest.promotion import check_strategy
from moneybutton.backtest.reports import write_report
from moneybutton.core.config import get_settings
from moneybutton.data.store import read_dataset
from moneybutton.features.common import parse_ts
from moneybutton.features.pipeline import default_as_of_before_close
from moneybutton.models.registry import get_active, load
from moneybutton.strategies.calibration_strat import make_scanner


def main() -> int:
    settings = get_settings()

    markets = read_dataset("markets")
    prices = read_dataset("prices")
    if markets.empty or prices.empty:
        print("No data on disk. Run the synthetic generator or the Kalshi backfill first.")
        return 1
    markets = markets[markets["result"].isin(["yes", "no"])].copy()

    entry = get_active("calibration")
    if entry is None:
        print("No active calibration model. Run scripts/train_calibration_v1.py first.")
        return 1
    model = load(entry)
    print(f"Loaded {entry.path.name} (feature_cols={len(model.feature_columns)})")

    # Use the same test window the model was evaluated on.
    meta = entry.metadata
    test_window = meta.get("test_window") or [None, None]
    window_start = parse_ts(test_window[0]) if test_window[0] else None
    window_end = parse_ts(test_window[1]) if test_window[1] else None
    print(f"test window: {window_start} -> {window_end}")

    scanner = make_scanner(
        model=model,
        min_edge_bps=settings.min_edge_bps_calibration,
        high_confidence_edge_bps=settings.min_edge_bps_calibration * 2,
        min_volume=0.0,
    )
    config = BacktestConfig(
        starting_equity_usd=settings.bankroll_usd,
        slippage_bps=settings.slippage_bps_backtest,
        fee_rate=0.07,
        min_edge_bps=0,  # threshold enforced by scanner
        max_position_usd=settings.max_position_usd,
    )

    result = run_backtest(
        markets=markets,
        prices=prices,
        scanner=scanner,
        config=config,
        as_of_fn=default_as_of_before_close(),
        window_start=window_start,
        window_end=window_end,
    )

    print()
    print("=== BACKTEST RESULT (calibration_v1, test window only) ===")
    print(f"n events scanned: {result.meta['n_events']}")
    print(f"n trades:         {result.num_trades}")
    print(f"hit rate:         {result.hit_rate:.3f}")
    print(f"expectancy/trade: ${result.expectancy_usd:.3f}")
    print(f"Sharpe (ann.):    {result.sharpe:.3f}")
    print(f"Sortino (ann.):   {result.sortino:.3f}")
    print(f"max drawdown:     {result.max_dd_pct:.3f}")
    print(f"Calmar:           {result.calmar:.3f}")
    print(f"final equity:     ${result.meta['final_equity_usd']:.2f}")
    print(f"total return:     {result.meta['total_return_pct']:.2f}%")

    decision = check_strategy("calibration", result)
    print()
    print(f"Promotion decision: {decision.outcome.value}")
    print(f"  reason: {decision.reason}")

    reports_dir = Path(settings.data_dir) / "reports" / "backtest" / "calibration"
    report_path = write_report(
        reports_dir,
        result,
        strategy="calibration",
        title="calibration v1 holdout backtest (synthetic)",
        notes={
            "data_source": "synthetic (sandbox egress blocked from Kalshi demo)",
            "model_path": str(entry.path),
            "model_metadata": {k: v for k, v in meta.items() if k not in ("metrics",)},
            "config": {
                "starting_equity_usd": config.starting_equity_usd,
                "slippage_bps": config.slippage_bps,
                "max_position_usd": config.max_position_usd,
                "min_edge_bps": settings.min_edge_bps_calibration,
            },
            "promotion_decision": decision.outcome.value,
            "promotion_reason": decision.reason,
        },
    )
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
