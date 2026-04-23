"""Walk-forward backtest of the calibration strategy (SPEC §16 step 15).

Expanding-window retrain. Evaluates the strategy under the same retrain
cadence the live system would use, so the promotion decision isn't held
hostage to one lucky/unlucky holdout window.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from moneybutton.backtest.engine import BacktestConfig
from moneybutton.backtest.promotion import check_strategy
from moneybutton.backtest.reports import write_report
from moneybutton.backtest.walk_forward import WalkForwardSpec, walk_forward
from moneybutton.core.config import get_settings
from moneybutton.data.store import read_dataset
from moneybutton.features.pipeline import build_training_frame, default_as_of_before_close
from moneybutton.strategies.calibration_strat import make_scanner


def _build(markets: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    return build_training_frame(markets, prices, default_as_of_before_close())


def main() -> int:
    settings = get_settings()
    markets = read_dataset("markets")
    prices = read_dataset("prices")
    if markets.empty or prices.empty:
        print("No data on disk.")
        return 1
    markets = markets[markets["result"].isin(["yes", "no"])].copy()
    close_ts = pd.to_datetime(markets["close_time"], utc=True, errors="coerce")
    start = close_ts.min().to_pydatetime()
    end = close_ts.max().to_pydatetime() + dt.timedelta(days=1)
    print(f"walk-forward window: {start} -> {end}")

    spec = WalkForwardSpec(
        start=start,
        end=end,
        initial_train_days=180,
        step_days=45,
        embargo_days=7,
    )

    def _scanner_factory(model):
        return make_scanner(
            model=model,
            min_edge_bps=settings.min_edge_bps_calibration,
            high_confidence_edge_bps=settings.min_edge_bps_calibration * 2,
        )

    result = walk_forward(
        markets=markets,
        prices=prices,
        build_training_frame_fn=_build,
        scanner_factory=_scanner_factory,
        spec=spec,
        config=BacktestConfig(
            starting_equity_usd=settings.bankroll_usd,
            slippage_bps=settings.slippage_bps_backtest,
            max_position_usd=settings.max_position_usd,
        ),
    )

    print()
    print("=== WALK-FORWARD BACKTEST RESULT (calibration) ===")
    print(f"folds run:        {result.meta.get('n_folds', 0)}")
    print(f"total trades:     {result.num_trades}")
    print(f"hit rate:         {result.hit_rate:.3f}")
    print(f"expectancy/trade: ${result.expectancy_usd:.3f}")
    print(f"Sharpe (ann.):    {result.sharpe:.3f}")
    print(f"Sortino (ann.):   {result.sortino:.3f}")
    print(f"max drawdown:     {result.max_dd_pct:.3f}")
    print(f"Calmar:           {result.calmar:.3f}")
    print(f"final equity:     ${result.meta['final_equity_usd']:.2f}")
    print(f"total return:     {result.meta['total_return_pct']:.2f}%")

    if result.meta.get("folds"):
        print()
        print("per-fold:")
        for f in result.meta["folds"]:
            print(
                f"  fold {f['fold']:>2}: trained on {f['n_train_markets']:>4} markets, "
                f"tested on {f['n_test_markets']:>3} -> {f['n_trades']:>3} trades, "
                f"hit {f['hit_rate']:.2f}, final ${f['final_equity']:.2f}"
            )

    decision = check_strategy("calibration", result)
    print()
    print(f"Promotion decision: {decision.outcome.value}")
    print(f"  reason: {decision.reason}")

    reports_dir = Path(settings.data_dir) / "reports" / "backtest" / "calibration"
    report_path = write_report(
        reports_dir,
        result,
        strategy="calibration",
        title="calibration walk-forward backtest (synthetic)",
        notes={
            "data_source": "synthetic",
            "folds": result.meta.get("folds"),
            "spec": {
                "initial_train_days": spec.initial_train_days,
                "step_days": spec.step_days,
                "embargo_days": spec.embargo_days,
            },
            "promotion_decision": decision.outcome.value,
        },
    )
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
