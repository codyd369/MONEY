"""Walk-forward backtest: retrain the model per fold (SPEC §11.1).

Holdout (step 14) is honest but pessimistic against the last fold only.
Walk-forward evaluates the strategy under the same retrain cadence the live
system uses, so the promotion decision reflects how the strategy performs
as it ages.

Scheme:
    fold 0: train on [T0, T0+train_window), test on [T0+train_window+embargo, T0+train_window+step)
    fold 1: train on [T0, T0+train_window+step), test on [T0+train_window+step+embargo, T0+train_window+2*step)
    ...
  The "step" slides the test window forward each fold while the train
  window grows (expanding window). Use fixed-window for shorter-memory
  strategies — not needed for calibration in v1.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from moneybutton.backtest.engine import BacktestConfig, run_backtest
from moneybutton.backtest.metrics import BacktestResult, summarize
from moneybutton.features.pipeline import default_as_of_before_close
from moneybutton.models.calibration import CalibratedClassifier, time_split, train


@dataclass
class WalkForwardSpec:
    start: dt.datetime
    end: dt.datetime
    initial_train_days: int = 180
    step_days: int = 45
    embargo_days: int = 7


TrainerFn = Callable[[pd.DataFrame, list[str]], CalibratedClassifier]


def default_trainer(frame: pd.DataFrame, feature_cols: list[str]) -> CalibratedClassifier:
    """Split the given frame 70/30 train/val and train a CalibratedClassifier."""
    ts = pd.to_datetime(frame["as_of_ts"], utc=True)
    train_end = ts.quantile(0.7).to_pydatetime()
    val_end = ts.max().to_pydatetime() + dt.timedelta(hours=1)
    # test here is empty; the outer walk-forward provides OOS evaluation.
    test_end = val_end
    split = time_split(
        frame,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        feature_columns=feature_cols,
        embargo=dt.timedelta(days=3),
    )
    model, _ = train(split)
    return model


def walk_forward(
    *,
    markets: pd.DataFrame,
    prices: pd.DataFrame,
    build_training_frame_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    scanner_factory: Callable[[CalibratedClassifier], Callable],
    trainer: TrainerFn = default_trainer,
    spec: WalkForwardSpec,
    config: BacktestConfig,
    as_of_fn: Callable[[dict], dt.datetime] | None = None,
) -> BacktestResult:
    """Run expanding-window walk-forward over [spec.start, spec.end)."""
    as_of_fn = as_of_fn or default_as_of_before_close()

    markets_with_dt = markets.copy()
    markets_with_dt["_close_dt"] = pd.to_datetime(
        markets_with_dt.get("close_time"), utc=True, errors="coerce"
    )

    all_trades: list[dict] = []
    equity = config.starting_equity_usd
    equity_curve: list[float] = [equity]
    fold_reports: list[dict] = []

    train_end = spec.start + dt.timedelta(days=spec.initial_train_days)
    fold_idx = 0
    while True:
        test_start = train_end + dt.timedelta(days=spec.embargo_days)
        test_end = test_start + dt.timedelta(days=spec.step_days)
        if test_start >= spec.end:
            break
        test_end = min(test_end, spec.end)

        train_markets = markets_with_dt[
            (markets_with_dt["_close_dt"] >= pd.Timestamp(spec.start))
            & (markets_with_dt["_close_dt"] < pd.Timestamp(train_end))
        ]
        test_markets = markets_with_dt[
            (markets_with_dt["_close_dt"] >= pd.Timestamp(test_start))
            & (markets_with_dt["_close_dt"] < pd.Timestamp(test_end))
        ]
        if train_markets.empty or test_markets.empty:
            train_end = test_end
            fold_idx += 1
            continue

        train_frame = build_training_frame_fn(train_markets, prices)
        id_cols = {"ticker", "as_of_ts", "label_resolved"}
        feature_cols: list[str] = []
        for c in train_frame.columns:
            if c in id_cols:
                continue
            train_frame[c] = pd.to_numeric(train_frame[c], errors="coerce")
            if train_frame[c].notna().any():
                feature_cols.append(c)

        model = trainer(train_frame, feature_cols)
        scanner = scanner_factory(model)

        # Bump equity of the per-fold run so the concatenated equity curve is
        # consistent (each fold starts where the last one ended).
        fold_config = BacktestConfig(
            starting_equity_usd=equity,
            slippage_bps=config.slippage_bps,
            fee_rate=config.fee_rate,
            min_edge_bps=config.min_edge_bps,
            max_position_usd=config.max_position_usd,
        )
        fold_result = run_backtest(
            markets=test_markets,
            prices=prices,
            scanner=scanner,
            config=fold_config,
            as_of_fn=as_of_fn,
            window_start=test_start,
            window_end=test_end,
        )
        # Skip the first entry of the fold's equity_curve (that's the starting
        # equity we already recorded at the end of the prior fold).
        for eq in fold_result.equity_curve[1:]:
            equity_curve.append(eq)
        all_trades.extend(fold_result.trades)
        equity = fold_result.meta["final_equity_usd"]

        fold_reports.append(
            {
                "fold": fold_idx,
                "train_window": (spec.start.isoformat(), train_end.isoformat()),
                "test_window": (test_start.isoformat(), test_end.isoformat()),
                "n_trades": fold_result.num_trades,
                "hit_rate": fold_result.hit_rate,
                "expectancy_usd": fold_result.expectancy_usd,
                "final_equity": equity,
                "n_train_markets": int(len(train_markets)),
                "n_test_markets": int(len(test_markets)),
            }
        )

        fold_idx += 1
        train_end = test_end  # expanding window

    if not all_trades:
        # Degenerate: no OOS trades across any fold. Return an empty result.
        return BacktestResult(
            sharpe=0.0,
            sortino=0.0,
            max_dd_pct=0.0,
            hit_rate=0.0,
            num_trades=0,
            expectancy_usd=0.0,
            calmar=0.0,
            equity_curve=equity_curve,
            trades=[],
            meta={"folds": fold_reports, "final_equity_usd": equity},
        )

    years = max(1 / 365.0, (spec.end - spec.start).days / 365.25)
    result = summarize(equity_curve, all_trades, years=years)
    result.meta = {
        "folds": fold_reports,
        "n_folds": len(fold_reports),
        "starting_equity_usd": config.starting_equity_usd,
        "final_equity_usd": equity,
        "total_return_pct": (equity / config.starting_equity_usd - 1) * 100.0,
        "window_start": spec.start.isoformat(),
        "window_end": spec.end.isoformat(),
    }
    return result
