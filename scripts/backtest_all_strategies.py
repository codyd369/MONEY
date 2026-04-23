"""End-to-end backtest sweep across all 5 strategies (§16 step 22).

For strategies with historical data on disk (calibration), runs the real
walk-forward backtest. For the structural strategies (arbitrage, consistency,
drift) and data-thin ones (news), emits a SHADOW-no-data result with a note
that real backtests require additional data sources.

Produces:
  - Console summary table (Sharpe, hit rate, DD, trades, decision)
  - Per-strategy HTML report under data/reports/backtest/<strategy>/
  - Updates strategies table in SQLite (metrics recorded; state unchanged
    unless PROMOTED).
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pandas as pd

from moneybutton.backtest.engine import BacktestConfig
from moneybutton.backtest.metrics import BacktestResult
from moneybutton.backtest.promotion import check_strategy
from moneybutton.backtest.reports import write_report
from moneybutton.backtest.walk_forward import WalkForwardSpec, walk_forward
from moneybutton.core.config import get_settings
from moneybutton.core.db import init_db
from moneybutton.data.store import read_dataset
from moneybutton.features.pipeline import build_training_frame, default_as_of_before_close
from moneybutton.strategies.calibration_strat import make_scanner


def _empty_result(note: str) -> BacktestResult:
    return BacktestResult(
        sharpe=0.0,
        sortino=0.0,
        max_dd_pct=0.0,
        hit_rate=0.0,
        num_trades=0,
        expectancy_usd=0.0,
        calmar=0.0,
        equity_curve=[],
        trades=[],
        meta={"note": note},
    )


def _run_calibration() -> BacktestResult:
    settings = get_settings()
    markets = read_dataset("markets")
    prices = read_dataset("prices")
    if markets.empty or prices.empty:
        return _empty_result("no data on disk for calibration backtest")
    markets = markets[markets["result"].isin(["yes", "no"])].copy()
    close_ts = pd.to_datetime(markets["close_time"], utc=True, errors="coerce")
    start = close_ts.min().to_pydatetime()
    end = close_ts.max().to_pydatetime() + dt.timedelta(days=1)

    def _build(m: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
        return build_training_frame(m, p, default_as_of_before_close())

    def _scanner_factory(model):
        return make_scanner(
            model=model,
            min_edge_bps=settings.min_edge_bps_calibration,
            high_confidence_edge_bps=settings.min_edge_bps_calibration * 2,
        )

    return walk_forward(
        markets=markets,
        prices=prices,
        build_training_frame_fn=_build,
        scanner_factory=_scanner_factory,
        spec=WalkForwardSpec(
            start=start,
            end=end,
            initial_train_days=180,
            step_days=45,
            embargo_days=7,
        ),
        config=BacktestConfig(
            starting_equity_usd=settings.bankroll_usd,
            slippage_bps=settings.slippage_bps_backtest,
            max_position_usd=settings.max_position_usd,
        ),
    )


def main() -> int:
    settings = get_settings()
    init_db()
    results: dict[str, BacktestResult] = {}

    print("running calibration walk-forward...")
    results["calibration"] = _run_calibration()

    # The four structural / data-thin strategies need data we don't have on
    # disk (Polymarket history, MEE cluster history, aligned news, etc.)
    # Produce an honest "no data" result so the DB + report system exercises
    # end-to-end.
    shadow_note = (
        "No historical data on disk for this strategy yet. "
        "Run the relevant scraper first: polymarket/odds/news. "
        "Scanner + unit tests (see tests/) verified."
    )
    for strategy in ("arbitrage", "consistency", "news", "drift"):
        results[strategy] = _empty_result(shadow_note)

    # Record metrics and run promotion gate for each.
    print()
    print(f"{'strategy':<15} {'trades':>8} {'hit':>6} {'Sharpe':>8} {'maxDD':>8} {'decision':<40}")
    print("-" * 95)
    for strategy, result in results.items():
        decision = check_strategy(strategy, result, persist_to=settings.sqlite_db_path)
        print(
            f"{strategy:<15} {result.num_trades:>8d} "
            f"{result.hit_rate:>6.3f} {result.sharpe:>8.3f} {result.max_dd_pct:>8.3f} "
            f"{decision.outcome.value:<40}"
        )
        reports_dir = Path(settings.data_dir) / "reports" / "backtest" / strategy
        try:
            write_report(
                reports_dir,
                result,
                strategy=strategy,
                title=f"{strategy} backtest (step 22 sweep)",
                notes={
                    "meta": result.meta,
                    "promotion_decision": decision.outcome.value,
                    "promotion_reason": decision.reason,
                },
            )
        except Exception as e:  # noqa: BLE001
            print(f"  (report failed for {strategy}: {e})")

    print()
    print("Current strategies table:")
    with sqlite3.connect(str(settings.sqlite_db_path)) as conn:
        rows = conn.execute(
            "SELECT name, state, allocation_pct, last_backtest_sharpe, "
            "last_backtest_hitrate, last_backtest_max_dd, last_backtest_num_trades "
            "FROM strategies ORDER BY name"
        ).fetchall()
    print(f"{'name':<15} {'state':<10} {'alloc':>6} {'Sharpe':>8} {'hit':>6} {'DD':>6} {'n':>6}")
    for row in rows:
        n, st, al, sh, hr, dd, nt = row
        sh_s = f"{sh:.3f}" if sh is not None else "-"
        hr_s = f"{hr:.3f}" if hr is not None else "-"
        dd_s = f"{dd:.3f}" if dd is not None else "-"
        nt_s = str(nt) if nt is not None else "-"
        print(f"{n:<15} {st:<10} {al:>6.2f} {sh_s:>8} {hr_s:>6} {dd_s:>6} {nt_s:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
