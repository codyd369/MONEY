"""HTML backtest report (SPEC §11.6)."""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
from pathlib import Path

import numpy as np

from moneybutton.backtest.metrics import BacktestResult


def render(
    result: BacktestResult,
    *,
    strategy: str,
    title: str | None = None,
    notes: dict | None = None,
) -> str:
    """Render a self-contained HTML report for `result`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _fig_to_b64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    # Equity curve
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if result.equity_curve:
        ax.plot(result.equity_curve)
    ax.set_title("Equity curve")
    ax.set_xlabel("event #")
    ax.set_ylabel("equity (USD)")
    img_eq = _fig_to_b64(fig)

    # Drawdown
    fig, ax = plt.subplots(figsize=(8, 2.5))
    if result.equity_curve:
        eq = np.array(result.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, 1) * -1
        ax.fill_between(range(len(dd)), dd, 0, step="pre")
    ax.set_title("Drawdown")
    ax.set_xlabel("event #")
    ax.set_ylabel("drawdown")
    img_dd = _fig_to_b64(fig)

    # Per-trade pnl scatter
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if result.trades:
        xs = [t.get("entry_price_cents", 0) for t in result.trades]
        ys = [t.get("pnl_usd", 0) for t in result.trades]
        ax.scatter(xs, ys, s=12, alpha=0.6)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title("P&L per trade vs entry price")
    ax.set_xlabel("entry price (cents)")
    ax.set_ylabel("P&L (USD)")
    img_scatter = _fig_to_b64(fig)

    meta = result.meta or {}
    notes_html = ""
    if notes:
        notes_html = "<h2>Notes</h2><pre>" + json.dumps(notes, indent=2, default=str) + "</pre>"

    title = title or f"backtest: {strategy}"
    return f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 980px; margin: 2em auto; color: #222; }}
h1, h2 {{ border-bottom: 1px solid #eee; padding-bottom: 4px; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
img {{ max-width: 100%; }}
.banner {{ padding: 10px; border-radius: 4px; margin: 1em 0; }}
.banner.overfit {{ background: #fff3cd; border: 1px solid #ffeeba; }}
</style></head>
<body>
<h1>{title}</h1>
<p>Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}</p>

<h2>Summary</h2>
<table>
<tr><th>metric</th><th>value</th></tr>
<tr><td>n trades</td><td>{result.num_trades}</td></tr>
<tr><td>hit rate</td><td>{result.hit_rate:.3f}</td></tr>
<tr><td>expectancy / trade</td><td>${result.expectancy_usd:.3f}</td></tr>
<tr><td>Sharpe (ann.)</td><td>{result.sharpe:.3f}</td></tr>
<tr><td>Sortino (ann.)</td><td>{result.sortino:.3f}</td></tr>
<tr><td>max drawdown</td><td>{result.max_dd_pct:.3f}</td></tr>
<tr><td>Calmar</td><td>{result.calmar:.3f}</td></tr>
<tr><td>final equity</td><td>${meta.get("final_equity_usd", 0):.2f}</td></tr>
<tr><td>total return</td><td>{meta.get("total_return_pct", 0):.2f}%</td></tr>
</table>

{('<div class="banner overfit">Sharpe &gt; 2.0 — this backtest is almost certainly overfit. Live performance will be materially worse. Require explicit operator override to promote.</div>' if result.sharpe > 2.0 else '')}

<h2>Equity curve</h2>
<img src="data:image/png;base64,{img_eq}" />

<h2>Drawdown</h2>
<img src="data:image/png;base64,{img_dd}" />

<h2>P&amp;L vs entry price</h2>
<img src="data:image/png;base64,{img_scatter}" />

{notes_html}
</body></html>
"""


def write_report(
    out_dir: str | Path,
    result: BacktestResult,
    *,
    strategy: str,
    title: str | None = None,
    notes: dict | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{strategy}_{ts}.html"
    path.write_text(render(result, strategy=strategy, title=title, notes=notes), encoding="utf-8")
    return path
