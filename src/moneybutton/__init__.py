"""The Money Button v2 — multi-strategy edge engine for Kalshi.

Safety first: nothing in this package places live orders or publishes content
unless (a) the strategy has been promoted via the backtest gate and the CLI,
(b) DRY_RUN is false, (c) KALSHI_ENV is prod, and (d) the .kill sentinel is
absent. All four must be true.
"""

__version__ = "0.1.0"
