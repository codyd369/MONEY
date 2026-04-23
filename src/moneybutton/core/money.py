"""Bankroll, fractional-Kelly sizing, per-strategy allocation (SPEC §5.2).

Sizing model:
  For a YES signal at market price p (cents -> fraction 0..1) with a model
  probability q = p + edge, the full Kelly fraction on this binary bet is

        f* = q - (1 - q) / b      where b = (1 - p) / p

  This is the f* that maximizes log bankroll growth. We never bet f* —
  we bet KELLY_FRACTION * f* (default 1/10, configurable up to 1/4),
  further scaled by a confidence multiplier and then CAPPED by:
    - MAX_POSITION_USD (per-trade hard cap)
    - remaining allocation room for the strategy
    - BANKROLL_USD (total)

  If the capped number falls below MIN_TRADE_USD, the trade is rejected
  (fees eat tiny positions).

  Callers must pass `edge_bps` that is positive when the SIGNALLED side has
  edge. For a NO signal, the caller computes edge as (1 - market_p) - (1 - q)
  and flips p accordingly before calling; this module doesn't re-derive it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from moneybutton.core.config import STRATEGY_NAMES, Settings

_CONFIDENCE_MULT: dict[str, float] = {"low": 0.5, "med": 1.0, "high": 1.0}


@dataclass(frozen=True)
class SizingDecision:
    size_usd: float
    reason: str
    allowed: bool
    full_kelly: float = 0.0
    fractional_kelly: float = 0.0


class CapitalAllocator:
    """Per-strategy exposure tracking and fractional-Kelly sizing.

    The DB is read on every call (small, indexed); this keeps the allocator
    stateless and always-fresh — there is no cache to get stale when an
    executor places a new order.
    """

    def __init__(self, db_path: str | Path, settings: Settings) -> None:
        self.db_path = Path(db_path)
        self.settings = settings

    # ------------------------------ helpers ------------------------------
    def _bankroll_usd(self) -> float:
        # Prefer the most recent bankroll snapshot from SQLite; fall back to
        # the configured BANKROLL_USD if the bankroll table is empty (fresh
        # install).
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                row = conn.execute(
                    "SELECT balance_usd FROM bankroll ORDER BY ts DESC LIMIT 1"
                ).fetchone()
            if row is not None and row[0] is not None:
                return float(row[0])
        except sqlite3.OperationalError:
            # bankroll table missing — treat as fresh install.
            pass
        return float(self.settings.bankroll_usd)

    # ------------------------------ public ------------------------------
    def allocation_cap_usd(self, strategy: str) -> float:
        if strategy not in self.settings.allocations:
            raise ValueError(f"unknown strategy: {strategy!r}")
        return self._bankroll_usd() * self.settings.allocations[strategy]

    def current_exposure(self, strategy: str) -> float:
        """Sum notional of open positions attributed to `strategy`.

        notional_usd = count * avg_entry_cents / 100
        A position with closed_ts != NULL is excluded.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(count * avg_entry_cents) / 100.0, 0.0) "
                "FROM positions WHERE strategy = ? AND closed_ts IS NULL",
                (strategy,),
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def room_usd(self, strategy: str) -> float:
        return max(0.0, self.allocation_cap_usd(strategy) - self.current_exposure(strategy))

    # ------------------------------ sizing ------------------------------
    def size_for_signal(
        self,
        strategy: str,
        edge_bps: int,
        entry_price_cents: int,
        confidence: str,
    ) -> SizingDecision:
        if strategy not in STRATEGY_NAMES:
            return SizingDecision(0.0, f"unknown strategy: {strategy!r}", False)
        if confidence not in _CONFIDENCE_MULT:
            return SizingDecision(0.0, f"unknown confidence: {confidence!r}", False)
        if not (1 <= entry_price_cents <= 99):
            return SizingDecision(
                0.0,
                f"entry_price_cents={entry_price_cents} out of range [1, 99]",
                False,
            )
        if edge_bps <= 0:
            return SizingDecision(0.0, "edge<=0 on signalled side", False)

        p = entry_price_cents / 100.0
        q = min(p + edge_bps / 10_000.0, 0.999)
        b = (1.0 - p) / p
        full_kelly = q - (1.0 - q) / b
        if full_kelly <= 0:
            # Defensive: edge could be sub-tick; don't bet.
            return SizingDecision(0.0, "full_kelly<=0", False, full_kelly=full_kelly)

        frac = full_kelly * self.settings.kelly_fraction * _CONFIDENCE_MULT[confidence]
        raw_size = frac * self._bankroll_usd()

        per_trade_cap = self.settings.max_position_usd
        allocation_room = self.room_usd(strategy)
        size = min(raw_size, per_trade_cap, allocation_room)

        if size < self.settings.min_trade_usd:
            return SizingDecision(
                0.0,
                f"sized ${size:.2f} < MIN_TRADE_USD=${self.settings.min_trade_usd:.2f}"
                f" (raw=${raw_size:.2f}, cap=${per_trade_cap:.2f}, room=${allocation_room:.2f})",
                False,
                full_kelly=full_kelly,
                fractional_kelly=frac,
            )

        return SizingDecision(
            size_usd=round(size, 2),
            reason=(
                f"sized ${size:.2f} (raw ${raw_size:.2f}, cap ${per_trade_cap:.2f},"
                f" room ${allocation_room:.2f}, kelly {full_kelly:.4f} x"
                f" {self.settings.kelly_fraction} x {_CONFIDENCE_MULT[confidence]})"
            ),
            allowed=True,
            full_kelly=full_kelly,
            fractional_kelly=frac,
        )
