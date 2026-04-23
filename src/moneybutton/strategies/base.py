"""Strategy ABC (SPEC §10.1)."""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Signal:
    """What a strategy emits to the orchestrator."""

    strategy: str
    ticker: str
    side: Literal["yes", "no"]
    edge_bps: int
    confidence: Literal["low", "med", "high"]
    suggested_size_usd: float
    reasoning: dict = field(default_factory=dict)
    expires_at: dt.datetime | None = None


class Strategy(ABC):
    """Every strategy runs the same protocol: scan -> Signal list, backtest -> BacktestResult."""

    name: str = "base"

    @abstractmethod
    def scan(self, now: dt.datetime) -> list[Signal]:
        """Produce live signals for `now`."""

    @abstractmethod
    def backtest(self, start: dt.date, end: dt.date):
        """Return a BacktestResult over [start, end)."""
