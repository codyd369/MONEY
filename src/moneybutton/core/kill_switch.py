"""Kill switch + daily-loss guard (SPEC §5.3, §5.4).

The kill switch is a single file on disk. Any component about to place an
order or publish content MUST call `guard_action()` first; if the sentinel
file exists, the call raises KillSwitchError and the action is refused.

Rationale for a file-on-disk sentinel:
- Operator can trip it from another shell in O(100ms) via `./kill.sh`.
- Survives process restarts — the bot re-reads it on every iteration.
- No dependency on the DB being healthy; the kill switch must work even
  if SQLite is corrupt.

DailyLossGuard walks the daily_pnl table and, when today's aggregate loss
exceeds DAILY_LOSS_LIMIT_USD, trips the kill switch automatically. It is
driven by the scheduler's safety.health_check job (SPEC §14).
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from moneybutton.core.config import Settings, get_settings


class KillSwitchError(RuntimeError):
    """Raised by guard_action() when the .kill sentinel is present."""


def _resolve_kill_path(kill_file_path: str | Path | None) -> Path:
    if kill_file_path is None:
        return get_settings().kill_file_path
    return Path(kill_file_path)


def is_killed(kill_file_path: str | Path | None = None) -> bool:
    """True if the .kill sentinel exists. Cheap to call repeatedly."""
    return _resolve_kill_path(kill_file_path).exists()


def touch_kill(
    kill_file_path: str | Path | None = None,
    reason: str = "",
    actor: str = "unknown",
) -> None:
    """Create the .kill sentinel. Idempotent; records reason and timestamp.

    Calling this twice appends a second entry; the operator can `cat .kill`
    to see the trip history of the current session.
    """
    path = _resolve_kill_path(kill_file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    line = f"[{ts}] actor={actor} reason={reason!r}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def clear_kill(kill_file_path: str | Path | None = None) -> None:
    """Remove the .kill sentinel. Idempotent."""
    path = _resolve_kill_path(kill_file_path)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def guard_action(
    actor: str,
    kind: str = "order",
    kill_file_path: str | Path | None = None,
) -> None:
    """Raise KillSwitchError if the .kill sentinel is present.

    Every executor / publisher entry point must call this before doing the
    thing. `kind` is 'order' or 'publish' for auditing.
    """
    if is_killed(kill_file_path):
        raise KillSwitchError(
            f"kill switch is engaged (.kill present); refusing {kind} action from {actor}"
        )


@dataclass
class DailyLossGuard:
    """Sum today's realized+unrealized P&L across all strategies; trip on breach.

    `clock` is a callable returning a timezone-aware datetime; injected so
    tests can pin time. `settings` and `kill_file_path` are plumbed so tests
    can use isolated temp files.
    """

    db_path: str | Path
    settings: Settings
    clock: Callable[[], dt.datetime] | None = None
    kill_file_path: str | Path | None = None

    def _now(self) -> dt.datetime:
        if self.clock is not None:
            return self.clock()
        return dt.datetime.now(dt.timezone.utc)

    def _today_str(self) -> str:
        return self._now().astimezone(dt.timezone.utc).date().isoformat()

    def current_loss_usd(self) -> float:
        """Return a non-negative loss number (0 if net positive).

        This intentionally collapses realized+unrealized and collapses across
        strategies: the daily loss cap is a whole-portfolio circuit breaker.
        """
        date_str = self._today_str()
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(realized_usd), 0) + COALESCE(SUM(unrealized_usd), 0) "
                "FROM daily_pnl WHERE date = ?",
                (date_str,),
            ).fetchone()
        total = float(row[0]) if row and row[0] is not None else 0.0
        # Loss is a positive number; net-positive days report 0.
        return max(0.0, -total)

    def check_and_maybe_trip(self) -> bool:
        """Return True iff this call created a new .kill file."""
        already_killed = is_killed(self.kill_file_path)
        loss = self.current_loss_usd()
        if loss >= self.settings.daily_loss_limit_usd:
            if not already_killed:
                reason = (
                    f"daily_loss_cap_breached: loss=${loss:.2f} "
                    f">= DAILY_LOSS_LIMIT_USD=${self.settings.daily_loss_limit_usd:.2f}"
                )
                touch_kill(
                    self.kill_file_path, reason=reason, actor="safety.daily_loss_guard"
                )
                return True
        return False
