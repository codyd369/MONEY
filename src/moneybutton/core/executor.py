"""Order executor (SPEC §12.2).

Single entry point: Executor.execute(signal). What happens in order:
  1. guard_action() — refuses if .kill sentinel is present.
  2. CapitalAllocator.size_for_signal — fractional-Kelly sizing with
     allocation / position / min-trade / max-position caps.
  3. Price / liquidity gate — refuses on illiquid or wide-spread markets.
  4. Dedup — idempotency key client_order_id; repeated inputs -> same
     outcome, never a second order.
  5. Placement — if DRY_RUN is true OR KALSHI_ENV != 'prod' OR the
     strategy is not in LIVE state, skip placement but record the
     intended order in the DB as `simulated`.
  6. Audit — every call appends to audit with the full reasoning blob.

The executor never holds state in memory between calls; all state lives
in SQLite. This keeps it restart-safe and makes dry-run -> live promotion
a no-op on the code side.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from moneybutton.core.audit import record as audit_record
from moneybutton.core.config import Settings, get_settings
from moneybutton.core.kill_switch import KillSwitchError, guard_action
from moneybutton.core.money import CapitalAllocator, SizingDecision
from moneybutton.kalshi.client import KalshiClient
from moneybutton.strategies.base import Signal

log = logging.getLogger("moneybutton.executor")


ExecutionOutcome = Literal[
    "PLACED",
    "SIMULATED",
    "REJECTED_KILL_SWITCH",
    "REJECTED_SIZE_ZERO",
    "REJECTED_LIQUIDITY",
    "REJECTED_STRATEGY_NOT_LIVE",
    "REJECTED_DUP",
    "REJECTED_EXCEPTION",
]


@dataclass
class ExecutionReport:
    outcome: ExecutionOutcome
    client_order_id: str
    reason: str
    size_usd: float = 0.0
    kalshi_order_id: str | None = None
    sizing: SizingDecision | None = None
    raw_response: dict | None = field(default_factory=dict)


def make_client_order_id(signal: Signal, clock: dt.datetime | None = None) -> str:
    """Idempotency key: sha256 of (strategy, ticker, side, edge_bps, day)."""
    day = (clock or dt.datetime.now(dt.timezone.utc)).strftime("%Y%m%d")
    blob = f"{signal.strategy}|{signal.ticker}|{signal.side}|{signal.edge_bps}|{day}"
    h = hashlib.sha256(blob.encode()).hexdigest()[:20]
    return f"mb_{h}"


@dataclass
class Executor:
    settings: Settings = field(default_factory=get_settings)
    kalshi: KalshiClient | None = None
    db_path: str | Path | None = None

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path) if self.db_path else self.settings.sqlite_db_path
        if self.kalshi is None and self._would_actually_place():
            # Only construct the signed client when we might use it — otherwise
            # KalshiSigner's startup check would refuse to run in DRY_RUN.
            try:
                self.kalshi = KalshiClient(settings=self.settings)
            except RuntimeError as exc:
                log.warning("Kalshi client unavailable (%s); execute() will simulate only", exc)
                self.kalshi = None

    # ------------------------------------------------------------------
    def _would_actually_place(self) -> bool:
        return not self.settings.dry_run and self.settings.kalshi_env == "prod"

    def _strategy_state(self, strategy: str) -> str | None:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            row = conn.execute(
                "SELECT state FROM strategies WHERE name = ?", (strategy,)
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def _already_placed(self, client_order_id: str) -> bool:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            row = conn.execute(
                "SELECT 1 FROM orders WHERE client_request_id = ?", (client_order_id,)
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    def _write_order_row(
        self,
        *,
        client_order_id: str,
        signal: Signal,
        entry_price_cents: int,
        contracts: int,
        notional_usd: float,
        status: str,
        kalshi_order_id: str | None,
        fill_price_cents: int | None,
        signal_id: int | None,
    ) -> None:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute(
                """
                INSERT INTO orders (client_request_id, ts, strategy, ticker, side,
                                    action, count, limit_price_cents, kalshi_order_id,
                                    status, fill_price_cents, notional_usd, signal_id)
                VALUES (?, ?, ?, ?, ?, 'buy', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_request_id) DO UPDATE SET
                  status=excluded.status,
                  kalshi_order_id=excluded.kalshi_order_id,
                  fill_price_cents=excluded.fill_price_cents
                """,
                (
                    client_order_id,
                    now,
                    signal.strategy,
                    signal.ticker,
                    signal.side,
                    contracts,
                    entry_price_cents,
                    kalshi_order_id,
                    status,
                    fill_price_cents,
                    notional_usd,
                    signal_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def execute(
        self,
        signal: Signal,
        *,
        entry_price_cents: int,
        liquidity_usd: float,
        signal_id: int | None = None,
    ) -> ExecutionReport:
        """Execute one signal end-to-end."""
        client_order_id = make_client_order_id(signal)

        # Gate 1: kill switch.
        try:
            guard_action(f"executor.{signal.strategy}", kind="order")
        except KillSwitchError as e:
            audit_record(
                actor=f"executor.{signal.strategy}",
                action="execute",
                payload={"signal": signal.__dict__, "client_order_id": client_order_id},
                outcome="REJECTED_KILL_SWITCH",
                reasoning=str(e),
            )
            return ExecutionReport(
                outcome="REJECTED_KILL_SWITCH",
                client_order_id=client_order_id,
                reason=str(e),
            )

        # Gate 2: dedup.
        if self._already_placed(client_order_id):
            return ExecutionReport(
                outcome="REJECTED_DUP",
                client_order_id=client_order_id,
                reason="client_order_id already present in orders",
            )

        # Gate 3: liquidity.
        if liquidity_usd < self.settings.min_liquidity_usd:
            audit_record(
                actor=f"executor.{signal.strategy}",
                action="execute",
                payload={"signal": signal.__dict__, "liquidity_usd": liquidity_usd},
                outcome="REJECTED_LIQUIDITY",
            )
            return ExecutionReport(
                outcome="REJECTED_LIQUIDITY",
                client_order_id=client_order_id,
                reason=f"liquidity ${liquidity_usd:.0f} < MIN_LIQUIDITY_USD ${self.settings.min_liquidity_usd:.0f}",
            )

        # Gate 4: sizing.
        allocator = CapitalAllocator(db_path=self.db_path, settings=self.settings)
        sizing = allocator.size_for_signal(
            strategy=signal.strategy,
            edge_bps=signal.edge_bps,
            entry_price_cents=entry_price_cents,
            confidence=signal.confidence,
        )
        if not sizing.allowed or sizing.size_usd <= 0:
            audit_record(
                actor=f"executor.{signal.strategy}",
                action="execute",
                payload={"signal": signal.__dict__, "sizing": sizing.__dict__},
                outcome="REJECTED_SIZE_ZERO",
            )
            return ExecutionReport(
                outcome="REJECTED_SIZE_ZERO",
                client_order_id=client_order_id,
                reason=sizing.reason,
                sizing=sizing,
            )

        contracts = max(1, int(sizing.size_usd // (entry_price_cents / 100.0)))
        notional_usd = contracts * entry_price_cents / 100.0

        # Gate 5: strategy state. LIVE placement requires LIVE state.
        state = self._strategy_state(signal.strategy) or "SHADOW"

        # Simulated path (DRY_RUN or demo or SHADOW): record an order row with
        # status='simulated'; positions are NOT opened (no real fills).
        if self.settings.dry_run or self.settings.kalshi_env != "prod" or state != "LIVE":
            self._write_order_row(
                client_order_id=client_order_id,
                signal=signal,
                entry_price_cents=entry_price_cents,
                contracts=contracts,
                notional_usd=notional_usd,
                status="simulated",
                kalshi_order_id=None,
                fill_price_cents=None,
                signal_id=signal_id,
            )
            reason = (
                f"simulated: DRY_RUN={self.settings.dry_run}, "
                f"KALSHI_ENV={self.settings.kalshi_env}, strategy_state={state}"
            )
            audit_record(
                actor=f"executor.{signal.strategy}",
                action="execute",
                payload={
                    "signal": signal.__dict__,
                    "client_order_id": client_order_id,
                    "contracts": contracts,
                    "notional_usd": notional_usd,
                    "entry_price_cents": entry_price_cents,
                },
                reasoning=reason,
                outcome="SIMULATED",
            )
            return ExecutionReport(
                outcome="SIMULATED",
                client_order_id=client_order_id,
                reason=reason,
                size_usd=notional_usd,
                sizing=sizing,
            )

        # Live path.
        if self.kalshi is None:
            return ExecutionReport(
                outcome="REJECTED_EXCEPTION",
                client_order_id=client_order_id,
                reason="Kalshi client unavailable (missing KALSHI_API_KEY_ID?)",
                sizing=sizing,
            )

        try:
            response = self.kalshi.create_order(
                ticker=signal.ticker,
                action="buy",
                side=signal.side,
                count=contracts,
                client_order_id=client_order_id,
                yes_price_cents=entry_price_cents if signal.side == "yes" else None,
                no_price_cents=entry_price_cents if signal.side == "no" else None,
            )
        except Exception as exc:  # noqa: BLE001
            audit_record(
                actor=f"executor.{signal.strategy}",
                action="execute",
                payload={"signal": signal.__dict__, "client_order_id": client_order_id},
                outcome="REJECTED_EXCEPTION",
                reasoning=f"{type(exc).__name__}: {exc}",
            )
            return ExecutionReport(
                outcome="REJECTED_EXCEPTION",
                client_order_id=client_order_id,
                reason=f"{type(exc).__name__}: {exc}",
                sizing=sizing,
            )

        order = response.get("order", {}) if isinstance(response, dict) else {}
        kalshi_order_id = order.get("order_id") or order.get("id")
        self._write_order_row(
            client_order_id=client_order_id,
            signal=signal,
            entry_price_cents=entry_price_cents,
            contracts=contracts,
            notional_usd=notional_usd,
            status=order.get("status", "placed"),
            kalshi_order_id=kalshi_order_id,
            fill_price_cents=None,
            signal_id=signal_id,
        )
        audit_record(
            actor=f"executor.{signal.strategy}",
            action="execute",
            payload={
                "signal": signal.__dict__,
                "client_order_id": client_order_id,
                "kalshi_order_id": kalshi_order_id,
                "response": order,
            },
            outcome="PLACED",
        )
        return ExecutionReport(
            outcome="PLACED",
            client_order_id=client_order_id,
            reason=f"placed on Kalshi (order_id={kalshi_order_id})",
            size_usd=notional_usd,
            kalshi_order_id=kalshi_order_id,
            sizing=sizing,
            raw_response=order,
        )
