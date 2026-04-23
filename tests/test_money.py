"""CapitalAllocator tests (SPEC §5.2, §5.4).

Guarantees under test:
  1. Per-strategy allocation cap is enforced: a strategy cannot open new
     exposure above BANKROLL_USD * ALLOCATION_X.
  2. Per-trade size is fractional-Kelly-on-edge, capped by MAX_POSITION_USD
     and the strategy's remaining allocation room.
  3. Below MIN_TRADE_USD => SizingDecision(allowed=False), zero size.
  4. Non-positive edge => allowed=False (no negative-EV trades).
  5. Invalid entry price (0 or >=100 cents) => allowed=False.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _seed_position(
    db_path: Path,
    ticker: str,
    strategy: str,
    side: str,
    count: int,
    entry_cents: int,
    opened: str = "2026-04-23T12:00:00+00:00",
) -> None:
    """Seed an open position. notional = count * entry_cents / 100 USD."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO positions (ticker, strategy, side, count, avg_entry_cents,
                                   opened_ts, unrealized_usd, realized_usd)
            VALUES (?, ?, ?, ?, ?, ?, 0, 0)
            """,
            (ticker, strategy, side, count, entry_cents, opened),
        )
        conn.commit()


# ------------------------------ allocation -------------------------------


def test_allocation_cap_for_each_strategy(tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.money import CapitalAllocator

    settings = get_settings()
    alloc = CapitalAllocator(db_path=tmp_db, settings=settings)
    # Bankroll $500 * allocation% per strategy.
    assert alloc.allocation_cap_usd("calibration") == pytest.approx(200.0)
    assert alloc.allocation_cap_usd("arbitrage") == pytest.approx(125.0)
    assert alloc.allocation_cap_usd("consistency") == pytest.approx(75.0)
    assert alloc.allocation_cap_usd("news") == pytest.approx(50.0)
    assert alloc.allocation_cap_usd("drift") == pytest.approx(50.0)


def test_current_exposure_sums_open_positions(tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.money import CapitalAllocator

    # 10 contracts at 40c = $4; 5 contracts at 60c = $3. Total $7 for calibration.
    _seed_position(tmp_db, "KX1", "calibration", "yes", 10, 40)
    _seed_position(tmp_db, "KX2", "calibration", "no", 5, 60)
    _seed_position(tmp_db, "KX3", "arbitrage", "yes", 20, 25)  # $5 for arbitrage

    settings = get_settings()
    alloc = CapitalAllocator(db_path=tmp_db, settings=settings)
    assert alloc.current_exposure("calibration") == pytest.approx(7.0)
    assert alloc.current_exposure("arbitrage") == pytest.approx(5.0)
    assert alloc.current_exposure("news") == pytest.approx(0.0)


def test_current_exposure_excludes_closed_positions(tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.money import CapitalAllocator

    with sqlite3.connect(tmp_db) as conn:
        conn.execute(
            """
            INSERT INTO positions (ticker, strategy, side, count, avg_entry_cents,
                                   opened_ts, closed_ts, unrealized_usd, realized_usd)
            VALUES ('KXCLOSED', 'calibration', 'yes', 10, 40,
                    '2026-04-22T00:00:00+00:00', '2026-04-22T18:00:00+00:00', 0, 1.0)
            """
        )
        conn.commit()

    settings = get_settings()
    alloc = CapitalAllocator(db_path=tmp_db, settings=settings)
    assert alloc.current_exposure("calibration") == pytest.approx(0.0)


def test_room_respects_cap(tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.money import CapitalAllocator

    # Calibration cap $200; seed $150 of exposure => room = $50.
    _seed_position(tmp_db, "KXBIG", "calibration", "yes", 300, 50)  # $150

    settings = get_settings()
    alloc = CapitalAllocator(db_path=tmp_db, settings=settings)
    assert alloc.room_usd("calibration") == pytest.approx(50.0)


# ------------------------------ sizing ----------------------------------


def test_positive_edge_produces_positive_size(tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.money import CapitalAllocator

    settings = get_settings()
    alloc = CapitalAllocator(db_path=tmp_db, settings=settings)
    # p = 40c, edge 1000 bps => q = 0.50 (coin flip vs market at 40c).
    decision = alloc.size_for_signal(
        strategy="calibration",
        edge_bps=1000,
        entry_price_cents=40,
        confidence="med",
    )
    assert decision.allowed is True
    assert decision.size_usd > 0


def test_negative_edge_blocked(tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.money import CapitalAllocator

    settings = get_settings()
    alloc = CapitalAllocator(db_path=tmp_db, settings=settings)
    decision = alloc.size_for_signal(
        strategy="calibration",
        edge_bps=-500,
        entry_price_cents=40,
        confidence="med",
    )
    assert decision.allowed is False
    assert decision.size_usd == 0.0


def test_zero_edge_blocked(tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.money import CapitalAllocator

    settings = get_settings()
    alloc = CapitalAllocator(db_path=tmp_db, settings=settings)
    decision = alloc.size_for_signal(
        strategy="calibration",
        edge_bps=0,
        entry_price_cents=40,
        confidence="med",
    )
    assert decision.allowed is False


@pytest.mark.parametrize("bad_price", [0, 100, 150, -1])
def test_invalid_entry_price_blocked(tmp_db, bad_price):
    from moneybutton.core.config import get_settings
    from moneybutton.core.money import CapitalAllocator

    settings = get_settings()
    alloc = CapitalAllocator(db_path=tmp_db, settings=settings)
    decision = alloc.size_for_signal(
        strategy="calibration",
        edge_bps=500,
        entry_price_cents=bad_price,
        confidence="med",
    )
    assert decision.allowed is False


def test_never_exceeds_max_position(tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.money import CapitalAllocator

    settings = get_settings()
    alloc = CapitalAllocator(db_path=tmp_db, settings=settings)
    # Huge edge, tiny price => raw Kelly size would be very large.
    decision = alloc.size_for_signal(
        strategy="calibration",
        edge_bps=4000,  # 40% edge
        entry_price_cents=5,
        confidence="high",
    )
    assert decision.allowed is True
    assert decision.size_usd <= settings.max_position_usd + 1e-9


def test_never_exceeds_allocation_room(tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.money import CapitalAllocator

    # Fill the calibration allocation to just $10 left.
    _seed_position(tmp_db, "KXFILL", "calibration", "yes", 380, 50)  # $190 used, $10 room
    settings = get_settings()
    alloc = CapitalAllocator(db_path=tmp_db, settings=settings)
    decision = alloc.size_for_signal(
        strategy="calibration",
        edge_bps=2000,
        entry_price_cents=40,
        confidence="high",
    )
    # Room is $10 but MIN_TRADE_USD=5 so $10 is OK.
    assert decision.allowed is True
    assert decision.size_usd <= 10.0 + 1e-9


def test_room_below_min_trade_blocks(tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.money import CapitalAllocator

    # $3 of room — below MIN_TRADE_USD=$5.
    _seed_position(tmp_db, "KXFILL", "calibration", "yes", 394, 50)  # $197 used, $3 room
    settings = get_settings()
    alloc = CapitalAllocator(db_path=tmp_db, settings=settings)
    decision = alloc.size_for_signal(
        strategy="calibration",
        edge_bps=2000,
        entry_price_cents=40,
        confidence="high",
    )
    assert decision.allowed is False
    assert decision.size_usd == 0.0
    assert "min" in decision.reason.lower()


def test_confidence_affects_size(tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.money import CapitalAllocator

    settings = get_settings()
    alloc = CapitalAllocator(db_path=tmp_db, settings=settings)
    # Use a 2000 bps edge so that even the low-confidence (0.5x) path
    # clears MIN_TRADE_USD; otherwise the property we're checking is masked.
    low = alloc.size_for_signal("calibration", 2000, 40, "low")
    med = alloc.size_for_signal("calibration", 2000, 40, "med")
    high = alloc.size_for_signal("calibration", 2000, 40, "high")
    assert low.allowed and med.allowed and high.allowed
    assert low.size_usd <= med.size_usd <= high.size_usd + 1e-9


def test_unknown_strategy_rejected(tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.money import CapitalAllocator

    settings = get_settings()
    alloc = CapitalAllocator(db_path=tmp_db, settings=settings)
    decision = alloc.size_for_signal("not_a_strategy", 1000, 40, "med")
    assert decision.allowed is False


def test_unknown_confidence_rejected(tmp_db):
    from moneybutton.core.config import get_settings
    from moneybutton.core.money import CapitalAllocator

    settings = get_settings()
    alloc = CapitalAllocator(db_path=tmp_db, settings=settings)
    decision = alloc.size_for_signal("calibration", 1000, 40, "super-high")
    assert decision.allowed is False
