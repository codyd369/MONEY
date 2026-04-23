"""Kalshi fee schedule (SPEC §11.3).

Kalshi's posted formula is roughly:
    fee_per_contract_cents = ceil( 0.07 * 100 * p * (1 - p) )
in cents, where p is the trade price as a decimal 0..1. This is symmetric
around p=0.5 and peaks at 1.75 cents per contract. Verify against the
current trading-api.readme.io fee page before going live (SPEC §17).

This module keeps the formula in one place so every component (backtest
fill model, executor pre-check, sizing sanity) uses the same number.
"""

from __future__ import annotations

import math


FEE_RATE = 0.07  # Kalshi's posted base rate; confirm at go-live time.


def fee_per_contract_cents(price_cents: int, fee_rate: float = FEE_RATE) -> int:
    """Per-contract fee in integer cents, rounded up to nearest cent.

    price_cents is 1..99 inclusive (Kalshi tick size is 1 cent).
    Returns 0 for invalid prices; the executor rejects those earlier.
    """
    if not (1 <= price_cents <= 99):
        return 0
    p = price_cents / 100.0
    raw = fee_rate * 100 * p * (1 - p)
    return int(math.ceil(raw))


def trade_fee_usd(price_cents: int, contracts: int, fee_rate: float = FEE_RATE) -> float:
    """Total fee in USD for buying `contracts` at `price_cents`."""
    if contracts <= 0:
        return 0.0
    per = fee_per_contract_cents(price_cents, fee_rate)
    return per * contracts / 100.0
