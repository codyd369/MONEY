"""Polymarket Gamma API read-only client (SPEC §7.2).

Gamma is Polymarket's public read-only REST layer. No auth required for
market data. Base URL: https://gamma-api.polymarket.com.

Used exclusively for READ — arbitrage execution on Polymarket requires
signing via their CLOB (not implemented here; SPEC §7.2 flagged as risk).

Live arbitrage requires cross-platform execution, which means actually
placing orders on both. Without Polymarket wallet signing we can only
emit SIGNALS (and rely on the operator to execute manually, or stay
Kalshi-only). The strategy file makes this explicit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger("moneybutton.polymarket")

BASE_URL = "https://gamma-api.polymarket.com"

_RETRIABLE = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)


class PolymarketHTTPError(RuntimeError):
    pass


@dataclass
class PolymarketClient:
    client: httpx.Client | None = None
    _owns: bool = False

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = httpx.Client(timeout=30.0)
            self._owns = True

    def close(self) -> None:
        if self._owns and self.client:
            self.client.close()

    def __enter__(self) -> "PolymarketClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(_RETRIABLE),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _get(self, path: str, params: dict | None = None) -> Any:
        assert self.client is not None
        r = self.client.get(f"{BASE_URL}{path}", params=params, headers={"accept": "application/json"})
        if 500 <= r.status_code < 600 or r.status_code == 429:
            raise httpx.ReadTimeout(f"polymarket {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            raise PolymarketHTTPError(f"polymarket {r.status_code} {path}: {r.text[:200]}")
        return r.json()

    def list_markets(
        self,
        *,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
        order: str | None = None,
    ) -> list[dict]:
        params: dict = {"closed": str(closed).lower(), "limit": limit, "offset": offset}
        if order:
            params["order"] = order
        data = self._get("/markets", params=params)
        if isinstance(data, list):
            return data
        return data.get("data", [])

    def get_market(self, condition_id: str) -> dict:
        return self._get(f"/markets/{condition_id}")

    def list_events(
        self,
        *,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        params = {"closed": str(closed).lower(), "limit": limit, "offset": offset}
        data = self._get("/events", params=params)
        return data if isinstance(data, list) else data.get("data", [])
