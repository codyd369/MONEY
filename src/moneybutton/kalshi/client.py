"""Kalshi /trade-api/v2 HTTP client (SPEC §16 step 8).

Read-only methods first per build order. Write methods (place order, cancel
order, portfolio endpoints) will be added in step 16 alongside the executor.

All public methods:
  - sign requests via KalshiSigner when credentials are present,
  - fall back to unsigned calls for public endpoints when unconfigured,
  - use httpx with a 30s default timeout,
  - retry (via tenacity) on network errors and 5xx/429, exponential backoff,
  - return decoded JSON dicts; callers get raw data and do their own shaping.

Note on URLs (verify at first run per SPEC §17):
    demo -> https://demo-api.kalshi.co
    prod -> https://api.elections.kalshi.com
Path is always /trade-api/v2/... — the exact scheme is the operator's to
confirm before going live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from moneybutton.core.config import Settings, get_settings
from moneybutton.kalshi.auth import KalshiSigner

log = logging.getLogger("moneybutton.kalshi")


class KalshiHTTPError(RuntimeError):
    def __init__(self, status: int, body: str, path: str) -> None:
        super().__init__(f"Kalshi {status} on {path}: {body[:300]}")
        self.status = status
        self.body = body
        self.path = path


_BASE_URLS: dict[str, str] = {
    "demo": "https://demo-api.kalshi.co",
    "prod": "https://api.elections.kalshi.com",
}

_API_PREFIX = "/trade-api/v2"

_RETRIABLE = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)


def _is_retriable_status(status: int) -> bool:
    return status == 429 or 500 <= status < 600


@dataclass
class KalshiClient:
    settings: Settings | None = None
    signer: KalshiSigner | None = None
    client: httpx.Client | None = None
    _owns_httpx: bool = False

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()
        if self.client is None:
            self.client = httpx.Client(timeout=30.0)
            self._owns_httpx = True

    # ----------------------------- lifecycle ----------------------------
    def close(self) -> None:
        if self._owns_httpx and self.client is not None:
            self.client.close()

    def __enter__(self) -> "KalshiClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ----------------------------- helpers ------------------------------
    def _base_url(self) -> str:
        env = (self.settings.kalshi_env or "demo").lower()
        if env not in _BASE_URLS:
            raise ValueError(f"KALSHI_ENV must be 'demo' or 'prod', got {env!r}")
        return _BASE_URLS[env]

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        signed: bool = True,
    ) -> dict[str, Any]:
        """Perform one HTTP call and return the decoded JSON body.

        Signed by default; unsigned when the signer is absent OR the caller
        explicitly passes signed=False (public endpoints only).
        """
        url = self._base_url() + path
        headers: dict[str, str] = {"accept": "application/json"}

        # httpx (unlike requests) sends None-valued params as empty strings,
        # which Kalshi rejects with "invalid syntax" on int-typed fields.
        # Drop Nones here so both httpx and the signing-string view match.
        clean_params: dict[str, Any] | None = None
        if params:
            clean_params = {k: v for k, v in params.items() if v is not None}
            if not clean_params:
                clean_params = None

        if signed:
            if self.signer is None:
                # Lazy-construct the signer only if it can succeed.
                if self.settings.kalshi_api_key_id:
                    self.signer = KalshiSigner(settings=self.settings)
            if self.signer is not None:
                # Kalshi signs path+query, not query-stripped path. httpx
                # rebuilds the query; we recompute the same string here.
                query = ""
                if clean_params:
                    query = "?" + "&".join(
                        f"{k}={_to_query_value(v)}" for k, v in clean_params.items()
                    )
                headers.update(self.signer.headers_for(method, path + query))

        return self._send_with_retry(
            method=method,
            url=url,
            headers=headers,
            params=clean_params,
            json_body=json_body,
            path=path,
        )

    @retry(
        retry=retry_if_exception_type(_RETRIABLE),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _send_with_retry(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
        path: str,
    ) -> dict[str, Any]:
        assert self.client is not None
        r = self.client.request(method, url, headers=headers, params=params, json=json_body)
        if _is_retriable_status(r.status_code):
            # Raise a retriable exception so tenacity retries us.
            raise httpx.ReadTimeout(
                f"retriable status {r.status_code} from {path}: {r.text[:200]}"
            )
        if r.status_code >= 400:
            raise KalshiHTTPError(r.status_code, r.text, path)
        return r.json()

    # ----------------------------- public API ---------------------------
    def exchange_status(self) -> dict[str, Any]:
        """Liveness check. Safe and cheap to call on startup."""
        return self._request("GET", f"{_API_PREFIX}/exchange/status", signed=False)

    def list_markets(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        cursor: str | None = None,
        limit: int = 200,
        event_ticker: str | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
    ) -> dict[str, Any]:
        """Paginated market listing. Pass the returned `cursor` in the next call."""
        params = {
            "status": status,
            "category": category,
            "cursor": cursor,
            "limit": limit,
            "event_ticker": event_ticker,
            "min_close_ts": min_close_ts,
            "max_close_ts": max_close_ts,
        }
        return self._request("GET", f"{_API_PREFIX}/markets", params=params, signed=False)

    def iter_settled_markets(
        self,
        *,
        category: str | None = None,
        limit_per_page: int = 200,
    ) -> Iterable[dict[str, Any]]:
        """Resume-friendly cursor iterator over status=settled markets.

        Caller persists the cursor to scraper_state (db.py) between runs so
        the multi-hour backfill can resume after a crash or rate-limit
        cooldown.
        """
        cursor: str | None = None
        while True:
            page = self.list_markets(
                status="settled",
                category=category,
                cursor=cursor,
                limit=limit_per_page,
            )
            for m in page.get("markets", []):
                yield m
            cursor = page.get("cursor")
            if not cursor:
                break

    def get_market(self, ticker: str) -> dict[str, Any]:
        return self._request("GET", f"{_API_PREFIX}/markets/{ticker}", signed=False)

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict[str, Any]:
        return self._request(
            "GET",
            f"{_API_PREFIX}/markets/{ticker}/orderbook",
            params={"depth": depth},
            signed=False,
        )

    def get_candlesticks(
        self,
        ticker: str,
        *,
        series_ticker: str | None = None,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> dict[str, Any]:
        """Candlestick price history. `period_interval` is in minutes."""
        path = f"{_API_PREFIX}/series/{series_ticker}/markets/{ticker}/candlesticks" \
               if series_ticker else f"{_API_PREFIX}/markets/{ticker}/candlesticks"
        return self._request(
            "GET",
            path,
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
            signed=False,
        )

    def list_events(
        self,
        *,
        status: str | None = None,
        cursor: str | None = None,
        limit: int = 200,
        series_ticker: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Event-level listing. Events carry `category` (and a clean taxonomy
        of Elections/Politics/Sports/Economics/...) — our events-first
        backfill uses this as the primary grouping instead of the now-
        unreliable /markets category filter."""
        return self._request(
            "GET",
            f"{_API_PREFIX}/events",
            params={
                "status": status,
                "cursor": cursor,
                "limit": limit,
                "series_ticker": series_ticker,
                "category": category,
            },
            signed=False,
        )

    # ============================== write API =============================
    # All write methods REQUIRE the signer (KALSHI_API_KEY_ID + private key).
    # They are only reachable via core/executor.py which gates them behind
    # the kill switch + DRY_RUN + KALSHI_ENV=prod checks.

    def create_order(
        self,
        *,
        ticker: str,
        action: str,
        side: str,
        count: int,
        client_order_id: str,
        type_: str = "limit",
        yes_price_cents: int | None = None,
        no_price_cents: int | None = None,
        buy_max_cost_cents: int | None = None,
        sell_position_floor: int | None = None,
        expiration_ts: int | None = None,
        post_only: bool = False,
    ) -> dict[str, Any]:
        """Create an order. Mandatory: ticker, action('buy'|'sell'),
        side('yes'|'no'), count, client_order_id (idempotency key — reuse
        the same id and Kalshi returns the same order record).

        For a YES-side limit buy at 42c of 10 contracts:
            create_order(ticker="KX-...", action="buy", side="yes",
                         count=10, yes_price_cents=42,
                         client_order_id="mb_...")
        """
        if action not in ("buy", "sell"):
            raise ValueError(f"action must be buy/sell, got {action!r}")
        if side not in ("yes", "no"):
            raise ValueError(f"side must be yes/no, got {side!r}")
        if count < 1:
            raise ValueError("count must be >= 1")

        body: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "client_order_id": client_order_id,
            "type": type_,
            "post_only": post_only,
        }
        if yes_price_cents is not None:
            body["yes_price"] = yes_price_cents
        if no_price_cents is not None:
            body["no_price"] = no_price_cents
        if buy_max_cost_cents is not None:
            body["buy_max_cost"] = buy_max_cost_cents
        if sell_position_floor is not None:
            body["sell_position_floor"] = sell_position_floor
        if expiration_ts is not None:
            body["expiration_ts"] = expiration_ts

        return self._request(
            "POST",
            f"{_API_PREFIX}/portfolio/orders",
            json_body=body,
            signed=True,
        )

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return self._request(
            "DELETE",
            f"{_API_PREFIX}/portfolio/orders/{order_id}",
            signed=True,
        )

    def get_portfolio_balance(self) -> dict[str, Any]:
        return self._request(
            "GET",
            f"{_API_PREFIX}/portfolio/balance",
            signed=True,
        )

    def get_portfolio_orders(
        self,
        *,
        ticker: str | None = None,
        status: str | None = None,
        cursor: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"{_API_PREFIX}/portfolio/orders",
            params={
                "ticker": ticker,
                "status": status,
                "cursor": cursor,
                "limit": limit,
            },
            signed=True,
        )

    def get_portfolio_positions(
        self,
        *,
        cursor: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"{_API_PREFIX}/portfolio/positions",
            params={"cursor": cursor, "limit": limit},
            signed=True,
        )


def _to_query_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
