"""the-odds-api client (SPEC §7.3).

Free-tier budget: 500 requests / month. The consistency + arbitrage
strategies cache every response locally (data/parquet/odds/) so
re-scans don't hit the API.

Docs: https://the-odds-api.com/liveapi/guides/v4/
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

from moneybutton.core.config import Settings, get_settings

log = logging.getLogger("moneybutton.oddsapi")

BASE_URL = "https://api.the-odds-api.com/v4"

_RETRIABLE = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)


class OddsAPIError(RuntimeError):
    pass


@dataclass
class OddsAPIClient:
    settings: Settings | None = None
    client: httpx.Client | None = None
    _owns: bool = False

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()
        if self.client is None:
            self.client = httpx.Client(timeout=30.0)
            self._owns = True

    def close(self) -> None:
        if self._owns and self.client:
            self.client.close()

    def __enter__(self) -> "OddsAPIClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _api_key(self) -> str:
        key = self.settings.oddsapi_key.get_secret_value()
        if not key:
            raise RuntimeError("ODDSAPI_KEY is empty; cannot call the-odds-api")
        return key

    @retry(
        retry=retry_if_exception_type(_RETRIABLE),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _get(self, path: str, params: dict | None = None) -> Any:
        assert self.client is not None
        full_params = {"apiKey": self._api_key(), **(params or {})}
        r = self.client.get(
            f"{BASE_URL}{path}",
            params=full_params,
            headers={"accept": "application/json"},
        )
        if 500 <= r.status_code < 600 or r.status_code == 429:
            raise httpx.ReadTimeout(f"oddsapi {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            raise OddsAPIError(f"oddsapi {r.status_code} {path}: {r.text[:200]}")
        return r.json()

    def list_sports(self, *, all_: bool = True) -> list[dict]:
        return self._get("/sports", params={"all": str(all_).lower()})

    def get_odds(
        self,
        sport_key: str,
        *,
        regions: str = "us",
        markets: str = "h2h",
        bookmakers: str | None = None,
        odds_format: str = "decimal",
    ) -> list[dict]:
        params: dict = {"regions": regions, "markets": markets, "oddsFormat": odds_format}
        if bookmakers:
            params["bookmakers"] = bookmakers
        return self._get(f"/sports/{sport_key}/odds", params=params)
