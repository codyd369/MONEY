"""Kalshi API v2 RSA-PSS request signing (SPEC §7.1, §16 step 7).

Kalshi's v2 trade API authenticates each request with three headers:

    KALSHI-ACCESS-KEY        <api_key_id>
    KALSHI-ACCESS-TIMESTAMP  <unix_time_ms>
    KALSHI-ACCESS-SIGNATURE  <base64(RSA-PSS-SHA256( "{timestamp_ms}{METHOD}{path}" ))>

The signing string is: <timestamp_ms><METHOD (uppercase)><path_only>.
`path_only` is the request path (without scheme/host), including the query
string if any — callers must pass whatever the httpx URL's `.raw_path`
produces.

We use PSS with MGF1(SHA-256) and salt_length = hash digest size (32 bytes);
this is the default Kalshi expects.

Operator workflow:
  1. Create a Kalshi API key via the dashboard; save the private key PEM
     to the path in KALSHI_PRIVATE_KEY_PATH (default data/kalshi_private_key.pem).
  2. Put KALSHI_API_KEY_ID in .env.
  3. Demo and prod use the same signing logic; only the base URL differs.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Mapping

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from moneybutton.core.config import Settings, get_settings

ACCESS_KEY_HEADER = "KALSHI-ACCESS-KEY"
ACCESS_TIMESTAMP_HEADER = "KALSHI-ACCESS-TIMESTAMP"
ACCESS_SIGNATURE_HEADER = "KALSHI-ACCESS-SIGNATURE"


def load_private_key(pem_path: str | Path) -> rsa.RSAPrivateKey:
    """Load an RSA private key from PEM. Raises if the path is missing or
    the PEM is wrong. Does not support passphrase-protected keys on purpose:
    the whole system is local, and a passphrase file would just be stored
    next to the key."""
    path = Path(pem_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Kalshi private key not found at {path}. "
            f"Export it from Kalshi's dashboard and save there, or update "
            f"KALSHI_PRIVATE_KEY_PATH."
        )
    data = path.read_bytes()
    key = serialization.load_pem_private_key(data, password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise TypeError(f"{path} did not contain an RSA private key")
    return key


def build_signing_string(timestamp_ms: int, method: str, path: str) -> str:
    """Kalshi v2: concatenation of <ts_ms><METHOD><path>, no separator."""
    return f"{timestamp_ms}{method.upper()}{path}"


def sign(private_key: rsa.RSAPrivateKey, signing_string: str) -> str:
    """RSA-PSS(SHA-256) base64-encoded signature of `signing_string`."""
    signature = private_key.sign(
        signing_string.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=hashes.SHA256.digest_size,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("ascii")


def sign_request(
    *,
    method: str,
    path: str,
    private_key: rsa.RSAPrivateKey,
    api_key_id: str,
    timestamp_ms: int | None = None,
) -> Mapping[str, str]:
    """Return the three Kalshi auth headers for the given request."""
    ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
    signing_string = build_signing_string(ts, method, path)
    signature = sign(private_key, signing_string)
    return {
        ACCESS_KEY_HEADER: api_key_id,
        ACCESS_TIMESTAMP_HEADER: str(ts),
        ACCESS_SIGNATURE_HEADER: signature,
    }


class KalshiSigner:
    """Stateful signer that caches the loaded private key + API key id.

    Instantiate once per process. If no explicit settings are passed, pulls
    from the validated Settings singleton.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        private_key: rsa.RSAPrivateKey | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if private_key is not None:
            self._private_key = private_key
        else:
            self._private_key = load_private_key(self.settings.kalshi_private_key_path)
        if not self.settings.kalshi_api_key_id:
            # Defer loudly: a signer without a key id can still sign, but
            # every request will 401. Surface this now rather than at runtime.
            # Callers that are intentionally unsigned (public endpoints)
            # should not instantiate KalshiSigner at all.
            raise RuntimeError(
                "KALSHI_API_KEY_ID is empty. Set it in .env, or use an "
                "unsigned httpx client for public endpoints."
            )

    def headers_for(self, method: str, path: str) -> Mapping[str, str]:
        return sign_request(
            method=method,
            path=path,
            private_key=self._private_key,
            api_key_id=self.settings.kalshi_api_key_id,
        )
