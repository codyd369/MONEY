"""RSA-PSS signing tests (SPEC §16 step 7).

We generate a throwaway RSA key in-test, sign a request, and round-trip
verify the signature with the public key. No network calls; no real
credentials. This ensures:
  1. The signing string format is exactly <ts_ms><METHOD><path>.
  2. The resulting signature validates under PSS/MGF1-SHA256.
  3. The three headers (ACCESS-KEY, ACCESS-TIMESTAMP, ACCESS-SIGNATURE)
     are set correctly.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


@pytest.fixture
def rsa_key_pair():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = priv.public_key()
    return priv, pub


def _verify(pub, signature_b64: str, signing_string: str) -> None:
    pub.verify(
        base64.b64decode(signature_b64),
        signing_string.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=hashes.SHA256.digest_size,
        ),
        hashes.SHA256(),
    )


def test_build_signing_string():
    from moneybutton.kalshi.auth import build_signing_string

    s = build_signing_string(1713880000000, "get", "/trade-api/v2/markets")
    assert s == "1713880000000GET/trade-api/v2/markets"


def test_sign_roundtrips_with_public_key(rsa_key_pair):
    from moneybutton.kalshi.auth import build_signing_string, sign

    priv, pub = rsa_key_pair
    s = build_signing_string(1713880000000, "GET", "/trade-api/v2/markets")
    sig_b64 = sign(priv, s)
    _verify(pub, sig_b64, s)


def test_sign_request_headers(rsa_key_pair):
    from moneybutton.kalshi.auth import (
        ACCESS_KEY_HEADER,
        ACCESS_SIGNATURE_HEADER,
        ACCESS_TIMESTAMP_HEADER,
        sign_request,
    )

    priv, pub = rsa_key_pair
    ts = 1_713_880_000_000
    headers = sign_request(
        method="GET",
        path="/trade-api/v2/markets?status=settled",
        private_key=priv,
        api_key_id="test_key_id_abc",
        timestamp_ms=ts,
    )
    assert headers[ACCESS_KEY_HEADER] == "test_key_id_abc"
    assert headers[ACCESS_TIMESTAMP_HEADER] == str(ts)
    # Signature verifies.
    _verify(
        pub,
        headers[ACCESS_SIGNATURE_HEADER],
        f"{ts}GET/trade-api/v2/markets?status=settled",
    )


def test_sign_request_generates_fresh_timestamp_when_omitted(rsa_key_pair):
    from moneybutton.kalshi.auth import ACCESS_TIMESTAMP_HEADER, sign_request

    priv, _ = rsa_key_pair
    before = int(time.time() * 1000)
    headers = sign_request(
        method="GET",
        path="/trade-api/v2/exchange/status",
        private_key=priv,
        api_key_id="kid",
    )
    after = int(time.time() * 1000)
    ts = int(headers[ACCESS_TIMESTAMP_HEADER])
    assert before - 10 <= ts <= after + 10  # clock skew tolerance


def test_load_private_key(tmp_path: Path, rsa_key_pair):
    from moneybutton.kalshi.auth import load_private_key

    priv, _ = rsa_key_pair
    pem_path = tmp_path / "test_key.pem"
    pem_path.write_bytes(
        priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    loaded = load_private_key(pem_path)
    assert isinstance(loaded, rsa.RSAPrivateKey)


def test_load_private_key_missing_path(tmp_path: Path):
    from moneybutton.kalshi.auth import load_private_key

    with pytest.raises(FileNotFoundError):
        load_private_key(tmp_path / "does_not_exist.pem")


def test_kalshi_signer_requires_key_id(tmp_path: Path, rsa_key_pair, tmp_env):
    """A signer without KALSHI_API_KEY_ID must refuse to be constructed."""
    from moneybutton.core.config import reset_settings_cache
    from moneybutton.kalshi.auth import KalshiSigner

    priv, _ = rsa_key_pair
    pem_path = tmp_path / "pk.pem"
    pem_path.write_bytes(
        priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    # tmp_env fixture left KALSHI_API_KEY_ID unset.
    reset_settings_cache()
    with pytest.raises(RuntimeError, match="KALSHI_API_KEY_ID"):
        KalshiSigner(private_key=priv)


def test_kalshi_signer_happy_path(tmp_path: Path, rsa_key_pair, tmp_env, monkeypatch):
    """With an API key id set, the signer produces verifying headers."""
    from moneybutton.core.config import reset_settings_cache
    from moneybutton.kalshi.auth import ACCESS_SIGNATURE_HEADER, KalshiSigner

    priv, pub = rsa_key_pair
    monkeypatch.setenv("KALSHI_API_KEY_ID", "kid_xyz")
    reset_settings_cache()

    signer = KalshiSigner(private_key=priv)
    headers = signer.headers_for("POST", "/trade-api/v2/portfolio/orders")
    ts = headers["KALSHI-ACCESS-TIMESTAMP"]
    signing_string = f"{ts}POST/trade-api/v2/portfolio/orders"
    _verify(pub, headers[ACCESS_SIGNATURE_HEADER], signing_string)
