"""Meta webhook X-Hub-Signature-256 verification.

The POST /api/whatsapp/webhook handler authenticates every inbound payload with
an HMAC-SHA256 over the RAW body keyed by the Meta App Secret. These tests pin
that logic: a valid signature passes, a forged/missing one is rejected, and an
unset secret falls back to skip (dev/test)."""
import hashlib
import hmac
import types

from app.api.routes.whatsapp import _verify_meta_signature


def _sig(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _settings(secret: str):
    return types.SimpleNamespace(meta_app_secret=secret)


def test_valid_signature_passes():
    body = b'{"object":"whatsapp_business_account","entry":[]}'
    secret = "app-secret-123"
    assert _verify_meta_signature(_settings(secret), body, _sig(secret, body)) is True


def test_forged_signature_rejected():
    body = b'{"object":"whatsapp_business_account"}'
    good = _settings("real-secret")
    # signed with the WRONG secret → must fail
    assert _verify_meta_signature(good, body, _sig("attacker-secret", body)) is False


def test_tampered_body_rejected():
    secret = "real-secret"
    signed_for = b'{"amount":1}'
    delivered = b'{"amount":9999}'           # body changed after signing
    assert _verify_meta_signature(_settings(secret), delivered, _sig(secret, signed_for)) is False


def test_missing_or_malformed_header_rejected_when_secret_set():
    body = b"{}"
    s = _settings("real-secret")
    assert _verify_meta_signature(s, body, None) is False
    assert _verify_meta_signature(s, body, "") is False
    assert _verify_meta_signature(s, body, "deadbeef") is False           # no sha256= prefix


def test_no_secret_configured_skips_verification():
    # dev/test: with no app secret set, verification is a no-op (returns True)
    body = b"{}"
    s = _settings("")
    assert _verify_meta_signature(s, body, None) is True
    assert _verify_meta_signature(s, body, "anything") is True
