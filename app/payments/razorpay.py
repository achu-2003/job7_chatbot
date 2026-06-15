"""Razorpay REST client — order creation + payment-signature verification.

Thin async wrapper over https://api.razorpay.com/v1/ (no SDK dependency — just
httpx + HTTP basic auth with the key id/secret). TEST keys (``rzp_test_…``) move
no real money; pay with Razorpay's test cards. The secret never leaves the
server: it signs nothing client-side and verifies the checkout callback.

Razorpay Checkout returns ``razorpay_order_id``, ``razorpay_payment_id`` and a
``razorpay_signature`` = HMAC_SHA256(order_id + "|" + payment_id, key_secret).
We recompute it and constant-time compare to confirm the payment is genuine
before granting anything.
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Any

import httpx

from app.config import get_settings
from app.core.logging import get_logger

log = get_logger("razorpay")

_API = "https://api.razorpay.com/v1"


class RazorpayError(RuntimeError):
    """A Razorpay API call failed (network, auth, or a 4xx/5xx response)."""


def _auth() -> tuple[str, str]:
    s = get_settings()
    if not s.razorpay_enabled:
        raise RazorpayError("Razorpay is not configured (missing key id/secret).")
    return s.razorpay_key_id, s.razorpay_key_secret


async def create_order(
    *, amount_paise: int, receipt: str, notes: dict[str, Any] | None = None,
    currency: str = "INR",
) -> dict[str, Any]:
    """Create a Razorpay order. ``amount_paise`` is the charge in the smallest
    currency unit (₹999 → 99900). Returns the order dict (incl. its ``id``)."""
    if amount_paise <= 0:
        raise RazorpayError("Order amount must be a positive number of paise.")
    payload = {
        "amount": int(amount_paise),
        "currency": currency,
        "receipt": receipt[:40],
        "notes": notes or {},
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(f"{_API}/orders", auth=_auth(), json=payload)
    except httpx.HTTPError as exc:
        raise RazorpayError(f"Razorpay request failed: {exc}") from exc
    if resp.status_code != 200:
        log.warning("razorpay_order_failed", status=resp.status_code, body=resp.text[:300])
        raise RazorpayError(f"Razorpay order failed ({resp.status_code}).")
    return resp.json()


def verify_payment_signature(
    *, order_id: str, payment_id: str, signature: str
) -> bool:
    """True iff ``signature`` is Razorpay's genuine HMAC over this order+payment.
    Constant-time compare; any blank input fails closed."""
    if not (order_id and payment_id and signature):
        return False
    _, secret = _auth()
    expected = hmac.new(
        secret.encode("utf-8"),
        f"{order_id}|{payment_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
