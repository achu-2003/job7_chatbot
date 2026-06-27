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


async def fetch_payment(payment_id: str) -> dict[str, Any]:
    """Fetch a payment object from Razorpay (status, amount, order_id, …)."""
    if not payment_id:
        raise RazorpayError("Missing payment id.")
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f"{_API}/payments/{payment_id}", auth=_auth())
    except httpx.HTTPError as exc:
        raise RazorpayError(f"Razorpay request failed: {exc}") from exc
    if resp.status_code != 200:
        log.warning("razorpay_fetch_failed", status=resp.status_code, body=resp.text[:300])
        raise RazorpayError(f"Razorpay payment fetch failed ({resp.status_code}).")
    return resp.json()


# A payment is only "good" once the money is actually held/taken — NOT 'failed',
# 'created', or 'refunded'. (Orders here are auto-captured, so success → 'captured';
# 'authorized' is accepted for manual-capture accounts where the money is held.)
_PAID_STATUSES = {"captured", "authorized"}


async def verify_payment(
    *, order_id: str, payment_id: str, signature: str,
    expected_amount_paise: int | None = None,
) -> tuple[bool, str]:
    """Full server-side payment check to run BEFORE granting anything. Returns
    ``(ok, reason)``. Confirms, in order: (1) the signature is genuine, (2) the
    payment ACTUALLY succeeded on Razorpay (status captured/authorized — not a
    failed/abandoned attempt), (3) it belongs to this order, and (4) the amount
    matches. Fails closed — if Razorpay can't be reached, we do NOT grant."""
    if not verify_payment_signature(order_id=order_id, payment_id=payment_id, signature=signature):
        return False, "bad_signature"
    try:
        pay = await fetch_payment(payment_id)
    except RazorpayError as exc:
        log.error("payment_verify_fetch_failed", payment_id=payment_id, error=str(exc)[:200])
        return False, "fetch_failed"
    status = pay.get("status")
    if status not in _PAID_STATUSES:
        log.warning("payment_not_captured", payment_id=payment_id, status=status)
        return False, "not_captured"
    if pay.get("order_id") and pay.get("order_id") != order_id:
        log.warning("payment_order_mismatch", payment_id=payment_id,
                    expected=order_id, got=pay.get("order_id"))
        return False, "order_mismatch"
    if expected_amount_paise is not None and int(pay.get("amount", -1)) != int(expected_amount_paise):
        log.warning("payment_amount_mismatch", payment_id=payment_id,
                    expected=expected_amount_paise, got=pay.get("amount"))
        return False, "amount_mismatch"
    return True, "ok"
