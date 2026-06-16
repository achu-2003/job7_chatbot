"""Employer job-posting credit model (mirrors the Jobs7 'Activate Job' screen).

Posting a job costs **job credits**, priced per the live catalog:

    credits required = (number of candidate districts) × (validity multiplier)
    validity:  15 days → 1×   ·   30 days → 2×   ·   45 days → 3×
    1 job credit = ₹649

The employer's balance ('Have') is read from the live ``credit_wallets`` table,
but the debit is simulated in Redis (test harness) — see the route layer. These
are pure, dependency-free helpers so the same numbers drive the server, the JS on
the Activate page, and the tests.
"""
from __future__ import annotations

from typing import Any

JOB_CREDIT_PRICE = 649                       # ₹ per job credit (live catalog)
WELCOME_JOB_CREDITS = 1                      # the free credit a new employer gets
# (validity in days, credit multiplier per district) — order = display order.
VALIDITY_OPTIONS: tuple[tuple[int, int], ...] = ((15, 1), (30, 2), (45, 3))
_MULTIPLIER = {days: mult for days, mult in VALIDITY_OPTIONS}
DEFAULT_VALIDITY_DAYS = 15


def validity_multiplier(days: Any) -> int:
    """Credits-per-district for a validity choice (15/30/45). Unknown → 1×."""
    try:
        return _MULTIPLIER.get(int(days), 1)
    except (TypeError, ValueError):
        return 1


def credits_required(num_districts: Any, validity_days: Any) -> int:
    """districts × multiplier (at least 1 district is always charged)."""
    try:
        n = int(num_districts)
    except (TypeError, ValueError):
        n = 0
    n = max(1, n)
    return n * validity_multiplier(validity_days)


def credit_quote(have: Any, need: Any) -> dict[str, int | bool]:
    """The 'Have / Need / Buy / Pay' box from the Activate screen.

    ``buy`` is what must be purchased to cover the gap; ``pay`` is its rupee
    cost; ``sufficient`` is True when the wallet already covers ``need``.
    """
    have = max(0, _int(have))
    need = max(0, _int(need))
    buy = max(0, need - have)
    return {
        "have": have,
        "need": need,
        "buy": buy,
        "pay": buy * JOB_CREDIT_PRICE,
        "sufficient": buy == 0,
    }


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Buy Credits catalog — the "Individual Credits" tab (à-la-carte per type)
# ---------------------------------------------------------------------------

# credit type → ordered list of (credits, total_price_₹). Mirrors the app's
# Individual Credits pricing (bulk tiers carry a "SAVE MORE" discount).
INDIVIDUAL_PACKS: dict[str, list[tuple[int, int]]] = {
    "JOB":    [(1, 649), (10, 5192)],
    "UNLOCK": [(1, 49), (10, 392), (100, 3920)],
    "BOOST":  [(1, 999), (10, 792)],
}
CREDIT_TYPE_LABELS = {
    "JOB": "Job Credits", "UNLOCK": "Unlock Candidates Credits", "BOOST": "Boost Job Credits",
}
CREDIT_TYPE_ICONS = {"JOB": "💼", "UNLOCK": "🔓", "BOOST": "🚀"}


def find_pack(ctype: Any, qty: Any) -> dict[str, int | str] | None:
    """Resolve an individual-credit pack by (type, quantity) → its price + grant.
    Returns None for an unknown combo (so a tampered request is rejected)."""
    ctype = _s(ctype).upper()
    try:
        qty = int(qty)
    except (TypeError, ValueError):
        return None
    for credits, price in INDIVIDUAL_PACKS.get(ctype, []):
        if credits == qty:
            return {"type": ctype, "credits": credits, "price": price}
    return None


def unit_price(ctype: str) -> int:
    """Per-credit price (the 1-unit tier) for display ('₹X / credit')."""
    packs = INDIVIDUAL_PACKS.get(_s(ctype).upper()) or [(1, 0)]
    return packs[0][1]


def _s(v: Any) -> str:
    return (v if isinstance(v, str) else "" if v is None else str(v)).strip()
