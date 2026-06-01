"""Compact, TOON-friendly rendering of tool results for LLM context.

Why this exists: naively TOON-encoding the raw ``tool_results`` is NOT smaller
than JSON — product dicts carry list fields (colors/sizes) and order rows have
varying keys, both of which break TOON's compact *table* form (it falls back to
verbose record-per-block). Here we flatten each result type into **uniform,
scalar-only rows** so the table form kicks in — that's where TOON's 30-50% token
saving actually comes from.
"""
from __future__ import annotations

import re
from typing import Any

from app.chatbot.toon import encode as toon_encode

# A turn that refers to the *currently discussed* product rather than naming a
# new one — "details about this", "what colours", "how much", "in stock?".
_REFERS_TO_CURRENT = re.compile(r"\b(this|it|that|these|those)\b", re.IGNORECASE)
_BARE_FACET = re.compile(
    r"^\s*(can you |could you |pls |please )?(give |tell |show )?\w{0,4}\s*"
    r"(detail|details|more|colou?r|colou?rs|size|sizes|price|cost|stock|fabric|"
    r"material|how much|available)\b",
    re.IGNORECASE,
)
_PRODUCT_NOUN = re.compile(
    r"\b(saree|sari|kurti|kurtis|kurta|lehenga|lehanga|blouse|dress|gown|top|skirt|"
    r"suit|frock|palazzo|dupatta|shawl|sandal|shoe|bag|jewell?ery|footwear|anarkali|"
    r"chudidaar|churidar)\b",
    re.IGNORECASE,
)
# Short WhatsApp affirmations / continuations that mean "about the current
# product" — "yes", "ok", "show more", "same in black", "order it".
_AFFIRM_CONTINUE = re.compile(
    r"^\s*(yes|yeah|yep|yup|ya|sure|ok(ay)?|k|fine|confirm\w*|proceed|go ahead|"
    r"do it|order( it| this)?|book( it)?|buy( it)?|i'?ll take it|continue|same|"
    r"same one|same in\b|show more|more|tell me more|interested|i want (it|this))\b",
    re.IGNORECASE,
)


def is_followup(query: str) -> bool:
    """True when the message refers to the product already on screen — a pronoun
    ("this"), a bare facet ("what colours"), or an affirmation/continuation
    ("yes", "show more", "same in black"). These resolve against the pinned
    current product, NOT a re-search or an older product/order."""
    if _PRODUCT_NOUN.search(query):
        return False  # names a different product → not a follow-up
    return bool(
        _REFERS_TO_CURRENT.search(query)
        or _BARE_FACET.search(query)
        or _AFFIRM_CONTINUE.search(query)
    )

# Fixed, uniform column set so every order renders as one TOON table row.
_ORDER_FIELDS = (
    "order_number", "order_status", "payment_status", "total_amount",
    "tracking_number", "courier_name", "estimated_delivery",
)


def toon_context(results: list[dict[str, Any]]) -> str:
    """Flatten tool results into uniform tables, then TOON-encode. Returns ''
    when there's nothing to show."""
    products: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    applications: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    docs: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    overview: dict[str, Any] = {}
    overview_cats: list[dict[str, Any]] = []

    for r in results or []:
        tool, res = r.get("tool"), r.get("result")
        if tool == "search_products" and isinstance(res, list):
            for p in res:
                if isinstance(p, dict):
                    products.append({
                        "title": p.get("title"),
                        "price": p.get("price"),
                        "mrp": p.get("mrp"),
                        "category": p.get("category"),
                        "colors": ", ".join(p.get("colors") or []) or "-",
                        "sizes": ", ".join(p.get("sizes") or []) or "-",
                        "stock": p.get("stock"),
                    })
        elif tool == "search_jobs" and isinstance(res, list):
            for j in res:
                if isinstance(j, dict):
                    jobs.append(_flat_job(j))
        elif tool == "list_jobs_overview" and isinstance(res, dict):
            # Count + per-category breakdown for "list all jobs". The model turns
            # this into "We have N open jobs across A, B, C — which interests you?"
            overview["total_open_jobs"] = res.get("total_open_jobs")
            for c in (res.get("categories") or []):
                if isinstance(c, dict) and c.get("category"):
                    overview_cats.append({"category": c["category"], "count": c.get("count")})
        elif tool == "get_application_status" and isinstance(res, dict):
            # Either {"applications": [...]} or {"application": {...}}; flatten
            # both into uniform rows so the model can read each status cleanly.
            apps = res.get("applications")
            if isinstance(apps, list):
                applications.extend(_flat_app(a) for a in apps if isinstance(a, dict))
            elif isinstance(res.get("application"), dict):
                applications.append(_flat_app(res["application"]))
            elif res.get("found") is False:
                applications.append({"job_title": "-", "status": "no applications found", "applied": "-"})
        elif tool == "get_order_status" and isinstance(res, dict) and isinstance(res.get("order"), dict):
            orders.append(_flat_order(res["order"]))
        elif tool == "get_recent_orders" and isinstance(res, list):
            orders.extend(_flat_order(o) for o in res if isinstance(o, dict))
        elif tool in {"search_policies", "search_faq"} and isinstance(res, list):
            for d in res:
                if isinstance(d, dict):
                    docs.append({"title": d.get("title") or "-", "text": (d.get("text") or "")[:200]})
        else:
            other.append({"tool": tool, "result": _scalar(res)})

    ctx: dict[str, Any] = {}
    if overview:
        ctx["open_jobs_total"] = overview.get("total_open_jobs")
    if overview_cats:
        ctx["job_categories"] = overview_cats
    if products:
        ctx["products"] = products
    if jobs:
        ctx["jobs"] = jobs
    if applications:
        ctx["applications"] = applications
    if orders:
        ctx["orders"] = orders
    if docs:
        ctx["docs"] = docs
    if other:
        ctx["other"] = other
    return toon_encode(ctx) if ctx else ""


# Uniform job row for the LLM context (scalar-only → TOON table form).
_JOB_FIELDS = ("title", "location", "employment_type", "department",
               "salary_min", "salary_max", "availability")


def _flat_job(job: dict[str, Any]) -> dict[str, Any]:
    return {k: (job.get(k) if job.get(k) not in (None, "") else "-") for k in _JOB_FIELDS}


def _flat_app(app: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_title": app.get("job_title") or app.get("job_ref") or "-",
        "status": app.get("status") or "-",
        "applied": (str(app.get("created_at"))[:10] if app.get("created_at") else "-"),
    }


def _flat_order(order: dict[str, Any]) -> dict[str, Any]:
    # Same keys for every row (missing → "-") so TOON uses table form.
    return {k: (order.get(k) if order.get(k) is not None else "-") for k in _ORDER_FIELDS}


def _scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)[:200]
