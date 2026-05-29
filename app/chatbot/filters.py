"""LLM-driven shopping-filter extraction.

Pulls structured filters out of free-form queries like::

    "kurtis under 1000 in red size M"
    "lehengas between 3000 and 5000 for kids"

The extractor uses the cheap router model with a tight JSON schema prompt so
it stays sub-300 ms. The returned filters are fed two places:

    * ``where`` clause on Qdrant vector searches (color, gender, category)
    * SQL params on :class:`ProductRepository.search` (min_price, max_price,
      size, color, etc.)

Both are advisory — if extraction fails we fall back to no filters and the
existing vector-only path. Never raises.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, TypedDict

from app.core.logging import get_logger
from app.llm.client import LLMClient

log = get_logger("filters")

_SYSTEM = (
    "You convert a customer's shopping message into structured filters. "
    "Output STRICT JSON only — no prose. Fields:\n"
    "  category    string|null   (e.g. \"saree\", \"kurti\", \"lehenga\", \"footwear\")\n"
    "  color       string|null   (single color word; lowercase)\n"
    "  size        string|null   (XS|S|M|L|XL|XXL|Free Size)\n"
    "  gender      string|null   (women|men|kids)\n"
    "  max_price   number|null   (rupees, integer)\n"
    "  min_price   number|null   (rupees, integer)\n"
    "Rules:\n"
    "- Omit fields the customer didn't mention; use null.\n"
    "- \"under 1000\" → max_price 1000\n"
    "- \"between 500 and 1500\" → min_price 500, max_price 1500\n"
    "- \"cheap\" / \"budget\" → max_price 1000\n"
    "- Never invent values. If unsure, use null.\n"
    "Return only the JSON object."
)


class ExtractedFilters(TypedDict, total=False):
    category: str | None
    color: str | None
    size: str | None
    gender: str | None
    max_price: float | None
    min_price: float | None


_EMPTY: ExtractedFilters = {}


async def extract_filters(query: str, *, llm: LLMClient) -> ExtractedFilters:
    """Best-effort filter extraction. Returns {} on any error."""
    try:
        obj, _usage = await llm.json_chat(
            purpose="filter_extract",
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": query},
            ],
            model=None,  # uses llm_model_chat — could switch to llm_model_router
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("filter_extract_failed", error=str(exc), query=query[:120])
        return dict(_EMPTY)

    out: ExtractedFilters = {}
    if isinstance(obj.get("category"), str) and obj["category"].strip():
        out["category"] = obj["category"].strip().lower()
    if isinstance(obj.get("color"), str) and obj["color"].strip():
        out["color"] = obj["color"].strip().lower()
    if isinstance(obj.get("size"), str) and obj["size"].strip():
        out["size"] = obj["size"].strip()
    if isinstance(obj.get("gender"), str) and obj["gender"].strip():
        out["gender"] = obj["gender"].strip().lower()
    out["max_price"] = _coerce_price(obj.get("max_price"))
    out["min_price"] = _coerce_price(obj.get("min_price"))
    # Drop None values so downstream `if v is not None` checks behave.
    return {k: v for k, v in out.items() if v is not None}


def _coerce_price(v: Any) -> float | None:
    if v is None or v is False:
        return None
    try:
        n = float(v)
        if n <= 0 or n > 10_000_000:
            return None
        return n
    except (TypeError, ValueError):
        return None


def vector_where(filters: ExtractedFilters) -> dict[str, Any]:
    """Build a Qdrant payload ``where`` map from extracted filters.

    Only fields that match the payload SHAPE exactly are safe to push here
    — Qdrant's ``MatchValue`` is exact-equals, no substring. ``category`` is
    stored as the full Postgres ``categories.name`` (e.g. ``"Half Sarees"``),
    so a normalised LLM extraction like ``"saree"`` would never match — we
    deliberately leave category out of the vector filter and narrow by
    category at the application layer (``narrow_sql_rows_by_category``)
    instead. ``gender`` is a clean enum and is safe to push.
    """
    where: dict[str, Any] = {}
    if filters.get("gender"):
        mapped = {"women": "WOMEN", "men": "MEN", "kids": "KIDS"}.get(
            filters["gender"].lower()
        )
        if mapped:
            where["gender"] = mapped
    return where


def narrow_sql_rows_by_category(
    rows: list[dict[str, Any]],
    filters: ExtractedFilters,
) -> list[dict[str, Any]]:
    """Application-layer category filter applied after SQL hydration.

    Matches via lowercase substring against ``category_name`` so the LLM's
    normalised ``"saree"`` still finds ``"Sarees"``, ``"Half Sarees"``,
    ``"Bridal Sarees"``. Falls open (returns original rows) when the filter
    is absent or no row matches — better to show vaguely-related products
    than to send an empty reply.
    """
    cat = (filters.get("category") or "").strip().lower()
    if not cat or not rows:
        return rows
    matched = [
        r for r in rows
        if cat in (r.get("category_name") or "").lower()
    ]
    return matched or rows


def sql_kwargs(filters: ExtractedFilters) -> dict[str, Any]:
    """Map extracted filters to ProductRepository.search keyword args."""
    out: dict[str, Any] = {}
    for k in ("category", "color", "size", "gender"):
        if filters.get(k):
            out[k] = filters[k]
    if filters.get("max_price") is not None:
        out["max_price"] = Decimal(str(filters["max_price"]))
    if filters.get("min_price") is not None:
        out["min_price"] = Decimal(str(filters["min_price"]))
    return out
