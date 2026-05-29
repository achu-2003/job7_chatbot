"""responder node — write the final, natural WhatsApp reply.

Grounded in memory + the gathered tool results, validated against the same
``HallucinationValidator`` the rest of the system uses. On an LLM failure it
degrades to a warm "try again" rather than crashing the turn.
"""
from __future__ import annotations

import re
from typing import Any

from app.agent.context import toon_context
from app.agent.prompts import PROMPT_VERSIONS, RESPONDER_SYSTEM
from app.agent.state import AgentState
from app.chatbot.validator import HallucinationValidator
from app.core import conversation_log as conv
from app.core.logging import get_logger
from app.core.metrics import AGENT_PROMPT_CALLS, HALLUCINATION_COUNTER

_PRICE_IN_DOC = re.compile(r"₹\s*([\d,]+(?:\.\d+)?)")
_SAFE_FALLBACK = (
    "Let me double-check that to be sure — could you tell me the role or job "
    "reference (JOB-XXXX) you mean?"
)

log = get_logger("agent_responder")

_BUSY_REPLY = (
    "I'm handling a lot of messages right now and couldn't get to yours — "
    "please send it again in a few seconds."
)


async def respond(
    state: AgentState,
    *,
    llm: Any,
    validator: HallucinationValidator,
    memory_context: str | None,
) -> dict[str, Any]:
    results = (state.get("working") or {}).get("tool_results") or []

    parts: list[str] = []
    if memory_context:
        parts.append(f"MEMORY:\n{memory_context}")
    ctx = toon_context(results)   # flattened → TOON table form (token-efficient)
    if ctx:
        parts.append("CONTEXT:\n" + ctx[:800])
    parts.append(f"CUSTOMER MESSAGE: {state['inbound_text']}\n\nWrite the reply.")

    messages: list[dict[str, Any]] = [{"role": "system", "content": RESPONDER_SYSTEM}]
    for t in (state.get("short_term") or [])[-3:]:
        if t.get("role") in {"user", "assistant"} and t.get("content"):
            messages.append({"role": t["role"], "content": t["content"]})
    messages.append({"role": "user", "content": "\n\n".join(parts)})

    AGENT_PROMPT_CALLS.labels(prompt="responder", version=PROMPT_VERSIONS["responder"]).inc()
    try:
        text, _ = await llm.chat(
            purpose="agent_respond", messages=messages, temperature=0.3, max_tokens=90,
        )
    except Exception as exc:  # noqa: BLE001 — never 500 the turn on an LLM error
        log.warning("responder_llm_unavailable", error=str(exc)[:200])
        return {"draft_response": _BUSY_REPLY, "used_llm": True}

    text = (text or "").strip() or (
        "I couldn't find an answer for that — could you give me a bit more detail?"
    )
    text = _shorten(text)   # deterministic guard: no wall of text, ever

    # Grounding = tool results + the pinned product (so a legit follow-up like
    # "it's ₹999" passes). If the model still quoted a price/order not in any of
    # them, it fabricated → REPLACE the reply (don't send fake confirmations).
    grounding = _grounding_rows(results) + _cached_grounding(state)
    verdict = validator.validate(
        text, sql_rows=grounding, vector_hits=[], customer_query=state.get("inbound_text"),
    )
    if not verdict.valid:
        log.warning("agent_response_blocked", offending=verdict.offending, draft=text[:160])
        HALLUCINATION_COUNTER.labels(result="blocked").inc()
        text = _SAFE_FALLBACK
    else:
        HALLUCINATION_COUNTER.labels(result="valid").inc()
    conv.note("respond", f"{len(text)} chars")
    return {"draft_response": text, "used_llm": True}


def _cached_grounding(state: AgentState) -> list[dict[str, Any]]:
    """Treat the pinned product's price as grounded, so follow-up answers about
    it ("Pink, ₹999") aren't falsely flagged."""
    doc = (state.get("cached_product") or {}).get("doc") or ""
    rows: list[dict[str, Any]] = []
    for m in _PRICE_IN_DOC.finditer(doc):
        try:
            rows.append({"price": float(m.group(1).replace(",", ""))})
        except ValueError:
            continue
    return rows


def _shorten(text: str, limit: int = 300) -> str:
    """Cap the reply so it can't become a paragraph, even if the model rambles.
    Trims back to the last sentence end / line break before ``limit``; product
    bullet lists (a few short lines) pass through untouched."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    best = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind("\n"))
    if best > limit * 0.5:
        return cut[: best + 1].strip()
    return cut.rstrip() + "…"


def _grounding_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pull authoritative rows out of tool results for the grounding check —
    maps the compact product shape onto the keys the validator understands."""
    rows: list[dict[str, Any]] = []
    for r in results:
        res = r.get("result")
        tool = r.get("tool")
        if tool == "search_products" and isinstance(res, list):
            for p in res:
                if isinstance(p, dict):
                    rows.append({
                        "title": p.get("title"),
                        "price": p.get("price"),
                        "suggested_mrp": p.get("mrp"),
                    })
        elif tool == "search_jobs" and isinstance(res, list):
            # The compact job shape already carries job_ref / title / salary_min /
            # salary_max — exactly the keys the validator grounds against — so the
            # rows pass through as-is. Without this branch every JOB-XXXX the model
            # quotes is flagged unsupported and the reply gets replaced.
            rows.extend(j for j in res if isinstance(j, dict))
        elif tool == "get_order_status" and isinstance(res, dict) and isinstance(res.get("order"), dict):
            rows.append(res["order"])
        elif tool == "get_recent_orders" and isinstance(res, list):
            rows.extend(o for o in res if isinstance(o, dict))
    return rows
