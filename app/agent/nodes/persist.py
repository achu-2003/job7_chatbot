"""persist node — write the turn back across short-term, episodic, semantic."""
from __future__ import annotations

from typing import Any

from app.agent.identity import extract_email, extract_name
from app.agent.state import AgentState
from app.core.logging import get_logger
from app.memory.gateway import MemoryGateway

log = get_logger("agent_persist")


async def persist(state: AgentState, *, gateway: MemoryGateway) -> dict[str, Any]:
    working = state.get("working") or {}
    await gateway.persist(
        tenant_id=state["tenant_id"],
        customer_id=state.get("customer_id", ""),
        conversation_id=state["conversation_id"],
        session_id=state["session_id"],
        user_text=state.get("inbound_text", ""),
        assistant_text=state.get("draft_response", "") or "",
        tool_calls=working.get("tool_calls") if isinstance(working, dict) else None,
    )

    # Remember name/email the candidate gave conversationally, so the next turn
    # already has them (via customer_facts) instead of asking again. A new value
    # stated this turn OVERWRITES the old one (handles "actually it's <new>").
    await _remember_identity(state, gateway)
    # Pin the product shown this turn as the "current product" for next turn.
    if state.get("last_product_id"):
        await gateway.set_focus_product(
            conversation_id=state["conversation_id"],
            tenant_id=state["tenant_id"],
            product_id=str(state["last_product_id"]),
            doc=state.get("last_product_doc") or "",
        )

    # Farewell → refresh this customer's session so the next chat starts clean
    # (no stale topic carryover). Durable facts are kept.
    if state.get("end_session"):
        await gateway.reset_session(
            tenant_id=state["tenant_id"],
            conversation_id=state["conversation_id"],
        )
    return {}


def _last_assistant_turn(state: AgentState) -> str | None:
    """The assistant line shown just before this message — used to tell whether
    a bare reply ("Sandhanapandiyan") is answering a 'what's your name?' ask."""
    for t in reversed(state.get("short_term") or []):
        if t.get("role") == "assistant" and t.get("content"):
            return t["content"]
    return None


async def _remember_identity(state: AgentState, gateway: MemoryGateway) -> None:
    """Best-effort: store name/email the candidate just gave. Never abort the
    turn on a storage hiccup — the reply has already been written."""
    text = state.get("inbound_text", "") or ""
    known = state.get("customer_facts") or {}
    tenant_id = state["tenant_id"]
    customer_id = state.get("customer_id", "")
    if not customer_id:
        return

    # extract_email / extract_name only return on an EXPLICIT statement (a valid
    # email, "my name is X", or a name-shaped reply right after we asked) — so a
    # returned value is confident enough to store. Write it whenever it's new or
    # differs from what's on file; this is what lets a correction ("actually use
    # my gmail") replace the stored value instead of looping on the mismatch.
    to_store: list[tuple[str, str, float]] = []
    email = extract_email(text)
    if email and email != (known.get("email") or "").lower():
        to_store.append(("email", email, 0.9))
    name = extract_name(text, assistant_prompt=_last_assistant_turn(state))
    if name and name != known.get("full_name"):
        to_store.append(("full_name", name, 0.8))

    for key, value, confidence in to_store:
        try:
            await gateway.semantic.remember_fact(
                tenant_id=tenant_id, customer_id=customer_id,
                key=key, value=value, confidence=confidence, source="chat",
            )
        except Exception as exc:  # noqa: BLE001 — identity capture is best-effort
            log.warning("identity_persist_failed", key=key, error=str(exc)[:200])
