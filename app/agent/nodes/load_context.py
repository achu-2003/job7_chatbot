"""load_context node — hydrate the AgentState from every memory tier."""
from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.core import conversation_log as conv
from app.memory.gateway import MemoryGateway


async def load_context(state: AgentState, *, gateway: MemoryGateway) -> dict[str, Any]:
    snapshot = await gateway.load(
        tenant_id=state["tenant_id"],
        customer_id=state.get("customer_id", ""),
        conversation_id=state["conversation_id"],
        query=state.get("inbound_text", ""),
    )
    cached = (snapshot.get("cached_product") or {}).get("product_id", "-")
    conv.note(
        "memory",
        f"session={snapshot['session_id'][:8]} status={snapshot['session_status']} "
        f"turns={len(snapshot['short_term'])} facts={len(snapshot['customer_facts'])} "
        f"recalled={len(snapshot['semantic_hits'])} pending={len(snapshot['pending_actions'])} "
        f"cached_product={cached}",
    )
    return snapshot
