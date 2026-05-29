"""summarizer node — fold long history into a rolling summary (token control)."""
from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.core import conversation_log as conv
from app.memory.gateway import MemoryGateway
from app.memory.repositories import AgentSessionRepository


async def summarize_if_needed(
    state: AgentState, *, gateway: MemoryGateway, llm: Any
) -> dict[str, Any]:
    short_term = state.get("short_term") or []
    if not gateway.should_summarize(short_term):
        return {}
    summary = await gateway.summarize(
        llm=llm, short_term=short_term, prior_summary=state.get("rolling_summary", ""),
    )
    session_id = state.get("session_id")
    if session_id:
        await AgentSessionRepository.update_summary(session_id, summary)
    conv.note("summary", f"{len(short_term)} turns → {len(summary)} chars")
    return {"rolling_summary": summary}
