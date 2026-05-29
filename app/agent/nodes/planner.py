"""planner node — turn the message + memory into a JSON plan (goal + tool steps).

Robust to a weak model: invalid/empty JSON degrades to a sensible default
(a single product search), and tools the model invents are dropped.
"""
from __future__ import annotations

from typing import Any

from app.agent.context import is_followup
from app.agent.prompts import PLANNER_SYSTEM, PROMPT_VERSIONS
from app.agent.state import AgentState, new_goal, new_step
from app.core import conversation_log as conv
from app.core.logging import get_logger
from app.core.metrics import AGENT_PROMPT_CALLS
from app.mcp.tools import ToolRegistry

log = get_logger("agent_planner")


async def plan(
    state: AgentState, *, llm: Any, registry: ToolRegistry, memory_context: str | None
) -> dict[str, Any]:
    # Deterministic shortcut: a follow-up about the pinned product ("details
    # about this", "what colours") is answered straight from memory — no plan,
    # no search (which would pull the wrong/older product), no LLM call here.
    cached = state.get("cached_product") or {}
    if cached.get("doc") and is_followup(state.get("inbound_text", "")):
        conv.note("plan", "direct (follow-up on current product)")
        goals = state.get("goals") or [new_goal(state["inbound_text"])]
        return {"goals": goals, "plan": [], "cursor": 0,
                "loop_count": state.get("loop_count", 0) + 1}

    tools_desc = "\n".join(
        f"- {s['function']['name']}: {s['function']['description']}"
        for s in registry.openai_schemas()
    )
    parts: list[str] = []
    if memory_context:
        parts.append(f"MEMORY:\n{memory_context}")
    recent = [
        f"{t.get('role')}: {t.get('content')}"
        for t in (state.get("short_term") or [])[-3:]
        if t.get("content")
    ]
    if recent:
        parts.append("RECENT:\n" + "\n".join(recent))
    parts.append(f"CUSTOMER MESSAGE: {state['inbound_text']}")

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM.format(tools=tools_desc)},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    AGENT_PROMPT_CALLS.labels(prompt="planner", version=PROMPT_VERSIONS["planner"]).inc()
    try:
        obj, _ = await llm.json_chat(purpose="agent_plan", messages=messages)
    except Exception as exc:  # noqa: BLE001
        log.warning("planner_failed", error=str(exc)[:200])
        obj = {}

    goal_text = (obj.get("goal") or state["inbound_text"]).strip()
    direct = bool(obj.get("direct_answer"))
    valid = set(registry.names())
    steps = [
        new_step(rs["tool"], tool=rs["tool"], args=rs.get("args") or {})
        for rs in (obj.get("steps") or [])
        if isinstance(rs, dict) and rs.get("tool") in valid
    ]
    # Fallback: neither a direct answer nor a valid step → search open jobs (the
    # most common need on this jobs board) so the reply is at least grounded.
    if not direct and not steps:
        steps = [new_step("search_jobs", tool="search_jobs",
                          args={"query": state["inbound_text"]})]

    # The goal is per-turn working memory: created once at the start of a
    # turn-chain and kept across replans (state persists within the turn). Not
    # written to agent_goals — durable "what happened" lives in agent_episodes,
    # and deferred work in agent_pending_actions; this avoids a stale active
    # goal sticking around and blocking new ones.
    goals = state.get("goals") or []
    if not goals:
        goals = [new_goal(goal_text)]

    conv.note(
        "plan",
        "direct answer" if not steps else f"{len(steps)} step(s): "
        + ", ".join(s["tool"] for s in steps),
    )
    return {
        "goals": goals,
        "plan": steps,
        "cursor": 0,
        "loop_count": state.get("loop_count", 0) + 1,
    }
