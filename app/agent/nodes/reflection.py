"""reflection node — critique the gathered results and decide what's next.

Verdict ``finish`` (we can answer well) or ``replan`` (a different tool/args
would clearly help). Biased toward finishing; the budget (loop cap + deadline)
in the runtime is the hard backstop. Robust to bad JSON (defaults to finish).
"""
from __future__ import annotations

from typing import Any

from app.agent.context import toon_context
from app.agent.prompts import PROMPT_VERSIONS, REFLECTION_SYSTEM
from app.agent.state import AgentState
from app.core import conversation_log as conv
from app.core.logging import get_logger
from app.core.metrics import AGENT_PROMPT_CALLS, AGENT_REFLECTIONS

log = get_logger("agent_reflection")


async def reflect(state: AgentState, *, llm: Any) -> dict[str, Any]:
    goals = state.get("goals") or []
    goal = goals[-1]["description"] if goals else state.get("inbound_text", "")
    results = (state.get("working") or {}).get("tool_results") or []
    payload = toon_context(results)[:800] or "(no results)"

    messages = [
        {"role": "system", "content": REFLECTION_SYSTEM},
        {"role": "user", "content": f"GOAL: {goal}\n\nTOOL RESULTS:\n{payload}\n\nVerdict?"},
    ]
    AGENT_PROMPT_CALLS.labels(prompt="reflection", version=PROMPT_VERSIONS["reflection"]).inc()
    try:
        obj, _ = await llm.json_chat(purpose="agent_reflect", messages=messages)
    except Exception as exc:  # noqa: BLE001
        log.warning("reflection_failed", error=str(exc)[:200])
        obj = {}

    nxt = obj.get("next")
    verdict = {
        "goal_satisfied": bool(obj.get("goal_satisfied", True)),
        "issues": obj.get("issues") or [],
        "next": nxt if nxt in {"finish", "replan"} else "finish",
    }
    AGENT_REFLECTIONS.labels(verdict=verdict["next"]).inc()
    conv.note("reflect", f"satisfied={verdict['goal_satisfied']} next={verdict['next']}")
    return {"reflections": (state.get("reflections") or []) + [verdict]}
