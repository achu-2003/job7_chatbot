"""executor / tool-router node — run the plan's pending tool steps.

Deterministic (no LLM): it dispatches each pending step through the MCP
registry, records the result, and accumulates results in working memory for the
reflection + responder nodes. ``ToolRegistry.dispatch`` never raises (failures
come back as ``{"error": ...}``), so this node can't crash the turn; a failed
step is left for reflection to decide whether to replan.

Identity (tenant + customer) is injected via ``ToolContext`` — never taken from
the model — so a customer can only ever touch their own orders.
"""
from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.core import conversation_log as conv
from app.core.metrics import AGENT_FOLLOWUPS, AGENT_TOOL_CALLS
from app.mcp.tools import ToolContext, ToolRegistry


async def execute(state: AgentState, *, registry: ToolRegistry) -> dict[str, Any]:
    ctx = ToolContext(
        tenant_id=state["tenant_id"],
        customer_external_id=state.get("customer_id"),
        request_id=state.get("request_id"),
        session_id=state.get("session_id"),
    )
    plan = state.get("plan") or []
    working = dict(state.get("working") or {})
    results: list[dict[str, Any]] = list(working.get("tool_results") or [])
    products = state.get("catalog_hits") or []

    for step in plan:
        if step.get("status") in {"done", "failed"} or not step.get("tool"):
            continue
        step["status"] = "running"
        step["attempts"] = step.get("attempts", 0) + 1
        result = await registry.dispatch(step["tool"], step.get("args") or {}, ctx)
        failed = isinstance(result, dict) and bool(result.get("error"))
        step["result"] = result
        step["status"] = "failed" if failed else "done"
        results.append({"tool": step["tool"], "args": step.get("args"), "result": result})
        if step["tool"] == "search_jobs" and isinstance(result, list) and result:
            products = result
        AGENT_TOOL_CALLS.labels(tool=step["tool"], result="error" if failed else "ok").inc()
        if step["tool"] == "schedule_followup" and not failed:
            AGENT_FOLLOWUPS.labels(event="scheduled").inc()
        conv.note("tool", f"{step['tool']} → {'error' if failed else 'ok'}")

    working["tool_results"] = results
    return {"plan": plan, "working": working, "catalog_hits": products}
