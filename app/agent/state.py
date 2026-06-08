"""Agent state model — the single object that flows through the LangGraph.

``AgentState`` is *assembled* from the durable stores at the ``load_context``
node and *flushed back* at ``persist``. Everything a node needs to reason about
the turn lives here. The small constructor/accessor helpers below keep node
code declarative and id/timestamp creation in one place.

See ``docs/AGENT_ARCHITECTURE.md`` §4 for the rationale behind each field.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal, TypedDict

GoalStatus = Literal["active", "blocked", "done", "abandoned"]
StepStatus = Literal["pending", "running", "done", "failed"]
ReflectVerdict = Literal["finish", "retry_step", "replan", "ask_user", "handoff"]


class Goal(TypedDict):
    id: str
    description: str
    status: GoalStatus
    priority: int
    created_at: float


class Step(TypedDict, total=False):
    id: str
    intent: str            # search_products | lookup_order | answer | ask_user | handoff
    tool: str | None       # MCP tool name, if this step calls a tool
    args: dict[str, Any]
    status: StepStatus
    result: Any
    attempts: int


class Reflection(TypedDict, total=False):
    goal_satisfied: bool
    grounded: bool
    issues: list[str]
    next: ReflectVerdict


class AgentState(TypedDict, total=False):
    # ---- identity / routing ----
    tenant_id: str
    customer_id: str            # WhatsApp sender phone (digits only)
    conversation_id: str
    session_id: str
    request_id: str
    inbound_text: str
    inbound_kind: str           # text | button | list | image | document
    button_id: str | None       # structured id of a tapped list row / button (e.g. "job:<ref>")
    attachment: dict[str, Any] | None  # an uploaded file (e.g. a resume document): {kind, media_id, filename}
    received_at: float

    # ---- memory (hydrated by load_context) ----
    short_term: list[dict[str, str]]      # recent role-tagged turns (Redis)
    rolling_summary: str                  # compressed older history
    semantic_hits: list[dict[str, Any]]   # durable facts recalled from vectors
    customer_facts: dict[str, Any]        # structured prefs (name, size, budget…)
    catalog_hits: list[dict[str, Any]]    # vector product retrieval this turn
    cached_product: dict[str, str] | None  # the pinned "current product" (focus)
    working: dict[str, Any]               # free scratchpad for this turn

    # ---- identity / onboarding (set by the identify node) ----
    is_known: bool                        # existing job-seeker OR name+email captured (routing)
    is_existing_user: bool                # found in the ACTIVE private_job_seekers table
    candidate_id: str | None              # active-table row id, for past-application lookups
    onboarding_prompt: str                # the ask-name / form-link / success line
    just_onboarded: bool                  # first turn after the form was submitted (success msg)
    did_browse: bool                      # this turn was a deterministic category listing
    did_menu: bool                        # this turn was a handled quick-reply menu tap

    # ---- session / continuity ----
    session_status: str                   # active | dormant | resumed
    last_active_at: float | None
    pending_actions: list[dict[str, Any]]
    intent: str
    intent_shifted: bool

    # ---- reasoning ----
    goals: list[Goal]
    plan: list[Step]
    cursor: int                           # index of the current step in plan
    reflections: list[Reflection]
    loop_count: int
    deadline: float                       # monotonic time the loop must stop by

    # ---- output ----
    draft_response: str
    message_chunks: list[str]
    single_bubble: bool                   # deliver the reply whole (e.g. a full category listing)
    delivery_plan: list[dict[str, Any]]   # [{text, typing_ms, delay_ms, image_url?}]
    # A fully-formed WhatsApp Cloud API interactive payload (e.g. the onboarding
    # cta_url "Open form" button). When set, the WhatsApp route sends THIS instead
    # of the text bubbles; the bubbles (draft_response) remain the web/fallback.
    whatsapp_interactive: dict[str, Any] | None
    # A SEQUENCE of interactive payloads sent one after another (e.g. one
    # Apply/Save/Share card per matching job). Takes priority over the single
    # whatsapp_interactive + text bubbles when present.
    whatsapp_messages: list[dict[str, Any]] | None
    used_llm: bool
    latency_ms: int
    # the product shown this turn, pinned as the "current product" for next turn
    last_product_id: str | None
    last_product_doc: str | None
    end_session: bool                     # farewell → reset this customer's memory in persist


# ---------------------------------------------------------------------------
# constructors / accessors (pure, side-effect free)
# ---------------------------------------------------------------------------


def new_goal(description: str, *, priority: int = 0, status: GoalStatus = "active") -> Goal:
    return {
        "id": _id("goal"),
        "description": description,
        "status": status,
        "priority": priority,
        "created_at": time.time(),
    }


def new_step(
    intent: str,
    *,
    tool: str | None = None,
    args: dict[str, Any] | None = None,
) -> Step:
    return {
        "id": _id("step"),
        "intent": intent,
        "tool": tool,
        "args": args or {},
        "status": "pending",
        "result": None,
        "attempts": 0,
    }


def current_step(state: AgentState) -> Step | None:
    """The step the executor should run next, or None when the plan is done."""
    plan = state.get("plan") or []
    cursor = state.get("cursor", 0)
    return plan[cursor] if 0 <= cursor < len(plan) else None


def active_goals(state: AgentState) -> list[Goal]:
    return [g for g in (state.get("goals") or []) if g.get("status") == "active"]


def plan_complete(state: AgentState) -> bool:
    """True when every step has terminated (done or failed)."""
    plan = state.get("plan") or []
    return bool(plan) and all(s.get("status") in {"done", "failed"} for s in plan)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
