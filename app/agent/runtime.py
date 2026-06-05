"""AgentRuntime — the production agent's LangGraph runtime.

Phase 2 wiring — explicit planner → executor → reflection loop:

    load_context → summarize → planner ─┬─(direct answer)──────────────► responder
                                        └─(tool steps)─► execute → reflect
                                                              │
                                              finish / budget │ replan
                                                  ▼           ▼
                                              responder ◄── planner
                                                  │
                                          humanize (chunk + pace) → persist → END

The loop is bounded by ``agent_max_loops`` + ``agent_deadline_seconds`` (the
``Budget`` semantics from Phase 0, tracked via ``loop_count`` + a ``deadline``
on the state). Reasoning is now made of inspectable nodes, not a black box.

Still **not wired into the live request path** — the webhook uses
``ChatGraphRunner`` until human-like delivery (Phase 3) lands and we switch over,
so each phase leaves the bot working.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from app.agent.context import is_followup
from app.agent.identity import extract_email, extract_name
from app.agent.nodes.humanizer import build_delivery_plan
from app.agent.nodes.load_context import load_context
from app.agent.nodes.persist import persist
from app.agent.nodes.planner import plan
from app.agent.nodes.reflection import reflect
from app.agent.nodes.responder import respond
from app.agent.nodes.summarizer import summarize_if_needed
from app.agent.nodes.tool_router import execute
from app.agent.state import AgentState
from app.chatbot.memory import ConversationMemory
from app.chatbot.validator import HallucinationValidator
from app.db.repositories import CandidateRepository
from app.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import AGENT_LOOPS
from app.core.tenancy import get_current_tenant_id
from app.llm.client import LLMClient
from app.mcp.tools import ToolRegistry
from app.vector.store import VectorStore
from app.memory.gateway import MemoryGateway

log = get_logger("agent_runtime")

# Application/status turns: don't pin a job as the "current job" focus (the
# candidate is asking about something they already applied to, not browsing).
_ORDER_HINT_RX = re.compile(
    r"\b(application|applications|applied|apply|status|withdraw\w*|"
    r"my\s+app|app[-_]?\w*|interview\w*|offer)\b",
    re.IGNORECASE,
)
# Pure greeting → fixed reply, 0 LLM.
_GREETING_RX = re.compile(
    r"^\s*(hi+|hello+|hey+|yo|namaste|good\s*(morning|afternoon|evening))[\s!.?]*$",
    re.IGNORECASE,
)
# Farewell / "that's all" → fixed reply + reset the session memory.
_CLOSER_RX = re.compile(
    r"^\s*(bye+|goodbye|good night|that'?s? all|thats all|no thanks?|nothing else|"
    r"i'?m done|im done|done for now|that'?ll be all|ok bye|thank you bye)[\s!.?]*$",
    re.IGNORECASE,
)

# ---- new-candidate onboarding copy --------------------------------------
# A number we've never seen (not in the job board AND no name/email captured
# yet) is asked to introduce itself before we help with roles. The name prompt
# MUST contain "full name"/"your name" so app.agent.identity.extract_name treats
# the bare reply as the answer (see _ASKED_FOR_NAME_RX).
_ONBOARD_ASK_NAME = (
    "Hi, and welcome! I don't think we've spoken before. "
    "Before I help you find roles, could you share your full name?"
)


def _first_name(name: str | None) -> str:
    return name.split()[0] if name else ""


def _onboard_ask_email(name: str | None) -> str:
    who = f", {_first_name(name)}" if name else ""
    return f"Thanks{who}! What's the best email address to reach you on?"


def _onboard_welcome(name: str | None) -> str:
    who = f", {_first_name(name)}" if name else ""
    return f"Great{who}, you're all set. What kind of role are you looking for?"


def _trim_result(res: Any) -> Any:
    """Compact a tool result for the UI: lists → count + a small sample."""
    if isinstance(res, list):
        return {"count": len(res), "sample": res[:3]}
    return res


def _step_detail(final: AgentState) -> dict[str, Any]:
    """Structured per-turn step detail for the flow UI (plan, tool calls with
    args + result previews, reflection verdicts, loop count)."""
    working = final.get("working") or {}
    return {
        "loop_count": final.get("loop_count", 0),
        "latency_ms": final.get("latency_ms", 0),
        "plan": [
            {"tool": s.get("tool"), "intent": s.get("intent"), "status": s.get("status")}
            for s in (final.get("plan") or [])
        ],
        "tools": [
            {"tool": r.get("tool"), "args": r.get("args"), "result": _trim_result(r.get("result"))}
            for r in (working.get("tool_results") or [])
        ],
        "reflections": final.get("reflections") or [],
    }


class AgentRuntime:
    def __init__(
        self,
        *,
        vector: VectorStore,
        memory: ConversationMemory,
        llm: LLMClient | None = None,
    ) -> None:
        self.vector = vector
        self.llm = llm or LLMClient()
        self.gateway = MemoryGateway(vector=vector, short_term=memory)
        self.tool_registry = ToolRegistry(vector=vector)
        self.validator = HallucinationValidator()
        # Injectable so tests can stub the job-board lookup without a DB.
        self._candidate_lookup = CandidateRepository.get
        self._graph = self._build_graph()

    # ---- nodes (bound coroutine methods) ----------------------------

    async def _load_context(self, state: AgentState) -> dict[str, Any]:
        return await load_context(state, gateway=self.gateway)

    async def _summarize(self, state: AgentState) -> dict[str, Any]:
        return await summarize_if_needed(state, gateway=self.gateway, llm=self.llm)

    async def _identify(self, state: AgentState) -> dict[str, Any]:
        """Decide whether the sender is a known person or a new number to onboard.

        Known = a registered job-seeker in the job board (lookup by phone) OR a
        number we've already captured BOTH name + email for (in customer_facts).
        Unknown numbers are routed to the onboarding response, which asks for the
        name then the email before any normal handling. Capture/storage itself is
        the persist node's job — we reuse the SAME extractors here so the field we
        decide to ask for is exactly the one persist will (or won't) store.
        """
        facts = dict(state.get("customer_facts") or {})
        phone = (state.get("customer_id") or "").strip()
        text = state.get("inbound_text", "") or ""

        # No phone (HTTP chat / non-WhatsApp callers) → we can't key on a number,
        # so don't gate — let the turn flow normally.
        if not phone:
            return {"is_known": True}

        # 1) Existence is decided by the ACTIVE job-board table, re-checked every
        # turn — never by the cache. A hit → known; the active record's name/email
        # take precedence over any (possibly stale) cached value.
        try:
            candidate = await self._candidate_lookup(
                tenant_id=state["tenant_id"], phone=phone
            )
        except Exception as exc:  # noqa: BLE001 — a DB blip must not hard-block; fall back to memory
            log.warning("identity_db_lookup_failed", error=str(exc)[:200])
            candidate = None
        if candidate:
            facts["full_name"] = candidate.get("full_name") or facts.get("full_name")
            facts["email"] = candidate.get("email") or facts.get("email")
            return {
                "is_known": True,
                "is_existing_user": True,
                "candidate_id": str(candidate.get("id")) if candidate.get("id") is not None else None,
                "customer_facts": facts,
            }

        # 2) NOT in the active table → a new number. The cache here only tracks
        # onboarding progress (so we don't re-ask), it never makes someone
        # "existing". Merge anything stated THIS turn to ask the next missing
        # field (and welcome them the moment both exist).
        had_both = bool(facts.get("full_name")) and bool(facts.get("email"))
        name = facts.get("full_name") or extract_name(
            text, assistant_prompt=self._last_assistant(state)
        )
        email = facts.get("email") or extract_email(text)

        if name and email:
            if had_both:
                return {"is_known": True}            # returning, already onboarded
            return {"is_known": False, "onboarding_prompt": _onboard_welcome(name)}
        if not name:
            return {"is_known": False, "onboarding_prompt": _ONBOARD_ASK_NAME}
        return {"is_known": False, "onboarding_prompt": _onboard_ask_email(name)}

    async def _onboarding_response(self, state: AgentState) -> dict[str, Any]:
        """0-LLM reply that asks a new number for their name/email (or welcomes
        them once both are in). The value they give is stored by the persist
        node via the shared identity extractors."""
        return {
            "intent": "onboarding",
            "draft_response": state.get("onboarding_prompt") or _ONBOARD_ASK_NAME,
            "used_llm": False,
        }

    @staticmethod
    def _last_assistant(state: AgentState) -> str | None:
        """The assistant line shown just before this message — lets the name
        extractor treat a bare reply as the answer to 'what's your name?'."""
        for turn in reversed(state.get("short_term") or []):
            if turn.get("role") == "assistant" and turn.get("content"):
                return turn["content"]
        return None

    async def _greeting_response(self, state: AgentState) -> dict[str, Any]:
        """0-LLM handling for pure greetings/farewells. A farewell also flags the
        session to be reset in persist (fresh start next time)."""
        # Known senders are greeted by name — identify() seeds full_name into
        # customer_facts from the job board (or earlier-captured memory).
        first = _first_name((state.get("customer_facts") or {}).get("full_name"))
        if _CLOSER_RX.search(state.get("inbound_text", "")):
            bye = f"Thanks for stopping by, {first}!" if first else "Thanks for stopping by!"
            return {
                "intent": "greeting",
                "draft_response": f"{bye} Message me anytime you need something.",
                "used_llm": False,
                "end_session": True,
            }
        hi = f"Hi {first}! What are you looking for today?" if first else "Hi! What are you looking for today?"
        return {
            "intent": "greeting",
            "draft_response": hi,
            "used_llm": False,
        }

    async def _planner(self, state: AgentState) -> dict[str, Any]:
        return await plan(
            state, llm=self.llm, registry=self.tool_registry,
            # planner gets a LEAN context (no rolling summary / recall) so it
            # never pollutes a search query with an older product.
            memory_context=self._memory_context(state, for_planner=True),
        )

    async def _execute(self, state: AgentState) -> dict[str, Any]:
        return await execute(state, registry=self.tool_registry)

    async def _reflect(self, state: AgentState) -> dict[str, Any]:
        return await reflect(state, llm=self.llm)

    async def _responder(self, state: AgentState) -> dict[str, Any]:
        return await respond(
            state, llm=self.llm, validator=self.validator,
            memory_context=self._memory_context(state),
        )

    async def _humanize(self, state: AgentState) -> dict[str, Any]:
        """Split the reply into paced bubbles. (Jobs have no images, so unlike
        the e-commerce version no image is attached to the first bubble.)"""
        s = get_settings()
        plan_ = build_delivery_plan(
            state.get("draft_response", "") or "",
            max_chars=s.agent_chunk_max_chars,
            max_chunks=s.agent_max_chunks,
            cps=s.agent_typing_cps,
            min_ms=s.agent_typing_min_ms,
            max_ms=s.agent_typing_max_ms,
        )
        out: dict[str, Any] = {
            "delivery_plan": plan_,
            "message_chunks": [b["text"] for b in plan_],
        }
        # Pin the job shown this turn so next turn's "this"/"that role"/"apply"
        # resolves to it. Skip application/status turns; a follow-up keeps the
        # existing pin.
        focus = self._focus_product(state)
        if focus:
            out["last_product_id"], out["last_product_doc"] = focus
        return out

    def _focus_product(self, state: AgentState) -> tuple[str, str] | None:
        """Pin the top job surfaced this turn as the 'current job' referent for
        deixis next turn. Skipped on application/status turns."""
        if _ORDER_HINT_RX.search(state.get("inbound_text", "")):
            return None
        jobs = state.get("catalog_hits") or []
        if not jobs or not jobs[0].get("id"):
            return None
        return str(jobs[0]["id"]), self._product_doc(jobs[0])

    @staticmethod
    def _product_doc(p: dict[str, Any]) -> str:
        """Compact one-line description of a job for the 'current job' pin."""
        bits = [str(p.get("title") or "Role")]
        if p.get("department"):
            bits.append(str(p["department"]))
        if p.get("location"):
            bits.append(str(p["location"]))
        if p.get("employment_type"):
            bits.append(str(p["employment_type"]))
        if p.get("salary_min") is not None or p.get("salary_max") is not None:
            lo, hi = p.get("salary_min"), p.get("salary_max")
            cur = p.get("salary_currency") or ""
            if lo is not None and hi is not None:
                bits.append(f"{cur} {lo}-{hi}".strip())
            elif hi is not None:
                bits.append(f"up to {cur} {hi}".strip())
        if p.get("job_ref"):
            bits.append(f"ref {p['job_ref']}")
        return " — ".join(bits)

    # ---- routers -----------------------------------------------------

    def _route_after_plan(self, state: AgentState) -> Literal["execute", "responder"]:
        """Tool steps → execute; a direct-answer plan → straight to the reply."""
        return "execute" if (state.get("plan") or []) else "responder"

    def _route_after_execute(self, state: AgentState) -> Literal["reflect", "responder"]:
        """Token saver: a single tool step that succeeded almost never needs a
        reflection LLM call — answer directly. Reflect only on multi-step plans
        or a failed step (where a replan/handoff might help)."""
        plan = state.get("plan") or []
        if len(plan) == 1 and plan[0].get("status") == "done":
            return "responder"
        return "reflect"

    def _route_after_reflect(self, state: AgentState) -> Literal["planner", "responder"]:
        reflections = state.get("reflections") or []
        wants_replan = bool(reflections) and reflections[-1].get("next") == "replan"
        return "planner" if (wants_replan and self._can_loop(state)) else "responder"

    def _can_loop(self, state: AgentState) -> bool:
        s = get_settings()
        within_loops = state.get("loop_count", 0) < s.agent_max_loops
        within_time = time.monotonic() < state.get("deadline", float("inf"))
        return within_loops and within_time

    async def _persist(self, state: AgentState) -> dict[str, Any]:
        await persist(state, gateway=self.gateway)
        AGENT_LOOPS.observe(state.get("loop_count", 0))
        latency = int((time.time() - state.get("received_at", time.time())) * 1000)
        return {"latency_ms": latency}

    # ---- helpers -----------------------------------------------------

    def _route_after_identify(
        self, state: AgentState
    ) -> Literal["onboarding", "greeting", "agent"]:
        """Unknown numbers go to onboarding (ask name/email). Known senders keep
        the existing split: pure greetings/farewells skip the LLM; everything
        else goes to the reasoning agent."""
        if not state.get("is_known", False):
            return "onboarding"
        q = state.get("inbound_text", "")
        if _GREETING_RX.match(q) or _CLOSER_RX.match(q):
            return "greeting"
        return "agent"

    def _memory_context(self, state: AgentState, *, for_planner: bool = False) -> str | None:
        """Compress durable memory into a short block.

        The *pinned current product* goes first, labelled as the referent for
        "this"/"it"/details. On a follow-up we drop the rolling summary +
        recall (they echo older products → wrong-item answers).

        ``for_planner=True`` returns a LEAN context — facts only — so the planner
        never injects an older product/colour into a fresh search query.
        """
        cached = state.get("cached_product") or {}
        followup = bool(cached.get("doc")) and is_followup(state.get("inbound_text", ""))
        facts = state.get("customer_facts") or {}

        if for_planner:
            # The follow-up shortcut already handles deixis before the LLM
            # planner, so the planner only needs customer facts here.
            return ("Customer: " + ", ".join(f"{k}={v}" for k, v in facts.items())) if facts else None

        # When this turn fetched fresh jobs/applications, answer from THOSE — drop
        # the rolling summary + recall so a stale topic (e.g. a job mentioned in an
        # earlier, since-corrected turn) can never leak back into a live answer.
        has_fresh_results = any(
            r.get("tool") in {"search_jobs", "get_application_status", "list_jobs_overview"}
            and r.get("result")
            for r in (state.get("working") or {}).get("tool_results") or []
        )

        bits: list[str] = []
        if cached.get("doc"):
            bits.append(
                "CURRENT PRODUCT — the customer is asking about THIS (resolve "
                "'this' / 'it' / details / colours / sizes / 'yes' to it):\n"
                + cached["doc"]
            )
        if facts:
            bits.append("Customer: " + ", ".join(f"{k}={v}" for k, v in facts.items()))

        if not followup and not has_fresh_results:
            if state.get("rolling_summary"):
                bits.append(f"Earlier in the chat: {state['rolling_summary']}")
            recalled = state.get("semantic_hits") or []
            if recalled:
                bits.append(
                    "Recalled: "
                    + " | ".join(h.get("text", "")[:120] for h in recalled[:3])
                )

        pending = state.get("pending_actions") or []
        if pending:
            bits.append(f"Open follow-ups for this customer: {len(pending)}.")
        if state.get("session_status") == "resumed":
            bits.append("(Returning customer resuming after a break — re-orient warmly.)")
        return "\n".join(bits) or None

    # ---- graph -------------------------------------------------------

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("load_context", self._load_context)
        g.add_node("identify", self._identify)
        g.add_node("onboarding_response", self._onboarding_response)
        g.add_node("greeting_response", self._greeting_response)
        g.add_node("summarize", self._summarize)
        g.add_node("planner", self._planner)
        g.add_node("execute", self._execute)
        g.add_node("reflect", self._reflect)
        g.add_node("responder", self._responder)
        g.add_node("humanize", self._humanize)
        g.add_node("persist", self._persist)

        g.add_edge(START, "load_context")
        g.add_edge("load_context", "identify")
        # unknown number → onboarding; known → greeting short-circuit or agent
        g.add_conditional_edges(
            "identify", self._route_after_identify,
            {
                "onboarding": "onboarding_response",
                "greeting": "greeting_response",
                "agent": "summarize",
            },
        )
        g.add_edge("onboarding_response", "humanize")
        g.add_edge("greeting_response", "humanize")
        g.add_edge("summarize", "planner")
        g.add_conditional_edges(
            "planner", self._route_after_plan,
            {"execute": "execute", "responder": "responder"},
        )
        g.add_conditional_edges(
            "execute", self._route_after_execute,
            {"reflect": "reflect", "responder": "responder"},
        )
        g.add_conditional_edges(
            "reflect", self._route_after_reflect,
            {"planner": "planner", "responder": "responder"},
        )
        g.add_edge("responder", "humanize")
        g.add_edge("humanize", "persist")
        g.add_edge("persist", END)
        return g.compile()

    # ---- public entry ------------------------------------------------

    async def handle(
        self,
        *,
        request_id: str,
        conversation_id: str | None,
        customer_query: str,
        customer_external_id: str | None,
        channel: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        tid = tenant_id or get_current_tenant_id()
        conv_id = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        initial: AgentState = {
            "tenant_id": tid,
            "conversation_id": conv_id,
            "customer_id": customer_external_id or "",
            "inbound_text": customer_query,
            "inbound_kind": "text",
            "request_id": request_id,
            "received_at": time.time(),
            # reasoning-loop budget (Phase 0 Budget semantics, tracked on state)
            "loop_count": 0,
            "deadline": time.monotonic() + get_settings().agent_deadline_seconds,
        }
        final = await self._graph.ainvoke(initial)
        log.info(
            "agent_done",
            request_id=request_id,
            tenant_id=tid,
            conversation_id=conv_id,
            session_status=final.get("session_status"),
            used_llm=final.get("used_llm"),
            latency_ms=final.get("latency_ms"),
        )
        return {
            # ChatResponse-compatible (HTTP chat endpoint)
            "request_id": request_id,
            "conversation_id": conv_id,
            "intent": final.get("session_status") or "agent",
            "response": final.get("draft_response", ""),
            "routing": None,
            "validation": "AGENT",
            "escalated": False,
            "latency_ms": final.get("latency_ms", 0),
            # agent extras (ignored by ChatResponse; used by the WhatsApp route)
            "delivery_plan": final.get("delivery_plan") or [],
            "session_id": final.get("session_id"),
            "session_status": final.get("session_status"),
            "used_llm": final.get("used_llm", True),
            # memory snapshot for the flow UI
            "memory": {
                "session_status": final.get("session_status"),
                "short_term_turns": len(final.get("short_term") or []),
                "cached_product": (final.get("cached_product") or {}).get("doc"),
                "customer_facts": final.get("customer_facts") or {},
                "pending": len(final.get("pending_actions") or []),
                "rolling_summary": (final.get("rolling_summary") or "")[:400],
            },
            # structured step detail for the flow UI (deep tool view)
            "steps": _step_detail(final),
        }
