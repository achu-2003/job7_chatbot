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

from app import onboarding
from app.agent import jobflow
from app.agent.browse import search_terms
from app.agent.context import is_followup
from app.agent.identity import extract_name, is_plausible_name
from app.agent.nodes.humanizer import build_delivery_plan
from app.agent.nodes.load_context import load_context
from app.agent.nodes.persist import persist
from app.agent.nodes.planner import plan
from app.agent.nodes.reflection import reflect
from app.agent.nodes.responder import respond
from app.agent.nodes.summarizer import summarize_if_needed
from app.agent.nodes.tool_router import execute
from app.agent.state import AgentState
from app.chatbot import wa_format as wa
from app.chatbot.memory import ConversationMemory
from app.chatbot.validator import HallucinationValidator
from app.db.repositories import CandidateRepository
from app.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import AGENT_LOOPS
from app.core.tenancy import get_current_tenant_id
from app.llm.client import LLMClient
from app.mcp.tools import (
    ToolRegistry,
    get_job_core,
    list_category_jobs_core,
    list_jobs_overview_core,
    recommend_jobs_core,
    search_jobs_by_title_core,
)
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

# ---- quick-reply menu (greeting / onboarding success) -------------------
# Three tappable buttons. A tap comes back as the button TITLE text (see the
# WhatsApp route), so the menu node + browse path detect these titles. Reply
# button titles are capped at 20 chars by Meta.
_MENU_BUTTONS = (
    ("menu_search", "Job Search"),
    ("menu_status", "Application Status"),
    ("menu_recommend", "Recommended Jobs"),
)
# Tapped/typed "Job Search" → show the category list. "Recommended Jobs" →
# profile-based recommendations. "Application Status" is left to the normal
# pipeline (planner → get_application_status), which already handles it.
_MENU_SEARCH_RX = re.compile(r"^\s*(job\s*search|search\s*jobs?)\s*$", re.IGNORECASE)
_MENU_RECOMMEND_RX = re.compile(
    r"^\s*(recommend(ed)?\s*jobs?|recommendations?|recommend)\s*$", re.IGNORECASE
)


def _menu_buttons_message(body: str) -> dict[str, Any]:
    """The greeting/onboarding quick-reply buttons wrapped as a WhatsApp
    interactive payload (text ``body`` doubles as the web/fallback reply)."""
    return wa.buttons_message(body, _MENU_BUTTONS)


# ---- lane selection (job seeker vs job creator) -------------------------
# A greeting now opens by asking which lane the sender is in. "Job Seeker"
# resumes the existing candidate flow (onboarding / menu); "Job Creator" is a
# placeholder until the recruiter side is built. The choice rides on a
# structured button id (role:seeker / role:creator) so it survives WhatsApp's
# title truncation; the exact title text doubles as the web/typed path.
_ROLE_BUTTONS = (
    ("role:seeker", "Job Seeker"),
    ("role:creator", "Job Creator"),
)
_ROLE_SEEKER_RX = re.compile(r"^\s*job\s*seeker\s*$", re.IGNORECASE)
_ROLE_CREATOR_RX = re.compile(r"^\s*job\s*creator\s*$", re.IGNORECASE)


def _role_choice_message(body: str) -> dict[str, Any]:
    """The Job Seeker / Job Creator quick-reply buttons wrapped as a WhatsApp
    interactive payload (``body`` doubles as the web/fallback reply)."""
    return wa.buttons_message(body, _ROLE_BUTTONS)


def _role_selection(state: AgentState) -> str | None:
    """Which lane the sender picked this turn: ``"seeker"``, ``"creator"``, or
    ``None``. Reads the structured button id first (role:seeker / role:creator);
    falls back to the exact title text for the web/typed path."""
    prefix, rest = jobflow.split_id(state.get("button_id"))
    if prefix == "role" and rest in {"seeker", "creator"}:
        return rest
    text = state.get("inbound_text", "") or ""
    if _ROLE_SEEKER_RX.match(text):
        return "seeker"
    if _ROLE_CREATOR_RX.match(text):
        return "creator"
    return None

# ---- new-candidate onboarding copy --------------------------------------
# A number we've never seen (not in the job board AND not onboarded yet) is
# asked to introduce itself before we help with roles: first the name in chat,
# then a short self-hosted web form (email/experience/role) we keep in Redis.
# The name prompt MUST contain "full name"/"your name" so
# app.agent.identity.extract_name treats the bare reply as the answer
# (see _ASKED_FOR_NAME_RX).
_ONBOARD_ASK_NAME = (
    "Hi, and welcome! I don't think we've spoken before. "
    "Before I help you find roles, could you share your full name?"
)


def _first_name(name: str | None) -> str:
    return name.split()[0] if name else ""


def _apply_reply(msg: str) -> dict[str, Any]:
    """Standard 0-LLM apply-flow turn (single bubble), short-circuited to delivery."""
    return {"intent": "apply", "did_browse": True, "used_llm": False, "draft_response": msg}


def _apply_prompt(msg: str) -> dict[str, Any]:
    """An apply-flow question with a tappable Skip button (the candidate can also
    type the answer / send a document)."""
    out = _apply_reply(msg)
    out["whatsapp_interactive"] = wa.buttons_message(msg, [("apply_skip", "Skip")])
    return out


def _onboard_ask_form(name: str | None, link: str) -> str:
    """Text version (with the raw link inline) — used for web and as the
    fallback if WhatsApp rejects the cta_url button."""
    who = f", {_first_name(name)}" if name else ""
    return (
        f"Thanks{who}! One quick step to finish setting up your profile — please "
        f"fill this short form (email, experience, preferred role/location):\n{link}"
        "\n\nOnce you've submitted it, message me here and we'll find you some roles."
    )


def _onboard_form_cta_body(name: str | None) -> str:
    """Body text for the WhatsApp cta_url button (the URL lives on the button, so
    it's omitted here)."""
    who = f", {_first_name(name)}" if name else ""
    return (
        f"Thanks{who}! One quick step to finish setting up your profile — tap "
        "below to fill a short form (email, experience, preferred role/location). "
        "Once you're done, message me here and we'll find you some roles."
    )


def _onboard_success(name: str | None) -> str:
    """One-time confirmation shown the first time a candidate messages after
    submitting the onboarding form (carries the Job Seeker / Job Creator lane
    choice, set in _onboarding_response)."""
    who = f" {_first_name(name)}" if name else ""
    return (
        f"You're registered successfully{who}! ✅\n\n"
        "Are you here to find a job, or to post jobs and hire?"
    )


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
        # Injectable so tests can stub the job-board lookups without a DB.
        self._candidate_lookup = CandidateRepository.get
        self._category_browse = list_category_jobs_core
        self._jobs_overview = list_jobs_overview_core
        self._recommend = recommend_jobs_core
        self._job_lookup = get_job_core
        self._title_search = search_jobs_by_title_core
        self._graph = self._build_graph()

    # ---- nodes (bound coroutine methods) ----------------------------

    async def _load_context(self, state: AgentState) -> dict[str, Any]:
        return await load_context(state, gateway=self.gateway)

    async def _summarize(self, state: AgentState) -> dict[str, Any]:
        return await summarize_if_needed(state, gateway=self.gateway, llm=self.llm)

    async def _browse(self, state: AgentState) -> dict[str, Any]:
        """The tappable job-browse state machine (deterministic, 0-LLM):

            category tapped/typed → list its roles (paged 10 at a time)
            "More roles" tapped   → next page
            a role tapped         → ask preferred location (tappable list)
            a location tapped     → matching job cards (Apply / Save / Share)
            an Apply/Save/Share tap → record + confirm

        Selection rides on the interactive ``button_id`` (so it survives WhatsApp's
        24-char title truncation); typed text is the web/fallback path. Anything
        that isn't a browse step returns ``{}`` to fall through to the planner.
        """
        # Mid apply-time collection? A typed reply / sent document is the answer
        # to the current question; the Skip button skips it; tapping a DIFFERENT
        # button abandons the apply flow.
        apply_state = await self._load_apply(state)
        if apply_state:
            bid = state.get("button_id") or ""
            if bid == "apply_skip":
                return await self._apply_answer(state, apply_state, skip=True)
            if bid:
                await self._clear_apply(state)
            else:
                return await self._apply_answer(state, apply_state)
        elif state.get("attachment"):
            # A file arrived outside an application — nudge them to apply first.
            return _apply_reply(
                "Thanks for the file! I can attach your resume once you tap Apply "
                "on a role. Want me to find you some jobs?"
            )

        text = state.get("inbound_text", "") or ""
        prefix, rest = jobflow.split_id(state.get("button_id"))

        if prefix in {"apply", "save", "share"}:
            return await self._job_action(state, prefix, rest)
        if prefix == "view":
            return await self._browse_one(state, rest)
        if prefix == "job":
            return await self._browse_role(state, rest)
        if prefix == "more":
            category, _, off = rest.rpartition(":")
            try:
                offset = int(off)
            except ValueError:
                offset = 0
            return await self._browse_category(state, category or rest, offset=offset)
        if prefix == "category":
            return await self._browse_category(state, rest, offset=0)

        # Typed path: an application/status turn must not be hijacked into browse.
        if _ORDER_HINT_RX.search(text):
            return {}
        # A typed category → its role list; otherwise try matching a specific role
        # by job title ("welder" → Welder openings). Only if both miss do we fall
        # through to the planner (FAQ / chit-chat / status are handled there).
        return (
            await self._browse_category(state, text, offset=0)
            or await self._browse_role_search(state, text)
        )

    async def _browse_role_search(self, state: AgentState, text: str) -> dict[str, Any]:
        """A typed specific role (not a category) → the matching job cards,
        straight from a deterministic title search (no LLM)."""
        query = search_terms(text)
        if len(query) < 3:
            return {}
        try:
            jobs = await self._title_search(
                tenant_id=state["tenant_id"], query=query, limit=5
            )
        except Exception as exc:  # noqa: BLE001 — a search miss must never break the turn
            log.warning("title_search_failed", error=str(exc)[:200])
            return {}
        if not jobs:
            return {}
        cards, body = jobflow.job_cards(jobs, limit=5)
        plural = "opening" if len(jobs) == 1 else "openings"
        header = f"{len(jobs)} {plural} matching “{query.title()}”:"
        await self._save_browse(state, {"stage": "results"})
        return {
            "intent": "browse", "did_browse": True, "used_llm": False,
            "draft_response": f"{header}\n\n{body}", "whatsapp_messages": cards,
            "catalog_hits": jobs,
        }

    async def _category_jobs(self, state: AgentState, category: str) -> tuple[str, list[dict[str, Any]]]:
        """Resolve a category phrase to (exact name, all its jobs), or ('', [])."""
        try:
            res = await self._category_browse(tenant_id=state["tenant_id"], text=category)
        except Exception as exc:  # noqa: BLE001 — a browse miss must never break the turn
            log.warning("category_browse_failed", error=str(exc)[:200])
            return "", []
        if not res or not res.get("jobs"):
            return "", []
        return res.get("category") or category, res["jobs"]

    async def _browse_category(
        self, state: AgentState, category_text: str, *, offset: int
    ) -> dict[str, Any]:
        """Show the tappable role list for a category (page ``offset``)."""
        category, jobs = await self._category_jobs(state, category_text)
        if not jobs:
            return {}  # not a known category → fall through to the planner
        payload, fallback = jobflow.role_list_message(jobs, category=category, offset=offset)
        await self._save_browse(state, {"stage": "roles", "category": category, "offset": offset})
        return {
            "intent": "browse", "did_browse": True, "used_llm": False, "single_bubble": True,
            "draft_response": fallback, "whatsapp_interactive": payload, "catalog_hits": jobs,
        }

    async def _browse_role(self, state: AgentState, ref: str) -> dict[str, Any]:
        """A role was tapped → show the detail cards (emoji + Apply/Save/Share)
        for EVERY open posting of that role, across all locations. e.g. tapping
        'Backend Developer' when 3 are open shows all 3 cards."""
        st = await self._load_browse(state)
        category = st.get("category")
        jobs = (await self._category_jobs(state, category))[1] if category else []

        role = next((j for j in jobs if jobflow.job_ref(j) == ref), None)
        if role is None:                       # tapped from recommendations / stale list
            role = await self._job_lookup(tenant_id=state["tenant_id"], ref=ref)
            if role and role.get("department") and not jobs:
                category, jobs = await self._category_jobs(state, role["department"])
        if role is None:
            return {}                          # unknown ref → fall through

        title = role.get("title") or "this role"
        matches = [j for j in jobs if jobflow.same_role_family(title, j.get("title"))]
        if not matches:
            matches = [role]

        cards, text = jobflow.job_cards(matches, limit=8)
        plural = "opening" if len(matches) == 1 else "openings"
        header = f"{len(matches)} {title} {plural}:"
        await self._save_browse(state, {"stage": "results", "category": category, "role_title": title})
        return {
            "intent": "browse", "did_browse": True, "used_llm": False, "single_bubble": True,
            "draft_response": f"{header}\n\n{text}", "whatsapp_messages": cards,
            "catalog_hits": matches,
        }

    async def _browse_one(self, state: AgentState, ref: str) -> dict[str, Any]:
        """Show the details of ONE specific job (e.g. a tapped recommended role)
        as a single card with Apply/Save/Share."""
        try:
            job = await self._job_lookup(tenant_id=state["tenant_id"], ref=ref)
        except Exception as exc:  # noqa: BLE001
            log.warning("job_lookup_failed", error=str(exc)[:200])
            job = None
        if not job:
            return {}  # unknown ref → fall through to the planner
        cards, text = jobflow.job_cards([job], limit=1)
        return {
            "intent": "browse", "did_browse": True, "used_llm": False, "single_bubble": True,
            "draft_response": text, "whatsapp_messages": cards, "catalog_hits": [job],
        }

    async def _job_action(self, state: AgentState, action: str, ref: str) -> dict[str, Any]:
        """Handle a Save / Share tap (Apply has its own collection flow)."""
        if action == "apply":
            return await self._apply_start(state, ref)
        job = await self._job_lookup(tenant_id=state["tenant_id"], ref=ref)
        title = (job or {}).get("title") or "that role"
        kw = {"tenant_id": state["tenant_id"], "conversation_id": state["conversation_id"]}
        if action == "save":
            await self._safe_store(self.gateway.save_job, ref=ref, job=job or {"job_ref": ref}, **kw)
            msg = f"Saved {title} to your list. Tap Apply on it whenever you're ready."
        else:  # share
            msg = jobflow.share_text(job) if job else f"Job reference: {ref}"
        return {"intent": "browse", "did_browse": True, "used_llm": False, "draft_response": msg}

    # ---- apply-time top-up (progressive profiling) -------------------

    async def _apply_start(self, state: AgentState, ref: str) -> dict[str, Any]:
        """Begin applying: ask only for the apply-time details the profile is
        still missing (resume, expected salary). If the profile already has them
        — or we have no staged profile — go straight to recording the apply."""
        job = await self._job_lookup(tenant_id=state["tenant_id"], ref=ref)
        title = (job or {}).get("title") or "this role"
        reg = await self._registration(state)
        if not job or not reg:
            # No job id / no staged profile → fall back to recording interest.
            await self._safe_store(
                self.gateway.record_interest, ref=ref, job=job or {"job_ref": ref},
                tenant_id=state["tenant_id"], conversation_id=state["conversation_id"],
            )
            return _apply_reply(f"Great — I've noted your interest in {title}. Our team will reach out.")

        pending = [
            f["key"] for f in onboarding.APPLY_FIELDS
            if f["key"] not in onboarding.known_apply_fields(reg)
        ]
        if not pending:
            return await self._apply_finalize(state, ref, job, reg, answers={})
        await self._save_apply(
            state, {"job_ref": ref, "job_title": title, "pending": pending, "answers": {}}
        )
        prompt = onboarding.APPLY_FIELD_BY_KEY[pending[0]]["prompt"]
        return _apply_prompt(f"Let's apply for {title}.\n\n{prompt}")

    async def _apply_answer(
        self, state: AgentState, apply_state: dict[str, Any], *, skip: bool = False
    ) -> dict[str, Any]:
        """Store the candidate's answer (typed value, sent document, or Skip) to
        the current apply-time question, then ask the next one or finalize."""
        text = (state.get("inbound_text") or "").strip()
        attachment = state.get("attachment") or {}
        pending = list(apply_state.get("pending") or [])
        answers = dict(apply_state.get("answers") or {})
        if not pending:
            await self._clear_apply(state)
            return {}
        key = pending[0]
        spec = onboarding.APPLY_FIELD_BY_KEY.get(key, {})

        if skip or onboarding.is_skip(text):
            answers[key] = None
        elif key == "resume" and attachment.get("kind") == "document":
            # A real resume upload — store a reference (filename + media id). On
            # real-DB wiring this is downloaded from Meta and hosted.
            fn = attachment.get("filename") or "resume"
            answers[key] = f"{fn} [wa-doc:{attachment.get('media_id')}]"
        elif attachment:
            # A file on a non-file question → ignore it and re-ask.
            return _apply_prompt(onboarding.APPLY_FIELD_BY_KEY[key]["prompt"])
        elif spec.get("numeric"):
            val = onboarding.parse_salary(text)
            if val is None:
                return _apply_prompt("Please reply with a number (e.g. 25000), or tap Skip.")
            answers[key] = val
        else:
            answers[key] = text

        pending = pending[1:]
        if pending:
            await self._save_apply(state, {**apply_state, "pending": pending, "answers": answers})
            return _apply_prompt(onboarding.APPLY_FIELD_BY_KEY[pending[0]]["prompt"])

        ref = apply_state.get("job_ref")
        job = await self._job_lookup(tenant_id=state["tenant_id"], ref=ref)
        reg = await self._registration(state)
        await self._clear_apply(state)
        if not job or not reg:
            return _apply_reply("Thanks! I've noted your details — our team will follow up.")
        return await self._apply_finalize(state, ref, job, reg, answers)

    async def _apply_finalize(
        self, state: AgentState, ref: str, job: dict[str, Any],
        reg: dict[str, Any], answers: dict[str, Any],
    ) -> dict[str, Any]:
        """Build + stage the application record (Redis only) and confirm."""
        built = onboarding.build_application_record(registration=reg, job=job, answers=answers)
        title = job.get("title") or "the role"
        kw = {"tenant_id": state["tenant_id"], "conversation_id": state["conversation_id"]}
        try:
            await self.gateway.save_application(ref=ref, record=built["application"], **kw)
            if built["profile_update"]:
                await self.gateway.update_registration_profile(fields=built["profile_update"], **kw)
        except Exception as exc:  # noqa: BLE001 — staging must not 500 the turn
            log.warning("apply_stage_failed", error=str(exc)[:200])
        return _apply_reply(
            f"✅ Applied to {title}! Our team will review your profile and get back "
            "to you. Anything else I can help with?"
        )

    async def _registration(self, state: AgentState) -> dict[str, Any] | None:
        try:
            return await self.gateway.registration(
                tenant_id=state["tenant_id"], conversation_id=state["conversation_id"]
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("registration_load_failed", error=str(exc)[:200])
            return None

    async def _load_apply(self, state: AgentState) -> dict[str, Any] | None:
        try:
            return await self.gateway.apply_state(
                tenant_id=state["tenant_id"], conversation_id=state["conversation_id"]
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("apply_state_load_failed", error=str(exc)[:200])
            return None

    async def _save_apply(self, state: AgentState, st: dict[str, Any]) -> None:
        try:
            await self.gateway.set_apply_state(
                tenant_id=state["tenant_id"], conversation_id=state["conversation_id"], state=st
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("apply_state_save_failed", error=str(exc)[:200])

    async def _clear_apply(self, state: AgentState) -> None:
        try:
            await self.gateway.clear_apply_state(
                tenant_id=state["tenant_id"], conversation_id=state["conversation_id"]
            )
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    async def _safe_store(fn, **kw) -> None:
        try:
            await fn(**kw)
        except Exception as exc:  # noqa: BLE001 — a Redis hiccup must not 500 the tap
            log.warning("job_action_store_failed", error=str(exc)[:200])

    async def _load_browse(self, state: AgentState) -> dict[str, Any]:
        try:
            return await self.gateway.browse_state(
                tenant_id=state["tenant_id"], conversation_id=state["conversation_id"]
            ) or {}
        except Exception as exc:  # noqa: BLE001
            log.warning("browse_state_load_failed", error=str(exc)[:200])
            return {}

    async def _save_browse(self, state: AgentState, data: dict[str, Any]) -> None:
        try:
            await self.gateway.set_browse_state(
                tenant_id=state["tenant_id"], conversation_id=state["conversation_id"], state=data
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("browse_state_save_failed", error=str(exc)[:200])

    def _route_after_browse(self, state: AgentState) -> Literal["humanize", "summarize"]:
        """A matched category browse goes straight to delivery; otherwise continue
        into the normal summarize → planner reasoning path."""
        return "humanize" if state.get("did_browse") else "summarize"

    async def _menu(self, state: AgentState) -> dict[str, Any]:
        """Handle the quick-reply menu taps (and their typed equivalents):

        * "Job Search"       → a tappable list of job categories (each row's
          title is the category name, so a tap flows into the browse node which
          lists every job in it).
        * "Recommended Jobs" → roles matched to the candidate's preferred role
          from onboarding, or the newest jobs when there's none on file.

        Everything else (incl. "Application Status") returns ``{}`` to fall
        through to the browse → planner pipeline that already handles it.
        """
        text = state.get("inbound_text", "") or ""
        if _MENU_SEARCH_RX.match(text):
            return await self._category_menu(state)
        if _MENU_RECOMMEND_RX.match(text):
            return await self._recommend_menu(state)
        return {}

    def _route_after_menu(self, state: AgentState) -> Literal["humanize", "browse"]:
        """A handled menu tap goes straight to delivery; otherwise continue into
        the category-browse / reasoning path."""
        return "humanize" if state.get("did_menu") else "browse"

    async def _category_menu(self, state: AgentState) -> dict[str, Any]:
        """Build the tappable category list for the 'Job Search' button."""
        try:
            overview = await self._jobs_overview(tenant_id=state["tenant_id"])
        except Exception as exc:  # noqa: BLE001 — a menu miss must never break the turn
            log.warning("menu_overview_failed", error=str(exc)[:200])
            return {}
        cats = overview.get("categories") or []
        if not cats:
            return {}
        total = overview.get("total_open_jobs")
        body = (
            f"We have {total} open jobs. Which area interests you?"
            if total else "Which area interests you?"
        )
        rows = [
            {
                "id": f"category:{c['category']}",
                "title": c["category"],
                "description": f"{c['count']} open role" + ("s" if c["count"] != 1 else ""),
            }
            for c in cats[:10]
        ]
        # Text fallback (web + when the interactive list is rejected) lists the
        # same areas; tapping/typing one runs the category browse.
        lines = [body, ""] + [f"• {c['category']} ({c['count']})" for c in cats[:10]]
        lines.append("\nReply with an area to see all its roles.")
        return {
            "intent": "menu",
            "did_menu": True,
            "used_llm": False,
            "single_bubble": True,
            "draft_response": "\n".join(lines),
            "whatsapp_interactive": wa.list_message(
                body=body, button_text="Choose area", rows=rows,
                header="Job categories", section_title="Areas",
            ),
        }

    async def _recommend_menu(self, state: AgentState) -> dict[str, Any]:
        """Profile-based recommendations for the 'Recommended Jobs' button."""
        role, location = await self._profile_pref(state)
        try:
            jobs = await self._recommend(
                self.vector, tenant_id=state["tenant_id"],
                role=role, location=location, limit=8,
            )
        except Exception as exc:  # noqa: BLE001 — never break the turn on a lookup miss
            log.warning("menu_recommend_failed", error=str(exc)[:200])
            jobs = []
        first = _first_name((state.get("customer_facts") or {}).get("full_name"))
        if not jobs:
            who = f", {first}" if first else ""
            return {
                "intent": "menu", "did_menu": True, "used_llm": False,
                "draft_response": (
                    f"I couldn't find roles to recommend just yet{who} — tell me a "
                    "role or area you're interested in and I'll pull some up."
                ),
            }
        payload, fallback = jobflow.recommend_list_message(jobs, name=first)
        return {
            "intent": "menu", "did_menu": True, "used_llm": False, "single_bubble": True,
            "catalog_hits": jobs,
            "draft_response": fallback,
            "whatsapp_interactive": payload,
        }

    async def _profile_pref(self, state: AgentState) -> tuple[str | None, str | None]:
        """The candidate's preferred role + location from the onboarding form
        (Redis), if any. Older records (before the form split) only carry the
        combined ``location`` field, so we fall back to it for the role."""
        try:
            form = await self.gateway.onboarding(
                tenant_id=state["tenant_id"], conversation_id=state["conversation_id"],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("menu_profile_lookup_failed", error=str(exc)[:200])
            form = None
        if not form:
            return None, None
        role = (form.get("preferred_role") or form.get("location") or "").strip() or None
        location = (form.get("location") or "").strip() or None
        return role, location

    async def _identify(self, state: AgentState) -> dict[str, Any]:
        """Decide whether the sender is a known person or a new number to onboard.

        Known = a registered job-seeker in the job board (lookup by phone) OR a
        new number that has completed onboarding: their name (captured in chat)
        plus a short self-hosted web form (email/experience/role) whose data is
        kept in Redis only. Until both are done the sender is routed to the
        onboarding response — first asked for the name, then handed the form
        link — and all normal job help is gated. Name capture/storage is the
        persist node's job; we reuse the SAME name extractor here so the field we
        ask for is exactly the one persist will (or won't) store.
        """
        facts = dict(state.get("customer_facts") or {})
        phone = (state.get("customer_id") or "").strip()
        text = state.get("inbound_text", "") or ""

        # Self-heal: a previously-stored full_name that isn't actually name-shaped
        # (legacy junk like "Searching Python Developer Job") must not greet the
        # sender or make them look already-onboarded — drop it and re-ask.
        if facts.get("full_name") and not is_plausible_name(facts["full_name"]):
            log.info("dropping_implausible_cached_name", value=str(facts["full_name"])[:40])
            facts.pop("full_name", None)

        # Remember the Job Seeker / Job Creator lane the MOMENT it's picked, so we
        # never ask again — later turns read it back from customer_facts and route
        # straight into that lane's flow.
        picked_lane = _role_selection(state)
        if picked_lane and picked_lane != facts.get("lane"):
            facts["lane"] = picked_lane
            if phone:
                try:
                    await self.gateway.semantic.remember_fact(
                        tenant_id=state["tenant_id"], customer_id=phone,
                        key="lane", value=picked_lane, confidence=1.0, source="chat",
                    )
                except Exception as exc:  # noqa: BLE001 — best-effort, never block the turn
                    log.warning("lane_persist_failed", error=str(exc)[:200])

        # No phone (HTTP chat / non-WhatsApp callers) → we can't key on a number,
        # so don't gate — let the turn flow normally.
        if not phone:
            return {"is_known": True, "customer_facts": facts}

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

        # 2) NOT in the active table → a new number we must ONBOARD. Step one is
        # the name (captured in chat); step two is a short self-hosted web form
        # (email, experience, preferred role/location) whose data we keep in
        # Redis only — never the business DB. Job help is gated until that form
        # is submitted.
        name = facts.get("full_name") or extract_name(
            text, assistant_prompt=self._last_assistant(state)
        )
        if not name:
            return {"is_known": False, "onboarding_prompt": _ONBOARD_ASK_NAME}

        form = await self.gateway.onboarding(
            tenant_id=state["tenant_id"], conversation_id=state["conversation_id"],
        )
        if form:
            # Form submitted → fully onboarded. Seed name/email from it so the
            # bot can greet/help right away. The FIRST turn after submission gets
            # a one-time success message; later turns proceed normally.
            facts["full_name"] = facts.get("full_name") or form.get("name") or name
            if form.get("email"):
                facts["email"] = facts.get("email") or form.get("email")
            if not form.get("welcomed"):
                await self.gateway.mark_onboarding_welcomed(
                    tenant_id=state["tenant_id"],
                    conversation_id=state["conversation_id"],
                )
                return {
                    "is_known": True,
                    "just_onboarded": True,
                    "onboarding_prompt": _onboard_success(facts.get("full_name") or name),
                    "customer_facts": facts,
                }
            return {"is_known": True, "customer_facts": facts}

        token = await self.gateway.onboarding_token(
            tenant_id=state["tenant_id"], customer_id=phone,
            conversation_id=state["conversation_id"], name=name,
        )
        link = f"{get_settings().public_base_url.rstrip('/')}/onboard/form?token={token}"
        out: dict[str, Any] = {
            "is_known": False,
            # text-with-link kept as the web reply + WhatsApp fallback
            "onboarding_prompt": _onboard_ask_form(name, link),
        }
        # Send a tappable "Open form" cta_url button on WhatsApp — but only for an
        # https link (Meta rejects cta_url with http/localhost). For a non-https
        # base URL we fall back to the inline-link text above.
        if link.startswith("https://"):
            out["whatsapp_interactive"] = wa.cta_url_message(
                body=_onboard_form_cta_body(name), display_text="Open form", url=link,
            )
        return out

    async def _onboarding_response(self, state: AgentState) -> dict[str, Any]:
        """0-LLM reply that asks a new number for their name/email (or welcomes
        them once both are in). The value they give is stored by the persist
        node via the shared identity extractors."""
        out: dict[str, Any] = {
            "intent": "onboarding",
            "draft_response": state.get("onboarding_prompt") or _ONBOARD_ASK_NAME,
            "used_llm": False,
        }
        # The one-time "profile complete" success message offers the Job Seeker /
        # Job Creator lane choice — a freshly-registered candidate picks where to
        # go next, the same hub a known sender gets on greeting. (Earlier
        # onboarding steps keep identify's form-link cta, never overwritten here.)
        if state.get("just_onboarded"):
            out["whatsapp_interactive"] = _role_choice_message(out["draft_response"])
        return out

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
            # Offer the quick-reply menu so the candidate can tap instead of type.
            "whatsapp_interactive": _menu_buttons_message(hi),
        }

    async def _role_select(self, state: AgentState) -> dict[str, Any]:
        """0-LLM opener shown on a greeting: ask whether the sender is here to
        find a job (Job Seeker → the existing candidate flow) or to hire (Job
        Creator → a stub until the recruiter side lands)."""
        first = _first_name((state.get("customer_facts") or {}).get("full_name"))
        who = f" {first}" if first else ""
        body = (
            f"Hi{who}! Welcome to Jobs7. 👋\n\n"
            "Are you here to find a job, or to post jobs and hire?\n\n"
            'Tap an option below (or reply "Job Seeker" / "Job Creator").'
        )
        return {
            "intent": "role_select",
            "draft_response": body,
            "used_llm": False,
            "single_bubble": True,
            "whatsapp_interactive": _role_choice_message(body),
        }

    async def _creator_response(self, state: AgentState) -> dict[str, Any]:
        """Placeholder for the Job Creator lane. The recruiter experience isn't
        built yet, so acknowledge and point them at the team — the routing is in
        place so building it out later is purely additive."""
        first = _first_name((state.get("customer_facts") or {}).get("full_name"))
        who = f", {first}" if first else ""
        body = (
            f"Thanks for your interest in hiring through Jobs7{who}! 🙌\n\n"
            "The job-poster experience is coming soon. For now, reply here and our "
            "team will help you post your roles and reach candidates."
        )
        return {
            "intent": "role_creator",
            "draft_response": body,
            "used_llm": False,
            "single_bubble": True,
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
        draft = state.get("draft_response", "") or ""
        if state.get("single_bubble"):
            # A deterministic listing (e.g. all jobs in a category) must arrive
            # whole — never chunked/truncated — so send it as one bubble.
            typing = int(min(s.agent_typing_max_ms,
                             max(s.agent_typing_min_ms, len(draft) / max(s.agent_typing_cps, 1.0) * 1000)))
            plan_ = [{"text": draft, "typing_ms": typing}]
        else:
            plan_ = build_delivery_plan(
                draft,
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
        if not jobs:
            for r in (state.get("working") or {}).get("tool_results") or []:
                if r.get("tool") == "search_jobs" and isinstance(r.get("result"), list):
                    jobs = r["result"]
                    break
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
    ) -> Literal["onboarding", "greeting", "role_select", "creator", "agent"]:
        """Routing gate, keyed on whether the number is already ours.

        * KNOWN number (in the job-board DB, or already onboarded) → a greeting
          opens the Job Seeker / Job Creator choice; picking one runs that flow.
        * NEW number (not in our DB) → straight to onboarding (ask name → form);
          the lane choice is only offered AFTER they've registered — that's the
          one-time post-form success turn (``just_onboarded``), which carries the
          lane buttons.
        """
        # First turn after the form is submitted → one-time success + lane choice.
        if state.get("just_onboarded"):
            return "onboarding"

        q = state.get("inbound_text", "")
        if _CLOSER_RX.match(q):
            return "greeting"

        # New numbers must register first — no lane choice until they're known.
        if not state.get("is_known", False):
            return "onboarding"

        # Lane is REMEMBERED: the tap this turn wins, otherwise the stored choice
        # from customer_facts. Once a lane is known we never re-ask.
        picked = _role_selection(state)
        lane = picked or (state.get("customer_facts") or {}).get("lane")
        if lane == "creator":
            return "creator"                       # recruiter flow, every turn
        if lane == "seeker":
            # Show the seeker hub (Job Search / Application Status / Recommended)
            # on the selection tap or on a greeting; otherwise reason normally.
            return "greeting" if (picked or _GREETING_RX.match(q)) else "agent"
        # No lane chosen yet → ask once on a greeting; otherwise proceed.
        if _GREETING_RX.match(q):
            return "role_select"
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
        g.add_node("role_select", self._role_select)
        g.add_node("creator_response", self._creator_response)
        g.add_node("menu", self._menu)
        g.add_node("browse", self._browse)
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
                "role_select": "role_select",
                "creator": "creator_response",
                "agent": "menu",
            },
        )
        # A handled menu tap (category list / recommendations) short-circuits to
        # delivery; anything else falls through to the category browse.
        g.add_conditional_edges(
            "menu", self._route_after_menu,
            {"humanize": "humanize", "browse": "browse"},
        )
        # A category browse short-circuits to delivery; anything else reasons on.
        g.add_conditional_edges(
            "browse", self._route_after_browse,
            {"humanize": "humanize", "summarize": "summarize"},
        )
        g.add_edge("onboarding_response", "humanize")
        g.add_edge("greeting_response", "humanize")
        g.add_edge("role_select", "humanize")
        g.add_edge("creator_response", "humanize")
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
        interactive_id: str | None = None,
        attachment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tid = tenant_id or get_current_tenant_id()
        conv_id = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        initial: AgentState = {
            "tenant_id": tid,
            "conversation_id": conv_id,
            "customer_id": customer_external_id or "",
            "inbound_text": customer_query,
            "inbound_kind": "document" if attachment else ("button" if interactive_id else "text"),
            # structured id of a tapped row/button (job:<ref>, loc:<x>, apply:<ref>…)
            "button_id": interactive_id,
            # an uploaded file (e.g. a resume document) when present
            "attachment": attachment,
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
            # interactive payload (e.g. onboarding cta_url button) — route sends
            # this instead of the text bubbles when present
            "whatsapp_interactive": final.get("whatsapp_interactive"),
            # a sequence of interactive cards (one per matching job) — route sends
            # each in turn; takes priority over whatsapp_interactive + bubbles
            "whatsapp_messages": final.get("whatsapp_messages"),
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
