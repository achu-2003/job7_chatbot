"""First-party MCP tools for the job-application assistant.

Each capability is declared **once** here as a :class:`ToolSpec` (name,
description, JSON-Schema for its arguments, and an async handler). Both the
MCP server (``app.mcp.server``) and the in-process agent loop build on these
specs, so the bot and any external MCP client expose an identical surface.

Design rules
------------
* **Catalog is read-only; candidate records are scoped writes.** Job postings
  are read-only (``JobRepository``). The agent CAN create a candidate's own
  application (``submit_application``) — a safe, idempotent write keyed to the
  authenticated phone, mirroring the agent-memory write pattern.
* **Tenant + identity are injected, never model-controlled.** The agent passes
  ``tenant_id`` and the candidate's phone through :class:`ToolContext` at
  dispatch time. The model can only ever see/modify the *calling* candidate's
  applications — it cannot read or apply on behalf of someone else, because the
  application tools don't expose ``candidate_ref`` in their schema at all.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Awaitable, Callable

from app.chatbot.escalation import EscalationService, EscalationTicket
from app.config import get_settings
from app.core.logging import get_logger
from app.db.repositories import (
    ApplicationRepository,
    CandidateRepository,
    JobRepository,
)
from app.memory.repositories import PendingActionRepository
from app.vector.store import VectorStore

log = get_logger("mcp_tools")


# ---------------------------------------------------------------------------
# context + spec types
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Per-turn identity the model is NOT allowed to set. Injected at dispatch.

    ``customer_external_id`` is the WhatsApp sender's phone (digits only).
    Application tools scope every query/write to it, so the model can never read
    or touch another candidate's applications. ``session_id`` lets action tools
    (e.g. follow-up scheduling) attach durable rows to the right conversation.
    """

    tenant_id: str
    customer_external_id: str | None = None
    request_id: str | None = None
    session_id: str | None = None


# A handler receives the model-supplied ``arguments`` (already JSON-parsed) and
# the injected :class:`ToolContext`; it returns any JSON-able value.
Handler = Callable[[dict[str, Any], ToolContext], Awaitable[Any]]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON-Schema object for the arguments
    handler: Handler


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _phone_variants(raw: str) -> list[str]:
    """[raw, stripped-CC] so we match stored candidates whether or not the
    sender's number carried a +91 prefix."""
    raw = raw.strip().lstrip("+")
    out = [raw]
    if len(raw) == 12 and raw.startswith("91"):
        out.append(raw[2:])
    elif len(raw) == 11 and raw.startswith("0"):
        out.append(raw[1:])
    return out


def _arg_query(args: dict[str, Any]) -> str:
    """The search text from a tool call, tolerant of a weak planner.

    The schema names it ``query``, but small models sometimes emit ``q`` /
    ``search`` / ``text`` instead, or omit it entirely. Return the first
    non-empty alias (stringified, trimmed) rather than KeyError the turn — an
    empty string is a harmless "no results" search, not a crash.
    """
    for key in ("query", "q", "search", "text"):
        v = args.get(key)
        if v not in (None, ""):
            return str(v).strip()
    return ""


def _jsonable(value: Any) -> Any:
    """Make repository rows JSON-serialisable (Decimal/datetime → primitives)."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):  # date / datetime
        return value.isoformat()
    return value


# Map the raw job status to a candidate-friendly availability label. Only the
# explicitly-open statuses are "open"; everything else is surfaced so the reply
# can flag it (e.g. "(expired)") and the candidate isn't misled into applying.
# An UNKNOWN status defaults to "unavailable" (NOT "open") — failing safe, so a
# new status value the schema adds later never silently looks applyable.
_OPEN_STATUSES = {"LIVE", "APPROVED", "PENDING"}
_AVAILABILITY = {
    "EXPIRED": "expired", "CLOSED": "closed", "PAUSED": "paused",
    "SUSPENDED": "unavailable", "REJECTED": "unavailable", "DRAFT": "unavailable",
}


def _availability(status: str) -> str:
    if status in _OPEN_STATUSES:
        return "open"
    return _AVAILABILITY.get(status, "unavailable")


def _compact_job(row: dict[str, Any]) -> dict[str, Any]:
    """The fields the model needs to answer, nothing more (keeps tokens low).
    ``id`` is included so the caller can re-hydrate; ``job_ref`` is the
    human-facing id the candidate uses to apply. ``availability`` lets the model
    flag jobs that aren't open (expired/closed) so it never tells a candidate to
    apply to a dead listing."""
    status = (row.get("status") or "").upper()
    return {
        "id": str(row["id"]) if row.get("id") is not None else None,
        "job_ref": row.get("job_ref"),
        "title": row.get("title"),
        "department": row.get("department_name"),
        "location": row.get("location"),
        "employment_type": row.get("employment_type"),
        "seniority": row.get("seniority"),
        "salary_min": _jsonable(row.get("salary_min")),
        "salary_max": _jsonable(row.get("salary_max")),
        "salary_currency": row.get("salary_currency"),
        "skills": row.get("skills") or [],
        "availability": _availability(status),
    }


def _matches_facets(
    row: dict[str, Any],
    *,
    location: str | None,
    employment_type: str | None,
    min_salary: float | None,
    max_salary: float | None,
) -> bool:
    """Post-filter a hydrated job row by the model's structured facets, done in
    Python on the already-hydrated row (which carries location/type/salary)."""
    if location:
        if location.lower() not in (row.get("location") or "").lower():
            return False
    if employment_type:
        if employment_type.lower().replace("-", "_").replace(" ", "_") != (
            row.get("employment_type") or ""
        ).lower():
            return False
    smin, smax = row.get("salary_min"), row.get("salary_max")
    if min_salary is not None and smax is not None and float(smax) < min_salary:
        return False
    if max_salary is not None and smin is not None and float(smin) > max_salary:
        return False
    return True


# ---------------------------------------------------------------------------
# core capabilities (shared by the registry and the standalone server)
# ---------------------------------------------------------------------------


async def search_jobs_core(
    vector: VectorStore,
    *,
    tenant_id: str,
    query: str,
    location: str | None = None,
    employment_type: str | None = None,
    min_salary: float | None = None,
    max_salary: float | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Semantic job search (hybrid vectors) hydrated from Postgres.

    Vector search for recall, then ``get_by_ids`` for authoritative
    title/location/salary, then optional facet post-filtering. Transactional
    truth (salary band, status) always comes from SQL, never vectors.
    """
    s = get_settings()
    hits = await vector.query(
        s.vector_collection_products, query, tenant_id=tenant_id,
    )
    # Prefer the explicit metadata.job_id, but fall back to the hit's own id —
    # which IS the job id (reindex stores each job under id=row["id"]). Requiring
    # metadata.job_id silently dropped every hit when that key wasn't present,
    # making search return nothing even though the vectors matched.
    job_ids = [
        (h.get("metadata") or {}).get("job_id") or h.get("id")
        for h in hits
        if (h.get("metadata") or {}).get("job_id") or h.get("id")
    ]
    if not job_ids:
        return []
    rows = await JobRepository.get_by_ids(job_ids, tenant_id=tenant_id)
    filtered = [
        r for r in rows
        if _matches_facets(
            r, location=location, employment_type=employment_type,
            min_salary=min_salary, max_salary=max_salary,
        )
    ]
    return [_compact_job(r) for r in filtered[:limit]]


async def list_jobs_overview_core(*, tenant_id: str) -> dict[str, Any]:
    """A count + category menu of open jobs — the answer to "list all jobs" so
    the candidate can pick a category instead of getting a wall of postings."""
    summary = await JobRepository.category_summary(tenant_id=tenant_id)
    # keep the top categories (by count) to keep the reply short
    cats = [
        {"category": c["category"], "count": c["n"]}
        for c in summary["categories"][:8]
    ]
    return {"total_open_jobs": summary["total"], "categories": cats}


async def submit_application_core(
    *,
    tenant_id: str,
    phone: str | None,
    job_ref: str,
    full_name: str | None = None,
    email: str | None = None,
    years_experience: float | None = None,  # noqa: ARG001 — kept for when real apply is re-enabled
    cover_note: str | None = None,           # noqa: ARG001 — kept for when real apply is re-enabled
) -> dict[str, Any]:
    """Acknowledge an apply request without writing to the DB.

    Submitting via WhatsApp would INSERT into the production ``users`` +
    ``applications`` tables of the live job board (jobs7uat), which needs a
    proper design (new-user role/status, dedup, notifications) we haven't built.
    Until then we don't write: we confirm the role is real (so the reply is
    grounded and specific) and point the candidate to apply on the portal.

    Still validates phone + name/email so the conversational flow that gathers
    them is unchanged for when real submission is enabled.
    """
    if not phone:
        return {"error": "no candidate identity on this channel"}
    if not full_name or not email:
        return {
            "error": "missing_details",
            "need": [f for f, v in (("full_name", full_name), ("email", email)) if not v],
            "message": "I still need your name and email before I can submit.",
        }
    job = await JobRepository.get_by_ref(job_ref, tenant_id=tenant_id)
    if not job:
        return {"error": "job_not_found", "job_ref": job_ref}

    return {
        "submitted": False,
        "apply_unavailable": True,
        "job_ref": job["job_ref"],
        "job_title": job["title"],
        "message": (
            f"I found the {job['title']} role for you. Applying straight from "
            "chat isn't available yet — I can connect you with our team to apply."
        ),
    }


async def get_application_status_core(
    *,
    tenant_id: str,
    phone: str,
    application_ref: str | None = None,
) -> dict[str, Any]:
    """Look up the calling candidate's application(s).

    With ``application_ref`` → that one application (only if it's theirs).
    Without → their most recent applications. Returns ``{"found": False}``
    rather than raising on a miss.
    """
    candidate = None
    for ref in _phone_variants(phone):
        candidate = await CandidateRepository.get(tenant_id=tenant_id, phone=ref)
        if candidate:
            break
    if not candidate:
        return {"found": False, "reason": "no_candidate_record"}

    cid = str(candidate["id"])
    if application_ref:
        row = await ApplicationRepository.get_by_ref(
            application_ref, tenant_id=tenant_id, candidate_id=cid,
        )
        if row:
            return {"found": True, "application": _jsonable(row)}
        return {"found": False, "application_ref": application_ref.upper()}

    rows = await ApplicationRepository.latest_for_candidate(cid, tenant_id=tenant_id)
    return {"found": bool(rows), "applications": [_jsonable(r) for r in rows]}


async def search_docs_core(
    vector: VectorStore,
    *,
    tenant_id: str,
    collection: str,
    query: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    hits = await vector.query(collection, query, tenant_id=tenant_id, top_k=limit)
    return [
        {
            "title": (h.get("metadata") or {}).get("title"),
            "text": (h.get("document") or "")[:600],
        }
        for h in hits
    ]


async def schedule_followup_core(
    *,
    tenant_id: str,
    session_id: str | None,
    customer_id: str | None,
    minutes: int,
    message: str,
) -> dict[str, Any]:
    """Queue a proactive follow-up to this candidate after ``minutes``. Used
    when the agent promises to check back (e.g. "I'll let you know when there's
    an update on your application")."""
    if not session_id:
        return {"error": "no session to attach the follow-up to"}
    run_after = datetime.now(timezone.utc) + timedelta(minutes=max(1, int(minutes)))
    await PendingActionRepository.add(
        session_id=session_id,
        tenant_id=tenant_id,
        kind="followup",
        payload={"message": message, "customer_id": customer_id},
        run_after=run_after,
    )
    return {"scheduled": True, "in_minutes": int(minutes)}


async def request_human_handoff_core(
    escalation: EscalationService,
    *,
    reason: str,
    summary: str,
    context: dict[str, Any] | None = None,
    severity: str = "normal",
) -> dict[str, Any]:
    """Hand the conversation to a human recruiter. The safe route for anything
    the agent shouldn't decide — withdrawals, complaints, hiring outcomes,
    special accommodations."""
    ticket = EscalationTicket(
        reason=reason, customer_message=summary, severity=severity,
    )
    await escalation.dispatch(ticket, context or {})
    return {"handed_off": True, "message": ticket.suggested_response}


# ---------------------------------------------------------------------------
# registry — wraps the core functions as model-callable tools (agent side)
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Builds the candidate-facing tool set for the in-process agent loop.

    The application tools deliberately omit any candidate identifier from their
    JSON schema: the agent injects the *authenticated sender's* phone from
    :class:`ToolContext`, so the model is structurally unable to read or modify
    another candidate's applications.
    """

    def __init__(
        self,
        *,
        vector: VectorStore,
        escalation: EscalationService | None = None,
    ) -> None:
        self.vector = vector
        self.escalation = escalation or EscalationService()
        self._specs: dict[str, ToolSpec] = {s.name: s for s in self._build_specs()}

    # -- public API ----------------------------------------------------

    def openai_schemas(self) -> list[dict[str, Any]]:
        """Tools formatted for the OpenAI chat-completions ``tools`` param."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._specs.values()
        ]

    def names(self) -> list[str]:
        return list(self._specs)

    async def dispatch(
        self, name: str, arguments: dict[str, Any], ctx: ToolContext
    ) -> Any:
        """Run a tool by name. Never raises — tool failures come back as an
        ``{"error": ...}`` payload the model can read and recover from."""
        spec = self._specs.get(name)
        if spec is None:
            return {"error": f"unknown tool: {name}"}
        start = time.perf_counter()
        try:
            result = await spec.handler(arguments or {}, ctx)
            log.info(
                "tool_call",
                tool=name,
                request_id=ctx.request_id,
                args=arguments,
                ms=int((time.perf_counter() - start) * 1000),
            )
            return result
        except Exception as exc:  # noqa: BLE001 — surface as data, not a 500
            log.warning("tool_error", tool=name, error=str(exc))
            return {"error": "tool execution failed", "detail": str(exc)[:200]}

    # -- spec construction --------------------------------------------

    def _build_specs(self) -> list[ToolSpec]:
        async def _search_jobs(args: dict[str, Any], ctx: ToolContext) -> Any:
            return await search_jobs_core(
                self.vector,
                tenant_id=ctx.tenant_id,
                # A weak planner sometimes omits `query` or names it q/search/text
                # — accept any of those rather than KeyError the whole turn.
                query=_arg_query(args),
                location=args.get("location"),
                employment_type=args.get("employment_type"),
                min_salary=args.get("min_salary"),
                max_salary=args.get("max_salary"),
                limit=int(args.get("limit") or 5),
            )

        async def _list_jobs_overview(args: dict[str, Any], ctx: ToolContext) -> Any:
            return await list_jobs_overview_core(tenant_id=ctx.tenant_id)

        async def _submit_application(args: dict[str, Any], ctx: ToolContext) -> Any:
            return await submit_application_core(
                tenant_id=ctx.tenant_id,
                phone=ctx.customer_external_id,  # injected, not from model
                job_ref=args["job_ref"],
                full_name=args.get("full_name"),
                email=args.get("email"),
                years_experience=args.get("years_experience"),
                cover_note=args.get("cover_note"),
            )

        async def _get_application_status(args: dict[str, Any], ctx: ToolContext) -> Any:
            if not ctx.customer_external_id:
                return {"error": "no candidate identity on this channel"}
            return await get_application_status_core(
                tenant_id=ctx.tenant_id,
                phone=ctx.customer_external_id,  # injected, not from model
                application_ref=args.get("application_ref"),
            )

        async def _search_policies(args: dict[str, Any], ctx: ToolContext) -> Any:
            return await search_docs_core(
                self.vector,
                tenant_id=ctx.tenant_id,
                collection=get_settings().vector_collection_policies,
                query=_arg_query(args),
            )

        async def _search_faq(args: dict[str, Any], ctx: ToolContext) -> Any:
            return await search_docs_core(
                self.vector,
                tenant_id=ctx.tenant_id,
                collection=get_settings().vector_collection_faq,
                query=_arg_query(args),
            )

        async def _handoff(args: dict[str, Any], ctx: ToolContext) -> Any:
            # `reason`/`summary` are schema-required, but a weak model sometimes
            # calls the tool with neither — don't KeyError the handoff (the one
            # path that exists to rescue a stuck turn). Fall back to the summary,
            # then a generic reason, so the escalation still goes through.
            reason = args.get("reason") or args.get("summary") or "unspecified"
            return await request_human_handoff_core(
                self.escalation,
                reason=reason,
                summary=args.get("summary") or reason,
                severity=args.get("severity", "normal"),
                context={
                    "tenant_id": ctx.tenant_id,
                    "candidate": ctx.customer_external_id,
                    "request_id": ctx.request_id,
                },
            )

        async def _schedule_followup(args: dict[str, Any], ctx: ToolContext) -> Any:
            return await schedule_followup_core(
                tenant_id=ctx.tenant_id,
                session_id=ctx.session_id,
                customer_id=ctx.customer_external_id,
                minutes=args.get("minutes", 60),
                message=args.get("message", ""),
            )

        return [
            ToolSpec(
                name="search_jobs",
                description=(
                    "Search open job postings matching a natural-language query "
                    "(e.g. 'remote backend engineering roles'). Returns the job "
                    "reference (JOB-XXXX), title, department, location, employment "
                    "type, seniority, salary range and skills. Salary and status "
                    "are authoritative — quote them verbatim. Optionally filter by "
                    "location, employment_type or salary."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What the candidate is looking for.",
                        },
                        "location": {"type": "string", "description": "Filter to this location."},
                        "employment_type": {
                            "type": "string",
                            "enum": ["full_time", "part_time", "contract", "intern"],
                        },
                        "min_salary": {"type": "number"},
                        "max_salary": {"type": "number"},
                        "limit": {
                            "type": "integer",
                            "description": "Max jobs to return (default 5).",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=_search_jobs,
            ),
            ToolSpec(
                name="list_jobs_overview",
                description=(
                    "Use when the candidate asks to LIST or SEE ALL jobs / what "
                    "jobs are available, without naming a role. Returns the total "
                    "number of open jobs and a breakdown by category, so you can "
                    "give a count and invite them to pick a category — instead of "
                    "dumping every posting. Prefer search_jobs when they name a "
                    "specific role/skill/location."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=_list_jobs_overview,
            ),
            ToolSpec(
                name="submit_application",
                description=(
                    "Submit the current candidate's application to a job, by its "
                    "JOB-XXXX reference. Requires the candidate's full name and "
                    "email (ask for these first if you don't have them). "
                    "years_experience and cover_note are optional. Idempotent — "
                    "re-submitting the same role returns the existing application. "
                    "Returns the new application reference (APP-XXXX)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "job_ref": {
                            "type": "string",
                            "description": "The job reference to apply to, e.g. JOB-AB1234.",
                        },
                        "full_name": {"type": "string", "description": "Candidate's full name."},
                        "email": {"type": "string", "description": "Candidate's email address."},
                        "years_experience": {
                            "type": "number",
                            "description": "Years of relevant experience.",
                        },
                        "cover_note": {
                            "type": "string",
                            "description": "Optional short note on why they're a fit.",
                        },
                    },
                    "required": ["job_ref", "full_name", "email"],
                    "additionalProperties": False,
                },
                handler=_submit_application,
            ),
            ToolSpec(
                name="get_application_status",
                description=(
                    "Check the current candidate's application(s). Give an "
                    "application_ref (APP-XXXX) to look up one specific application, "
                    "or omit it to list their most recent applications. Only ever "
                    "returns this candidate's own applications."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "application_ref": {
                            "type": "string",
                            "description": "Optional application reference, e.g. APP-CD5678.",
                        },
                    },
                    "additionalProperties": False,
                },
                handler=_get_application_status,
            ),
            ToolSpec(
                name="search_policies",
                description=(
                    "Search hiring policy documents (eligibility, equal opportunity, "
                    "privacy, interview process, withdrawal). Use to answer policy "
                    "questions accurately."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=_search_policies,
            ),
            ToolSpec(
                name="search_faq",
                description="Search frequently-asked-question documents about applying and hiring.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=_search_faq,
            ),
            ToolSpec(
                name="request_human_handoff",
                description=(
                    "Hand the conversation to a human recruiter. Use this for "
                    "anything you cannot or should not do yourself — WITHDRAWING an "
                    "application, complaints, questions about a hiring decision, "
                    "accommodation requests, or anything the other tools can't "
                    "resolve. Summarise what the candidate needs."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Short reason, e.g. 'withdraw application', 'complaint'.",
                        },
                        "summary": {
                            "type": "string",
                            "description": "One-line summary of what the candidate wants.",
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["normal", "high"],
                        },
                    },
                    "required": ["reason", "summary"],
                    "additionalProperties": False,
                },
                handler=_handoff,
            ),
            ToolSpec(
                name="schedule_followup",
                description=(
                    "Schedule a proactive follow-up message to THIS candidate after "
                    "a delay. Use when you promise to check back later (e.g. 'I'll "
                    "let you know when there's an update'). Sent automatically when due."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "minutes": {
                            "type": "integer",
                            "description": "Delay before sending, in minutes.",
                        },
                        "message": {
                            "type": "string",
                            "description": "The follow-up message to send the candidate.",
                        },
                    },
                    "required": ["minutes", "message"],
                    "additionalProperties": False,
                },
                handler=_schedule_followup,
            ),
        ]
