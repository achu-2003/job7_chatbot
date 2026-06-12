"""responder node — write the final, natural WhatsApp reply.

Grounded in memory + the gathered tool results, validated against the same
``HallucinationValidator`` the rest of the system uses. On an LLM failure it
degrades to a warm "try again" rather than crashing the turn.
"""
from __future__ import annotations

import re
from typing import Any

from app.agent.browse import build_job_list_text
from app.agent.context import toon_context
from app.agent.prompts import PROMPT_VERSIONS, RESPONDER_SYSTEM
from app.agent.state import AgentState
from app.chatbot import wa_format as wa
from app.chatbot.validator import HallucinationValidator
from app.core import conversation_log as conv
from app.core.logging import get_logger
from app.core.metrics import AGENT_PROMPT_CALLS, HALLUCINATION_COUNTER

_PRICE_IN_DOC = re.compile(r"₹\s*([\d,]+(?:\.\d+)?)")
_SAFE_FALLBACK = (
    "Let me double-check that to be sure — could you tell me the role or job "
    "reference (JOB-XXXX) you mean?"
)

log = get_logger("agent_responder")

_BUSY_REPLY = (
    "I'm handling a lot of messages right now and couldn't get to yours — "
    "please send it again in a few seconds."
)

# Strict regeneration prompt. The first draft tripped the grounding validator
# (usually a fabricated JOB-XXXX reference or a salary the model invented), so
# rather than discard the whole reply we ask the model once to rewrite it using
# only verifiable facts. Matches the documented "one strict regenerate, then
# fall back" strategy (README §7) that the agent responder was missing.
_REGEN_INSTRUCTION = (
    "Your last reply included details I can't verify against the CONTEXT. "
    "Rewrite it using ONLY facts shown in CONTEXT/MEMORY. Do NOT include any job "
    "code or reference (never write 'JOB-1234' or similar), and do NOT state any "
    "salary or number that isn't present in CONTEXT. In particular, do NOT state "
    "any count of jobs/roles/openings (e.g. 'we have 69 open jobs') unless CONTEXT "
    "has an 'open_jobs_total' — if it doesn't, tell them to tap 'Job Search' to see "
    "what's open. If CONTEXT lists no matching roles, simply say you couldn't find "
    "any right now. Max 3 short lines, no emojis."
)


def _clean(text: str | None) -> str:
    """Strip + default an empty draft, then cap length (no wall of text, ever)."""
    text = (text or "").strip() or (
        "I couldn't find an answer for that — could you give me a bit more detail?"
    )
    return _shorten(text)


async def respond(
    state: AgentState,
    *,
    llm: Any,
    validator: HallucinationValidator,
    memory_context: str | None,
) -> dict[str, Any]:
    results = (state.get("working") or {}).get("tool_results") or []

    # Deterministic short-circuit for application status: the candidate is
    # already identified by their number, so when the lookup returned data we
    # format the reply ourselves rather than letting a weak model wander off and
    # ask for name/email (it does so inconsistently). Grounded by construction.
    direct = _application_status_reply(results)
    if direct is not None:
        conv.note("respond", f"app-status {len(direct)} chars")
        # single_bubble: the whole status list must arrive as ONE bubble — never
        # chunked at the small per-bubble char cap (which split it mid-card).
        return {"draft_response": direct, "used_llm": False, "single_bubble": True}

    # Apply confirmation: the candidate is identified and confirmed a real role,
    # so submit_application returned the Jobs7 app link. Format it ourselves (the
    # weak model otherwise asks for name/email or claims "no matching roles") and
    # attach a tappable "Open in Jobs7" cta button. Grounded by construction.
    apply = _apply_link_reply(results, state)
    if apply is not None:
        conv.note("respond", "apply-link (Jobs7 app)")
        return apply

    # Category browse: the candidate asked to see a whole category, so list every
    # job deterministically. The LLM responder caps at ~3 lines and can't, so we
    # format it ourselves and flag single_bubble so delivery doesn't truncate it.
    listing = _job_listing_reply(results)
    if listing is not None:
        conv.note("respond", f"job-listing {len(listing)} chars")
        return {"draft_response": listing, "used_llm": False, "single_bubble": True}

    parts: list[str] = []
    if memory_context:
        parts.append(f"MEMORY:\n{memory_context}")
    ctx = toon_context(results)   # flattened → TOON table form (token-efficient)
    if ctx:
        parts.append("CONTEXT:\n" + ctx[:800])
    parts.append(f"CUSTOMER MESSAGE: {state['inbound_text']}\n\nWrite the reply.")

    messages: list[dict[str, Any]] = [{"role": "system", "content": RESPONDER_SYSTEM}]
    for t in (state.get("short_term") or [])[-3:]:
        if t.get("role") in {"user", "assistant"} and t.get("content"):
            messages.append({"role": t["role"], "content": t["content"]})
    messages.append({"role": "user", "content": "\n\n".join(parts)})

    AGENT_PROMPT_CALLS.labels(prompt="responder", version=PROMPT_VERSIONS["responder"]).inc()
    try:
        text, _ = await llm.chat(
            purpose="agent_respond", messages=messages, temperature=0.3, max_tokens=90,
        )
    except Exception as exc:  # noqa: BLE001 — never 500 the turn on an LLM error
        log.warning("responder_llm_unavailable", error=str(exc)[:200])
        return {"draft_response": _BUSY_REPLY, "used_llm": True}

    text = _clean(text)   # deterministic guard: no wall of text, ever

    # Grounding = tool results + the pinned product (so a legit follow-up like
    # "it's ₹999" passes). If the model quoted a salary/reference not in any of
    # them, it fabricated → try ONE strict regenerate before giving up, so a
    # single stray token can't discard an otherwise-correct job listing.
    grounding = _grounding_rows(results) + _cached_grounding(state)
    text, result_label = await _validate_or_regenerate(
        text, state=state, llm=llm, validator=validator,
        messages=messages, grounding=grounding,
        allowed_counts=_allowed_job_counts(results),
    )
    HALLUCINATION_COUNTER.labels(result=result_label).inc()
    conv.note("respond", f"{len(text)} chars ({result_label})")
    return {"draft_response": text, "used_llm": True}


async def _validate_or_regenerate(
    text: str,
    *,
    state: AgentState,
    llm: Any,
    validator: HallucinationValidator,
    messages: list[dict[str, Any]],
    grounding: list[dict[str, Any]],
    allowed_counts: set[int] | None = None,
) -> tuple[str, str]:
    """Validate the draft; on a grounding failure, regenerate once under a strict
    instruction and re-validate. Returns ``(reply, metric_label)`` where the
    label is ``valid`` / ``regenerated`` / ``blocked``."""
    query = state.get("inbound_text")
    verdict = validator.validate(
        text, sql_rows=grounding, vector_hits=[], customer_query=query,
        allowed_counts=allowed_counts,
    )
    if verdict.valid:
        return text, "valid"

    log.warning("agent_response_blocked", offending=verdict.offending, draft=text[:160])
    try:
        retry, _ = await llm.chat(
            purpose="agent_respond_strict",
            messages=messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": _REGEN_INSTRUCTION},
            ],
            temperature=0.0, max_tokens=90,
        )
    except Exception as exc:  # noqa: BLE001 — regenerate is best-effort
        log.warning("responder_regen_unavailable", error=str(exc)[:200])
        return _SAFE_FALLBACK, "blocked"

    retry = _clean(retry)
    verdict = validator.validate(
        retry, sql_rows=grounding, vector_hits=[], customer_query=query,
        allowed_counts=allowed_counts,
    )
    if verdict.valid:
        return retry, "regenerated"

    log.warning("agent_response_blocked_after_regen", offending=verdict.offending, draft=retry[:160])
    return _SAFE_FALLBACK, "blocked"


def _application_status_reply(results: list[dict[str, Any]]) -> str | None:
    """If this turn looked up the candidate's applications, build the reply
    deterministically: list each "Job — Status", or a clear "no applications"
    line. Returns None when there's no application-status result, so normal
    (job search / chat) turns fall through to the LLM responder."""
    app_res = next(
        (r.get("result") for r in results if r.get("tool") == "get_application_status"),
        None,
    )
    if not isinstance(app_res, dict):
        return None

    apps = app_res.get("applications")
    if not isinstance(apps, list):
        one = app_res.get("application")
        apps = [one] if isinstance(one, dict) else []

    if not apps:
        return "You don't have any applications on record yet. Want me to find you some jobs?"

    # A blank line ("") becomes a "\n\n" paragraph break that the humanizer strips
    # + re-merges away, so apps end up packed together. A Hangul-filler line renders
    # as a blank line WhatsApp keeps — giving a clear gap between applications.
    # Lines start at the margin (no indent) and are kept short so they don't wrap.
    gap = "ㅤ"
    lines = [f"*📋 Your Applications ({len(apps)})*"]
    for a in apps[:5]:
        title = a.get("job_title") or a.get("job_ref") or "a role"
        status_raw = str(a.get("status") or "PENDING")
        status = status_raw.replace("_", " ").title()
        lines.append(gap)                                        # spacer before each app
        lines.append(f"{_status_badge(status_raw)} *{title}* — {status}")
        loc = " · ".join(x for x in (a.get("company"), a.get("location")) if x)
        if loc:
            lines.append(f"🏢 {loc}")
        salary = _app_salary(a)
        if salary:
            lines.append(f"💰 {salary}")
        applied = _fmt_app_date(a.get("created_at"))
        if applied:
            lines.append(f"📅 Applied {applied}")
        note = _status_note(status_raw)                          # only for progress statuses
        if note:
            lines.append(note)
    return "\n".join(lines)


def _status_badge(status: str) -> str:
    return {
        "PENDING": "⏳", "VIEWED": "👀", "SHORTLISTED": "⭐",
        "INTERVIEW_SCHEDULED": "📅", "INTERVIEWED": "🎤",
        "SELECTED": "✅", "APPROVED": "✅", "REJECTED": "❌",
        "ACTIVE": "🟢", "INACTIVE": "⚪",
    }.get((status or "").upper(), "•")


def _status_note(status: str) -> str | None:
    """A short note only for PROGRESS statuses — Pending/Viewed need none (the
    badge + label already say it), keeping those cards compact."""
    return {
        "SHORTLISTED": "⭐ You've been shortlisted!",
        "INTERVIEW_SCHEDULED": "📅 Interview scheduled",
        "INTERVIEWED": "🎤 Interview done",
        "SELECTED": "🎉 You've been selected!",
        "REJECTED": "Not selected this time",
    }.get((status or "").upper())


def _app_salary(a: dict[str, Any]) -> str | None:
    lo, hi = a.get("salary_min"), a.get("salary_max")
    period = (a.get("salary_period") or "MONTHLY").lower()
    per = {"monthly": "/month", "yearly": "/year", "daily": "/day", "hourly": "/hr"}.get(period, "")

    def _fmt(n: Any) -> str | None:
        try:
            return f"₹{float(n):,.0f}"
        except (TypeError, ValueError):
            return None

    lo_s, hi_s = _fmt(lo), _fmt(hi)
    if lo_s and hi_s:
        return f"{lo_s}–{hi_s}{per}" if lo_s != hi_s else f"{lo_s}{per}"
    if lo_s:
        return f"{lo_s}+{per}"
    return None


def _fmt_app_date(value: Any) -> str | None:
    """'12 Jun' from a datetime / ISO string / date — None if unparseable."""
    if not value:
        return None
    try:
        from datetime import datetime
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00")[:19])
        return value.strftime("%d %b")
    except (ValueError, TypeError):
        return None


def _first_name(name: str | None) -> str:
    return name.split()[0] if name else ""


def _apply_link_reply(
    results: list[dict[str, Any]], state: AgentState
) -> dict[str, Any] | None:
    """When the candidate confirmed applying to a real role, ``submit_application``
    returns the Jobs7 app link. Build the hand-over reply deterministically: a
    grounded text bubble (link inline, used as the web/fallback) plus a tappable
    ``cta_url`` "Open in Jobs7" button for WhatsApp. Returns None when this turn
    wasn't an apply confirmation, so other turns fall through to the LLM."""
    res = next(
        (r.get("result") for r in results if r.get("tool") == "submit_application"),
        None,
    )
    if not isinstance(res, dict) or not res.get("apply_via_app"):
        return None

    title = res.get("job_title") or "this role"
    url = res.get("app_url") or ""
    name = _first_name((state.get("customer_facts") or {}).get("full_name"))
    who = f", {name}" if name else ""
    out: dict[str, Any] = {
        "draft_response": (
            f"You're all set to apply for the {title} role{who}! Finish in the "
            f"Jobs7 app — we've got your details ready to carry over:\n{url}"
        ),
        "used_llm": False,
        "single_bubble": True,
    }
    # Tappable button only for an https link (Meta rejects http/localhost). The
    # text above keeps the link inline as the web reply + WhatsApp fallback.
    if url.startswith("https://"):
        out["whatsapp_interactive"] = wa.cta_url_message(
            body=(
                f"You're all set to apply for the {title} role{who}! Tap below to "
                "open the Jobs7 app and finish — we've got your details ready."
            ),
            display_text="Open in Jobs7",
            url=url,
        )
    return out


def _job_listing_reply(results: list[dict[str, Any]]) -> str | None:
    """When a category browse returned a sizable set of jobs, list them ALL as a
    compact, grounded reply. Returns None for ordinary top-k searches (<= 5 hits),
    which keep their natural LLM-written reply. Bullets (not '1.' numbering) so the
    bubble splitter doesn't mistake the numbers for sentence boundaries."""
    jobs = next(
        (r.get("result") for r in results
         if r.get("tool") == "search_jobs" and isinstance(r.get("result"), list)),
        None,
    )
    if not jobs or len(jobs) <= 5:
        return None

    dept = next((j.get("department") for j in jobs if j.get("department")), None)
    return build_job_list_text(jobs, dept)


def _allowed_job_counts(results: list[dict[str, Any]]) -> set[int]:
    """The job counts the responder is allowed to state this turn — derived from
    the actual tool results. A list result grounds its length (e.g. 3 search
    hits → "3 roles"); an overview result grounds its total + per-category counts.
    Anything else the model says (a count from memory or the prompt example) is a
    fabrication the validator will reject."""
    counts: set[int] = set()
    for r in results:
        res = r.get("result")
        if isinstance(res, list):
            counts.add(len(res))
        elif isinstance(res, dict):
            for key in ("total_open_jobs", "total", "count"):
                if isinstance(res.get(key), int):
                    counts.add(res[key])
            for c in res.get("categories") or []:
                if isinstance(c, dict) and isinstance(c.get("count"), int):
                    counts.add(c["count"])
            for v in res.values():            # nested lists (e.g. applications)
                if isinstance(v, list):
                    counts.add(len(v))
    return counts


def _cached_grounding(state: AgentState) -> list[dict[str, Any]]:
    """Treat the pinned product's price as grounded, so follow-up answers about
    it ("Pink, ₹999") aren't falsely flagged."""
    doc = (state.get("cached_product") or {}).get("doc") or ""
    rows: list[dict[str, Any]] = []
    for m in _PRICE_IN_DOC.finditer(doc):
        try:
            rows.append({"price": float(m.group(1).replace(",", ""))})
        except ValueError:
            continue
    return rows


def _shorten(text: str, limit: int = 300) -> str:
    """Cap the reply so it can't become a paragraph, even if the model rambles.
    Trims back to the last sentence end / line break before ``limit``; product
    bullet lists (a few short lines) pass through untouched."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    best = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind("\n"))
    if best > limit * 0.5:
        return cut[: best + 1].strip()
    return cut.rstrip() + "…"


def _grounding_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pull authoritative rows out of tool results for the grounding check —
    maps the compact product shape onto the keys the validator understands."""
    rows: list[dict[str, Any]] = []
    for r in results:
        res = r.get("result")
        tool = r.get("tool")
        if tool == "search_products" and isinstance(res, list):
            for p in res:
                if isinstance(p, dict):
                    rows.append({
                        "title": p.get("title"),
                        "price": p.get("price"),
                        "suggested_mrp": p.get("mrp"),
                    })
        elif tool == "search_jobs" and isinstance(res, list):
            # The compact job shape already carries job_ref / title / salary_min /
            # salary_max — exactly the keys the validator grounds against — so the
            # rows pass through as-is. Without this branch every JOB-XXXX the model
            # quotes is flagged unsupported and the reply gets replaced.
            for j in res:
                if isinstance(j, dict):
                    r_copy = dict(j)
                    if r_copy.get("salary_min") is not None:
                        r_copy["price"] = r_copy["salary_min"]
                    if r_copy.get("salary_max") is not None:
                        r_copy["suggested_mrp"] = r_copy["salary_max"]
                    rows.append(r_copy)
        elif tool == "get_order_status" and isinstance(res, dict) and isinstance(res.get("order"), dict):
            rows.append(res["order"])
        elif tool == "get_recent_orders" and isinstance(res, list):
            rows.extend(o for o in res if isinstance(o, dict))
    return rows
