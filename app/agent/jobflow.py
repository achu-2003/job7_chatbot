"""Interactive job-browse flow — pure builders for the tappable WhatsApp UI.

The browse flow is a small state machine driven by structured interactive ids
(carried on every list row / button so the selection survives WhatsApp's 24-char
title truncation):

    category:<name>     a category was tapped  → show its roles (paged list)
    more:<name>:<off>   "More roles" tapped     → next page of the role list
    job:<ref>           a role was tapped       → show all its job cards
    apply|save|share:<ref>   a card action      → record/confirm

Everything here is pure (data in, payload/text out) so it's trivially testable;
the Redis state + DB lookups live in the ``_browse`` node on the runtime.
"""
from __future__ import annotations

import re
from typing import Any

from app.chatbot import wa_format as wa

# WhatsApp interactive list: max 10 rows. When a page overflows we keep 9 roles
# and use the 10th row as a "More roles" pager.
_PAGE = 10

_NUM_EMOJI = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]

# A location string that means "no fixed office" → render as Work From Home.
_WFH_RX = re.compile(
    r"\b(remote|work\s*from\s*home|wfh|anywhere|fully\s*remote|home\s*based)\b",
    re.IGNORECASE,
)
_EMPLOYMENT_LABEL = {
    "full_time": "Full-time", "part_time": "Part-time", "contract": "Contract",
    "internship": "Internship", "intern": "Internship", "freelance": "Freelance",
}


def _trunc(s: str | None, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: max(0, n - 1)].rstrip() + "…"


def infer_wfh(location: str | None) -> bool:
    """True when the location text reads as remote/work-from-home."""
    return bool(location and _WFH_RX.search(location))


def _amount(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def salary_to_lpa(salary_min: Any, salary_max: Any) -> str | None:
    """Format an ANNUAL salary band (rupees) as 'lakhs per annum', e.g.
    1_000_000 → '10 LPA', 1_500_000–3_000_000 → '15–30 LPA'. Returns None when
    no usable figure is present."""
    def fmt(x: float) -> str:
        return f"{x / 100_000:.1f}".rstrip("0").rstrip(".")

    lo, hi = _amount(salary_min), _amount(salary_max)
    if lo and hi and round(lo) != round(hi):
        return f"{fmt(lo)}–{fmt(hi)} LPA"
    v = hi or lo
    return f"{fmt(v)} LPA" if v else None


def salary_display(salary_min: Any, salary_max: Any) -> str | None:
    """Human salary string that adapts to the figure's scale: values under ₹1L
    are a MONTHLY figure (the live job board stores monthly pay), shown as
    '₹20,000–50,000/month'; larger values are annual, shown in LPA. Returns None
    when there's no usable figure."""
    lo, hi = _amount(salary_min), _amount(salary_max)
    top = hi or lo
    if top is None:
        return None
    if top < 100_000:  # below ₹1L → a monthly figure, not annual
        if lo and hi and round(lo) != round(hi):
            return f"₹{lo:,.0f}–{hi:,.0f}/month"
        return f"₹{top:,.0f}/month"
    return salary_to_lpa(salary_min, salary_max)


# Generic role words carry no role identity — stripped before deciding whether
# two titles are the "same role" (so "Frontend Developer" ≠ "Backend Developer",
# but "Python Developer" == "Senior Python Developer").
_GENERIC_ROLE = {
    "developer", "engineer", "manager", "executive", "officer", "analyst",
    "specialist", "lead", "senior", "junior", "sr", "jr", "associate",
    "consultant", "intern", "trainee", "head", "assistant", "coordinator",
    "representative", "agent", "staff", "support",
}


def _role_keywords(title: str | None) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", (title or "").lower())
    return {t for t in toks if len(t) >= 2 and t not in _GENERIC_ROLE}


def same_role_family(a: str | None, b: str | None) -> bool:
    """True when two titles name the same kind of role (share a distinctive,
    non-generic keyword). Falls back to a full-title match when neither has a
    distinctive word (e.g. a bare 'Developer')."""
    ka, kb = _role_keywords(a), _role_keywords(b)
    if not ka or not kb:
        return (a or "").strip().lower() == (b or "").strip().lower()
    return bool(ka & kb)


def _ref(job: dict[str, Any]) -> str:
    return str(job.get("job_ref") or job.get("id") or "")


def job_ref(job: dict[str, Any]) -> str:
    """Public accessor for a job's reference (slug) used in action ids."""
    return _ref(job)


_WORK_MODE_LABEL = {
    "OFFICE": "Office", "WORK_FROM_HOME": "Work From Home", "WFH": "Work From Home",
    "HYBRID": "Hybrid", "FIELD": "Field work", "REMOTE": "Remote",
}
_QUALIFICATION_LABEL = {
    "BELOW_10TH": "Below 10th", "10TH_ABOVE": "10th & above",
    "12TH_ABOVE": "12th & above", "DIPLOMA_ABOVE": "Diploma & above",
    "DEGREE_ABOVE": "Degree & above", "GRADUATE": "Graduate", "ANY": "Any qualification",
}
# WhatsApp interactive body cap is 1024 chars; keep headroom for safety.
_CARD_MAX = 1000


def _experience_label(jmin: Any, jmax: Any) -> str | None:
    lo = _int_or_none(jmin)
    hi = _int_or_none(jmax)
    if lo is None and hi is None:
        return None
    if (lo in (None, 0)) and (hi in (None, 0)):
        return "Any experience"
    if lo and hi:
        return f"{lo}–{hi} yrs experience"
    if hi:
        return f"Up to {hi} yrs experience"
    return f"{lo}+ yrs experience"


def _age_label(amin: Any, amax: Any) -> str | None:
    lo, hi = _int_or_none(amin), _int_or_none(amax)
    if lo and hi:
        return f"Age {lo}–{hi} yrs"
    if hi:
        return f"Age up to {hi} yrs"
    if lo:
        return f"Age {lo}+ yrs"
    return None


def _int_or_none(v: Any) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _clean_description(desc: str | None) -> str:
    """Collapse blank-line runs but keep the description's line structure."""
    if not desc:
        return ""
    out: list[str] = []
    for ln in str(desc).splitlines():
        ln = ln.strip()
        if not ln and (not out or not out[-1]):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def job_card_text(job: dict[str, Any], idx: int | None = None) -> str:
    """The detailed emoji card for one job — title, location/work-mode,
    employment type, experience, qualification, English, age, salary, vacancies,
    reference, and a (space-permitting) description excerpt. Capped to the
    WhatsApp interactive body limit; the structured lines always survive and the
    description fills whatever room is left."""
    title = job.get("title") or "Role"
    lines = [f"{_num(idx)} {title}".strip() if idx else title]

    loc = job.get("location")
    wm = _WORK_MODE_LABEL.get(str(job.get("work_mode") or "").upper(), "")
    if wm == "Work From Home" or (not wm and infer_wfh(loc)):
        lines.append("🏠 Work From Home")
    elif loc or wm:
        lines.append("📍 " + " · ".join(x for x in (loc, wm) if x))

    etype = (job.get("employment_type") or "").lower()
    if etype:
        lines.append(f"💼 {_EMPLOYMENT_LABEL.get(etype, etype.replace('_', ' ').title())}")
    exp = _experience_label(job.get("experience_min"), job.get("experience_max"))
    if exp:
        lines.append(f"🧰 {exp}")
    qual = _QUALIFICATION_LABEL.get(str(job.get("qualification_level") or "").upper()) or (
        ", ".join(str(q) for q in (job.get("qualifications") or [])) or None
    )
    if qual:
        lines.append(f"🎓 {qual}")
    eng = job.get("english_level")
    if eng:
        lines.append(f"🗣️ English: {str(eng).title()}")
    age = _age_label(job.get("age_min"), job.get("age_max"))
    if age:
        lines.append(f"🎂 {age}")
    pay = salary_display(job.get("salary_min"), job.get("salary_max"))
    if pay:
        lines.append(f"💰 {pay}")
    vac = _int_or_none(job.get("vacancies"))
    if vac:
        lines.append(f"👥 {vac} vacanc{'y' if vac == 1 else 'ies'}")
    ref = _ref(job)
    if ref:
        lines.append(f"🆔 {ref}")

    card = "\n".join(lines)
    desc = _clean_description(job.get("description"))
    if desc:
        budget = _CARD_MAX - len(card) - 6
        if budget > 60:
            card += "\n\n📋 " + _trunc(desc, budget)
    return card


def _num(idx: int | None) -> str:
    if not idx:
        return ""
    return _NUM_EMOJI[idx - 1] if 1 <= idx <= len(_NUM_EMOJI) else f"{idx}."


def _role_row_desc(job: dict[str, Any]) -> str:
    bits: list[str] = []
    loc = job.get("location")
    bits.append("Work From Home" if infer_wfh(loc) else (loc or "—"))
    pay = salary_display(job.get("salary_min"), job.get("salary_max"))
    if pay:
        bits.append(pay)
    return _trunc(" · ".join(bits), 72)


def role_list_message(
    jobs: list[dict[str, Any]], *, category: str, offset: int = 0
) -> tuple[dict[str, Any], str]:
    """A tappable list of roles in a category, paged 10 at a time. Returns the
    interactive payload + a text fallback (web / cta-rejected)."""
    total = len(jobs)
    has_more = total > offset + _PAGE
    take = (_PAGE - 1) if has_more else _PAGE
    window = jobs[offset:offset + take]

    rows = [
        {
            "id": f"job:{_ref(j)}",
            "title": _trunc(j.get("title") or "Role", 24),
            "description": _role_row_desc(j),
        }
        for j in window
    ]
    if has_more:
        nxt = offset + len(window)
        rows.append({
            "id": f"more:{category}:{nxt}",
            "title": "More roles ▸",
            "description": f"{total - nxt} more",
        })

    shown_to = offset + len(window)
    body = (
        f"{total} {category} roles. Tap one to see the openings"
        + (f" (showing {offset + 1}–{shown_to})" if total > _PAGE else "")
        + ":"
    )
    payload = wa.list_message(
        body=body, button_text="View roles", rows=rows,
        header=_trunc(category, 60), section_title="Roles",
    )
    text_lines = [body, ""] + [
        f"• {j.get('title') or 'Role'}" + (f" — {j['location']}" if j.get("location") else "")
        for j in window
    ]
    if has_more:
        text_lines.append(f"\n…and {total - shown_to} more. Reply 'more' to see them.")
    return payload, "\n".join(text_lines)


def recommend_list_message(
    jobs: list[dict[str, Any]], *, name: str | None = None
) -> tuple[dict[str, Any], str]:
    """A tappable list of recommended roles. Each row's id is ``view:<ref>`` so a
    tap shows THAT specific job's card (not the whole role family). Returns the
    interactive payload + a text fallback."""
    who = f", {name}" if name else ""
    body = f"Based on your profile{who}, here are roles you might like — tap one for details:"
    rows = [
        {
            "id": f"view:{_ref(j)}",
            "title": _trunc(j.get("title") or "Role", 24),
            "description": _role_row_desc(j),
        }
        for j in jobs[:10]
    ]
    payload = wa.list_message(
        body=body, button_text="View roles", rows=rows,
        header="Recommended for you", section_title="Roles",
    )
    text_lines = [body, ""] + [
        f"• {j.get('title') or 'Role'}" + (f" — {j['location']}" if j.get("location") else "")
        for j in jobs[:10]
    ]
    return payload, "\n".join(text_lines)


def job_cards(jobs: list[dict[str, Any]], *, limit: int = 8) -> tuple[list[dict[str, Any]], str]:
    """Per-job interactive cards (emoji body + Apply / Save / Share buttons) plus
    one combined text fallback listing the same cards. Capped at ``limit`` so we
    never spam a long thread."""
    shown = jobs[:limit]
    messages = [
        wa.buttons_message(
            job_card_text(j, idx=i),
            (
                (f"apply:{_ref(j)}", "Apply"),
                (f"save:{_ref(j)}", "Save"),
                (f"share:{_ref(j)}", "Share"),
            ),
        )
        for i, j in enumerate(shown, 1)
    ]
    text = "\n\n".join(job_card_text(j, idx=i) for i, j in enumerate(shown, 1))
    if len(jobs) > limit:
        text += f"\n\n…and {len(jobs) - limit} more matching roles."
    return messages, text


def share_text(job: dict[str, Any]) -> str:
    """A self-contained, forwardable description of a job (for the Share action)."""
    return "Check out this role 👇\n\n" + job_card_text(job)


def split_id(button_id: str | None) -> tuple[str, str]:
    """Split a structured interactive id into (prefix, rest). 'job:abc' →
    ('job', 'abc'); a bare/absent id → ('', '')."""
    if not button_id or ":" not in button_id:
        return "", (button_id or "")
    prefix, rest = button_id.split(":", 1)
    return prefix, rest
