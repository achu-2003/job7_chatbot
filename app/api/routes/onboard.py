"""Self-hosted onboarding form for new WhatsApp candidates.

A new number is asked for their name in chat, then handed a tokenised link to
this form (email, years of experience, preferred role/location). The submission
is stored in **Redis only** (never the business DB) and acts as the gate that
lets the agent start helping with jobs.

    GET  /onboard/form?token=...   → the HTML form (token resolves to the
                                     candidate; 404 if unknown/expired)
    POST /onboard/submit           → validate + store in Redis, show a thank-you

Form bodies are parsed manually (urlencoded) so we don't depend on
``python-multipart``.
"""
from __future__ import annotations

import asyncio
import html
import json
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from app.api.deps import get_memory
from app.chatbot import wa_format as wa
from app.config import get_settings
from app.core.logging import get_logger
from app.db.repositories import (
    ApplicationRepository,
    CandidateRepository,
    JobSeekerRepository,
    LookupRepository,
)
from app.onboarding import build_application_record, prepare_registration
from app.validation import MAX_RESUME_BYTES, valid_resume_filename, validate_registration
from app.whatsapp import delivery as wa_delivery

# The /onboard form is the SEEKER lane (the employer side has its own form), so
# after a submission we push the seeker hub directly — the conversation just
# continues with Job Search / Application Status / Recommended Jobs. Ids must
# match app.agent.runtime._MENU_BUTTONS.
_SEEKER_HUB_BUTTONS = (
    ("menu_search", "Job Search"),
    ("menu_status", "Application Status"),
    ("menu_recommend", "Recommended Jobs"),
)

router = APIRouter()
log = get_logger("onboard")

# Uploaded resumes are stored here and served read-only at /uploads (mounted in
# app.main). Local disk is fine for testing; point this at cloud storage later.
_RESUME_DIR = Path("uploads/resumes")
_RESUME_EXTS = {".pdf", ".doc", ".docx", ".rtf", ".odt", ".png", ".jpg", ".jpeg"}


async def _register_with_retry(payload: dict[str, Any], *, tries: int = 3) -> dict[str, Any]:
    """Write the registration, retrying briefly on transient connection saturation
    ('too many clients already') from the shared cluster. Re-raises other errors."""
    last: Exception | None = None
    for attempt in range(tries):
        try:
            return await JobSeekerRepository.create(payload, commit=True)
        except Exception as exc:  # noqa: BLE001
            last = exc
            transient = "too many clients" in str(exc).lower()
            if transient and attempt < tries - 1:
                log.warning("registration_db_retry", attempt=attempt + 1, error=str(exc)[:120])
                await asyncio.sleep(0.6 * (attempt + 1))
                continue
            raise
    assert last is not None
    raise last


async def _write_application_live(
    tenant_id: str, phone: str, job: dict[str, Any], built: dict[str, Any]
) -> None:
    """Persist an application (incl. the uploaded resume) to the live
    private_job_applications. Resolves the real jobSeekerId/profileId by phone
    (the staged registration ids may not be the live rows). Idempotent + best-
    effort — an error never fails the resume upload."""
    app = built.get("application") or {}
    try:
        ids = await CandidateRepository.application_ids(phone=phone)
        if not (ids or {}).get("jobSeekerId"):
            log.info("apply_resume_db_skipped_no_live_seeker", phone=phone[-10:])
            return
        app["jobId"] = job["id"]
        app["jobSeekerId"] = ids["jobSeekerId"]
        app["profileId"] = ids.get("profileId")
        res = await ApplicationRepository.create(built, commit=True)
        log.info("apply_resume_db_written", inserted=res["inserted"],
                 application_id=res["application_id"], has_resume=bool(app.get("resume")))
    except Exception as exc:  # noqa: BLE001 — never fail the upload on a DB error
        log.error("apply_resume_db_write_failed", error=str(exc)[:300])


async def _save_resume_upload(upload: Any, token: str) -> str:
    """Save a selected resume file to local storage and return its served URL.
    Returns "" when no file was attached. Best-effort — a failure never blocks
    the submission (the candidate can still register without a resume)."""
    filename = getattr(upload, "filename", None)
    if not filename:                       # str field / None → no file selected
        return ""
    if not valid_resume_filename(filename):   # reject wrong file types (not a CV)
        log.info("resume_upload_bad_type", file=str(filename)[:60])
        return ""
    try:
        data = await upload.read()
        if not data:
            return ""
        if len(data) > MAX_RESUME_BYTES:      # reject oversized uploads (> 5 MB)
            log.info("resume_upload_too_large", bytes=len(data))
            return ""
        clean = re.sub(r"[^A-Za-z0-9._-]", "_", filename)[-50:].lstrip("._") or "resume"
        fname = f"{uuid.uuid4().hex[:10]}_{clean}"
        _RESUME_DIR.mkdir(parents=True, exist_ok=True)
        (_RESUME_DIR / fname).write_bytes(data)
        base = get_settings().public_base_url.rstrip("/")
        log.info("resume_uploaded", file=fname, bytes=len(data))
        return f"{base}/uploads/resumes/{fname}"
    except Exception as exc:  # noqa: BLE001 — upload is best-effort
        log.warning("resume_upload_failed", error=str(exc)[:200])
        return ""

# Dropdown sources fetched to render the form (id-valued options).
_OPTION_KINDS = (
    "states", "districts", "education_levels", "courses", "specializations",
    "experience_levels", "skills", "roles", "categories", "other_states",
    "languages",
)

# Fixed enum options (stored as plain text / array columns on the seeker/profile).
_I_AM_A = (("STUDENT", "Student"), ("FRESHER", "Fresher - First Job"), ("EXPERIENCED", "Experienced"))
_GENDER = (("MALE", "Male"), ("FEMALE", "Female"), ("OTHER", "Other"))
_MARITAL = (("SINGLE", "Single"), ("MARRIED", "Married"))
_ENGLISH = (("BASIC", "Basic"), ("INTERMEDIATE", "Intermediate"), ("FLUENT", "Fluent"))
_JOB_TYPES = (
    ("FULL_TIME", "Full-time"), ("PART_TIME", "Part-time"), ("CONTRACT", "Contract"),
    ("INTERNSHIP", "Internship"), ("FREELANCE", "Freelance"), ("TEMPORARY", "Temporary"),
    ("WORK_FROM_HOME", "Work From Home"), ("WALK_IN", "Walk-In"),
)
_WORK_MODE = (("ON_SITE", "On-site"), ("REMOTE", "Remote"), ("HYBRID", "Hybrid"))
_YESNO = (("yes", "Yes"), ("no", "No"))
# value = the lower bound (₹/month), stored as expected_salary.
_SALARY = (
    ("5000", "₹5,000 - ₹10,000"), ("10000", "₹10,000 - ₹15,000"),
    ("15000", "₹15,000 - ₹20,000"), ("20000", "₹20,000 - ₹25,000"),
    ("25000", "₹25,000 - ₹35,000"), ("35000", "₹35,000 - ₹50,000"),
    ("50000", "₹50,000 - ₹75,000"), ("75000", "₹75,000 - ₹1,00,000"),
    ("100000", "₹1,00,000+"),
)


async def _load_options() -> dict[str, list[dict[str, Any]]]:
    """Fetch every dropdown's options (read-only reference data).

    Loaded SEQUENTIALLY, not concurrently: firing all lookups at once opened a
    Postgres connection per query and tripped 'sorry, too many clients already',
    so some lists silently came back empty (skills/categories/roles/specs blank).
    One connection at a time is plenty fast for a form render and never starves
    the pool.
    """
    return {k: await LookupRepository.options(k) for k in _OPTION_KINDS}


@router.get("/form", response_class=HTMLResponse)
async def onboarding_form(request: Request, token: str = Query(default="")) -> HTMLResponse:
    identity = await get_memory(request).get_onboarding_identity(token) if token else None
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    opts = await _load_options()
    return HTMLResponse(
        _form_html(token, identity.get("name") or "", identity.get("customer_id") or "", opts)
    )


@router.post("/submit", response_class=HTMLResponse)
async def onboarding_submit(request: Request) -> HTMLResponse:
    # The form now carries a file input (resume), so it's multipart/form-data —
    # parse via request.form() (handles both text fields and the upload).
    posted = await request.form()

    def one(key: str) -> str:
        v = posted.get(key)
        return v.strip() if isinstance(v, str) else ""

    def many(key: str) -> list[str]:
        return [x.strip() for x in posted.getlist(key) if isinstance(x, str) and x.strip()]

    token = one("token")
    if not token:
        return HTMLResponse(_expired_html(), status_code=400)

    # Resume: a selected file is saved to local storage and its served URL stored;
    # a pasted link (resume_url) is honored as a fallback when no file is attached.
    resume = await _save_resume_upload(posted.get("resume"), token) or one("resume_url")

    name = one("full_name") or one("name")
    # Languages: each can be marked Speak and/or Write (checkboxes by language id).
    speak_ids, write_ids = set(many("lang_speak")), set(many("lang_write"))
    languages = [
        {
            "languageId": lid,
            "speak": "FLUENT" if lid in speak_ids else None,
            "write": "FLUENT" if lid in write_ids else None,
        }
        for lid in (speak_ids | write_ids)
    ]

    # The wizard's "Skip for now" lets any step be skipped, so the server accepts
    # partial data — the client enforces the per-step required (*) fields on Next.
    form = {
        # name kept under both keys (identify reads "name"; staging reads "full_name")
        "name": name,
        "full_name": name,
        "email": one("email"),
        "current_status": one("current_status"),
        "gender": one("gender"),
        "marital_status": one("marital_status"),
        "date_of_birth": one("date_of_birth"),
        "state_id": one("state_id"),
        "district_id": one("district_id"),
        "city": one("city"),
        "education_level_id": one("education_level_id"),
        "course_id": one("course_id"),
        "specialization_id": one("specialization_id"),
        "institution": one("institution"),
        "year_of_passing": one("year_of_passing"),
        "experience_level_id": one("experience_level_id"),
        "current_salary": one("current_salary"),
        "expected_salary": one("expected_salary"),
        "resume": resume,
        "work_mode": one("work_mode"),
        "job_types": many("job_types"),
        "interested_in_abroad": one("interested_in_abroad"),
        "skill_ids": many("skill_ids"),
        "preferred_category_ids": many("preferred_category_ids"),
        "preferred_role_ids": many("preferred_role_ids"),
        "preferred_location_ids": many("preferred_location_ids"),
        "other_state_ids": many("other_state_ids"),
        "languages": languages,
    }
    # Back-compat text fields the recommendation menu reads (first selection).
    first_role = (form["preferred_role_ids"] or [None])[0]
    form["preferred_role"] = await LookupRepository.name_for("roles", first_role) or ""
    first_loc = (form["preferred_location_ids"] or [None])[0]
    form["location"] = await LookupRepository.name_for("districts", first_loc) or ""

    memory = get_memory(request)
    identity = await memory.get_onboarding_identity(token)
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)

    # SERVER-SIDE VALIDATION (authoritative backstop): reject bad/incomplete data
    # so nothing malformed is ever staged or written. The form mirrors these rules
    # in JS, so this normally only triggers if the client check is bypassed —
    # re-show the form with the first error.
    errors = validate_registration(form)
    if errors:
        opts = await _load_options()
        phone = re.sub(r"\D", "", identity.get("customer_id") or "")
        return HTMLResponse(
            _form_html(token, name or identity.get("name") or "", phone, opts,
                       error=next(iter(errors.values()))),
            status_code=400,
        )

    identity = await memory.save_onboarding(token, form)
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)

    # Build the DB-ready registration payload (private_job_seekers +
    # job_seeker_profiles + child rows) and STAGE it in Redis (always — it's the
    # safe backup / inspection copy).
    payload: dict[str, Any] | None = None
    try:
        payload = await prepare_registration(identity=identity, form=form)
        await memory.save_registration(
            identity["conversation_id"], payload, tenant_id=identity["tenant_id"]
        )
    except Exception as exc:  # noqa: BLE001 — staging must never fail the submission
        log.warning("registration_stage_failed", error=str(exc)[:200])

    # Write to the LIVE job board only when explicitly enabled (REGISTER_IN_DB).
    # A DB failure is logged but never fails the form — the Redis copy still holds
    # (and the seeker is treated as registered via the Redis gate either way).
    #
    # IDEMPOTENCY: the form can be slow, so candidates double-tap Submit. Two
    # guards stop duplicate private_job_seekers rows: (1) an ATOMIC NX lock keyed
    # by phone so only the FIRST of N concurrent/duplicate submits writes, and
    # (2) a "already on the job board?" check so a re-registration is a no-op.
    if payload is not None and get_settings().register_in_db:
        phone_key = re.sub(r"\D", "", identity.get("customer_id") or "")
        lock = f"onboard_db:{phone_key or identity['conversation_id']}"
        if await memory.mark_seen(lock, ttl=120):
            try:
                existing = await CandidateRepository.get(
                    tenant_id=identity["tenant_id"], phone=identity.get("customer_id") or ""
                )
            except Exception as exc:  # noqa: BLE001 — a blip must not block the write
                log.warning("registration_dup_check_failed", error=str(exc)[:200])
                existing = None
            if existing:
                log.info("registration_db_already_exists_skipped", phone=phone_key)
            else:
                try:
                    res = await _register_with_retry(payload)
                    log.info("registration_db_written",
                             seeker_id=res["seeker_id"], children=res["children"])
                except Exception as exc:  # noqa: BLE001
                    log.error("registration_db_write_failed", error=str(exc)[:300])
                    # write failed → release the lock so a genuine retry can write
                    await memory.clear_seen(lock)
        else:
            log.info("registration_db_duplicate_submit_skipped", phone=phone_key)

    settings = get_settings()
    # PROACTIVELY push the "registration successful" message + the seeker hub to
    # WhatsApp, so it appears the moment they return to the chat — no typing
    # needed. They reached this form via the Job Seeker lane, so we continue as a
    # seeker (search / status / recommendations) rather than re-asking the lane.
    # Mark onboarding welcomed so the bot doesn't also send its own one-time
    # success on the next message.
    phone = re.sub(r"\D", "", identity.get("customer_id") or "")
    if phone:
        first = (identity.get("name") or "").split()[0] if identity.get("name") else ""
        who = f", {first}" if first else ""
        body = (
            f"🎉 Registration successful{who}!\n\nYou're all set. Here's what I can "
            "help you with — just tap an option below."
        )
        try:
            await wa_delivery.send_message(settings, phone, wa.buttons_message(body, _SEEKER_HUB_BUTTONS))
            await memory.mark_onboarding_welcomed(
                identity["conversation_id"], tenant_id=identity["tenant_id"]
            )
        except Exception as exc:  # noqa: BLE001 — proactive push is best-effort
            log.warning("onboard_push_failed", error=str(exc)[:200])

    number = re.sub(r"\D", "", settings.whatsapp_business_number or "")
    return HTMLResponse(_success_html(identity.get("name") or "", business_number=number))


# ---------------------------------------------------------------------------
# Apply-time resume upload (the friendly "Upload Resume" web button)
# ---------------------------------------------------------------------------


@router.get("/resume", response_class=HTMLResponse)
async def resume_form(request: Request, token: str = Query(default="")) -> HTMLResponse:
    identity = await get_memory(request).get_apply_identity(token) if token else None
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    return HTMLResponse(_resume_html(token))


@router.post("/resume/submit", response_class=HTMLResponse)
async def resume_submit(request: Request) -> HTMLResponse:
    posted = await request.form()
    tok = posted.get("token")
    token = tok.strip() if isinstance(tok, str) else ""
    if not token:
        return HTMLResponse(_expired_html(), status_code=400)
    memory = get_memory(request)
    identity = await memory.get_apply_identity(token)
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    tid, conv = identity["tenant_id"], identity["conversation_id"]

    resume_url = await _save_resume_upload(posted.get("resume"), token)
    if not resume_url:
        return HTMLResponse(
            _resume_html(token, error="Please upload a valid PDF/DOC file (under 5 MB)."),
            status_code=400,
        )

    apply_state = await memory.get_apply_state(conv, tenant_id=tid)
    reg = await memory.get_registration(conv, tenant_id=tid)
    settings = get_settings()
    # Finalize the application if there's an active apply with a staged job; else
    # just save the resume onto the staged profile.
    if apply_state and reg and apply_state.get("job"):
        job = apply_state["job"]
        ref = apply_state.get("job_ref")
        title = apply_state.get("job_title") or "the role"
        try:
            built = build_application_record(
                registration=reg, job=job, answers={"resume": resume_url}
            )
            await memory.save_application(conv, ref, built["application"], tenant_id=tid)
            if built["profile_update"]:
                await memory.update_registration_profile(conv, built["profile_update"], tenant_id=tid)
            await memory.clear_apply_state(conv, tenant_id=tid)
        except Exception as exc:  # noqa: BLE001 — staging must not 500 the upload
            log.warning("apply_resume_finalize_failed", error=str(exc)[:200])
            built = None
        # LIVE write (flag-gated): persist the application + uploaded resume to
        # private_job_applications, with the real live jobSeekerId/profileId.
        if built and job.get("id") and get_settings().register_in_db:
            await _write_application_live(tid, re.sub(r"\D", "", conv), job, built)
        push_body = (
            f"✅ Resume received — you've applied to {title}! Our team will review "
            "your profile and get back to you."
        )
        sub = f"Your resume is attached and you've applied to {title}."
    else:
        if reg:
            try:
                await memory.update_registration_profile(conv, {"resume": resume_url}, tenant_id=tid)
            except Exception as exc:  # noqa: BLE001
                log.warning("apply_resume_profile_save_failed", error=str(exc)[:200])
        push_body = "📎 Resume uploaded and saved to your profile."
        sub = "Your resume has been saved to your profile."

    phone = re.sub(r"\D", "", conv)          # conv id is wa_<number>
    if phone:
        try:
            await wa_delivery.send_message(settings, phone, wa.text_message(push_body))
        except Exception as exc:  # noqa: BLE001 — push is best-effort
            log.warning("apply_resume_push_failed", error=str(exc)[:200])

    number = re.sub(r"\D", "", settings.whatsapp_business_number or "")
    return HTMLResponse(_resume_done_html(sub, business_number=number))


# ---------------------------------------------------------------------------
# HTML (inline; no template engine dependency)
# ---------------------------------------------------------------------------

_PAGE = """\
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Complete your profile</title>
<style>
  body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;background:#0b141a;color:#e9edef;
       margin:0;display:flex;min-height:100vh;align-items:flex-start;justify-content:center;padding:18px 0}}
  .card{{background:#111b21;max-width:460px;width:92%;padding:24px 22px;border-radius:14px;
         box-shadow:0 10px 30px rgba(0,0,0,.4)}}
  h1{{font-size:20px;margin:0 0 4px}} p.sub{{color:#8696a0;margin:0 0 16px;font-size:14px}}
  fieldset{{border:1px solid #2a3942;border-radius:10px;margin:0 0 16px;padding:6px 14px 16px}}
  legend{{padding:0 8px;font-size:13px;font-weight:600;color:#00d3a7;text-transform:uppercase;letter-spacing:.04em}}
  label{{display:block;font-size:13px;color:#aebac1;margin:14px 0 6px}}
  label .req{{color:#ff8a8a}}
  input,select{{width:100%;box-sizing:border-box;padding:11px 12px;border-radius:8px;border:1px solid #2a3942;
         background:#202c33;color:#e9edef;font-size:15px}}
  select[multiple]{{height:auto;min-height:104px}}
  .hint{{font-size:12px;color:#6b7d88;margin:6px 0 0}}
  button,a.btn{{display:block;width:100%;box-sizing:border-box;margin-top:8px;padding:12px;border:0;
          border-radius:8px;background:#00a884;color:#fff;font-size:16px;font-weight:600;cursor:pointer;
          text-align:center;text-decoration:none}}
  .err{{background:#3a1d1d;color:#ffb4b4;padding:10px 12px;border-radius:8px;font-size:13px;margin-bottom:12px}}
  .ok{{text-align:center}} .ok .tick{{font-size:44px}}
  .ts{{position:relative}}
  .ts-box{{display:flex;flex-wrap:wrap;gap:6px;min-height:44px;padding:7px 8px;border:1px solid #2a3942;
           border-radius:8px;background:#202c33;align-items:center;cursor:text}}
  .ts-chip{{display:inline-flex;align-items:center;gap:7px;background:#005c4b;color:#e9edef;border-radius:14px;
            padding:3px 6px 3px 11px;font-size:13px}}
  .ts-chip b{{cursor:pointer;font-weight:700;opacity:.85;font-size:15px;line-height:1}}
  .ts-input{{flex:1;min-width:90px;border:0;background:transparent;color:#e9edef;font-size:15px;outline:none;padding:4px 2px}}
  .ts-menu{{position:absolute;left:0;right:0;top:calc(100% + 4px);z-index:30;background:#0b141a;border:1px solid #2a3942;
            border-radius:8px;max-height:220px;overflow:auto;box-shadow:0 8px 24px rgba(0,0,0,.5)}}
  .ts-opt{{padding:9px 12px;font-size:14px;cursor:pointer}}
  .ts-opt.active,.ts-opt:hover{{background:#202c33}}
  .ts-empty{{padding:9px 12px;color:#6b7d88;font-size:13px}}
  .prog{{display:flex;gap:12px;margin:0 0 22px;flex-wrap:wrap;align-items:center}}
  .prog span{{width:28px;height:28px;border-radius:50%;background:#202c33;color:#8696a0;display:flex;
              align-items:center;justify-content:center;font-size:13px;font-weight:700;border:1px solid #2a3942;
              transition:background .15s}}
  .prog span.on{{background:#00a884;color:#fff;border-color:#00a884}}
  .step{{display:none}} .step.on{{display:block}}
  .chips{{display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 4px}}
  .chip{{padding:9px 14px;border:1px solid #2a3942;border-radius:20px;background:#202c33;color:#e9edef;
         font-size:14px;cursor:pointer;user-select:none}}
  .chip.on{{background:#005c4b;border-color:#00a884;color:#fff}}
  .chips.invalid,.ts.invalid .ts-box,.dob.invalid select,select.invalid,input.invalid{{border-color:#ff6b6b}}
  .nav{{display:flex;gap:10px;align-items:center;margin-top:24px}}
  .nav button{{margin-top:0}}
  .nav .back{{background:#2a3942;color:#e9edef;width:auto;flex:none;padding:12px 16px}}
  .nav .skip{{background:transparent;color:#8696a0;width:auto;flex:none;padding:12px 4px;font-weight:500}}
  .nav .next{{flex:1}}
  .dob{{display:flex;gap:8px}}
  .lang{{border:1px solid #2a3942;border-radius:10px;padding:10px 12px;margin:8px 0}}
  .lang .nm{{font-weight:600;margin-bottom:8px}}
  .lang label{{display:inline-flex;align-items:center;gap:6px;margin:0 16px 0 0;color:#cfd9de;font-size:14px}}
  .lang input{{width:auto}}
  .ro{{background:#161f25;color:#8696a0}}
</style></head><body><div class="card">{body}</div></body></html>"""


def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _options_html(items: "list[dict[str, Any]] | tuple", *, placeholder: str | None = None) -> str:
    """``<option>`` tags. ``items`` is either id/name dicts (value=id) or
    (value, label) tuples (fixed enums). ``placeholder`` adds an empty first row."""
    out: list[str] = []
    if placeholder is not None:
        out.append(f'<option value="">{_esc(placeholder)}</option>')
    for it in items:
        val, lab = (it["id"], it["name"]) if isinstance(it, dict) else (it[0], it[1])
        out.append(f'<option value="{_esc(val)}">{_esc(lab)}</option>')
    return "".join(out)


def _js_rows(items: list[dict[str, Any]]) -> str:
    """``[[id, name, parentId], …]`` as a JS-safe JSON literal — feeds both the
    cascading selects and the type-to-search token pickers."""
    rows = [[it["id"], it["name"], it.get("parent")] for it in items]
    return json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")


# Inline, dependency-free form behaviour: (1) cascade — a parent <select> fills
# its child (state→district, course→specialization); (2) tokenSelect — a
# type-to-search multi picker that renders chips + hidden inputs (skills,
# locations, categories, roles). Raw string so Python leaves the JS untouched.
_FORM_JS = r"""
function _esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function fillSelect(sel, rows, ph){
  if(!sel) return;
  sel.innerHTML = "";
  var first = document.createElement("option"); first.value = ""; first.textContent = ph;
  sel.appendChild(first);
  rows.forEach(function(r){
    var o = document.createElement("option"); o.value = r[0]; o.textContent = r[1]; sel.appendChild(o);
  });
}
function cascade(parentId, childId, rows, ph){
  var p = document.getElementById(parentId), c = document.getElementById(childId);
  if(!p || !c) return;
  p.addEventListener("change", function(){
    fillSelect(c, rows.filter(function(r){ return String(r[2]) === String(p.value); }), ph);
  });
}
function tokenSelect(host, name, options, ph, onChange){
  if(!host) return;
  host.classList.add("ts");
  var box = document.createElement("div"); box.className = "ts-box";
  var input = document.createElement("input");
  input.className = "ts-input"; input.type = "text"; input.placeholder = ph; input.autocomplete = "off";
  var menu = document.createElement("div"); menu.className = "ts-menu"; menu.style.display = "none";
  box.appendChild(input); host.appendChild(box); host.appendChild(menu);
  var chosen = {}, els = {}, filtered = [], active = -1;
  box.addEventListener("click", function(){ input.focus(); });
  function notify(){ if(onChange){ onChange(Object.keys(chosen)); } }
  function remove(id){
    if(!els[id]) return;
    box.removeChild(els[id].chip); host.removeChild(els[id].hid);
    delete els[id]; delete chosen[id];
  }
  function add(opt){
    if(chosen[opt[0]]) return;
    chosen[opt[0]] = 1;
    var chip = document.createElement("span"); chip.className = "ts-chip";
    chip.appendChild(document.createTextNode(opt[1]));
    var x = document.createElement("b"); x.textContent = "×"; chip.appendChild(x);
    var hid = document.createElement("input"); hid.type = "hidden"; hid.name = name; hid.value = opt[0];
    els[opt[0]] = {chip: chip, hid: hid};
    x.addEventListener("click", function(e){
      e.stopPropagation(); remove(opt[0]); host.classList.remove("invalid"); render(); notify();
    });
    box.insertBefore(chip, input); host.appendChild(hid);
    input.value = ""; host.classList.remove("invalid"); render(); input.focus(); notify();
  }
  // Replace the available options (used by the state→district cascade); drop any
  // already-chosen items that are no longer valid for the new option set.
  function setOptions(newOpts){
    options = newOpts || [];
    var valid = {}; options.forEach(function(o){ valid[o[0]] = 1; });
    Object.keys(chosen).forEach(function(id){ if(!valid[id]){ remove(id); } });
    render(); notify();
  }
  function render(){
    var q = input.value.trim().toLowerCase();
    filtered = options.filter(function(o){
      if(chosen[o[0]]) return false;
      return q === "" ? true : o[1].toLowerCase().indexOf(q) >= 0;
    }).slice(0, 12);
    if(!filtered.length){ menu.innerHTML = '<div class="ts-empty">No matches</div>'; return; }
    menu.innerHTML = filtered.map(function(o, i){
      var sub = o[2] ? ' <span style="color:#6b7d88">- ' + _esc(o[2]) + '</span>' : '';
      return '<div class="ts-opt' + (i === active ? ' active' : '') + '" data-i="' + i + '">'
        + _esc(o[1]) + sub + '</div>';
    }).join("");
  }
  function openMenu(){ active = -1; render(); menu.style.display = "block"; }
  input.addEventListener("focus", openMenu);
  input.addEventListener("input", openMenu);
  input.addEventListener("keydown", function(e){
    if(e.key === "ArrowDown"){ active = Math.min(active + 1, filtered.length - 1); render(); e.preventDefault(); }
    else if(e.key === "ArrowUp"){ active = Math.max(active - 1, 0); render(); e.preventDefault(); }
    else if(e.key === "Enter"){ if(active >= 0 && filtered[active]) add(filtered[active]); e.preventDefault(); }
    else if(e.key === "Backspace" && input.value === ""){
      var c = box.querySelectorAll(".ts-chip"); if(c.length) c[c.length - 1].querySelector("b").click();
    }
  });
  menu.addEventListener("mousedown", function(e){
    var t = e.target.closest(".ts-opt"); if(t){ add(filtered[+t.getAttribute("data-i")]); e.preventDefault(); }
  });
  document.addEventListener("click", function(e){ if(!host.contains(e.target)) menu.style.display = "none"; });
  return { setOptions: setOptions };
}

cascade("state_id", "district_id", DISTRICTS, "Select district...");
cascade("course_id", "specialization_id", SPECS, "Select specialization...");

// Chrome may autofill the State select from a saved address — force it back to
// the "Select state…" placeholder so nothing is pre-chosen until the user picks.
(function(){
  var st = document.getElementById("state_id");
  if(!st) return;
  function reset(){ if(st.value){ st.value = ""; } }
  reset();
  setTimeout(reset, 200); setTimeout(reset, 600);
})();

var STATE_NAME = {}; STATES.forEach(function(s){ STATE_NAME[s[0]] = s[1]; });
// district options carry [id, name, stateName, stateId] so they can be filtered
// by the chosen states.
var DISTRICT_OPTS = DISTRICTS.map(function(d){ return [d[0], d[1], STATE_NAME[d[2]] || "", d[2]]; });

tokenSelect(document.getElementById("ts_skills"), "skill_ids", SKILLS, "Type a skill...");
tokenSelect(document.getElementById("ts_categories"), "preferred_category_ids", CATEGORIES, "Type a category...");
tokenSelect(document.getElementById("ts_roles"), "preferred_role_ids", ROLES, "Type a job role...");

// Preferred work locations: pick STATE(s) first (multi), then only those states'
// DISTRICTS are offered (multi). Selecting a state populates the district list;
// removing a state drops its districts.
var prefDistricts = tokenSelect(
  document.getElementById("ts_locations"), "preferred_location_ids", [],
  "Select a state above first..."
);
tokenSelect(
  document.getElementById("ts_pref_states"), "preferred_state_ids", STATES,
  "Type a state (e.g. Tamil Nadu)...",
  function(stateIds){
    var set = {}; stateIds.forEach(function(id){ set[id] = 1; });
    prefDistricts.setOptions(DISTRICT_OPTS.filter(function(d){ return set[d[3]]; }));
  }
);

// ---- chip groups (single / multi-select → hidden inputs) ----
function chipGroup(group){
  var name = group.getAttribute("data-name"), multi = group.getAttribute("data-multi") === "1";
  group.querySelectorAll(".chip").forEach(function(chip){
    chip.addEventListener("click", function(){
      if(multi){ chip.classList.toggle("on"); }
      else { group.querySelectorAll(".chip").forEach(function(c){ c.classList.remove("on"); }); chip.classList.add("on"); }
      group.classList.remove("invalid");
      group.querySelectorAll("input.cv").forEach(function(h){ h.remove(); });
      group.querySelectorAll(".chip.on").forEach(function(c){
        var h = document.createElement("input"); h.type = "hidden"; h.className = "cv";
        h.name = name; h.value = c.getAttribute("data-val"); group.appendChild(h);
      });
    });
  });
}
document.querySelectorAll(".chips").forEach(chipGroup);

// ---- date of birth combiner (Day / Month / Year → YYYY-MM-DD) ----
(function(){
  var d = document.getElementById("dob_d"), m = document.getElementById("dob_m"),
      y = document.getElementById("dob_y"), hid = document.getElementById("date_of_birth");
  if(!d) return;
  for(var i=1;i<=31;i++){ var o=document.createElement("option"); o.value=i; o.textContent=i; d.appendChild(o); }
  ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"].forEach(function(nm,i){
    var o=document.createElement("option"); o.value=i+1; o.textContent=nm; m.appendChild(o); });
  var ny=new Date().getFullYear(); for(var yr=ny-15; yr>=ny-70; yr--){ var o=document.createElement("option"); o.value=yr; o.textContent=yr; y.appendChild(o); }
  function pad(n){ return String(n).length<2 ? "0"+n : ""+n; }
  function upd(){ hid.value=(d.value&&m.value&&y.value)?(y.value+"-"+pad(m.value)+"-"+pad(d.value)):""; if(hid.value) d.parentNode.classList.remove("invalid"); }
  [d,m,y].forEach(function(s){ s.addEventListener("change", upd); });
})();

// ---- step wizard (Next validates required, Skip / Back navigate) ----
var steps = [].slice.call(document.querySelectorAll(".step"));
var dots = [].slice.call(document.querySelectorAll(".prog span"));
var cur = 0;
function show(i){
  cur = i;
  steps.forEach(function(s,k){ s.classList.toggle("on", k===i); });
  dots.forEach(function(s,k){ s.classList.toggle("on", k<=i); });
  window.scrollTo(0,0);
  document.getElementById("nextBtn").textContent = (i===steps.length-1) ? "Finish Setup" : "Next";
  document.getElementById("backBtn").style.visibility = i===0 ? "hidden" : "visible";
}
function valid(step){
  var ok = true, msg = "";
  step.querySelectorAll("[data-req]").forEach(function(el){
    var bad = false;
    if(el.classList.contains("chips")) bad = !el.querySelector(".chip.on");
    else if(el.classList.contains("ts")) bad = !el.querySelector("input[type=hidden]");
    else if(el.classList.contains("dob")) bad = !document.getElementById("date_of_birth").value;
    else if(el.tagName === "SELECT") bad = !el.value;
    else bad = !((el.value||"").trim());
    el.classList.toggle("invalid", bad); if(bad) ok = false;
  });
  // Format checks — only when the field has a value (blanks are handled above).
  step.querySelectorAll("[data-fmt]").forEach(function(el){
    var v = (el.value||"").trim(); if(!v) return;
    var f = el.getAttribute("data-fmt"), bad = false, m = "";
    if(f==="email"){ bad = !/^[^@\s]+@[^@\s]+\.[^@\s]{2,}$/.test(v); m = "Please enter a valid email address."; }
    else if(f==="year"){ bad = !/^\d{4}$/.test(v) || +v < 1950 || +v > 2035; m = "Enter a valid 4-digit passing year."; }
    else if(f==="salary"){ bad = !/^\d+(\.\d+)?$/.test(v) || +v < 0; m = "Salary must be a valid number."; }
    if(bad){ el.classList.add("invalid"); ok = false; if(!msg) msg = m; }
  });
  // Date of birth → must be a real date for a plausible working age (14–80).
  var dob = document.getElementById("date_of_birth");
  if(dob && dob.value && step.contains(dob)){
    var d = new Date(dob.value), t = new Date();
    var age = t.getFullYear()-d.getFullYear()-((t.getMonth()<d.getMonth()||(t.getMonth()===d.getMonth()&&t.getDate()<d.getDate()))?1:0);
    if(isNaN(age) || age < 14 || age > 80){ ok = false; if(!msg) msg = "Enter a valid date of birth (age 14–80)."; }
  }
  if(!ok && msg) alert(msg);
  return ok;
}
var submitting = false;
function advance(){
  if(cur===steps.length-1){
    if(submitting) return;                       // guard against double-submit
    submitting = true;
    var nb=document.getElementById("nextBtn"); nb.disabled=true; nb.textContent="Submitting…";
    var sb=document.getElementById("skipBtn"); if(sb){ sb.style.pointerEvents="none"; sb.style.opacity=".5"; }
    document.getElementById("profForm").submit();
  } else show(cur+1);
}
document.getElementById("nextBtn").addEventListener("click", function(){ if(!submitting && valid(steps[cur])) advance(); });
document.getElementById("skipBtn").addEventListener("click", function(){ if(!submitting) advance(); });
document.getElementById("backBtn").addEventListener("click", function(){ if(cur>0) show(cur-1); });
show(0);
"""


def _chips(name: str, options, *, multi: bool = False, req: bool = False) -> str:
    """A pill chip-group → hidden inputs (single or multi select)."""
    attrs = f' data-name="{_esc(name)}"'
    if multi:
        attrs += ' data-multi="1"'
    if req:
        attrs += ' data-req="1"'
    btns = "".join(f'<span class="chip" data-val="{_esc(v)}">{_esc(l)}</span>' for v, l in options)
    return f'<div class="chips"{attrs}>{btns}</div>'


def _R() -> str:
    return '<span class="req">*</span>'


def _form_html(
    token: str, name: str, phone: str, opts: dict[str, list[dict[str, Any]]], *, error: str = ""
) -> str:
    safe_name = _esc(name)
    err = f'<div class="err">{_esc(error)}</div>' if error else ""
    o = opts  # shorthand
    dots = "".join(f"<span>{i}</span>" for i in range(1, 10))
    langs = "".join(
        f'<div class="lang"><div class="nm">{_esc(l["name"])}</div>'
        f'<label><input type="checkbox" name="lang_speak" value="{_esc(l["id"])}"> Speak</label>'
        f'<label><input type="checkbox" name="lang_write" value="{_esc(l["id"])}"> Write</label></div>'
        for l in o["languages"]
    )

    body = f"""\
<h1>Complete Your Profile</h1>
<div class="prog">{dots}</div>
{err}
<form method="post" action="/onboard/submit" id="profForm" autocomplete="off" enctype="multipart/form-data">
  <input type="hidden" name="token" value="{_esc(token)}">
  <input type="hidden" name="date_of_birth" id="date_of_birth">

  <section class="step"><h1>Personal Info</h1>
    <label>I am a {_R()}</label>
    {_chips("current_status", _I_AM_A, req=True)}
    <label>Full Name {_R()}</label>
    <input type="text" name="full_name" value="{safe_name}" placeholder="Your full name" data-req>
    <label>Mobile Number</label>
    <input type="text" class="ro" value="{_esc(phone)}" readonly>
    <label>Email</label>
    <input type="email" name="email" data-fmt="email" placeholder="you@example.com">
    <label>Resume <span class="hint">(PDF, DOC — tap to select a file)</span></label>
    <input type="file" name="resume" accept=".pdf,.doc,.docx,.rtf,.odt,.png,.jpg,.jpeg">
    <label>Gender {_R()}</label>
    {_chips("gender", _GENDER, req=True)}
    <label>Marital Status {_R()}</label>
    {_chips("marital_status", _MARITAL, req=True)}
  </section>

  <section class="step"><h1>Birth &amp; Location</h1>
    <label>Date of Birth {_R()}</label>
    <div class="dob" data-req>
      <select id="dob_d"><option value="">Day</option></select>
      <select id="dob_m"><option value="">Month</option></select>
      <select id="dob_y"><option value="">Year</option></select>
    </div>
    <label>State {_R()}</label>
    <select name="state_id" id="state_id" data-req autocomplete="off">{_options_html(o["states"], placeholder="Select state…")}</select>
    <label>District {_R()}</label>
    <select name="district_id" id="district_id" data-req autocomplete="off"><option value="">Select a state first…</option></select>
    <label>City / Area</label>
    <input type="text" name="city" placeholder="Enter city name" autocomplete="off">
  </section>

  <section class="step"><h1>Education</h1>
    <label>Education Level {_R()}</label>
    <select name="education_level_id" data-req>{_options_html(o["education_levels"], placeholder="Select…")}</select>
    <label>Course / Degree</label>
    <select name="course_id" id="course_id">{_options_html(o["courses"], placeholder="Select…")}</select>
    <label>Specialization {_R()}</label>
    <select name="specialization_id" id="specialization_id" data-req><option value="">Select a course first…</option></select>
    <label>Institution / College</label>
    <input type="text" name="institution" placeholder="e.g. Anna University">
    <label>Year of Passing</label>
    <input type="number" name="year_of_passing" data-fmt="year" min="1970" max="2035" placeholder="e.g. 2024">
  </section>

  <section class="step"><h1>Skills</h1>
    <p class="sub">Select at least 1 skill to get better job matches.</p>
    <label>Selected Skills {_R()}</label>
    <div id="ts_skills" data-req></div>
  </section>

  <section class="step"><h1>Job Preferences</h1>
    <label>Preferred Job Categories {_R()}</label>
    <div id="ts_categories" data-req></div>
    <label>Job Type</label>
    {_chips("job_types", _JOB_TYPES, multi=True)}
    <label>Work Mode Preference</label>
    {_chips("work_mode", _WORK_MODE)}
  </section>

  <section class="step"><h1>Salary &amp; Experience</h1>
    <label>Expected Monthly Salary {_R()}</label>
    {_chips("expected_salary", _SALARY, req=True)}
    <label>Current Monthly Salary (₹) <span class="hint">(if working)</span></label>
    <input type="number" name="current_salary" data-fmt="salary" min="0" step="500" placeholder="e.g. 18000">
    <label>Do you have work experience?</label>
    {_chips("has_experience", _YESNO)}
    <label>Experience level <span class="hint">(if experienced)</span></label>
    <select name="experience_level_id">{_options_html(o["experience_levels"], placeholder="Select…")}</select>
  </section>

  <section class="step"><h1>Preferred Roles</h1>
    <p class="sub">Pick the roles you'd like us to match you with.</p>
    <label>Preferred Job Roles</label>
    <div id="ts_roles"></div>
  </section>

  <section class="step"><h1>Preferred Work Locations</h1>
    <label>States {_R()} <span class="hint">(pick one or more)</span></label>
    <div id="ts_pref_states" data-req></div>
    <label>Districts {_R()} <span class="hint">(districts of the states you picked)</span></label>
    <div id="ts_locations" data-req></div>
    <label>Interested in working abroad?</label>
    {_chips("interested_in_abroad", _YESNO)}
  </section>

  <section class="step"><h1>Language Mastery</h1>
    <p class="sub">Which languages can you speak and write? This helps us match you.</p>
    {langs}
  </section>

  <div class="nav">
    <button type="button" class="back" id="backBtn">← Back</button>
    <button type="button" class="skip" id="skipBtn">Skip for now</button>
    <button type="button" class="next" id="nextBtn">Next</button>
  </div>
</form>
<script>
const SKILLS = {_js_rows(o["skills"])};
const ROLES = {_js_rows(o["roles"])};
const CATEGORIES = {_js_rows(o["categories"])};
const STATES = {_js_rows(o["states"])};
const DISTRICTS = {_js_rows(o["districts"])};
const SPECS = {_js_rows(o["specializations"])};
const OTHER_STATES = {_js_rows(o["other_states"])};
</script>
<script>{_FORM_JS}</script>"""
    return _PAGE.format(body=body)


def _success_html(name: str, *, business_number: str = "") -> str:
    who = f", {html.escape(name)}" if name else ""
    # A "Back to chat" button that returns the candidate to WhatsApp. With the
    # business number configured it's a wa.me deep link that simply reopens the
    # chat — NO pre-filled text, because the bot already pushed the registration
    # success + menu, so the conversation just continues. Otherwise it's a
    # best-effort window.close(). This button only appears AFTER a submission, so
    # a candidate who merely opens the form and leaves is never let past the gate.
    if business_number:
        # Reliable: navigate back to WhatsApp (the app reopens the chat). A web
        # page cannot force-close a tab the user navigated to, so we redirect
        # rather than call window.close().
        close = (
            f'<a class="btn" href="https://wa.me/{business_number}">'
            "Back to chat</a>"
        )
    else:
        # No business number configured → we can't deep-link back. window.close()
        # is blocked for user-opened tabs, so we attempt it but ALSO tell the
        # candidate how to return, instead of leaving a dead button.
        close = (
            '<button onclick="window.close()">Close</button>'
            '<p class="sub" style="margin-top:14px">All done! Return to WhatsApp '
            "(tap the back arrow or &#10005; at the top) and send us a message.</p>"
        )
    body = f"""\
<div class="ok">
  <div class="tick">&#10003;</div>
  <h1>Registration successful{who}!</h1>
  <p class="sub">Your profile is saved. Tap below to head back to WhatsApp —
  your menu is already waiting in the chat.</p>
  {close}
</div>"""
    return _PAGE.format(body=body)


def _expired_html() -> str:
    body = """\
<h1>Link expired</h1>
<p class="sub">This form link is invalid or has expired. Please go back to
WhatsApp and message us so we can send you a fresh one.</p>"""
    return _PAGE.format(body=body)


def _resume_html(token: str, *, error: str = "") -> str:
    """One-screen resume picker — tap to select a PDF/DOC, then Submit."""
    err = f'<div class="err">{_esc(error)}</div>' if error else ""
    body = f"""\
<h1>Upload your resume</h1>
<p class="sub">Choose your resume (PDF or DOC) to finish applying.</p>
{err}
<form method="post" action="/onboard/resume/submit" enctype="multipart/form-data">
  <input type="hidden" name="token" value="{_esc(token)}">
  <label>Resume <span class="hint">(PDF, DOC — tap to select a file)</span></label>
  <input type="file" name="resume" accept=".pdf,.doc,.docx,.rtf,.odt,.png,.jpg,.jpeg" required>
  <button class="btn" type="submit" style="margin-top:18px">Submit resume</button>
</form>"""
    return _PAGE.format(body=body)


def _resume_done_html(sub: str, *, business_number: str = "") -> str:
    if business_number:
        close = f'<a class="btn" href="https://wa.me/{business_number}">Back to chat</a>'
    else:
        close = (
            '<button onclick="window.close()">Close</button>'
            '<p class="sub" style="margin-top:14px">All done! Return to WhatsApp '
            "and continue the chat.</p>"
        )
    body = f"""\
<div class="ok">
  <div class="tick">&#10003;</div>
  <h1>Resume uploaded!</h1>
  <p class="sub">{_esc(sub)}</p>
  {close}
</div>"""
    return _PAGE.format(body=body)
