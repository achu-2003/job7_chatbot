"""Self-hosted employer (job-poster) forms for the WhatsApp recruiter lane.

Mirrors the candidate onboarding form, but for the *employer* side and shaped
to the live ``private_employers`` table. Three tokenised forms, all keyed to one
per-phone employer token:

    GET  /employer/register?token=...   → company profile form  (Stage 1)
    POST /employer/register/submit      → stage the employer record in Redis
    GET  /employer/kyc?token=...        → KYC / business-proof form (Stage 2)
    POST /employer/kyc/submit           → merge KYC, auto-verify (test mode)
    GET  /employer/post-job?token=...   → post-a-job form (Stage 4)
    POST /employer/post-job/submit      → append the job to the employer record

Everything is stored in **Redis only** (never the business DB) — this is the
test harness for the employer flow. Each submit proactively pushes the next
step's prompt/menu back to WhatsApp so the chat continues without typing.
"""
from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import parse_qs

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from app.api.deps import get_memory
from app.chatbot import wa_format as wa
from app.config import get_settings
from app.core.logging import get_logger
from app.db.repositories import LookupRepository
from app.employer import (
    APPLY_MODES,
    CANDIDATE_DISTANCES,
    COMPANY_SIZES,
    ENGLISH_LEVELS,
    EXPERIENCE_TYPES,
    GENDER_PREFS,
    JOB_LANGUAGES,
    JOB_LOCATION_TYPES,
    JOB_TYPES,
    JOB_WORK_MODES,
    KYC_DOC_TYPES,
    MARITAL_PREFS,
    QUALIFICATION_LEVELS,
    REQUIRED_ASSETS,
    SALARY_PERIODS,
    apply_kyc,
    build_employer_record,
    build_job_record,
)
from app.whatsapp import delivery as wa_delivery

router = APIRouter()
log = get_logger("employer")

# Employer menu buttons pushed after KYC verification (ids routed in runtime).
_EMP_MENU = (("emp:post", "Post a Job"), ("emp:candidates", "View Candidates"),
             ("emp:myjobs", "My Jobs"))


def _esc(v: Any) -> str:
    return html.escape(str(v if v is not None else ""), quote=True)


async def _reg_options() -> dict[str, list[dict[str, Any]]]:
    """Reference data for the registration form (loaded sequentially to avoid
    opening many Postgres connections at once)."""
    kinds = ("industries", "designations", "states", "districts")
    return {k: await LookupRepository.options(k) for k in kinds}


async def _job_options() -> dict[str, list[dict[str, Any]]]:
    kinds = ("categories", "districts", "states")
    return {k: await LookupRepository.options(k) for k in kinds}


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@router.get("/register", response_class=HTMLResponse)
async def register_form(request: Request, token: str = Query(default="")) -> HTMLResponse:
    identity = await get_memory(request).get_employer_identity(token) if token else None
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    opts = await _reg_options()
    return HTMLResponse(_register_html(token, identity.get("customer_id") or "", opts))


@router.post("/register/submit", response_class=HTMLResponse)
async def register_submit(request: Request) -> HTMLResponse:
    raw = parse_qs((await request.body()).decode("utf-8"))

    def one(key: str) -> str:
        v = raw.get(key) or []
        return v[0].strip() if v else ""

    token = one("token")
    if not token:
        return HTMLResponse(_expired_html(), status_code=400)

    memory = get_memory(request)
    identity = await memory.get_employer_identity(token)
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)

    form = {
        "company_name": one("company_name"),
        "industry_id": one("industry_id"),
        "company_size": one("company_size"),
        "contact_person": one("contact_person"),
        "designation_id": one("designation_id"),
        "email": one("email"),
        "primary_phone": identity.get("customer_id") or one("primary_phone"),
        "website": one("website"),
        "address": one("address"),
        "district_id": one("district_id"),
        "city": one("city"),
        "pincode": one("pincode"),
        "description": one("description"),
    }
    record = build_employer_record(identity=identity, form=form)
    record.setdefault("paid", False)
    record.setdefault("jobs", [])
    phone = identity.get("customer_id") or ""
    await memory.save_employer(phone, record, tenant_id=identity["tenant_id"])

    company = record["private_employers"].get("companyName") or "your company"
    # PROACTIVELY push the next step (KYC) so the chat continues without typing.
    settings = get_settings()
    digits = re.sub(r"\D", "", phone)
    if digits:
        body = (
            f"✅ Company profile created for *{company}*!\n\nOne more step — verify "
            "your business to unlock candidate details. Tap below to submit your KYC."
        )
        try:
            await wa_delivery.send_message(
                settings, digits, wa.buttons_message(body, [("emp:kyc", "Verify Business")])
            )
        except Exception as exc:  # noqa: BLE001 — proactive push is best-effort
            log.warning("employer_register_push_failed", error=str(exc)[:200])

    number = re.sub(r"\D", "", settings.whatsapp_business_number or "")
    return HTMLResponse(_success_html(
        "Company profile created!",
        "Your details are saved. Head back to WhatsApp to verify your business "
        "and start posting jobs.",
        business_number=number,
    ))


@router.get("/kyc", response_class=HTMLResponse)
async def kyc_form(request: Request, token: str = Query(default="")) -> HTMLResponse:
    identity = await get_memory(request).get_employer_identity(token) if token else None
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    return HTMLResponse(_kyc_html(token))


@router.post("/kyc/submit", response_class=HTMLResponse)
async def kyc_submit(request: Request) -> HTMLResponse:
    raw = parse_qs((await request.body()).decode("utf-8"))

    def one(key: str) -> str:
        v = raw.get(key) or []
        return v[0].strip() if v else ""

    token = one("token")
    if not token:
        return HTMLResponse(_expired_html(), status_code=400)
    memory = get_memory(request)
    identity = await memory.get_employer_identity(token)
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    phone = identity.get("customer_id") or ""
    record = await memory.get_employer(phone, tenant_id=identity["tenant_id"])
    if not record:
        return HTMLResponse(_expired_html(), status_code=404)

    settings = get_settings()
    apply_kyc(
        record,
        doc_type=one("kyc_document_type"),
        doc_url=one("kyc_document_url"),
        gst=one("gst_number"),
        pan=one("pan_number"),
        auto_verify=settings.employer_kyc_auto_verify,
    )
    await memory.save_employer(phone, record, tenant_id=identity["tenant_id"])

    verified = record["private_employers"].get("kycStatus") == "VERIFIED"
    digits = re.sub(r"\D", "", phone)
    if digits:
        if verified:
            body = (
                "✅ Business verified! You're all set.\n\nWhat would you like to do "
                "— post a job, or view candidates?"
            )
            payload = wa.buttons_message(body, _EMP_MENU)
        else:
            body = (
                "📋 KYC submitted — it's now under review. We'll notify you here once "
                "your business is verified, then you can view full candidate details."
            )
            payload = wa.text_message(body)
        try:
            await wa_delivery.send_message(settings, digits, payload)
        except Exception as exc:  # noqa: BLE001
            log.warning("employer_kyc_push_failed", error=str(exc)[:200])

    number = re.sub(r"\D", "", settings.whatsapp_business_number or "")
    sub = ("Your business is verified — head back to WhatsApp to post a job or view "
           "candidates." if verified else
           "KYC submitted. We'll verify your business and notify you on WhatsApp.")
    return HTMLResponse(_success_html(
        "Business verified!" if verified else "KYC submitted!", sub, business_number=number,
    ))


@router.get("/post-job", response_class=HTMLResponse)
async def post_job_form(request: Request, token: str = Query(default="")) -> HTMLResponse:
    identity = await get_memory(request).get_employer_identity(token) if token else None
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    opts = await _job_options()
    return HTMLResponse(_post_job_html(token, opts))


@router.post("/post-job/submit", response_class=HTMLResponse)
async def post_job_submit(request: Request) -> HTMLResponse:
    raw = parse_qs((await request.body()).decode("utf-8"))

    def one(key: str) -> str:
        v = raw.get(key) or []
        return v[0].strip() if v else ""

    def many(key: str) -> list[str]:
        return [x.strip() for x in (raw.get(key) or []) if x.strip()]

    token = one("token")
    if not token:
        return HTMLResponse(_expired_html(), status_code=400)
    memory = get_memory(request)
    identity = await memory.get_employer_identity(token)
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    phone = identity.get("customer_id") or ""

    employer = await memory.get_employer(phone, tenant_id=identity["tenant_id"])
    if not employer:
        return HTMLResponse(_expired_html(), status_code=404)
    employer_id = (employer.get("private_employers") or {}).get("id")

    title = one("title")
    form = {
        # Job Details
        "title": title,
        "job_type": one("job_type"),
        "description": one("description"),
        # Experience & Salary
        "experience_type": one("experience_type"),
        "experience_min": one("experience_min"),
        "experience_max": one("experience_max"),
        "salary_period": one("salary_period"),
        "salary_min": one("salary_min"),
        "salary_max": one("salary_max"),
        "salary_negotiable": one("salary_negotiable"),
        "vacancies": one("vacancies"),
        # Job Location
        "job_location_type": one("job_location_type"),
        "work_mode": one("work_mode"),
        "state_id": one("state_id"),
        "district_id": one("district_id"),
        "city": one("city"),
        "work_from_home": one("work_from_home"),
        # Candidate Requirements
        "qualification_level": one("qualification_level"),
        "gender_preference": one("gender_preference"),
        "marital_status_preference": one("marital_status_preference"),
        "english_level": one("english_level"),
        "age_min": one("age_min"),
        "age_max": one("age_max"),
        # Location Preferences
        "candidate_distance": one("candidate_distance"),
        "candidate_distance_custom": one("candidate_distance_custom"),
        "willing_to_relocate": one("willing_to_relocate"),
        # Security Deposit
        "has_security_deposit": one("has_security_deposit"),
        "security_deposit_amt": one("security_deposit_amt"),
        "security_deposit_reason": one("security_deposit_reason"),
        # Work Timings + Interview
        "work_start_time": one("work_start_time"),
        "work_end_time": one("work_end_time"),
        "interview_date": one("interview_date"),
        "interview_time": one("interview_time"),
        # Skills & Languages
        "category_id": one("category_id"),
        "skills": one("skills"),
        "preferred_languages": many("preferred_languages"),
        "required_assets": many("required_assets"),
        # Apply Methods
        "apply_modes": many("apply_modes"),
        "contact_phone": one("contact_phone"),
        "contact_whatsapp": one("contact_whatsapp"),
    }
    # Build the DB-ready private_jobs record (status PENDING — awaiting approval)
    # and stage it on the employer's Redis record. No live private_jobs write yet.
    job = build_job_record(employer_id=employer_id, form=form, status="PENDING")
    record = await memory.add_employer_job(phone, job, tenant_id=identity["tenant_id"])
    if record is None:
        return HTMLResponse(_expired_html(), status_code=404)

    settings = get_settings()
    digits = re.sub(r"\D", "", phone)
    if digits:
        body = (
            f"✅ Job posted: *{title}* ({job['ref']}).\n\nCandidates can be matched to "
            "it now. What next?"
        )
        try:
            await wa_delivery.send_message(settings, digits, wa.buttons_message(body, _EMP_MENU))
        except Exception as exc:  # noqa: BLE001
            log.warning("employer_postjob_push_failed", error=str(exc)[:200])

    number = re.sub(r"\D", "", settings.whatsapp_business_number or "")
    return HTMLResponse(_success_html(
        "Job posted!", f"“{title}” is live ({job['ref']}). Head back to WhatsApp to "
        "view candidates.", business_number=number,
    ))


# ---------------------------------------------------------------------------
# HTML (inline; no template engine dependency)
# ---------------------------------------------------------------------------

_STYLE = """
  body{font-family:system-ui,Segoe UI,Roboto,sans-serif;background:#0b141a;color:#e9edef;
       margin:0;display:flex;min-height:100vh;align-items:flex-start;justify-content:center;padding:18px 0}
  .card{background:#111b21;max-width:480px;width:92%;padding:24px 22px;border-radius:14px;
        box-shadow:0 10px 30px rgba(0,0,0,.4)}
  h1{font-size:20px;margin:0 0 4px} p.sub{color:#8696a0;margin:0 0 16px;font-size:14px}
  label{display:block;font-size:13px;color:#aebac1;margin:14px 0 6px}
  label .req{color:#ff8a8a} label .hint{color:#6b7d88;font-weight:400}
  input,select,textarea{width:100%;box-sizing:border-box;padding:11px 12px;border-radius:8px;
        border:1px solid #2a3942;background:#202c33;color:#e9edef;font-size:15px}
  textarea{min-height:74px;resize:vertical}
  .row{display:flex;gap:10px} .row>div{flex:1}
  .btn{margin-top:20px;width:100%;padding:13px;border:0;border-radius:9px;background:#00a884;
       color:#04150f;font-size:16px;font-weight:600;cursor:pointer}
  .tick{width:54px;height:54px;border-radius:50%;background:#00a884;color:#04150f;font-size:30px;
        display:flex;align-items:center;justify-content:center;margin:0 auto 14px}
  .ok{text-align:center}
  /* multi-step wizard */
  .dots{display:flex;gap:6px;justify-content:center;margin:0 0 8px;flex-wrap:wrap}
  .dots span{width:9px;height:9px;border-radius:50%;background:#2a3942}
  .dots span.on{background:#00a884}
  .stepname{text-align:center;color:#00d3a7;font-size:12px;text-transform:uppercase;
        letter-spacing:.04em;margin:0 0 14px}
  .step{display:none} .step.active{display:block}
  .opts label.opt{display:flex;align-items:center;gap:10px;border:1px solid #2a3942;border-radius:10px;
        padding:11px 12px;margin:8px 0;cursor:pointer}
  .opts label.opt input{width:auto;accent-color:#00a884}
  .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}
  .chips label.chip{border:1px solid #2a3942;border-radius:20px;padding:7px 13px;cursor:pointer;font-size:14px}
  .chips label.chip input{display:none}
  .chips label.chip input:checked + span{color:#04150f}
  .chips label.chip:has(input:checked){background:#00a884;border-color:#00a884}
  .toggle{display:flex;justify-content:space-between;align-items:center;gap:10px;margin:14px 0}
  .toggle small{display:block;color:#6b7d88;font-size:12px;margin-top:2px}
  .toggle input{width:auto;transform:scale(1.4);accent-color:#00a884}
  .nav{display:flex;gap:10px;margin-top:18px}
  .nav button{flex:1;padding:13px;border:0;border-radius:9px;font-size:15px;font-weight:600;cursor:pointer}
  .ghost{background:#202c33;color:#e9edef;border:1px solid #2a3942 !important}
"""


def _page(title: str, inner: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{_STYLE}</style></head>"
        f'<body><div class="card">{inner}</div></body></html>'
    )


def _options_html(items: "list[dict[str, Any]] | tuple", *, placeholder: str | None = None) -> str:
    out: list[str] = []
    if placeholder is not None:
        out.append(f'<option value="">{_esc(placeholder)}</option>')
    for it in items:
        val, lab = (it["id"], it["name"]) if isinstance(it, dict) else (it[0], it[1])
        out.append(f'<option value="{_esc(val)}">{_esc(lab)}</option>')
    return "".join(out)


def _radios(name: str, options: "tuple | list", *, default: str | None = None) -> str:
    """Card-style radio group (one choice)."""
    out = []
    for val, lab in options:
        chk = " checked" if default is not None and val == default else ""
        out.append(
            f'<label class="opt"><input type="radio" name="{name}" value="{_esc(val)}"{chk}>'
            f"<span>{_esc(lab)}</span></label>"
        )
    return f'<div class="opts">{"".join(out)}</div>'


def _chips(name: str, options: "tuple | list") -> str:
    """Pill-style checkbox group (multi-select). Options may be (val,label) or str."""
    out = []
    for o in options:
        val, lab = (o, o) if isinstance(o, str) else o
        out.append(
            f'<label class="chip"><input type="checkbox" name="{name}" value="{_esc(val)}">'
            f"<span>{_esc(lab)}</span></label>"
        )
    return f'<div class="chips">{"".join(out)}</div>'


def _toggle(name: str, label: str, hint: str = "") -> str:
    return (
        f'<label class="toggle"><span><b>{_esc(label)}</b>'
        f'<small>{_esc(hint)}</small></span>'
        f'<input type="checkbox" name="{name}" value="yes"></label>'
    )


def _js_rows(items: list[dict[str, Any]]) -> str:
    rows = [[_esc(i["id"]), _esc(i["name"]), _esc(i.get("parent") or "")] for i in items]
    cells = ",".join(
        "[" + ",".join(f'"{c}"' for c in r) + "]" for r in rows
    )
    return f"[{cells}]"


def _register_html(token: str, phone: str, o: dict[str, list[dict[str, Any]]]) -> str:
    inner = f"""
<h1>Register your company</h1>
<p class="sub">A few details about your business so candidates know who's hiring.</p>
<form method="post" action="/employer/register/submit" autocomplete="off">
  <input type="hidden" name="token" value="{_esc(token)}">
  <label>Company name <span class="req">*</span></label>
  <input name="company_name" required placeholder="e.g. Acme Technologies">
  <div class="row">
    <div><label>Industry</label>
      <select name="industry_id" autocomplete="off">{_options_html(o['industries'], placeholder='Select…')}</select></div>
    <div><label>Company size</label>
      <select name="company_size" autocomplete="off">{_options_html(COMPANY_SIZES, placeholder='Select…')}</select></div>
  </div>
  <div class="row">
    <div><label>Contact person</label><input name="contact_person" placeholder="Your name"></div>
    <div><label>Designation</label>
      <select name="designation_id" autocomplete="off">{_options_html(o['designations'], placeholder='Select…')}</select></div>
  </div>
  <label>Work email</label>
  <input type="email" name="email" placeholder="hr@company.com">
  <label>Phone</label>
  <input name="primary_phone" value="{_esc(phone)}" readonly>
  <label>Website</label>
  <input type="url" name="website" placeholder="https://…">
  <label>Address</label>
  <input name="address" placeholder="Office address">
  <div class="row">
    <div><label>State</label>
      <select name="state_id" id="state_id" autocomplete="off">{_options_html(o['states'], placeholder='Select state…')}</select></div>
    <div><label>District</label>
      <select name="district_id" id="district_id" autocomplete="off"><option value="">Select a state first…</option></select></div>
  </div>
  <div class="row">
    <div><label>City</label><input name="city" placeholder="City"></div>
    <div><label>Pincode</label><input name="pincode" inputmode="numeric" placeholder="600001"></div>
  </div>
  <label>About the company</label>
  <textarea name="description" placeholder="What your company does (optional)"></textarea>
  <button class="btn" type="submit">Create profile</button>
</form>
<script>
var DISTRICTS = {_js_rows(o['districts'])};
function fillDistricts(){{
  var st = document.getElementById('state_id').value;
  var c = document.getElementById('district_id');
  var rows = DISTRICTS.filter(function(r){{ return String(r[2]) === String(st); }});
  c.innerHTML = '<option value="">Select district…</option>' +
    rows.map(function(r){{ return '<option value="'+r[0]+'">'+r[1]+'</option>'; }}).join('');
}}
document.getElementById('state_id').addEventListener('change', fillDistricts);
// Chrome may autofill State — force it back to the placeholder on load.
(function(){{ var s=document.getElementById('state_id');
  function reset(){{ if(s.value){{ s.value=''; }} }} reset(); setTimeout(reset,300); }})();
</script>
"""
    return _page("Register your company", inner)


def _kyc_html(token: str) -> str:
    inner = f"""
<h1>Verify your business</h1>
<p class="sub">Submit a business proof to unlock full candidate details. Your
documents are used only for verification.</p>
<form method="post" action="/employer/kyc/submit" autocomplete="off">
  <input type="hidden" name="token" value="{_esc(token)}">
  <label>Document type <span class="req">*</span></label>
  <select name="kyc_document_type" required>{_options_html(KYC_DOC_TYPES, placeholder='Select…')}</select>
  <div class="row">
    <div><label>GST number</label><input name="gst_number" placeholder="22AAAAA0000A1Z5"></div>
    <div><label>PAN number</label><input name="pan_number" placeholder="AAAAA0000A"></div>
  </div>
  <label>Document link <span class="req">*</span></label>
  <input type="url" name="kyc_document_url" required placeholder="https://… (Drive/Dropbox link to the proof)">
  <button class="btn" type="submit">Submit for verification</button>
</form>
"""
    return _page("Verify your business", inner)


_JOB_STEPS = (
    "Job Details", "Experience & Salary", "Job Location", "Candidate Requirements",
    "Preferences", "Timings & Interview", "Skills & Languages", "Apply Methods",
)


def _post_job_html(token: str, o: dict[str, list[dict[str, Any]]]) -> str:
    n = len(_JOB_STEPS)
    steps = "\n".join([
        # 1 — Job Details
        f"""<section class="step"><h1>Job Details</h1>
  <label>Job Title <span class="req">*</span></label>
  <input name="title" required placeholder="e.g. Software Developer">
  <label>Job Type</label>
  {_radios("job_type", JOB_TYPES, default="FULL_TIME")}
  <label>Description</label>
  <textarea name="description" placeholder="Role responsibilities, requirements…"></textarea>
</section>""",
        # 2 — Experience & Salary
        f"""<section class="step"><h1>Experience &amp; Salary</h1>
  <label>Experience Required</label>
  {_radios("experience_type", EXPERIENCE_TYPES, default="ANY")}
  <div class="row">
    <div><label>Min years</label><input name="experience_min" inputmode="numeric" placeholder="0"></div>
    <div><label>Max years</label><input name="experience_max" inputmode="numeric" placeholder="3"></div>
  </div>
  <label>Salary Range</label>
  {_radios("salary_period", SALARY_PERIODS, default="MONTHLY")}
  <div class="row">
    <div><label>Min (₹)</label><input name="salary_min" inputmode="numeric" placeholder="Min"></div>
    <div><label>Max (₹)</label><input name="salary_max" inputmode="numeric" placeholder="Max"></div>
  </div>
  {_toggle("salary_negotiable", "Salary negotiable", "Open to discussion")}
  <label>Number of Vacancies</label>
  <input name="vacancies" inputmode="numeric" placeholder="e.g. 5">
</section>""",
        # 3 — Job Location
        f"""<section class="step"><h1>Job Location</h1>
  <label>Job Location <span class="req">*</span></label>
  {_radios("job_location_type", JOB_LOCATION_TYPES, default="COMPANY_ADDRESS")}
  <label>Work mode</label>
  {_radios("work_mode", JOB_WORK_MODES, default="OFFICE")}
  <div class="row">
    <div><label>State</label>
      <select name="state_id" autocomplete="off">{_options_html(o['states'], placeholder='Select…')}</select></div>
    <div><label>District</label>
      <select name="district_id" autocomplete="off">{_options_html(o['districts'], placeholder='Select…')}</select></div>
  </div>
  <label>City / area</label>
  <input name="city" placeholder="Street, area, landmark">
</section>""",
        # 4 — Candidate Requirements
        f"""<section class="step"><h1>Candidate Requirements</h1>
  <label>Qualification Level</label>
  {_radios("qualification_level", QUALIFICATION_LEVELS)}
  <label>Gender Preference</label>
  {_radios("gender_preference", GENDER_PREFS, default="BOTH")}
  <label>Marital Status</label>
  {_radios("marital_status_preference", MARITAL_PREFS, default="ANY")}
  <label>English Level</label>
  {_radios("english_level", ENGLISH_LEVELS, default="NO_NEED")}
  <label>Age Range <span class="hint">(optional)</span></label>
  <div class="row">
    <div><input name="age_min" inputmode="numeric" placeholder="Min age"></div>
    <div><input name="age_max" inputmode="numeric" placeholder="Max age"></div>
  </div>
</section>""",
        # 5 — Preferences
        f"""<section class="step"><h1>Preferences</h1>
  <label>Preferred Candidate Distance</label>
  {_radios("candidate_distance", CANDIDATE_DISTANCES)}
  <label>Custom distance (km) <span class="hint">— if Custom selected</span></label>
  <input name="candidate_distance_custom" inputmode="numeric" placeholder="e.g. 50">
  {_toggle("willing_to_relocate", "Willing to Relocate", "Accept candidates willing to relocate")}
  {_toggle("work_from_home", "Work from Home", "This job can be done from home")}
  {_toggle("has_security_deposit", "Security Deposit Required", "Candidate needs to pay a deposit")}
  <div class="row">
    <div><label>Deposit amount (₹)</label><input name="security_deposit_amt" inputmode="numeric" placeholder="0"></div>
    <div><label>Reason</label><input name="security_deposit_reason" placeholder="e.g. Tools"></div>
  </div>
</section>""",
        # 6 — Timings & Interview
        f"""<section class="step"><h1>Timings &amp; Interview</h1>
  <label>Work Timings</label>
  <div class="row">
    <div><label>Start time</label><input type="time" name="work_start_time"></div>
    <div><label>End time</label><input type="time" name="work_end_time"></div>
  </div>
  <label>Interview Details</label>
  <div class="row">
    <div><label>Date</label><input type="date" name="interview_date"></div>
    <div><label>Time</label><input type="time" name="interview_time"></div>
  </div>
</section>""",
        # 7 — Skills & Languages
        f"""<section class="step"><h1>Skills &amp; Languages</h1>
  <label>Job Category <span class="req">*</span></label>
  <select name="category_id" autocomplete="off">{_options_html(o['categories'], placeholder='Select a job category')}</select>
  <label>Skills <span class="hint">(comma-separated)</span></label>
  <input name="skills" placeholder="e.g. Welding, Fitting">
  <label>Preferred Languages</label>
  {_chips("preferred_languages", JOB_LANGUAGES)}
  <label>Required Assets</label>
  {_chips("required_assets", REQUIRED_ASSETS)}
</section>""",
        # 8 — Apply Methods
        f"""<section class="step"><h1>Apply Methods</h1>
  <p class="sub">Choose how candidates can reach you for this job.</p>
  {_chips("apply_modes", APPLY_MODES)}
  <label>Contact Phone Number <span class="hint">(optional)</span></label>
  <input name="contact_phone" inputmode="numeric" placeholder="e.g. 9876543210">
  <label>WhatsApp Number <span class="hint">(optional)</span></label>
  <input name="contact_whatsapp" inputmode="numeric" placeholder="e.g. 9876543210">
</section>""",
    ])

    names_js = "[" + ", ".join(f'"{_esc(s)}"' for s in _JOB_STEPS) + "]"
    inner = f"""
<div class="dots" id="dots"></div>
<div class="stepname" id="stepname"></div>
<form method="post" action="/employer/post-job/submit" autocomplete="off" id="jobForm">
  <input type="hidden" name="token" value="{_esc(token)}">
  {steps}
  <div class="nav">
    <button type="button" class="ghost" id="back">← Back</button>
    <button type="button" class="btn" id="next" style="margin-top:0">Next →</button>
    <button type="submit" class="btn" id="post" style="margin-top:0;display:none">Post Job</button>
  </div>
</form>
<script>
var NAMES = {names_js};
var steps = [].slice.call(document.querySelectorAll('.step'));
var cur = 0;
var dots = document.getElementById('dots'), nm = document.getElementById('stepname');
var back = document.getElementById('back'), next = document.getElementById('next'), post = document.getElementById('post');
function render(){{
  steps.forEach(function(s,i){{ s.classList.toggle('active', i===cur); }});
  dots.innerHTML = steps.map(function(_,i){{ return '<span class="'+(i<=cur?'on':'')+'"></span>'; }}).join('');
  nm.textContent = 'Step '+(cur+1)+' of {n} — '+NAMES[cur];
  back.style.visibility = cur===0 ? 'hidden' : 'visible';
  next.style.display = cur===steps.length-1 ? 'none' : 'block';
  post.style.display = cur===steps.length-1 ? 'block' : 'none';
  window.scrollTo(0,0);
}}
function valid(){{
  if(cur===0){{ var t=document.querySelector('[name=title]'); if(!t.value.trim()){{ t.focus(); alert('Please enter a job title'); return false; }} }}
  if(NAMES[cur]==='Skills & Languages'){{ var c=document.querySelector('[name=category_id]'); if(!c.value){{ c.focus(); alert('Please select a job category'); return false; }} }}
  return true;
}}
next.onclick = function(){{ if(valid()){{ cur=Math.min(cur+1,steps.length-1); render(); }} }};
back.onclick = function(){{ cur=Math.max(cur-1,0); render(); }};
document.getElementById('jobForm').addEventListener('submit', function(e){{
  // ensure required category is set before the final submit
  var c=document.querySelector('[name=category_id]');
  if(!c.value){{ e.preventDefault(); cur=NAMES.indexOf('Skills & Languages'); render(); alert('Please select a job category'); }}
}});
render();
</script>
"""
    return _page("Post a job", inner)


def _success_html(title: str, sub: str, *, business_number: str = "") -> str:
    if business_number:
        close = f'<a class="btn" href="https://wa.me/{_esc(business_number)}">Back to chat</a>'
    else:
        close = ('<button class="btn" onclick="window.close()">Close</button>'
                 '<p class="sub" style="margin-top:14px">All done! Return to WhatsApp '
                 'and continue the chat.</p>')
    return _page(title, f"""
<div class="ok">
  <div class="tick">&#10003;</div>
  <h1>{_esc(title)}</h1>
  <p class="sub">{_esc(sub)}</p>
  {close}
</div>""")


def _expired_html() -> str:
    return _page("Link expired", """
<div class="ok">
  <h1>This link has expired</h1>
  <p class="sub">Please head back to WhatsApp and request a fresh link.</p>
</div>""")
