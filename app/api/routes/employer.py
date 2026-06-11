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
    COMPANY_SIZES,
    EXPERIENCE_TYPES,
    JOB_LOCATION_TYPES,
    INTERN_PAYMENT_TYPES,
    JOB_TYPES,
    KYC_DOC_TYPES,
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
    # District is REQUIRED — it becomes the job's location fallback (private_jobs
    # .districtId is NOT NULL), so an employer must have one. Re-show the form on a
    # bypass of the client-side check.
    if not form["company_name"] or not form["district_id"]:
        opts = await _reg_options()
        return HTMLResponse(
            _register_html(
                token, identity.get("customer_id") or "", opts,
                error="Please fill the required fields — company name, state and district.",
            ),
            status_code=400,
        )
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
    memory = get_memory(request)
    identity = await memory.get_employer_identity(token) if token else None
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    opts = await _job_options()
    employer = await memory.get_employer(identity.get("customer_id") or "", tenant_id=identity["tenant_id"])
    pe = (employer or {}).get("private_employers") or {}
    return HTMLResponse(_post_job_html(
        token, opts, company_address=_company_address(employer),
        company_district_id=pe.get("districtId") or "",
    ))


def _company_address(employer: dict[str, Any] | None) -> str:
    """A one-line display of the employer's registered address (for the
    'Company Address' job-location option)."""
    pe = (employer or {}).get("private_employers") or {}
    parts = [pe.get("address"), pe.get("city"), pe.get("pincode")]
    return ", ".join(str(p).strip() for p in parts if p and str(p).strip())


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
        "intern_payment_type": one("intern_payment_type"),
        "intern_stipend": one("intern_stipend"),
        "training_fee": one("training_fee"),
        "intern_duration_months": one("intern_duration_months"),
        "salary_period": one("salary_period"),
        "salary_min": one("salary_min"),
        "salary_max": one("salary_max"),
        "vacancies": one("vacancies"),
        # Job Location (work mode is derived from the location type)
        "job_location_type": one("job_location_type"),
        "state_id": one("state_id"),
        "district_id": one("district_id"),
        "city": one("city"),
        # Candidate Location Preference (where candidates should be from)
        "preferred_state_id": one("preferred_state_id"),
        "preferred_district_ids": many("preferred_district_ids"),
        # Apply Methods
        "apply_modes": many("apply_modes"),
        "contact_phone": one("contact_phone"),
        "contact_whatsapp": one("contact_whatsapp"),
    }
    # "Company Address" → the job's location IS the employer's registered
    # location, so fill it from the company profile (the form hides those inputs).
    pe = employer.get("private_employers") or {}
    if form["job_location_type"] == "COMPANY_ADDRESS":
        form["district_id"] = pe.get("districtId") or ""
        form["city"] = pe.get("city") or ""
    # private_jobs.districtId is NOT NULL, but a Remote job (or any unset pick)
    # has no district — fall back to the employer's registered district so the
    # row is always storable. The job stays flagged Remote via jobLocationType.
    if not form.get("district_id"):
        form["district_id"] = pe.get("districtId") or ""

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
  .opts label.opt{display:flex;align-items:center;gap:11px;border:1.5px solid #2a3942;border-radius:12px;
        padding:13px 14px;margin:9px 0;cursor:pointer;transition:all .12s}
  .opts label.opt input{width:18px;height:18px;accent-color:#00a884;flex:0 0 auto}
  .opts label.opt:has(input:checked){border-color:#00a884;background:rgba(0,168,132,.14)}
  .opts label.opt:has(input:checked) span{color:#00d3a7;font-weight:600}
  .subblock{background:#0e2a25;border:1px solid #1f4d44;border-radius:12px;padding:4px 14px 14px;margin:10px 0}
  .subblock > label:first-child{color:#9ad9cb}
  .addrbox{border:1px solid #2a3942;border-radius:10px;padding:13px 14px;margin:8px 0;
        background:#202c33;color:#cfd8dc;font-size:14px;line-height:1.4}
  .qs{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0}
  .qspill{border:1px solid #2a3942;background:#202c33;color:#e9edef;border-radius:20px;
        padding:9px 14px;font-size:14px;cursor:pointer}
  .qspill.on{border-color:#00a884;background:rgba(0,168,132,.16);color:#00d3a7;font-weight:600}
  .chips2{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
  .chip2{background:rgba(0,168,132,.16);border:1px solid #00a884;color:#9ce6d4;border-radius:16px;
        padding:6px 11px;font-size:13px}
  .chip2 b{cursor:pointer;color:#00d3a7;margin-left:5px;font-weight:700}
  .info{background:rgba(0,168,132,.10);border:1px solid #1f4d44;border-radius:10px;
        padding:11px 13px;margin-top:12px;color:#9ad9cb;font-size:13px}
  [hidden]{display:none !important}
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


def _pill_radios(name: str, options: "tuple | list") -> str:
    """Pill-style single-choice radio group (horizontal chips)."""
    out = []
    for val, lab in options:
        out.append(
            f'<label class="chip"><input type="radio" name="{name}" value="{_esc(val)}">'
            f"<span>{_esc(lab)}</span></label>"
        )
    return f'<div class="chips">{"".join(out)}</div>'


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


def _year_select(name: str) -> str:
    """A 0-15 'years' dropdown (used for the Experienced min/max year pickers)."""
    opts = "".join(
        f'<option value="{i}">{i} year{"" if i == 1 else "s"}</option>' for i in range(0, 16)
    )
    return f'<select name="{name}" autocomplete="off"><option value="">Select…</option>{opts}</select>'


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


def _register_html(
    token: str, phone: str, o: dict[str, list[dict[str, Any]]], *, error: str = "",
) -> str:
    err = f'<div class="addrbox" style="border-color:#a33;color:#ffb3b3">{_esc(error)}</div>' if error else ""
    inner = f"""
<h1>Register your company</h1>
<p class="sub">A few details about your business so candidates know who's hiring.</p>
{err}
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
    <div><label>State <span class="req">*</span></label>
      <select name="state_id" id="state_id" autocomplete="off" required>{_options_html(o['states'], placeholder='Select state…')}</select></div>
    <div><label>District <span class="req">*</span></label>
      <select name="district_id" id="district_id" autocomplete="off" required><option value="">Select a state first…</option></select></div>
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
    "Job Details", "Experience & Salary", "Job Location",
    "Candidate Location", "Apply Methods",
)
# A few major Tamil Nadu districts used by the "Top Cities" quick-select.
_TOP_DISTRICT_NAMES = (
    "Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli",
    "Tirunelveli", "Erode", "Tiruppur",
)


def _post_job_html(
    token: str, o: dict[str, list[dict[str, Any]]], *,
    company_address: str = "", company_district_id: str = "",
) -> str:
    n = len(_JOB_STEPS)
    addr_display = _esc(company_address) or "Your registered company address will be used."
    steps = "\n".join([
        # 1 — Job Details
        f"""<section class="step"><h1>Job Details</h1>
  <label>Job Title <span class="req">*</span></label>
  <input name="title" required placeholder="e.g. Software Developer">
  <label>Job Type</label>
  {_radios("job_type", JOB_TYPES)}
  <label>Description</label>
  <textarea name="description" placeholder="Role responsibilities, requirements…"></textarea>
</section>""",
        # 2 — Experience & Salary (conditional: years for Experienced, payment for Intern)
        f"""<section class="step"><h1>Experience &amp; Salary</h1>
  <label>Experience Required</label>
  {_radios("experience_type", EXPERIENCE_TYPES)}

  <div class="subblock" id="expYears" hidden>
    <label>Years of Experience</label>
    <div class="row">
      <div><label>Min years</label>{_year_select("experience_min")}</div>
      <div><label>Max years</label>{_year_select("experience_max")}</div>
    </div>
  </div>

  <div class="subblock" id="internBlock" hidden>
    <label>Intern Payment Type</label>
    {_radios("intern_payment_type", INTERN_PAYMENT_TYPES)}
    <div id="stipendInput" hidden>
      <label>Monthly Stipend (₹)</label>
      <input name="intern_stipend" inputmode="numeric" placeholder="e.g. 10000">
    </div>
    <div id="trainingInput" hidden>
      <label>Training Fee (₹)</label>
      <input name="training_fee" inputmode="numeric" placeholder="e.g. 25000">
    </div>
    <label>Duration (months)</label>
    <input name="intern_duration_months" inputmode="numeric" placeholder="e.g. 6">
  </div>

  <div id="salaryBlock">
    <label>Salary Range</label>
    {_radios("salary_period", SALARY_PERIODS)}
    <div class="row">
      <div><label>Min (₹)</label><input name="salary_min" inputmode="numeric" placeholder="Min"></div>
      <div><label>Max (₹)</label><input name="salary_max" inputmode="numeric" placeholder="Max"></div>
    </div>
  </div>

  <label>Number of Vacancies</label>
  <input name="vacancies" inputmode="numeric" placeholder="e.g. 5">
</section>""",
        # 3 — Job Location (conditional: Specific → state+district, Company → address)
        f"""<section class="step"><h1>Job Location</h1>
  <label>Job Location <span class="req">*</span></label>
  {_pill_radios("job_location_type", JOB_LOCATION_TYPES)}

  <div class="subblock" id="locSpecific" hidden>
    <label>State</label>
    <select name="state_id" id="job_state_id" autocomplete="off">{_options_html(o['states'], placeholder='Select a state…')}</select>
    <label>District</label>
    <select name="district_id" id="job_district_id" autocomplete="off"><option value="">Select a state first…</option></select>
    <label>City / area</label>
    <input name="city" placeholder="Street, area, landmark">
  </div>

  <div class="subblock" id="locCompany" hidden>
    <label>Company address</label>
    <div class="addrbox">🏢 {addr_display}</div>
  </div>

  <div class="subblock" id="locRemote" hidden>
    <div class="addrbox">🌐 This is a remote job — candidates can work from anywhere.</div>
  </div>
</section>""",
        # 4 — Candidate Location Preference (quick-select + district chips + credits)
        f"""<section class="step"><h1>Candidate Location Preference</h1>
  <p class="sub">Select where candidates should be from.</p>
  <label>Quick Select</label>
  <div class="qs">
    <button type="button" class="qspill" data-qs="company">🏢 Company District</button>
    <button type="button" class="qspill" data-qs="nearby">📍 Nearby</button>
    <button type="button" class="qspill" data-qs="all">▦ All Districts</button>
    <button type="button" class="qspill" data-qs="top">🏙 Top Cities</button>
    <button type="button" class="qspill" data-qs="custom">⚙ Custom</button>
  </div>
  <label>State</label>
  <select id="pref_state" autocomplete="off">{_options_html(o['states'], placeholder='Select a state…')}</select>
  <input type="hidden" name="preferred_state_id" id="pref_state_hidden">
  <label>Districts <span class="req">*</span></label>
  <select id="pref_add" autocomplete="off"><option value="">+ Add a district…</option></select>
  <div class="chips2" id="pref_chips"></div>
  <div id="pref_hidden"></div>
  <div class="info" id="creditInfo">0 districts selected = 0 credits per 15 days</div>
</section>""",
        # 5 — Apply Methods (In-App default; Phone/WhatsApp reveal a contact input)
        f"""<section class="step"><h1>Apply Methods</h1>
  <p class="sub">Choose how candidates can reach you for this job.</p>
  <div class="opts">
    <label class="opt"><input type="checkbox" name="apply_modes" value="APPLY" id="am_apply" checked><span>📲 In-App Apply</span></label>
    <label class="opt"><input type="checkbox" name="apply_modes" value="CALL" id="am_call"><span>📞 Phone Call</span></label>
    <label class="opt"><input type="checkbox" name="apply_modes" value="WHATSAPP" id="am_wa"><span>💬 WhatsApp</span></label>
  </div>
  <div id="phoneInput" hidden>
    <label>Contact Phone Number</label>
    <input name="contact_phone" inputmode="numeric" placeholder="e.g. 9876543210">
  </div>
  <div id="waInput" hidden>
    <label>WhatsApp Number</label>
    <input name="contact_whatsapp" inputmode="numeric" placeholder="e.g. 9876543210">
  </div>
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
  return true;
}}
next.onclick = function(){{ if(valid()){{ cur=Math.min(cur+1,steps.length-1); render(); }} }};
back.onclick = function(){{ cur=Math.max(cur-1,0); render(); }};

// --- Experience & Salary conditional fields ---
function picked(name){{ var el=document.querySelector('input[name="'+name+'"]:checked'); return el?el.value:''; }}
function toggleEl(id,on){{ var e=document.getElementById(id); if(e) e.hidden=!on; }}
function expChange(){{
  var v=picked('experience_type');
  toggleEl('expYears', v==='EXPERIENCED');   // year selectors only for Experienced
  toggleEl('internBlock', v==='INTERN');     // intern payment only for Intern
  toggleEl('salaryBlock', v!=='INTERN');     // salary shown by default; hidden for Intern
}}
function internChange(){{
  var p=picked('intern_payment_type');
  toggleEl('stipendInput', p==='STIPEND');
  toggleEl('trainingInput', p==='TRAINING_FEE');
}}
[].forEach.call(document.querySelectorAll('input[name=experience_type]'), function(r){{ r.addEventListener('change', expChange); }});
[].forEach.call(document.querySelectorAll('input[name=intern_payment_type]'), function(r){{ r.addEventListener('change', internChange); }});
expChange(); internChange();

// --- Job Location conditional + State→District cascade ---
var JOB_DISTRICTS = {_js_rows(o['districts'])};
function fillJobDistricts(){{
  var st = document.getElementById('job_state_id').value;
  var c = document.getElementById('job_district_id');
  var rows = JOB_DISTRICTS.filter(function(r){{ return String(r[2]) === String(st); }});
  c.innerHTML = '<option value="">Select district…</option>' +
    rows.map(function(r){{ return '<option value="'+r[0]+'">'+r[1]+'</option>'; }}).join('');
}}
function locChange(){{
  var v = picked('job_location_type');
  toggleEl('locSpecific', v==='SPECIFIC');     // state then district
  toggleEl('locCompany', v==='COMPANY_ADDRESS'); // show the company address
  toggleEl('locRemote', v==='REMOTE');           // remote note
}}
document.getElementById('job_state_id').addEventListener('change', fillJobDistricts);
[].forEach.call(document.querySelectorAll('input[name=job_location_type]'), function(r){{ r.addEventListener('change', locChange); }});
locChange();

// --- Candidate Location Preference: quick-select + district chips + credits ---
var COMPANY_DISTRICT = "{_esc(company_district_id)}";
var TOP_NAMES = {"[" + ", ".join(f'"{_esc(t)}"' for t in _TOP_DISTRICT_NAMES) + "]"};
var prefState = document.getElementById('pref_state');
var prefAdd = document.getElementById('pref_add');
var prefChips = document.getElementById('pref_chips');
var prefHidden = document.getElementById('pref_hidden');
var prefStateHidden = document.getElementById('pref_state_hidden');
var creditInfo = document.getElementById('creditInfo');
var prefSelected = [];
function dName(id){{ var d=JOB_DISTRICTS.filter(function(r){{return r[0]===id;}})[0]; return d?d[1]:id; }}
function dState(id){{ var d=JOB_DISTRICTS.filter(function(r){{return r[0]===id;}})[0]; return d?d[2]:''; }}
function fillPrefAdd(){{
  var st=prefState.value;
  var rows=JOB_DISTRICTS.filter(function(r){{ return String(r[2])===String(st) && prefSelected.indexOf(r[0])<0; }});
  prefAdd.innerHTML='<option value="">+ Add a district…</option>'+
    rows.map(function(r){{ return '<option value="'+r[0]+'">'+r[1]+'</option>'; }}).join('');
}}
function renderPref(){{
  prefChips.innerHTML=prefSelected.map(function(id){{ return '<span class="chip2">'+dName(id)+' <b data-id="'+id+'">×</b></span>'; }}).join('');
  prefHidden.innerHTML=prefSelected.map(function(id){{ return '<input type="hidden" name="preferred_district_ids" value="'+id+'">'; }}).join('');
  prefStateHidden.value=prefState.value;
  var nd=prefSelected.length;
  creditInfo.textContent=nd+' district'+(nd===1?'':'s')+' selected = '+nd+' credit'+(nd===1?'':'s')+' per 15 days';
  [].forEach.call(prefChips.querySelectorAll('b'), function(b){{ b.onclick=function(){{ var id=b.getAttribute('data-id'); prefSelected=prefSelected.filter(function(x){{return x!==id;}}); fillPrefAdd(); renderPref(); }}; }});
  fillPrefAdd();
}}
prefAdd.addEventListener('change', function(){{ if(prefAdd.value && prefSelected.indexOf(prefAdd.value)<0){{ prefSelected.push(prefAdd.value); renderPref(); }} }});
prefState.addEventListener('change', function(){{ renderPref(); }});
function quickSelect(mode){{
  if(mode==='company'||mode==='nearby'){{ if(COMPANY_DISTRICT){{ prefState.value=dState(COMPANY_DISTRICT)||prefState.value; prefSelected=[COMPANY_DISTRICT]; }} }}
  else if(mode==='all'){{ prefSelected=JOB_DISTRICTS.filter(function(r){{return String(r[2])===String(prefState.value);}}).map(function(r){{return r[0];}}); }}
  else if(mode==='top'){{ prefSelected=JOB_DISTRICTS.filter(function(r){{ return String(r[2])===String(prefState.value) && TOP_NAMES.indexOf(r[1])>=0; }}).map(function(r){{return r[0];}}); }}
  // 'custom' → leave the current selection for manual editing
  renderPref();
}}
[].forEach.call(document.querySelectorAll('.qspill'), function(b){{ b.onclick=function(){{
  [].forEach.call(document.querySelectorAll('.qspill'), function(x){{x.classList.remove('on');}});
  b.classList.add('on'); quickSelect(b.getAttribute('data-qs'));
}}; }});
if(COMPANY_DISTRICT){{ prefState.value=dState(COMPANY_DISTRICT)||prefState.value; }}
renderPref();

// --- Apply Methods: Phone/WhatsApp reveal a contact input ---
function applyChange(){{
  toggleEl('phoneInput', document.getElementById('am_call').checked);
  toggleEl('waInput', document.getElementById('am_wa').checked);
}}
['am_call','am_wa'].forEach(function(id){{ document.getElementById(id).addEventListener('change', applyChange); }});
applyChange();

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
