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
import json
import re
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.deps import get_memory
from app.chatbot import wa_format as wa
from app.config import get_settings
from app.core.logging import get_logger
from app.credits import (
    CREDIT_TYPE_ICONS,
    CREDIT_TYPE_LABELS,
    INDIVIDUAL_PACKS,
    JOB_CREDIT_PRICE,
    VALIDITY_OPTIONS,
    WELCOME_JOB_CREDITS,
    credit_quote,
    credits_required,
    find_pack,
)
from app.db.repositories import (
    CreditBundleRepository,
    CreditWalletRepository,
    LookupRepository,
    SubscriptionPlanRepository,
)
from app.payments import razorpay as rzp
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
from app.validation import validate_employer, validate_job_post, validate_kyc
from app.whatsapp import delivery as wa_delivery

router = APIRouter()
log = get_logger("employer")

# Employer menu buttons pushed after an action (reply buttons cap at 3).
_EMP_MENU = (("emp:post", "Post a Job"), ("emp:candidates", "View Candidates"),
             ("emp:myjobs", "My Jobs"))
# Full hub as a LIST (up to 10 rows) so the wallet / credits options fit too.
_EMP_MENU_ROWS = (
    {"id": "emp:post", "title": "Post a Job", "description": "Create a new job listing"},
    {"id": "emp:candidates", "title": "View Candidates", "description": "Browse matching candidates"},
    {"id": "emp:myjobs", "title": "My Jobs", "description": "Your posted jobs"},
    {"id": "emp:wallet", "title": "🪪 Credits & Wallet", "description": "Manage your credits"},
    {"id": "emp:buy", "title": "💳 Buy Credits", "description": "View pricing and bundles"},
)


def _emp_menu_list(body: str) -> dict[str, Any]:
    return wa.list_message(body=body, button_text="Menu",
                           rows=list(_EMP_MENU_ROWS), section_title="Employer menu")


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


def _district_names(ids: list[str], *, opts: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Map selected candidate-district ids → their display names (for the Activate
    screen header). Unknown ids fall back to the id so the count is never wrong."""
    by_id = {str(d.get("id")): d.get("name") for d in (opts.get("districts") or [])}
    return [by_id.get(str(i)) or str(i) for i in ids]


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
    # SERVER-SIDE VALIDATION: company name + district required (district is the
    # job-location fallback — private_jobs.districtId is NOT NULL), email/website/
    # PIN validated when filled. Re-show the form with the error on any failure.
    errors = validate_employer(form)
    if errors:
        opts = await _reg_options()
        return HTMLResponse(
            _register_html(token, identity.get("customer_id") or "", opts,
                           error=next(iter(errors.values()))),
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

    # SERVER-SIDE VALIDATION: a document type + link are required; GST/PAN
    # validated when filled. Re-show the KYC form with the error.
    kyc_form_data = {
        "kyc_document_type": one("kyc_document_type"),
        "kyc_document_url": one("kyc_document_url"),
        "gst_number": one("gst_number"),
        "pan_number": one("pan_number"),
    }
    errors = validate_kyc(kyc_form_data)
    if errors:
        return HTMLResponse(_kyc_html(token, error=next(iter(errors.values()))), status_code=400)

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
                "✅ Business verified! You're all set.\n\nWhat would you like to do? "
                "Tap *Menu* to post a job, view candidates, or buy credits."
            )
            payload = _emp_menu_list(body)
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
    # NB: never backfill for a Specific Location — there the district is the
    # employer's explicit choice and a blank one must fail validation, not be
    # silently replaced by the company district.
    if form["job_location_type"] != "SPECIFIC" and not form.get("district_id"):
        form["district_id"] = pe.get("districtId") or ""

    # SERVER-SIDE VALIDATION: title required; salaries / vacancies / contacts /
    # intern amounts validated (and salary max ≥ min). Re-show the wizard on error.
    errors = validate_job_post(form)
    if errors:
        opts = await _job_options()
        return HTMLResponse(
            _post_job_html(token, opts, company_address=_company_address(employer),
                           company_district_id=pe.get("districtId") or "",
                           error=next(iter(errors.values()))),
            status_code=400,
        )

    # Build the DB-ready private_jobs record (status DRAFT — not posted until the
    # employer activates it with credits) and stage it as a draft keyed by token.
    # No live private_jobs write. The job is finalized on /post-job/activate.
    job = build_job_record(employer_id=employer_id, form=form, status="DRAFT")
    await memory.stage_job_draft(token, job)

    # 'Have' = the employer's REAL live job-credit balance, mirrored into Redis on
    # first read (then debited there). New/test employers with no live wallet get
    # the welcome grant. Reading live billing is safe (read-only).
    live = await CreditWalletRepository.balances(employer_id)
    seed = live if live is not None else {"job": WELCOME_JOB_CREDITS, "unlock": 0, "boost": 0}
    bal = await memory.ensure_wallet(phone, tenant_id=identity["tenant_id"], seed=seed,
                                     welcome=(live is None))
    have = bal["job"]

    district_names = _district_names(form.get("preferred_district_ids") or [], opts=await _job_options())
    settings = get_settings()
    return HTMLResponse(_activate_job_html(
        token, title=title, district_names=district_names, have=have,
        key_id=settings.razorpay_key_id, test_mode=settings.razorpay_test_mode,
        prefill_name=identity.get("name") or "", prefill_phone=re.sub(r"\D", "", phone)[-10:],
    ))


def _clean_validity(value: str) -> str:
    """Coerce a posted validity to a known option (15/30/45), default the first."""
    return value if value in {str(d) for d, _ in VALIDITY_OPTIONS} else str(VALIDITY_OPTIONS[0][0])


async def _job_credit_quote(memory, phone: str, tenant_id: str, job: dict[str, Any], validity: str):
    """(need, quote) for a staged job at a chosen validity, vs the live balance."""
    districts = (job.get("private_jobs") or {}).get("preferredDistrictIds") or []
    need = credits_required(len(districts), validity)
    have = int((await memory.get_employer(phone, tenant_id=tenant_id) or {}).get("walletJobCredits") or 0)
    return need, credit_quote(have, need)


async def _finalize_job(memory, phone: str, tenant_id: str, token: str, validity: str) -> dict | None:
    """Debit ``need`` job credits, flip the staged draft to PENDING (posted),
    append it to the employer, push the WhatsApp confirmation, and return a
    summary. Wallet top-up (the purchased credits) must already be applied."""
    job = await memory.get_job_draft(token)
    if job is None:
        return None
    districts = (job.get("private_jobs") or {}).get("preferredDistrictIds") or []
    need = credits_required(len(districts), validity)
    title = (job.get("private_jobs") or {}).get("title") or "your job"
    await memory.adjust_job_credits(phone, -need, tenant_id=tenant_id,
                                    description=f"Posted “{title}” ({validity} days)")
    job["private_jobs"]["status"] = "PENDING"
    job["private_jobs"]["validityDays"] = int(validity)
    job["creditsCharged"] = need
    job["validityDays"] = int(validity)
    await memory.add_employer_job(phone, job, tenant_id=tenant_id)
    summary = {"title": title, "ref": job.get("ref"), "validity": int(validity), "need": need}
    await memory.update_employer(phone, {"lastActivated": summary}, tenant_id=tenant_id)
    await memory.clear_job_draft(token)

    settings = get_settings()
    digits = re.sub(r"\D", "", phone)
    if digits:
        bal = (await memory.wallet_balances(phone, tenant_id=tenant_id))["job"]
        body = (
            f"✅ Job activated: *{title}* ({job['ref']}) for {validity} days.\n\n"
            f"💳 {need} job credit{'s' if need != 1 else ''} used — balance {bal}. What next?"
        )
        try:
            await wa_delivery.send_message(settings, digits, _emp_menu_list(body))
        except Exception as exc:  # noqa: BLE001
            log.warning("employer_postjob_push_failed", error=str(exc)[:200])
    return summary


def _job_activated_html(summary: dict | None) -> HTMLResponse:
    s = summary or {}
    title = s.get("title") or "Your job"
    ref = s.get("ref") or ""
    days = s.get("validity") or ""
    need = s.get("need") or 0
    number = re.sub(r"\D", "", get_settings().whatsapp_business_number or "")
    return HTMLResponse(_success_html(
        "Job activated!", f"“{title}” is live for {days} days ({ref}). "
        f"{need} credit{'s' if need != 1 else ''} used. Head back to WhatsApp to view candidates.",
        business_number=number,
    ))


@router.post("/post-job/activate", response_class=HTMLResponse)
async def post_job_activate(request: Request) -> HTMLResponse:
    """The 'Activate Now' path — used ONLY when the wallet already covers the cost
    (no payment). A shortfall goes through Razorpay (/post-job/credits/*)."""
    raw = parse_qs((await request.body()).decode("utf-8"))
    token = (raw.get("token") or [""])[0].strip()
    validity = _clean_validity((raw.get("validity_days") or [""])[0].strip())
    if not token:
        return HTMLResponse(_expired_html(), status_code=400)
    memory = get_memory(request)
    identity = await memory.get_employer_identity(token)
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    phone = identity.get("customer_id") or ""
    tenant_id = identity["tenant_id"]
    job = await memory.get_job_draft(token)
    if job is None:
        return HTMLResponse(_expired_html(), status_code=404)
    _, quote = await _job_credit_quote(memory, phone, tenant_id, job, validity)
    if not quote["sufficient"]:
        # Shouldn't happen (the button routes to Razorpay) — re-show the screen.
        return HTMLResponse(_page("Insufficient credits",
            '<div class="ok"><h1>Not enough credits</h1><p class="sub">Please go back '
            'and pay for the required credits.</p></div>'), status_code=402)
    summary = await _finalize_job(memory, phone, tenant_id, token, validity)
    return _job_activated_html(summary)


@router.post("/post-job/credits/order")
async def post_job_credits_order(request: Request) -> JSONResponse:
    """Create a Razorpay order for the JOB-CREDIT shortfall (buy × ₹649)."""
    raw = parse_qs((await request.body()).decode("utf-8"))
    token = (raw.get("token") or [""])[0].strip()
    validity = _clean_validity((raw.get("validity_days") or [""])[0].strip())
    memory = get_memory(request)
    identity = await memory.get_employer_identity(token) if token else None
    if not identity:
        return JSONResponse({"error": "expired"}, status_code=404)
    if not get_settings().razorpay_enabled:
        return JSONResponse({"error": "payments_unavailable"}, status_code=503)
    phone = identity.get("customer_id") or ""
    tenant_id = identity["tenant_id"]
    job = await memory.get_job_draft(token)
    if job is None:
        return JSONResponse({"error": "expired"}, status_code=404)
    _, quote = await _job_credit_quote(memory, phone, tenant_id, job, validity)
    if quote["sufficient"]:
        return JSONResponse({"sufficient": True})   # no payment needed
    try:
        order = await rzp.create_order(
            amount_paise=int(quote["pay"]) * 100,
            receipt=f"jobcr_{re.sub(r'[^0-9]', '', phone)[-10:]}",
            notes={"kind": "job_credits", "buy": quote["buy"], "phone": phone},
        )
    except rzp.RazorpayError as exc:
        log.warning("jobcredits_order_failed", error=str(exc)[:200])
        return JSONResponse({"error": "order_failed"}, status_code=502)
    await memory.stage_payment_order(order["id"], {
        "kind": "job_credits", "token": token, "validity": validity,
        "buy": quote["buy"], "phone": phone, "tenant_id": tenant_id,
    })
    return JSONResponse({
        "order_id": order["id"], "amount": order["amount"], "currency": order["currency"],
        "key_id": get_settings().razorpay_key_id, "buy": quote["buy"],
        "prefill_name": identity.get("name") or "", "prefill_phone": re.sub(r"\D", "", phone)[-10:],
    })


@router.post("/post-job/credits/verify")
async def post_job_credits_verify(request: Request) -> JSONResponse:
    """Verify the Razorpay payment, credit the purchased job credits, then
    activate the job (Redis test harness)."""
    raw = parse_qs((await request.body()).decode("utf-8"))

    def one(k: str) -> str:
        return (raw.get(k) or [""])[0].strip()

    order_id, payment_id, signature = one("razorpay_order_id"), one("razorpay_payment_id"), one("razorpay_signature")
    if not rzp.verify_payment_signature(order_id=order_id, payment_id=payment_id, signature=signature):
        return JSONResponse({"error": "bad_signature"}, status_code=400)
    memory = get_memory(request)
    pending = await memory.get_payment_order(order_id)
    if not pending or pending.get("kind") != "job_credits":
        return JSONResponse({"error": "unknown_order"}, status_code=404)
    phone, tenant_id = pending["phone"], pending["tenant_id"]
    # Credit the purchased shortfall, then activate (which debits the cost).
    await memory.adjust_job_credits(phone, int(pending["buy"]), tenant_id=tenant_id,
                                    description=f"Purchased {pending['buy']} job credit"
                                    f"{'s' if int(pending['buy']) != 1 else ''}")
    summary = await _finalize_job(memory, phone, tenant_id, pending["token"], pending["validity"])
    await memory.clear_payment_order(order_id)
    if summary is None:
        return JSONResponse({"error": "expired"}, status_code=404)
    return JSONResponse({"ok": True, "redirect": f"/employer/post-job/done?token={pending['token']}"})


@router.get("/post-job/done", response_class=HTMLResponse)
async def post_job_done(request: Request, token: str = Query(default="")) -> HTMLResponse:
    memory = get_memory(request)
    identity = await memory.get_employer_identity(token) if token else None
    employer = await memory.get_employer(identity.get("customer_id") or "", tenant_id=identity["tenant_id"]) if identity else None
    return _job_activated_html((employer or {}).get("lastActivated"))


# ---------------------------------------------------------------------------
# Subscription plans + Razorpay checkout
# ---------------------------------------------------------------------------

# billingCycle enum → (human label, validity days) for the plan card + expiry.
_BILLING = {
    "DAYS_15": ("15 days", 15), "DAYS_30": ("30 days", 30),
    "DAYS_90": ("90 days", 90), "YEARLY": ("1 year", 365),
}


@router.get("/subscribe", response_class=HTMLResponse)
async def subscribe_page(request: Request, token: str = Query(default="")) -> HTMLResponse:
    """Stage: the employer 'Upgrade Plan' page — the live subscription_plans with
    a Razorpay checkout (test mode)."""
    memory = get_memory(request)
    identity = await memory.get_employer_identity(token) if token else None
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    settings = get_settings()
    if not settings.razorpay_enabled:
        return HTMLResponse(_page("Payments unavailable",
            '<div class="ok"><h1>Payments not set up</h1><p class="sub">Razorpay '
            'keys are not configured. Please try again later.</p></div>'), status_code=503)
    plans = await SubscriptionPlanRepository.list_active()
    employer = await memory.get_employer(identity.get("customer_id") or "", tenant_id=identity["tenant_id"])
    current = (employer or {}).get("subscription") or {}
    return HTMLResponse(_subscribe_html(
        token, plans, key_id=settings.razorpay_key_id, test_mode=settings.razorpay_test_mode,
        prefill_name=identity.get("name") or "", prefill_phone=identity.get("customer_id") or "",
        current_type=current.get("planType") or "",
    ))


@router.post("/subscribe/order")
async def subscribe_order(request: Request) -> JSONResponse:
    """Create a Razorpay order for the chosen plan (price read SERVER-SIDE from
    the catalog — never trust a client amount). Free plan → activate immediately."""
    raw = parse_qs((await request.body()).decode("utf-8"))
    token = (raw.get("token") or [""])[0].strip()
    plan_id = (raw.get("plan_id") or [""])[0].strip()
    memory = get_memory(request)
    identity = await memory.get_employer_identity(token) if token else None
    if not identity:
        return JSONResponse({"error": "expired"}, status_code=404)
    plan = await SubscriptionPlanRepository.get(plan_id)
    if not plan:
        return JSONResponse({"error": "unknown_plan"}, status_code=400)
    phone = identity.get("customer_id") or ""
    tenant_id = identity["tenant_id"]
    price = float(plan.get("price") or 0)

    # Free plan: no payment — activate now and tell the client to redirect.
    if price <= 0:
        await _activate_subscription(memory, phone, tenant_id, plan)
        return JSONResponse({"free": True, "redirect": f"/employer/subscribe/done?token={token}"})

    try:
        order = await rzp.create_order(
            amount_paise=int(round(price * 100)),
            receipt=f"sub_{plan.get('type','')}_{re.sub(r'[^0-9]', '', phone)[-10:]}",
            notes={"plan_id": plan_id, "plan_type": plan.get("type", ""), "phone": phone},
        )
    except rzp.RazorpayError as exc:
        log.warning("subscribe_order_failed", error=str(exc)[:200])
        return JSONResponse({"error": "order_failed"}, status_code=502)

    await memory.stage_payment_order(order["id"], {
        "token": token, "plan_id": plan_id, "phone": phone, "tenant_id": tenant_id,
        "amount": order["amount"],
    })
    return JSONResponse({
        "order_id": order["id"], "amount": order["amount"], "currency": order["currency"],
        "key_id": get_settings().razorpay_key_id, "plan_name": plan.get("name", "Plan"),
        "prefill_name": identity.get("name") or "", "prefill_phone": re.sub(r"\D", "", phone)[-10:],
    })


@router.post("/subscribe/verify")
async def subscribe_verify(request: Request) -> JSONResponse:
    """Verify the Razorpay payment signature, then activate the subscription
    (Redis test harness) + grant the plan's monthly credits."""
    raw = parse_qs((await request.body()).decode("utf-8"))

    def one(k: str) -> str:
        return (raw.get(k) or [""])[0].strip()

    order_id, payment_id, signature = one("razorpay_order_id"), one("razorpay_payment_id"), one("razorpay_signature")
    if not rzp.verify_payment_signature(order_id=order_id, payment_id=payment_id, signature=signature):
        return JSONResponse({"error": "bad_signature"}, status_code=400)
    memory = get_memory(request)
    pending = await memory.get_payment_order(order_id)
    if not pending:
        return JSONResponse({"error": "unknown_order"}, status_code=404)
    plan = await SubscriptionPlanRepository.get(pending.get("plan_id"))
    if not plan:
        return JSONResponse({"error": "unknown_plan"}, status_code=400)
    await _activate_subscription(memory, pending["phone"], pending["tenant_id"], plan,
                                 payment_id=payment_id)
    await memory.clear_payment_order(order_id)
    return JSONResponse({"ok": True, "redirect": f"/employer/subscribe/done?token={pending.get('token','')}"})


@router.get("/subscribe/done", response_class=HTMLResponse)
async def subscribe_done(request: Request, token: str = Query(default="")) -> HTMLResponse:
    memory = get_memory(request)
    identity = await memory.get_employer_identity(token) if token else None
    name = re.sub(r"\D", "", get_settings().whatsapp_business_number or "")
    employer = await memory.get_employer(identity.get("customer_id") or "", tenant_id=identity["tenant_id"]) if identity else None
    sub = (employer or {}).get("subscription") or {}
    plan = sub.get("planName") or "your plan"
    return HTMLResponse(_success_html(
        "Subscription active!", f"You're now on the {plan} plan. Head back to WhatsApp to continue.",
        business_number=name,
    ))


async def _activate_subscription(memory, phone: str, tenant_id: str, plan: dict[str, Any],
                                 *, payment_id: str = "") -> None:
    """Record the plan on the employer record + grant its monthly credits, then
    push a WhatsApp confirmation. Redis-only (never the live subscriptions table)."""
    label, days = _BILLING.get(plan.get("billingCycle"), ("subscription", 30))
    sub = {
        "planId": plan.get("id"), "planType": plan.get("type"), "planName": plan.get("name"),
        "billingCycle": plan.get("billingCycle"), "validityLabel": label, "validityDays": days,
        "maxActiveJobs": plan.get("maxActiveJobs"), "maxLocationsPerJob": plan.get("maxLocationsPerJob"),
        "price": plan.get("price"), "paymentId": payment_id, "test": get_settings().razorpay_test_mode,
    }
    await memory.activate_subscription(
        phone, sub, grant_unlock=int(plan.get("monthlyCredits") or 0),
        grant_boost=int(plan.get("monthlyBoosts") or 0), tenant_id=tenant_id,
    )
    settings = get_settings()
    digits = re.sub(r"\D", "", phone)
    if digits:
        grants = []
        if plan.get("monthlyCredits"):
            grants.append(f"{plan['monthlyCredits']} unlock credits")
        if plan.get("monthlyBoosts"):
            grants.append(f"{plan['monthlyBoosts']} boosts")
        extra = (" You received " + " and ".join(grants) + ".") if grants else ""
        body = (
            f"✅ *{plan.get('name')}* plan activated ({label}).{extra}\n\nWhat next?"
        )
        try:
            await wa_delivery.send_message(settings, digits, _emp_menu_list(body))
        except Exception as exc:  # noqa: BLE001
            log.warning("subscribe_push_failed", error=str(exc)[:200])


# ---------------------------------------------------------------------------
# Buy Credits — bundles (credit_bundles) + individual credits + Razorpay
# ---------------------------------------------------------------------------


async def _resolve_buy_item(item: str) -> dict[str, Any] | None:
    """Resolve a Buy-Credits selection → {price, grant{job,unlock,boost}, label}.
    ``item`` is 'bundle:<id>' or 'pack:<TYPE>:<qty>'. Price comes from the
    SERVER (catalog), never the client."""
    if item.startswith("bundle:"):
        b = await CreditBundleRepository.get(item.split(":", 1)[1])
        if not b:
            return None
        return {
            "price": float(b.get("price") or 0), "label": f"{b.get('name')} bundle",
            "grant": {"job": int(b.get("jobCredits") or 0), "unlock": int(b.get("unlockCredits") or 0),
                      "boost": int(b.get("boostCredits") or 0)},
        }
    if item.startswith("pack:"):
        _, ctype, qty = (item.split(":") + ["", ""])[:3]
        pack = find_pack(ctype, qty)
        if not pack:
            return None
        bucket = {"JOB": "job", "UNLOCK": "unlock", "BOOST": "boost"}[pack["type"]]
        return {
            "price": float(pack["price"]),
            "label": f"{pack['credits']} {CREDIT_TYPE_LABELS.get(pack['type'], '')}".strip(),
            "grant": {"job": 0, "unlock": 0, "boost": 0, bucket: int(pack["credits"])},
        }
    return None


@router.get("/buy-credits", response_class=HTMLResponse)
async def buy_credits_page(request: Request, token: str = Query(default="")) -> HTMLResponse:
    """The 'Buy Credits' page: Bundles + Individual Credits tabs, current balance,
    Razorpay checkout (test mode)."""
    memory = get_memory(request)
    identity = await memory.get_employer_identity(token) if token else None
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    settings = get_settings()
    if not settings.razorpay_enabled:
        return HTMLResponse(_page("Payments unavailable",
            '<div class="ok"><h1>Payments not set up</h1><p class="sub">Razorpay keys '
            'are not configured.</p></div>'), status_code=503)
    phone = identity.get("customer_id") or ""
    employer = await memory.get_employer(phone, tenant_id=identity["tenant_id"])
    employer_id = (employer or {}).get("private_employers", {}).get("id")
    live = await CreditWalletRepository.balances(employer_id)
    seed = live if live is not None else {"job": WELCOME_JOB_CREDITS, "unlock": 0, "boost": 0}
    bal = await memory.ensure_wallet(phone, tenant_id=identity["tenant_id"], seed=seed,
                                     welcome=(live is None))
    bundles = await CreditBundleRepository.list_active()
    return HTMLResponse(_buy_credits_html(
        token, bundles=bundles, balance=bal, key_id=settings.razorpay_key_id,
        test_mode=settings.razorpay_test_mode,
        prefill_name=identity.get("name") or "", prefill_phone=re.sub(r"\D", "", phone)[-10:],
    ))


@router.post("/buy-credits/order")
async def buy_credits_order(request: Request) -> JSONResponse:
    """Create a Razorpay order for a bundle or individual-credit pack."""
    raw = parse_qs((await request.body()).decode("utf-8"))
    token = (raw.get("token") or [""])[0].strip()
    item = (raw.get("item") or [""])[0].strip()
    memory = get_memory(request)
    identity = await memory.get_employer_identity(token) if token else None
    if not identity:
        return JSONResponse({"error": "expired"}, status_code=404)
    if not get_settings().razorpay_enabled:
        return JSONResponse({"error": "payments_unavailable"}, status_code=503)
    resolved = await _resolve_buy_item(item)
    if not resolved or resolved["price"] <= 0:
        return JSONResponse({"error": "unknown_item"}, status_code=400)
    phone, tenant_id = identity.get("customer_id") or "", identity["tenant_id"]
    try:
        order = await rzp.create_order(
            amount_paise=int(round(resolved["price"] * 100)),
            receipt=f"buy_{re.sub(r'[^0-9]', '', phone)[-10:]}",
            notes={"item": item, "phone": phone},
        )
    except rzp.RazorpayError as exc:
        log.warning("buy_credits_order_failed", error=str(exc)[:200])
        return JSONResponse({"error": "order_failed"}, status_code=502)
    await memory.stage_payment_order(order["id"], {
        "kind": "buy_credits", "token": token, "phone": phone, "tenant_id": tenant_id,
        "grant": resolved["grant"], "label": resolved["label"],
    })
    return JSONResponse({
        "order_id": order["id"], "amount": order["amount"], "currency": order["currency"],
        "key_id": get_settings().razorpay_key_id, "label": resolved["label"],
        "prefill_name": identity.get("name") or "", "prefill_phone": re.sub(r"\D", "", phone)[-10:],
    })


@router.post("/buy-credits/verify")
async def buy_credits_verify(request: Request) -> JSONResponse:
    """Verify the Razorpay payment, then grant the purchased credits (Redis)."""
    raw = parse_qs((await request.body()).decode("utf-8"))

    def one(k: str) -> str:
        return (raw.get(k) or [""])[0].strip()

    order_id, payment_id, signature = one("razorpay_order_id"), one("razorpay_payment_id"), one("razorpay_signature")
    if not rzp.verify_payment_signature(order_id=order_id, payment_id=payment_id, signature=signature):
        return JSONResponse({"error": "bad_signature"}, status_code=400)
    memory = get_memory(request)
    pending = await memory.get_payment_order(order_id)
    if not pending or pending.get("kind") != "buy_credits":
        return JSONResponse({"error": "unknown_order"}, status_code=404)
    g = pending.get("grant") or {}
    phone, tenant_id = pending["phone"], pending["tenant_id"]
    bal = await memory.grant_credits(phone, tenant_id=tenant_id,
                                     job=int(g.get("job", 0)), unlock=int(g.get("unlock", 0)),
                                     boost=int(g.get("boost", 0)),
                                     description=f"Purchased {pending.get('label', 'credits')}")
    await memory.clear_payment_order(order_id)
    settings = get_settings()
    digits = re.sub(r"\D", "", phone)
    if digits and bal is not None:
        body = (
            f"✅ Purchase complete: *{pending.get('label')}*.\n\n"
            f"💳 Balance — 💼 {bal['job']} job · 🔓 {bal['unlock']} unlock · 🚀 {bal['boost']} boost. What next?"
        )
        try:
            await wa_delivery.send_message(settings, digits, _emp_menu_list(body))
        except Exception as exc:  # noqa: BLE001
            log.warning("buy_credits_push_failed", error=str(exc)[:200])
    return JSONResponse({"ok": True, "redirect": f"/employer/buy-credits?token={pending.get('token','')}"})


@router.get("/wallet", response_class=HTMLResponse)
async def wallet_page(request: Request, token: str = Query(default="")) -> HTMLResponse:
    """Credits & Wallet (Credit History): the balance card + transaction list."""
    memory = get_memory(request)
    identity = await memory.get_employer_identity(token) if token else None
    if not identity:
        return HTMLResponse(_expired_html(), status_code=404)
    phone, tenant_id = identity.get("customer_id") or "", identity["tenant_id"]
    employer = await memory.get_employer(phone, tenant_id=tenant_id)
    employer_id = (employer or {}).get("private_employers", {}).get("id")
    live = await CreditWalletRepository.balances(employer_id)
    seed = live if live is not None else {"job": WELCOME_JOB_CREDITS, "unlock": 0, "boost": 0}
    bal = await memory.ensure_wallet(phone, tenant_id=tenant_id, seed=seed, welcome=(live is None))
    ledger = await memory.wallet_ledger(phone, tenant_id=tenant_id)
    return HTMLResponse(_wallet_html(token, balance=bal, ledger=ledger))


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
  /* inline validation errors (red border + message below the field) */
  input.invalid,select.invalid,textarea.invalid{border-color:#ff5b5b !important;
        box-shadow:0 0 0 1px rgba(255,91,91,.35)}
  .field-err{color:#ff7a7a;font-size:12.5px;margin:6px 0 2px;line-height:1.3}
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
  /* Activate Job screen */
  .jobhdr{background:linear-gradient(135deg,#5b3df5,#7b5cff);border-radius:14px;padding:16px 16px 14px;margin-bottom:14px}
  .jobttl{font-size:18px;font-weight:700;color:#fff}
  .jobsub{color:#dcd6ff;font-size:13px;margin-top:6px}
  .achips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
  .achip{background:rgba(255,255,255,.18);color:#fff;border-radius:20px;padding:4px 11px;font-size:12px}
  .acard{background:#111b21;border:1px solid #2a3942;border-radius:12px;padding:14px;margin-bottom:12px}
  .acard>label{margin:0 0 10px;color:#e9edef;font-size:14px;font-weight:600}
  .vpills{display:flex;gap:8px}
  .vpill{flex:1;padding:12px 0;border:1.5px solid #2a3942;border-radius:10px;background:#202c33;
         color:#e9edef;font-size:14px;font-weight:600;cursor:pointer;display:flex;flex-direction:column;align-items:center}
  .vpill small{color:#8696a0;font-weight:400;font-size:11px;margin-top:2px}
  .vpill.on{background:#5b3df5;border-color:#5b3df5;color:#fff}
  .vpill.on small{color:#dcd6ff}
  .vnote{color:#8696a0;font-size:12px;margin:10px 0 0;text-align:center}
  .calc{display:flex;align-items:center;justify-content:space-around;background:#202c33;border-radius:10px;padding:12px 8px}
  .calc>div{display:flex;flex-direction:column;align-items:center}
  .calc b{font-size:22px} .calc small{color:#8696a0;font-size:11px;margin-top:2px}
  .calc .op{color:#6b7d88;font-size:16px;font-weight:400} .calc .accent{color:#a78bfa}
  .boxes{display:flex;gap:8px;margin-top:12px}
  .box{flex:1;border-radius:10px;padding:10px 0;text-align:center;border:1px solid #2a3942}
  .box small{display:block;color:#8696a0;font-size:11px} .box b{font-size:18px}
  .box.have{background:rgba(0,168,132,.12)} .box.have b{color:#00d3a7}
  .box.need{background:rgba(167,139,250,.12)} .box.need b{color:#a78bfa}
  .box.buy{background:rgba(255,107,107,.12)} .box.buy b{color:#ff6b6b}
  .box.buy.zero{background:rgba(0,168,132,.12)} .box.buy.zero b{color:#00d3a7}
  .bar{height:7px;background:#2a3942;border-radius:6px;overflow:hidden;margin-top:12px}
  .bar span{display:block;height:100%;background:#5b3df5;width:0;transition:width .2s}
  .tc{display:flex;align-items:center;gap:9px;color:#aebac1;font-size:12px;margin-top:14px}
  .tc input{width:auto;transform:scale(1.3);accent-color:#5b3df5}
  /* Subscription plans */
  .testbadge{font-size:11px;background:#3a2f00;color:#ffcf33;border:1px solid #6b5a00;
        border-radius:6px;padding:2px 7px;vertical-align:middle;margin-left:6px}
  .plan{display:block;border:1.5px solid #2a3942;border-radius:12px;padding:13px 14px;margin:10px 0;cursor:pointer}
  .plan.on{border-color:#5b3df5;background:rgba(91,61,245,.08)}
  .prow{display:flex;align-items:center;gap:9px}
  .prow input{width:auto;accent-color:#5b3df5} .prow b{font-size:15px;flex:0 0 auto}
  .pprice{margin-left:auto;font-size:17px;font-weight:700;color:#fff;text-align:right}
  .pprice small{display:block;color:#8696a0;font-size:11px;font-weight:400}
  .pfeats{display:flex;flex-wrap:wrap;gap:6px;margin-top:9px}
  .pfeat{background:#202c33;color:#aebac1;border-radius:16px;padding:3px 10px;font-size:12px}
  .curtag{font-size:11px;color:#00d3a7;font-weight:400} .poptag{font-size:11px;color:#ffb020;margin-left:4px}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  /* Buy Credits */
  .balcard{background:linear-gradient(135deg,#5b3df5,#7b5cff);border-radius:14px;padding:15px 16px;margin-bottom:14px}
  .ballbl{color:#e3ddff;font-size:13px;margin-bottom:10px}
  .balrow{display:flex;justify-content:space-around;text-align:center}
  .balrow b{font-size:20px;color:#fff;display:block} .balrow small{color:#dcd6ff;font-size:12px}
  .tabs{display:flex;background:#202c33;border-radius:10px;padding:4px;margin-bottom:14px}
  .tab{flex:1;padding:10px 0;border:0;border-radius:8px;background:transparent;color:#aebac1;
       font-size:14px;font-weight:600;cursor:pointer}
  .tab.on{background:#5b3df5;color:#fff}
  .sechdr{color:#cdb6ff;font-size:13px;font-weight:700;margin:16px 2px 8px}
  .savetag{font-size:10px;background:#063a2c;color:#39e6a8;border-radius:10px;padding:2px 8px;margin-left:6px}
  .buyitem.on{border-color:#5b3df5;background:rgba(91,61,245,.08)}
  /* Credits & Wallet (Credit History) */
  .wallethdr{background:linear-gradient(135deg,#5b3df5,#7b5cff);border-radius:16px;padding:16px;margin-bottom:14px}
  .wrow{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
  .wlbl{color:#fff;font-size:15px;font-weight:600}
  .recharge{background:rgba(255,255,255,.2);color:#fff;border-radius:20px;padding:7px 14px;
        font-size:13px;text-decoration:none;font-weight:600}
  .filters{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
  .filt{border:1px solid #2a3942;background:#202c33;color:#aebac1;border-radius:20px;
        padding:8px 14px;font-size:13px;cursor:pointer}
  .filt.on{background:#5b3df5;border-color:#5b3df5;color:#fff;font-weight:600}
  .txrow{display:flex;align-items:flex-start;gap:11px;background:#111b21;border:1px solid #2a3942;
        border-radius:12px;padding:13px;margin-bottom:10px}
  .txicon{width:38px;height:38px;border-radius:50%;background:rgba(0,168,132,.14);
        display:flex;align-items:center;justify-content:center;font-size:18px;flex:0 0 auto}
  .txmid{flex:1;min-width:0} .txmid b{font-size:14px} .txmid p{margin:3px 0;color:#aebac1;font-size:13px}
  .txmid small{color:#6b7d88;font-size:12px}
  .txamt{text-align:right;flex:0 0 auto} .txcredit{color:#00d3a7;font-weight:700;font-size:16px}
  .txdebit{color:#ff7a7a;font-weight:700;font-size:16px}
  .txbal{display:block;margin-top:6px;background:#202c33;color:#8696a0;border-radius:8px;
        padding:2px 8px;font-size:11px}
"""


def _page(title: str, inner: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{_STYLE}</style></head>"
        f'<body><div class="card">{inner}</div></body></html>'
    )


# Drop-in inline validator for the NATIVE single-page forms (register / KYC):
# disables the browser's popup bubbles (novalidate) and instead shows a red border
# + a message under each bad field, driven by the existing required/pattern/type
# attributes. Plain string (not an f-string) so its braces stay literal.
_FORM_VALIDATE_JS = r"""<script>
(function(){
  function wire(form){
    form.setAttribute('novalidate','novalidate');
    form.addEventListener('submit', function(e){
      [].forEach.call(form.querySelectorAll('.field-err'), function(x){ x.parentNode.removeChild(x); });
      [].forEach.call(form.querySelectorAll('.invalid'), function(x){ x.classList.remove('invalid'); });
      var first=null;
      function err(el,msg){ el.classList.add('invalid'); var d=document.createElement('div');
        d.className='field-err'; d.textContent=msg; el.insertAdjacentElement('afterend', d); if(!first) first=el; }
      [].forEach.call(form.querySelectorAll('input,select,textarea'), function(el){
        if(el.type==='hidden'||el.readOnly||el.disabled||el.offsetParent===null) return;
        var v=(el.value||'').trim();
        if(el.hasAttribute('required') && !v){ err(el,'This field is required.'); return; }
        if(!v) return;
        if(el.type==='email' && !/^[^@\s]+@[^@\s]+\.[^@\s]{2,}$/.test(v)){ err(el,'Please enter a valid email address.'); return; }
        if(el.type==='url' && !/^(https?:\/\/)?([a-z0-9-]+\.)+[a-z]{2,}(\/\S*)?$/i.test(v)){ err(el,'Please enter a valid link (e.g. https://…).'); return; }
        var pat=el.getAttribute('pattern');
        if(pat){ try{ if(!(new RegExp('^(?:'+pat+')$')).test(v)){ err(el, el.getAttribute('title')||'Please match the requested format.'); } }catch(_){ } }
      });
      if(first){ e.preventDefault(); if(first.focus){ try{ first.focus(); }catch(_){ } } first.scrollIntoView({block:'center'}); }
    });
  }
  [].forEach.call(document.querySelectorAll('form[data-validate]'), wire);
})();
</script>"""


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
<form method="post" action="/employer/register/submit" autocomplete="off" data-validate>
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
    <div><label>Pincode</label><input name="pincode" inputmode="numeric" pattern="[1-9][0-9]{{5}}" title="6-digit PIN code" placeholder="600001"></div>
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
{_FORM_VALIDATE_JS}
"""
    return _page("Register your company", inner)


def _kyc_html(token: str, *, error: str = "") -> str:
    err = f'<div class="addrbox" style="border-color:#a33;color:#ffb3b3">{_esc(error)}</div>' if error else ""
    inner = f"""
<h1>Verify your business</h1>
<p class="sub">Submit a business proof to unlock full candidate details. Your
documents are used only for verification.</p>
{err}
<form method="post" action="/employer/kyc/submit" autocomplete="off" data-validate>
  <input type="hidden" name="token" value="{_esc(token)}">
  <label>Document type <span class="req">*</span></label>
  <select name="kyc_document_type" required>{_options_html(KYC_DOC_TYPES, placeholder='Select…')}</select>
  <div class="row">
    <div><label>GST number</label>
      <input name="gst_number" pattern="[0-9]{{2}}[A-Za-z]{{5}}[0-9]{{4}}[A-Za-z][0-9A-Za-z]Z[0-9A-Za-z]"
        title="15-character GST, e.g. 22AAAAA0000A1Z5" placeholder="22AAAAA0000A1Z5"></div>
    <div><label>PAN number</label>
      <input name="pan_number" pattern="[A-Za-z]{{5}}[0-9]{{4}}[A-Za-z]"
        title="10-character PAN, e.g. AAAAA0000A" placeholder="AAAAA0000A"></div>
  </div>
  <label>Document link <span class="req">*</span></label>
  <input type="url" name="kyc_document_url" required placeholder="https://… (Drive/Dropbox link to the proof)">
  <button class="btn" type="submit">Submit for verification</button>
</form>
{_FORM_VALIDATE_JS}
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
    company_address: str = "", company_district_id: str = "", error: str = "",
) -> str:
    n = len(_JOB_STEPS)
    addr_display = _esc(company_address) or "Your registered company address will be used."
    err = f'<div class="addrbox" style="border-color:#a33;color:#ffb3b3">{_esc(error)}</div>' if error else ""
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
      <div><label>Min years <span class="req">*</span></label>{_year_select("experience_min")}</div>
      <div><label>Max years <span class="req">*</span></label>{_year_select("experience_max")}</div>
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
    <label>Salary Range <span class="req">*</span></label>
    {_radios("salary_period", SALARY_PERIODS)}
    <div class="row">
      <div><label>Min (₹) <span class="req">*</span></label><input name="salary_min" data-req="1" data-fmt="num" inputmode="numeric" placeholder="Min"></div>
      <div><label>Max (₹) <span class="req">*</span></label><input name="salary_max" data-req="1" data-fmt="num" inputmode="numeric" placeholder="Max"></div>
    </div>
  </div>

  <label>Number of Vacancies <span class="req">*</span></label>
  <input name="vacancies" data-req="1" data-fmt="int" inputmode="numeric" placeholder="e.g. 5">
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
    <input name="contact_phone" data-fmt="phone" inputmode="numeric" placeholder="e.g. 9876543210">
  </div>
  <div id="waInput" hidden>
    <label>WhatsApp Number</label>
    <input name="contact_whatsapp" data-fmt="phone" inputmode="numeric" placeholder="e.g. 9876543210">
  </div>
</section>""",
    ])

    # JSON-encode (NOT html-escape) — these are JS string literals inside a
    # <script>; HTML-escaping the "&" in "Experience & Salary" would make the
    # runtime value "Experience &amp; Salary" and silently break the NAMES[cur]
    # comparison in valid() (and garble the step header).
    names_js = json.dumps(list(_JOB_STEPS))
    inner = f"""
{err}
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
// --- inline validation helpers (red border + message below the field) ---
function clearErrs(scope){{
  [].forEach.call(scope.querySelectorAll('.invalid'), function(x){{ x.classList.remove('invalid'); }});
  [].forEach.call(scope.querySelectorAll('.field-err'), function(x){{ x.parentNode.removeChild(x); }});
}}
function showErr(el, msg){{
  if(!el) return false;
  var t = el.tagName;
  if(t==='INPUT'||t==='SELECT'||t==='TEXTAREA') el.classList.add('invalid');
  var e = document.createElement('div'); e.className='field-err'; e.textContent = msg;
  el.insertAdjacentElement('afterend', e);
  if(el.focus) try {{ el.focus(); }} catch(_){{}}
  el.scrollIntoView({{block:'center'}});
  return false;
}}
function grp(name){{ var r = steps[cur].querySelector('[name='+name+']'); return r ? r.closest('.opts, .chips') : null; }}
function valid(){{
  clearErrs(steps[cur]);
  if(cur===0){{ var t=document.querySelector('[name=title]'); if(!t.value.trim()) return showErr(t,'Please enter a job title.'); }}
  // Experience & Salary: conditionally-required fields by experience type.
  if(steps[cur].querySelector('[name=experience_type]')){{
    var exp=picked('experience_type');
    if(exp==='EXPERIENCED'){{
      var mn=document.querySelector('[name=experience_min]'), mx=document.querySelector('[name=experience_max]');
      if(!mn.value) return showErr(mn,'Select the minimum years of experience.');
      if(!mx.value) return showErr(mx,'Select the maximum years of experience.');
      if(+mx.value < +mn.value) return showErr(mx,"Max years can't be less than min years.");
    }} else if(exp==='INTERN'){{
      var pay=picked('intern_payment_type');
      if(!pay) return showErr(grp('intern_payment_type'),'Please choose an intern payment type.');
      if(pay==='STIPEND'){{ var sp=document.querySelector('[name=intern_stipend]'); if(!sp.value.trim()) return showErr(sp,'Please enter the monthly stipend.'); }}
      if(pay==='TRAINING_FEE'){{ var tf=document.querySelector('[name=training_fee]'); if(!tf.value.trim()) return showErr(tf,'Please enter the training fee.'); }}
    }}
    // Salary Range (Monthly / Annual) is required whenever the salary block shows.
    var sb=document.getElementById('salaryBlock');
    if(sb && !sb.hidden && !picked('salary_period'))
      return showErr(grp('salary_period'),'Please choose Monthly or Annual.');
  }}
  // Job Location: a location type must be chosen; Specific needs state + district.
  if(steps[cur].querySelector('[name=job_location_type]')){{
    var lt=picked('job_location_type');
    if(!lt) return showErr(grp('job_location_type'),'Please choose a job location.');
    if(lt==='SPECIFIC'){{
      var st=document.getElementById('job_state_id'), di=document.getElementById('job_district_id');
      if(!st.value) return showErr(st,'Please select the job state.');
      if(!di.value) return showErr(di,'Please select the job district.');
    }}
  }}
  // Candidate Location Preference: at least one district must be selected.
  if(steps[cur].querySelector('#pref_add')){{
    if(steps[cur].querySelectorAll('[name=preferred_district_ids]').length===0)
      return showErr(document.getElementById('pref_add'),'Please select at least one candidate district.');
  }}
  // Apply Methods: a chosen Phone Call / WhatsApp needs its number.
  if(steps[cur].querySelector('[name=apply_modes]')){{
    if(document.getElementById('am_call').checked){{
      var p=document.querySelector('[name=contact_phone]'); if(!p.value.trim()) return showErr(p,'Please enter the contact phone number.');
    }}
    if(document.getElementById('am_wa').checked){{
      var w=document.querySelector('[name=contact_whatsapp]'); if(!w.value.trim()) return showErr(w,'Please enter the WhatsApp number.');
    }}
  }}
  // Required fields on this step (data-req) — skipping any inside a hidden block.
  var reqBad=null;
  [].forEach.call(steps[cur].querySelectorAll('[data-req]'), function(el){{
    if(reqBad || el.offsetParent===null) return;        // visible-only
    if(!(el.value||'').trim()) reqBad=el;
  }});
  if(reqBad) return showErr(reqBad,'This field is required.');
  // format checks for the step's numeric/phone fields (only when filled + visible)
  var bad=null, msg='';
  [].forEach.call(steps[cur].querySelectorAll('[data-fmt]'), function(el){{
    if(bad || el.offsetParent===null) return;
    var v=(el.value||'').trim(); if(!v) return;
    var f=el.getAttribute('data-fmt');
    if(f==='num' && !/^\\d+(\\.\\d+)?$/.test(v)){{ bad=el; msg="Please enter a valid number."; }}
    else if(f==='int' && !/^\\d+$/.test(v)){{ bad=el; msg="Please enter a whole number."; }}
    else if(f==='phone'){{ var d=v.replace(/\\D/g,''); if(d.length>10) d=d.slice(-10); if(!(d.length===10 && /[6-9]/.test(d[0]))){{ bad=el; msg="Enter a valid 10-digit mobile number."; }} }}
  }});
  if(bad) return showErr(bad,msg);
  // salary max >= min (when both given)
  var mn2=steps[cur].querySelector('[name=salary_min]'), mx2=steps[cur].querySelector('[name=salary_max]');
  if(mn2 && mx2 && mn2.value.trim() && mx2.value.trim() && +mx2.value < +mn2.value)
    return showErr(mx2,"Max salary can't be less than the minimum.");
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

// The "Post Job" button is type=submit (last step) — guard it through valid().
document.getElementById('jobForm').addEventListener('submit', function(e){{ if(!valid()){{ e.preventDefault(); }} }});

render();
</script>
"""
    return _page("Post a job", inner)


def _fmt_ts(ts: Any) -> str:
    """Epoch seconds → 'Jun 15, 2026 · 2:39 PM' (falls back gracefully)."""
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%b %d, %Y · %-I:%M %p")
    except (TypeError, ValueError, OSError):
        try:                       # Windows strftime has no %-I
            return datetime.fromtimestamp(int(ts)).strftime("%b %d, %Y · %I:%M %p").replace(" 0", " ")
        except (TypeError, ValueError, OSError):
            return ""


# Credit-type → wallet icon (matches the balance card / Buy Credits).
_WALLET_ICONS = {"JOB": "💼", "UNLOCK": "🔓", "BOOST": "🚀"}


def _wallet_html(token: str, *, balance: dict[str, int], ledger: list[dict[str, Any]]) -> str:
    """Credits & Wallet — a Credit Wallet balance card + transaction history with
    All / Unlocks / Boosts filters (the 'Credit History' screen)."""
    rows = []
    for tx in ledger:
        ctype = str(tx.get("creditType", "JOB")).upper()
        credit = tx.get("action") != "debit"
        sign = "+" if credit else "−"
        amt_cls = "txcredit" if credit else "txdebit"
        rows.append(
            f'<div class="txrow" data-type="{ctype}">'
            f'<div class="txicon">{_WALLET_ICONS.get(ctype, "💳")}</div>'
            f'<div class="txmid"><b>Transaction</b>'
            f'<p>{_esc(tx.get("description") or "")}</p>'
            f'<small>{_esc(_fmt_ts(tx.get("createdAt")))}</small></div>'
            f'<div class="txamt"><span class="{amt_cls}">{sign}{int(tx.get("amount", 0))}</span>'
            f'<span class="txbal">Bal: {int(tx.get("balance", 0))}</span></div></div>'
        )
    empty = '<p class="sub" id="txEmpty" style="text-align:center;margin-top:24px">No transactions yet.</p>'
    inner = f"""
<div class="wallethdr">
  <div class="wrow"><span class="wlbl">🪪 Credit Wallet</span>
    <a class="recharge" href="/employer/buy-credits?token={_esc(token)}">＋ Recharge</a></div>
  <div class="balrow">
    <div><b>💼 {balance.get('job', 0)}</b><small>Job</small></div>
    <div><b>🔓 {balance.get('unlock', 0)}</b><small>Unlock</small></div>
    <div><b>🚀 {balance.get('boost', 0)}</b><small>Boost</small></div>
  </div>
</div>
<div class="filters">
  <button type="button" class="filt on" data-f="ALL">All Transactions</button>
  <button type="button" class="filt" data-f="UNLOCK">Unlocks</button>
  <button type="button" class="filt" data-f="BOOST">Boosts</button>
</div>
<div id="txlist">{''.join(rows) or empty}</div>
<a class="btn" href="/employer/buy-credits?token={_esc(token)}" style="display:block;text-align:center;text-decoration:none">Add Credits</a>
<script>
[].forEach.call(document.querySelectorAll('.filt'), function(b){{
  b.addEventListener('click', function(){{
    [].forEach.call(document.querySelectorAll('.filt'), function(x){{ x.classList.remove('on'); }});
    b.classList.add('on');
    var f = b.getAttribute('data-f');
    [].forEach.call(document.querySelectorAll('.txrow'), function(r){{
      r.style.display = (f === 'ALL' || r.getAttribute('data-type') === f) ? '' : 'none';
    }});
  }});
}});
</script>
"""
    return _page("Credits & Wallet", inner)


def _buy_credits_html(token: str, *, bundles: list[dict[str, Any]], balance: dict[str, int],
                      key_id: str, test_mode: bool, prefill_name: str, prefill_phone: str) -> str:
    """Buy Credits: a Current Balance card + Bundles / Individual Credits tabs,
    each item a selectable card; pay the selected one via Razorpay checkout."""
    # Bundles tab (credit_bundles).
    bcards = []
    for b in bundles:
        price = float(b.get("price") or 0)
        feats = (f'<span class="pfeat">💼 {b.get("jobCredits", 0)} Job</span>'
                 f'<span class="pfeat">🔓 {b.get("unlockCredits", 0)} Unlocks</span>'
                 f'<span class="pfeat">🚀 {b.get("boostCredits", 0)} Boost</span>')
        pop = '<span class="poptag">★ Popular</span>' if str(b.get("bundleType")) == "GROWTH" else ""
        bcards.append(
            f'<label class="plan buyitem" data-item="bundle:{_esc(b.get("id"))}" data-price="{price}">'
            f'<div class="prow"><input type="radio" name="buy"><b>{_esc(b.get("name"))}</b>{pop}'
            f'<span class="pprice">₹{int(price) if price == int(price) else price}</span></div>'
            f'<div class="pfeats">{feats}</div></label>'
        )
    # Individual Credits tab (per-type packs).
    icards = []
    for ctype, packs in INDIVIDUAL_PACKS.items():
        icards.append(f'<div class="sechdr">{CREDIT_TYPE_ICONS.get(ctype, "")} {_esc(CREDIT_TYPE_LABELS.get(ctype, ctype))}</div>')
        for i, (qty, price) in enumerate(packs):
            per = round(price / qty, 1) if qty else price
            save = '<span class="savetag">SAVE MORE</span>' if i > 0 else ""
            icards.append(
                f'<label class="plan buyitem" data-item="pack:{ctype}:{qty}" data-price="{price}">'
                f'<div class="prow"><input type="radio" name="buy">'
                f'<b>{qty} Credit{"s" if qty != 1 else ""}</b>{save}'
                f'<span class="pprice">₹{price}<small>₹{per} / credit</small></span></div></label>'
            )
    badge = '<span class="testbadge">TEST MODE</span>' if test_mode else ""
    inner = f"""
<h1>💳 Buy Credits {badge}</h1>
<div class="balcard">
  <div class="ballbl">Current Balance</div>
  <div class="balrow">
    <div><b>💼 {balance.get('job', 0)}</b><small>Job</small></div>
    <div><b>🔓 {balance.get('unlock', 0)}</b><small>Unlock</small></div>
    <div><b>🚀 {balance.get('boost', 0)}</b><small>Boost</small></div>
  </div>
</div>
<div class="tabs">
  <button type="button" class="tab on" id="tabBtnB">Bundles</button>
  <button type="button" class="tab" id="tabBtnI">Individual Credits</button>
</div>
<div id="tabBundles">{''.join(bcards)}</div>
<div id="tabIndividual" hidden>{''.join(icards)}</div>
<button type="button" class="btn" id="buyBtn" disabled>Select an option</button>
<p class="vnote" id="buyNote"></p>
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
<script>
var TOKEN = {json.dumps(token)};
var PREFILL = {{name: {json.dumps(prefill_name)}, contact: {json.dumps(prefill_phone)}}};
var sel = null, buyBtn = document.getElementById('buyBtn'), note = document.getElementById('buyNote');
function selTab(which){{
  document.getElementById('tabBundles').hidden = which !== 'b';
  document.getElementById('tabIndividual').hidden = which !== 'i';
  document.getElementById('tabBtnB').classList.toggle('on', which === 'b');
  document.getElementById('tabBtnI').classList.toggle('on', which === 'i');
}}
document.getElementById('tabBtnB').addEventListener('click', function(){{ selTab('b'); }});
document.getElementById('tabBtnI').addEventListener('click', function(){{ selTab('i'); }});
[].forEach.call(document.querySelectorAll('.buyitem'), function(el){{
  el.addEventListener('click', function(){{
    [].forEach.call(document.querySelectorAll('.buyitem'), function(x){{ x.classList.remove('on'); var r=x.querySelector('input'); if(r) r.checked=false; }});
    el.classList.add('on'); var r=el.querySelector('input'); if(r) r.checked=true;
    sel = el.getAttribute('data-item');
    var price = parseFloat(el.getAttribute('data-price'));
    buyBtn.disabled = false;
    buyBtn.textContent = 'Pay ₹' + (price % 1 ? price : price.toFixed(0));
  }});
}});
function post(url, data){{
  var body = Object.keys(data).map(function(k){{ return encodeURIComponent(k)+'='+encodeURIComponent(data[k]); }}).join('&');
  return fetch(url, {{method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}}, body:body}});
}}
buyBtn.addEventListener('click', function(){{
  if(!sel) return;
  buyBtn.disabled = true; note.textContent = 'Starting secure checkout…';
  post('/employer/buy-credits/order', {{token: TOKEN, item: sel}})
    .then(function(r){{ return r.json(); }})
    .then(function(o){{
      if(o.error){{ note.textContent = 'Could not start checkout. Please try again.'; buyBtn.disabled = false; return; }}
      var rzp = new Razorpay({{
        key: o.key_id, order_id: o.order_id, amount: o.amount, currency: o.currency,
        name: 'Jobs7', description: o.label,
        prefill: {{name: o.prefill_name || PREFILL.name, contact: o.prefill_phone || PREFILL.contact}},
        theme: {{color: '#5b3df5'}},
        handler: function(resp){{
          note.textContent = 'Confirming payment…';
          post('/employer/buy-credits/verify', {{
            razorpay_order_id: resp.razorpay_order_id,
            razorpay_payment_id: resp.razorpay_payment_id,
            razorpay_signature: resp.razorpay_signature
          }}).then(function(r){{ return r.json(); }}).then(function(v){{
            if(v.ok){{ window.location = v.redirect; }}
            else {{ note.textContent = 'Payment verification failed. If charged, contact support.'; buyBtn.disabled = false; }}
          }});
        }},
        modal: {{ondismiss: function(){{ note.textContent = 'Payment cancelled.'; buyBtn.disabled = false; }}}}
      }});
      rzp.on('payment.failed', function(){{ note.textContent = 'Payment failed. Please try again.'; buyBtn.disabled = false; }});
      rzp.open();
    }})
    .catch(function(){{ note.textContent = 'Network error. Please try again.'; buyBtn.disabled = false; }});
}});
</script>
"""
    return _page("Buy credits", inner)


def _subscribe_html(token: str, plans: list[dict[str, Any]], *, key_id: str,
                    test_mode: bool, prefill_name: str, prefill_phone: str,
                    current_type: str) -> str:
    """The 'Upgrade Plan' page: subscription_plans as cards + Razorpay Checkout."""
    cards = []
    for p in plans:
        ptype = str(p.get("type") or "")
        price = float(p.get("price") or 0)
        label, _ = _BILLING.get(p.get("billingCycle"), ("subscription", 30))
        feats = []
        if p.get("maxActiveJobs"):
            feats.append(f"📋 {p['maxActiveJobs']} active job{'s' if p['maxActiveJobs'] != 1 else ''}")
        if p.get("monthlyCredits"):
            feats.append(f"🔓 {p['monthlyCredits']} unlocks")
        if p.get("monthlyBoosts"):
            feats.append(f"🚀 {p['monthlyBoosts']} boosts")
        chips = "".join(f'<span class="pfeat">{_esc(f)}</span>' for f in feats)
        price_txt = "Free" if price <= 0 else f"₹{int(price) if price == int(price) else price}"
        cur = ' <span class="curtag">Current</span>' if ptype == current_type else ""
        popular = '<span class="poptag">★ Popular</span>' if ptype == "GROWTH" else ""
        cards.append(
            f'<label class="plan" data-plan="{_esc(p.get("id"))}" data-price="{price}">'
            f'<div class="prow"><input type="radio" name="plan" value="{_esc(p.get("id"))}">'
            f'<b>{_esc(p.get("name"))}{cur}</b>{popular}'
            f'<span class="pprice">{price_txt}<small>/{_esc(label)}</small></span></div>'
            f'<div class="pfeats">{chips}</div></label>'
        )
    badge = '<span class="testbadge">TEST MODE</span>' if test_mode else ""
    inner = f"""
<h1>💎 Upgrade Plan {badge}</h1>
<p class="sub">Pick a plan — billed securely via Razorpay.</p>
<div id="plans">{''.join(cards)}</div>
<button type="button" class="btn" id="payBtn" disabled>Select a plan</button>
<p class="sub" id="payNote" style="margin-top:12px;text-align:center"></p>
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
<script>
var TOKEN = {json.dumps(token)};
var PREFILL = {{name: {json.dumps(prefill_name)}, contact: {json.dumps(prefill_phone)}}};
var sel = null, payBtn = document.getElementById('payBtn'), note = document.getElementById('payNote');
[].forEach.call(document.querySelectorAll('.plan'), function(el){{
  el.addEventListener('click', function(){{
    [].forEach.call(document.querySelectorAll('.plan'), function(x){{ x.classList.remove('on'); }});
    el.classList.add('on'); el.querySelector('input').checked = true;
    sel = el.getAttribute('data-plan');
    var price = parseFloat(el.getAttribute('data-price'));
    payBtn.disabled = false;
    payBtn.textContent = price > 0 ? ('Pay ₹' + (price % 1 ? price : price.toFixed(0)) + ' & Subscribe') : 'Activate Free Plan';
  }});
}});
function post(url, data){{
  var body = Object.keys(data).map(function(k){{ return encodeURIComponent(k)+'='+encodeURIComponent(data[k]); }}).join('&');
  return fetch(url, {{method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}}, body:body}});
}}
payBtn.addEventListener('click', function(){{
  if(!sel) return;
  payBtn.disabled = true; note.textContent = 'Starting secure checkout…';
  post('/employer/subscribe/order', {{token: TOKEN, plan_id: sel}})
    .then(function(r){{ return r.json(); }})
    .then(function(o){{
      if(o.free){{ window.location = o.redirect; return; }}
      if(o.error){{ note.textContent = 'Could not start checkout. Please try again.'; payBtn.disabled = false; return; }}
      var rzp = new Razorpay({{
        key: o.key_id, order_id: o.order_id, amount: o.amount, currency: o.currency,
        name: 'Jobs7', description: o.plan_name + ' plan',
        prefill: {{name: o.prefill_name || PREFILL.name, contact: o.prefill_phone || PREFILL.contact}},
        theme: {{color: '#5b3df5'}},
        handler: function(resp){{
          note.textContent = 'Confirming payment…';
          post('/employer/subscribe/verify', {{
            razorpay_order_id: resp.razorpay_order_id,
            razorpay_payment_id: resp.razorpay_payment_id,
            razorpay_signature: resp.razorpay_signature
          }}).then(function(r){{ return r.json(); }}).then(function(v){{
            if(v.ok){{ window.location = v.redirect; }}
            else {{ note.textContent = 'Payment verification failed. If you were charged, contact support.'; payBtn.disabled = false; }}
          }});
        }},
        modal: {{ondismiss: function(){{ note.textContent = 'Payment cancelled.'; payBtn.disabled = false; }}}}
      }});
      rzp.on('payment.failed', function(){{ note.textContent = 'Payment failed. Please try again.'; payBtn.disabled = false; }});
      rzp.open();
    }})
    .catch(function(){{ note.textContent = 'Network error. Please try again.'; payBtn.disabled = false; }});
}});
</script>
"""
    return _page("Upgrade plan", inner)


def _activate_job_html(token: str, *, title: str, district_names: list[str], have: int,
                       key_id: str = "", test_mode: bool = False,
                       prefill_name: str = "", prefill_phone: str = "") -> str:
    """The 'Activate Job' screen: pick a validity (15/30/45 days → 1×/2×/3×),
    see credits required (#districts × multiplier) vs the wallet balance, and
    activate. A shortfall opens **Razorpay checkout** for buy × ₹649 (test mode)."""
    chips = "".join(f'<span class="achip">{_esc(n)}</span>' for n in district_names) or \
        '<span class="achip">—</span>'
    pills = "".join(
        f'<button type="button" class="vpill{" on" if days == VALIDITY_OPTIONS[0][0] else ""}" '
        f'data-days="{days}" data-mult="{mult}">{days} Days<small>{mult}x</small></button>'
        for days, mult in VALIDITY_OPTIONS
    )
    mult_js = "{" + ", ".join(f"{d}: {m}" for d, m in VALIDITY_OPTIONS) + "}"
    names_js = json.dumps(district_names)
    default_days = VALIDITY_OPTIONS[0][0]
    inner = f"""
<div class="jobhdr">
  <div class="jobttl">💼 {_esc(title)}</div>
  <div class="jobsub">Districts ({len(district_names)})</div>
  <div class="achips">{chips}</div>
</div>

<form method="post" action="/employer/post-job/activate" id="actForm">
  <input type="hidden" name="token" value="{_esc(token)}">
  <input type="hidden" name="validity_days" id="validity_days" value="{default_days}">

  <div class="acard">
    <label>🕒 Job Validity</label>
    <div class="vpills">{pills}</div>
    <p class="vnote" id="vnote">{default_days} Days · 1 credit/district</p>
  </div>

  <div class="acard">
    <label>🧮 Credits Required</label>
    <div class="calc">
      <div><b id="cDistricts">{len(district_names)}</b><small>districts</small></div>
      <div class="op">×</div>
      <div><b id="cMult">1</b><small>multiplier</small></div>
      <div class="op">=</div>
      <div><b id="cNeed" class="accent">0</b><small>credits</small></div>
    </div>
    <div class="boxes">
      <div class="box have"><small>Have</small><b id="bHave">{int(have)}</b></div>
      <div class="box need"><small>Need</small><b id="bNeed">0</b></div>
      <div class="box buy" id="buyBox"><small>Buy</small><b id="bBuy">0</b></div>
    </div>
    <div class="bar"><span id="barFill"></span></div>
    <p class="vnote" id="statusNote"></p>
  </div>

  <label class="tc"><input type="checkbox" id="tc" checked> I accept the Terms, Privacy &amp; Refund policy</label>
  <button type="submit" class="btn" id="actBtn">Activate Now</button>
  <p class="vnote" id="payNote"></p>
</form>
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
<script>
var PRICE = {JOB_CREDIT_PRICE};
var HAVE = {int(have)};
var MULT = {mult_js};
var NAMES = {names_js};
var TOKEN = {json.dumps(token)};
var PREFILL = {{name: {json.dumps(prefill_name)}, contact: {json.dumps(prefill_phone)}}};
var days = {default_days};
function recompute(){{
  var districts = NAMES.length, m = MULT[days] || 1;
  var need = Math.max(1, districts) * m;
  var buy = Math.max(0, need - HAVE), pay = buy * PRICE;
  document.getElementById('cDistricts').textContent = districts;
  document.getElementById('cMult').textContent = m;
  document.getElementById('cNeed').textContent = need;
  document.getElementById('bHave').textContent = HAVE;
  document.getElementById('bNeed').textContent = need;
  document.getElementById('bBuy').textContent = buy;
  document.getElementById('validity_days').value = days;
  document.getElementById('vnote').textContent = days + ' Days · ' + m + ' credit/district';
  var pct = need > 0 ? Math.min(100, Math.round(HAVE / need * 100)) : 100;
  document.getElementById('barFill').style.width = pct + '%';
  var ok = buy === 0;
  document.getElementById('buyBox').classList.toggle('zero', ok);
  document.getElementById('statusNote').textContent = ok
    ? '✅ Sufficient credits available' : ('Need ' + buy + ' more credit' + (buy===1?'':'s'));
  var btn = document.getElementById('actBtn');
  btn.textContent = ok ? 'Activate Now' : ('Pay ₹' + pay + ' & Activate');
}}
[].forEach.call(document.querySelectorAll('.vpill'), function(p){{
  p.addEventListener('click', function(){{
    [].forEach.call(document.querySelectorAll('.vpill'), function(x){{ x.classList.remove('on'); }});
    p.classList.add('on');
    days = parseInt(p.getAttribute('data-days'), 10);
    recompute();
  }});
}});
function post(url, data){{
  var body = Object.keys(data).map(function(k){{ return encodeURIComponent(k)+'='+encodeURIComponent(data[k]); }}).join('&');
  return fetch(url, {{method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}}, body:body}});
}}
var note = document.getElementById('payNote'), actBtn = document.getElementById('actBtn');
document.getElementById('actForm').addEventListener('submit', function(e){{
  if(!document.getElementById('tc').checked){{ e.preventDefault(); note.textContent = 'Please accept the Terms to continue.'; note.style.color = '#ff7a7a'; return; }}
  // Enough credits → let the normal POST to /post-job/activate go through.
  var need = Math.max(1, NAMES.length) * (MULT[days] || 1);
  if(need - HAVE <= 0) return;
  // Shortfall → pay the difference via Razorpay, then activate.
  e.preventDefault();
  actBtn.disabled = true; note.textContent = 'Starting secure checkout…';
  post('/employer/post-job/credits/order', {{token: TOKEN, validity_days: days}})
    .then(function(r){{ return r.json(); }})
    .then(function(o){{
      if(o.sufficient){{ document.getElementById('actForm').submit(); return; }}
      if(o.error){{ note.textContent = 'Could not start checkout. Please try again.'; actBtn.disabled = false; return; }}
      var rzp = new Razorpay({{
        key: o.key_id, order_id: o.order_id, amount: o.amount, currency: o.currency,
        name: 'Jobs7', description: o.buy + ' job credit' + (o.buy===1?'':'s'),
        prefill: {{name: o.prefill_name || PREFILL.name, contact: o.prefill_phone || PREFILL.contact}},
        theme: {{color: '#5b3df5'}},
        handler: function(resp){{
          note.textContent = 'Confirming payment…';
          post('/employer/post-job/credits/verify', {{
            razorpay_order_id: resp.razorpay_order_id,
            razorpay_payment_id: resp.razorpay_payment_id,
            razorpay_signature: resp.razorpay_signature
          }}).then(function(r){{ return r.json(); }}).then(function(v){{
            if(v.ok){{ window.location = v.redirect; }}
            else {{ note.textContent = 'Payment verification failed. If charged, contact support.'; actBtn.disabled = false; }}
          }});
        }},
        modal: {{ondismiss: function(){{ note.textContent = 'Payment cancelled.'; actBtn.disabled = false; }}}}
      }});
      rzp.on('payment.failed', function(){{ note.textContent = 'Payment failed. Please try again.'; actBtn.disabled = false; }});
      rzp.open();
    }})
    .catch(function(){{ note.textContent = 'Network error. Please try again.'; actBtn.disabled = false; }});
}});
recompute();
</script>
"""
    return _page("Activate job", inner)


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
