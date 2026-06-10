"""Build a DB-ready *employer record* from a submitted registration form.

This assembles the row that WOULD be inserted to register a job-poster on the
live job board — the ``private_employers`` table — with the dropdown selections
already FK ids. Nothing here writes to the business DB: the caller stages the
record in **Redis only** (for testing) so the field shape + KYC gate can be
exercised before any real INSERTs are turned on.

Stage 2 (KYC) merges ``gstNumber`` / ``panNumber`` / ``kycDocument*`` /
``kycStatus`` back onto this same record — they're columns on the same table.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

# private_employers enums (verified against the live jobs7uat schema).
COMPANY_SIZES = (
    ("STARTUP", "Startup (1-10)"),
    ("SMALL", "Small (11-50)"),
    ("MEDIUM", "Medium (51-200)"),
    ("LARGE", "Large (201-1000)"),
    ("ENTERPRISE", "Enterprise (1000+)"),
)
# KYC document types accepted at Stage 2.
KYC_DOC_TYPES = (
    ("GST", "GST Certificate"),
    ("PAN", "Company PAN"),
    ("INCORPORATION", "Certificate of Incorporation"),
    ("UDYAM", "Udyam / MSME"),
    ("OTHER", "Other business proof"),
)

# private_jobs enums (verified against the live jobs7uat schema). Form values ARE
# the DB enum labels, so no mapping is needed at insert time.
JOB_TYPES = (
    ("FULL_TIME", "Full-time"), ("PART_TIME", "Part-time"), ("CONTRACT", "Contract"),
    ("INTERNSHIP", "Internship"), ("FREELANCE", "Freelance"),
)
JOB_WORK_MODES = (("OFFICE", "On-site / Office"), ("REMOTE", "Remote"), ("HYBRID", "Hybrid"))
SALARY_PERIODS = (
    ("MONTHLY", "Monthly"), ("YEARLY", "Annual"),
    ("DAILY", "Per day"), ("HOURLY", "Per hour"),
)
# Value labels verified against existing private_jobs rows on jobs7uat.
EXPERIENCE_TYPES = (
    ("ANY", "Any"), ("FRESHER", "Fresher Only"), ("INTERN", "Intern"), ("EXPERIENCED", "Experienced"),
)
QUALIFICATION_LEVELS = (
    ("BELOW_10TH", "<10th pass"), ("10TH_ABOVE", "10th above"), ("12TH_PASS", "12th pass"),
    ("12TH_ABOVE", "12th above"), ("DEGREE", "Degree"),
)
GENDER_PREFS = (("BOTH", "Both"), ("MALE", "Male"), ("FEMALE", "Female"))
MARITAL_PREFS = (("ANY", "Any"), ("UNMARRIED", "Unmarried"), ("MARRIED", "Married"))
ENGLISH_LEVELS = (("NO_NEED", "No need"), ("INTERMEDIATE", "Intermediate"), ("GOOD", "Good English"))
JOB_LOCATION_TYPES = (
    ("SPECIFIC", "Specific Location"), ("COMPANY_ADDRESS", "Company Address"), ("REMOTE", "Remote"),
)
CANDIDATE_DISTANCES = (
    ("10", "10 km"), ("15", "15 km"), ("30", "30 km"), ("CUSTOM", "Custom"), ("ANYWHERE", "Anywhere"),
)
JOB_LANGUAGES = ("Tamil", "English", "Hindi", "Telugu", "Malayalam", "Kannada")
REQUIRED_ASSETS = (
    "Bike", "Licence", "Aadhar", "PAN", "Laptop", "Camera", "Smartphone", "Car", "Passport", "Bank Account",
)
APPLY_MODES = (("APPLY", "In-App Apply"), ("CALL", "Phone Call"), ("WHATSAPP", "WhatsApp"))


def _cuid() -> str:
    """Placeholder id for a staged (Redis-only) record; the live DB mints a real
    cuid on actual insert."""
    return "c" + uuid.uuid4().hex[:24]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _str(form: dict[str, Any], key: str) -> str | None:
    v = form.get(key)
    v = v.strip() if isinstance(v, str) else v
    return v or None


def _slugify(name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in (name or "")).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug


def build_employer_record(*, identity: dict[str, Any], form: dict[str, Any]) -> dict[str, Any]:
    """Pure assembly of the ``private_employers`` row (no I/O). The form's
    dropdowns are id-valued, so industry/designation/district map straight onto
    their FK columns. ``status`` / ``kycStatus`` start at the table defaults."""
    now = _now_iso()
    emp_id = _cuid()
    company = (form.get("company_name") or "").strip()
    phone = (identity.get("customer_id") or form.get("primary_phone") or "").strip() or None
    size = (form.get("company_size") or "").strip().upper() or None

    record = {
        "id": emp_id,
        "userId": None,
        "companyName": company,
        "slug": _slugify(company) or emp_id,
        "contactPerson": _str(form, "contact_person"),
        "designationId": _str(form, "designation_id"),
        "primaryPhone": phone,
        "email": _str(form, "email"),
        "website": _str(form, "website"),
        "logo": _str(form, "logo"),
        "description": _str(form, "description"),
        "address": _str(form, "address"),
        "city": _str(form, "city"),
        "districtId": _str(form, "district_id"),
        "pincode": _str(form, "pincode"),
        "industryId": _str(form, "industry_id"),
        "companySize": size,
        "gstNumber": _str(form, "gst_number"),
        "panNumber": _str(form, "pan_number"),
        "status": "PENDING",            # EmployerStatus default
        "kycStatus": "NOT_SUBMITTED",   # KycStatus default
        "emailVerified": False,
        "phoneVerified": bool(phone),
        "registrationSource": "WHATSAPP_BOT",
        "createdAt": now,
        "updatedAt": now,
    }
    return {"private_employers": record}


def _to_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _to_num(value: Any) -> float | None:
    try:
        v = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return v if v >= 0 else None


def _csv(value: Any) -> list[str]:
    """A comma-separated text field → a clean list (for skills / benefits)."""
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [s.strip() for s in str(value or "").split(",") if s.strip()]


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"yes", "true", "1", "on"}


def _enum(form: dict[str, Any], key: str, default: str | None = None) -> str | None:
    v = (form.get(key) or "").strip().upper()
    return v or default


def build_job_record(
    *, employer_id: str | None, form: dict[str, Any], status: str = "PENDING"
) -> dict[str, Any]:
    """Pure assembly of a DB-ready ``private_jobs`` row from the (sectioned)
    post-job form. Keyed by table name (like the seeker/employer records) and
    staged in Redis; the form's controls already carry the right enum labels / FK
    ids, so this maps straight onto the columns. ``status`` defaults to PENDING.
    """
    now = _now_iso()
    jid = _cuid()
    title = (form.get("title") or "").strip()

    # Candidate distance: 10/15/30 are radii (km); CUSTOM uses a typed value;
    # ANYWHERE has no radius.
    dist = (form.get("candidate_distance") or "").strip().upper()
    if dist in {"10", "15", "30"}:
        radius, radius_type = _to_int(dist), dist
    elif dist == "CUSTOM":
        radius, radius_type = _to_int(form.get("candidate_distance_custom")), "CUSTOM"
    elif dist == "ANYWHERE":
        radius, radius_type = None, "ANYWHERE"
    else:
        radius, radius_type = None, None

    loc_type = _enum(form, "job_location_type", "COMPANY_ADDRESS")
    job = {
        "id": jid,
        "title": title,
        "slug": _slugify(title) or jid,
        "description": (form.get("description") or "").strip(),
        "employerId": employer_id,
        "categoryId": _str(form, "category_id"),
        "districtId": _str(form, "district_id"),
        "jobStateId": _str(form, "state_id"),
        "locationDetails": _str(form, "city"),
        "jobLocationType": loc_type,
        "jobType": _enum(form, "job_type", "FULL_TIME"),
        "workMode": _enum(form, "work_mode", "OFFICE"),
        "isWorkFromHome": _truthy(form.get("work_from_home")),
        # experience
        "experienceType": _enum(form, "experience_type"),
        "experienceMin": _to_int(form.get("experience_min")) or 0,
        "experienceMax": _to_int(form.get("experience_max")),
        # salary
        "salaryMin": _to_num(form.get("salary_min")),
        "salaryMax": _to_num(form.get("salary_max")),
        "salaryPeriod": _enum(form, "salary_period", "MONTHLY"),
        "salaryNegotiable": _truthy(form.get("salary_negotiable")),
        # candidate requirements
        "qualificationLevel": _enum(form, "qualification_level"),
        "genderPreference": _enum(form, "gender_preference"),
        "maritalStatusPreference": _enum(form, "marital_status_preference"),
        "englishLevel": _enum(form, "english_level"),
        "ageMin": _to_int(form.get("age_min")),
        "ageMax": _to_int(form.get("age_max")),
        # location preferences
        "candidateRadius": radius,
        "candidateRadiusType": radius_type,
        "willingToRelocate": _truthy(form.get("willing_to_relocate")),
        # security deposit
        "hasSecurityDeposit": _truthy(form.get("has_security_deposit")),
        "securityDepositAmt": _to_num(form.get("security_deposit_amt")),
        "securityDepositReason": _str(form, "security_deposit_reason"),
        # timings + interview
        "workStartTime": _str(form, "work_start_time"),
        "workEndTime": _str(form, "work_end_time"),
        "interviewDate": _str(form, "interview_date"),
        "interviewTime": _str(form, "interview_time"),
        # skills + languages + assets (multi-selects arrive as lists; CSV-safe too)
        "skills": _csv(form.get("skills")),
        "preferredLanguages": _csv(form.get("preferred_languages")),
        "requiredAssets": _csv(form.get("required_assets")),
        # apply methods
        "applyModes": _csv(form.get("apply_modes")) or ["APPLY"],
        "contactPhone": _str(form, "contact_phone"),
        "contactWhatsapp": _str(form, "contact_whatsapp"),
        "vacancies": _to_int(form.get("vacancies")) or 1,
        "status": status,
        "createdAt": now,
        "updatedAt": now,
    }
    return {"ref": f"JOB-{jid[-6:].upper()}", "private_jobs": job}


def apply_kyc(record: dict[str, Any], *, doc_type: str | None, doc_url: str | None,
              gst: str | None, pan: str | None, auto_verify: bool) -> dict[str, Any]:
    """Merge a Stage-2 KYC submission onto a staged employer record (in place) and
    advance ``kycStatus``. With ``auto_verify`` the gate flips straight to
    VERIFIED (test mode); otherwise it sits at PENDING ('under review')."""
    emp = record.setdefault("private_employers", {})
    now = _now_iso()
    if gst:
        emp["gstNumber"] = gst.strip()
    if pan:
        emp["panNumber"] = pan.strip()
    if doc_type:
        emp["kycDocumentType"] = doc_type.strip()
    if doc_url:
        emp["kycDocumentUrl"] = doc_url.strip()
    emp["kycSubmittedAt"] = now
    if auto_verify:
        emp["kycStatus"] = "VERIFIED"
        emp["kycVerifiedAt"] = now
        emp["status"] = "APPROVED"
    else:
        emp["kycStatus"] = "PENDING"
    emp["updatedAt"] = now
    return record
