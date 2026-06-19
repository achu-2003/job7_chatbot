"""Shared input validators for the job-seeker workflow.

Pure, dependency-free predicates + a ``validate_registration`` that returns
field-level errors. Used server-side (the authoritative backstop in the onboard
submit + build_registration_records); the onboarding form mirrors the same rules
in JS so the candidate gets immediate feedback. Keeping the rules here means the
two never drift in intent.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

# A pragmatic email check (not RFC-perfect, but rejects the obvious junk).
_EMAIL_RX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")
_PINCODE_RX = re.compile(r"^[1-9]\d{5}$")        # Indian 6-digit PIN (no leading 0)
# Accept any http(s) link (Drive/Dropbox/etc.) — the point is to reject plain
# text like "i'll send it later", not to enforce a TLD.
_URL_RX = re.compile(r"^https?://\S{2,}$", re.IGNORECASE)
# A plausible person name: letters/spaces/.'- , at least 2 letters, not all digits.
_NAME_RX = re.compile(r"^[A-Za-z][A-Za-z .'\-]{1,59}$")
# Indian GST (15) and PAN (10) formats.
_GST_RX = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_PAN_RX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
# A domain (with optional scheme/path) — websites are often typed without http://.
_DOMAIN_RX = re.compile(r"^([a-z0-9-]+\.)+[a-z]{2,}(/\S*)?$", re.IGNORECASE)

_MIN_AGE, _MAX_AGE = 14, 80
_MIN_YEAR, _MAX_YEAR = 1950, date.today().year + 1
_RESUME_EXTS = {".pdf", ".doc", ".docx"}   # resumes only — no images/other formats
MAX_RESUME_BYTES = 5 * 1024 * 1024               # 5 MB


def _s(v: Any) -> str:
    return (v if isinstance(v, str) else "" if v is None else str(v)).strip()


def valid_email(value: Any) -> bool:
    v = _s(value)
    return bool(v) and len(v) <= 254 and bool(_EMAIL_RX.match(v))


def valid_pincode(value: Any) -> bool:
    return bool(_PINCODE_RX.match(_s(value)))


def valid_url(value: Any) -> bool:
    return bool(_URL_RX.match(_s(value)))


def valid_name(value: Any) -> bool:
    v = _s(value)
    return bool(_NAME_RX.match(v)) and any(c.isalpha() for c in v)


def valid_year(value: Any) -> bool:
    """A 4-digit passing year within a sane range."""
    v = _s(value)
    if not v.isdigit() or len(v) != 4:
        return False
    return _MIN_YEAR <= int(v) <= _MAX_YEAR


def valid_salary(value: Any) -> bool:
    """A non-negative numeric salary."""
    try:
        return float(_s(value)) >= 0
    except (TypeError, ValueError):
        return False


def parse_dob(value: Any) -> date | None:
    """Parse 'YYYY-MM-DD' (the form's date input) to a date, else None."""
    v = _s(value)
    if not v:
        return None
    try:
        return datetime.strptime(v[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def age_for(dob: date, *, today: date | None = None) -> int:
    today = today or date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def valid_dob(value: Any) -> bool:
    """A real date that makes the candidate a plausible working age."""
    d = parse_dob(value)
    if d is None:
        return False
    return _MIN_AGE <= age_for(d) <= _MAX_AGE


def valid_resume_filename(name: Any) -> bool:
    n = _s(name).lower()
    return any(n.endswith(ext) for ext in _RESUME_EXTS)


def valid_phone(value: Any) -> bool:
    """A 10-digit Indian mobile (6-9 start), with or without a country code."""
    d = re.sub(r"\D", "", _s(value))
    if len(d) > 10:
        d = d[-10:]
    return len(d) == 10 and d[0] in "6789"


def valid_gst(value: Any) -> bool:
    return bool(_GST_RX.match(_s(value).upper()))


def valid_pan(value: Any) -> bool:
    return bool(_PAN_RX.match(_s(value).upper()))


def valid_website(value: Any) -> bool:
    v = re.sub(r"^https?://", "", _s(value), flags=re.IGNORECASE)
    return bool(v) and bool(_DOMAIN_RX.match(v))


def _pos_int(value: Any) -> bool:
    v = _s(value)
    return v.isdigit() and int(v) >= 1


# ---------------------------------------------------------------------------
# Registration-level validation (the authoritative server backstop)
# ---------------------------------------------------------------------------

# Fields that, if provided, MUST be well-formed (else the submission is rejected
# and the form is re-shown with the error). Optional fields left blank are fine.
def validate_registration(form: dict[str, Any]) -> dict[str, str]:
    """Return ``{field: message}`` for every invalid field. Empty ⇒ all good.
    Only the *name* is strictly required; the rest are validated only when the
    candidate actually filled them in (the form allows skipping)."""
    errors: dict[str, str] = {}
    name = _s(form.get("full_name") or form.get("name"))
    if not valid_name(name):
        errors["full_name"] = "Please enter a valid full name."
    if _s(form.get("email")) and not valid_email(form.get("email")):
        errors["email"] = "Please enter a valid email address."
    if _s(form.get("pincode")) and not valid_pincode(form.get("pincode")):
        errors["pincode"] = "PIN code must be 6 digits."
    if _s(form.get("year_of_passing")) and not valid_year(form.get("year_of_passing")):
        errors["year_of_passing"] = "Enter a valid 4-digit passing year."
    if _s(form.get("date_of_birth")) and not valid_dob(form.get("date_of_birth")):
        errors["date_of_birth"] = "Enter a valid date of birth (age 14–80)."
    for key, label in (("current_salary", "Current"), ("expected_salary", "Expected")):
        if _s(form.get(key)) and not valid_salary(form.get(key)):
            errors[key] = f"{label} salary must be a number."
    return errors


# ---------------------------------------------------------------------------
# Employer-side validation (registration, KYC, post-a-job)
# ---------------------------------------------------------------------------


def validate_employer(form: dict[str, Any]) -> dict[str, str]:
    """Company registration: company name + district required; email/website/PIN
    validated only when filled."""
    errors: dict[str, str] = {}
    if len(_s(form.get("company_name"))) < 2:
        errors["company_name"] = "Please enter your company name."
    if not _s(form.get("district_id")):
        errors["district_id"] = "Please select a state and district."
    if _s(form.get("email")) and not valid_email(form.get("email")):
        errors["email"] = "Please enter a valid work email."
    if _s(form.get("website")) and not valid_website(form.get("website")):
        errors["website"] = "Please enter a valid website (e.g. acme.com)."
    if _s(form.get("pincode")) and not valid_pincode(form.get("pincode")):
        errors["pincode"] = "PIN code must be 6 digits."
    return errors


def validate_kyc(form: dict[str, Any]) -> dict[str, str]:
    """KYC: a document type + a document link are required; GST/PAN validated
    only when provided."""
    errors: dict[str, str] = {}
    if not _s(form.get("kyc_document_type")):
        errors["kyc_document_type"] = "Please choose a document type."
    if not valid_url(form.get("kyc_document_url")):
        errors["kyc_document_url"] = "Please paste a valid document link (https://…)."
    if _s(form.get("gst_number")) and not valid_gst(form.get("gst_number")):
        errors["gst_number"] = "Enter a valid 15-character GST number."
    if _s(form.get("pan_number")) and not valid_pan(form.get("pan_number")):
        errors["pan_number"] = "Enter a valid 10-character PAN (e.g. AAAAA0000A)."
    return errors


def validate_job_post(form: dict[str, Any]) -> dict[str, str]:
    """Post-a-job: title + category required; salary/vacancies/experience numeric
    (and max ≥ min); contact numbers and intern amounts validated when filled."""
    errors: dict[str, str] = {}
    if len(_s(form.get("title"))) < 2:
        errors["title"] = "Please enter a job title."
    if not _s(form.get("category_id")):
        errors["category_id"] = "Please select a job category."
    lo, hi = _s(form.get("salary_min")), _s(form.get("salary_max"))
    for key, val in (("salary_min", lo), ("salary_max", hi)):
        if val and not valid_salary(val):
            errors[key] = "Salary must be a number."
    if lo and hi and valid_salary(lo) and valid_salary(hi) and float(hi) < float(lo):
        errors["salary_max"] = "Max salary can't be less than the minimum."
    if _s(form.get("vacancies")) and not _pos_int(form.get("vacancies")):
        errors["vacancies"] = "Vacancies must be a whole number (1 or more)."
    # A location type must be chosen; a Specific Location needs state + district.
    loc = _s(form.get("job_location_type")).upper()
    if not loc:
        errors["job_location_type"] = "Please choose a job location."
    elif loc == "SPECIFIC":
        if not _s(form.get("state_id")):
            errors["state_id"] = "Please select the job state."
        if not _s(form.get("district_id")):
            errors["district_id"] = "Please select the job district."
    # Candidate Location Preference: at least one district must be selected.
    districts = form.get("preferred_district_ids") or []
    if isinstance(districts, str):
        districts = [districts]
    if not [d for d in (_s(x) for x in districts) if d]:
        errors["preferred_district_ids"] = "Select at least one candidate district."
    # Apply methods: a chosen Phone Call / WhatsApp needs a valid number.
    modes = form.get("apply_modes") or []
    if isinstance(modes, str):
        modes = [modes]
    modes = [_s(m).upper() for m in modes]
    if "CALL" in modes and not valid_phone(form.get("contact_phone")):
        errors["contact_phone"] = "Enter a valid 10-digit contact phone number."
    if "WHATSAPP" in modes and not valid_phone(form.get("contact_whatsapp")):
        errors["contact_whatsapp"] = "Enter a valid 10-digit WhatsApp number."
    # Conditional requirements driven by the experience type.
    exp = _s(form.get("experience_type")).upper()
    if exp == "EXPERIENCED":
        if not _s(form.get("experience_min")) or not _s(form.get("experience_max")):
            errors["experience_min"] = "Select the min and max years of experience."
    elif exp == "INTERN":
        pay = _s(form.get("intern_payment_type")).upper()
        if not pay:
            errors["intern_payment_type"] = "Choose an intern payment type."
        elif pay == "STIPEND" and not _s(form.get("intern_stipend")):
            errors["intern_stipend"] = "Enter the monthly stipend."
        elif pay == "TRAINING_FEE" and not _s(form.get("training_fee")):
            errors["training_fee"] = "Enter the training fee."
    for key in ("experience_min", "experience_max", "intern_stipend", "training_fee",
                "intern_duration_months", "age_min", "age_max"):
        if _s(form.get(key)) and not valid_salary(form.get(key)):   # non-negative number
            errors[key] = "Please enter a valid number."
    for key, label in (("contact_phone", "Contact phone"), ("contact_whatsapp", "WhatsApp")):
        if _s(form.get(key)) and not valid_phone(form.get(key)):
            errors[key] = f"{label} number must be a valid 10-digit mobile."
    return errors
