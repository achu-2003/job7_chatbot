"""Build a DB-ready *registration payload* from a submitted onboarding form.

This assembles the exact rows that WOULD be inserted to register a new candidate
on the live job board — ``private_job_seekers`` + ``job_seeker_profiles`` + the
preferred-role / location child rows — with the free-text role/location resolved
to their FK ids by READING the live lookup tables.

Nothing here writes to the business DB. The caller stages the payload in Redis so
the data shape + FK resolution can be verified before flipping to real INSERTs.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Progressive-profiling field plan
# ---------------------------------------------------------------------------
# INITIAL  — collected in the onboarding form (Step 1-5): enough to create the
#            seeker + profile + preference child rows and start browsing.
# APPLY    — the gaps we top up at apply-time, asked ONE at a time and only when
#            the profile doesn't already have them. resume lives on the
#            application; expected_salary lives on the profile.
INITIAL_FIELDS = (
    "full_name", "email", "gender", "marital_status", "state_id", "district_id",
    "current_status", "current_year_of_study", "education_level_id", "course_id",
    "specialization_id", "experience_level_id", "current_salary", "expected_salary",
    "english_proficiency", "skill_ids", "preferred_location_ids",
    "preferred_category_ids", "preferred_role_ids",
)

APPLY_FIELDS = (
    {
        "key": "resume",
        "numeric": False,
        "prompt": "Almost done! 📎 Tap the attachment (clip) icon and send your "
                  "resume as a document — PDF or DOC. You can also paste a link, "
                  "or tap Skip.",
    },
    {
        "key": "expected_salary",
        "numeric": True,
        "prompt": "What monthly salary are you expecting (in ₹)? Reply a number, "
                  "or 'skip'.",
    },
)
APPLY_FIELD_BY_KEY = {f["key"]: f for f in APPLY_FIELDS}
_SKIP_WORDS = {"skip", "no", "none", "n/a", "na", "later"}


def is_skip(text: str | None) -> bool:
    return (text or "").strip().lower() in _SKIP_WORDS


def parse_salary(text: str | None) -> float | None:
    """Pull a number out of a salary reply ('25k', '₹25,000', '25000')."""
    raw = (text or "").strip().lower().replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(k|l|lakh|lpa)?", raw)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "k":
        val *= 1_000
    elif unit in {"l", "lakh", "lpa"}:
        val *= 100_000
    return val if val > 0 else None


def _cuid() -> str:
    """A placeholder id for a staged (Redis-only) record. The live DB generates a
    real cuid on actual insert; this just makes the staged row complete."""
    return "c" + uuid.uuid4().hex[:24]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _to_num(value: Any) -> float | None:
    """Parse a numeric salary field; None for blank/invalid/negative."""
    try:
        v = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return v if v >= 0 else None


def _id(form: dict[str, Any], key: str) -> str | None:
    v = (form.get(key) or "").strip() if isinstance(form.get(key), str) else form.get(key)
    return (v or None) if isinstance(v, str) else (str(v) if v else None)


def _enum(form: dict[str, Any], key: str) -> str | None:
    """A fixed-enum text field, normalised to the UPPER form the DB stores."""
    v = (form.get(key) or "").strip().upper()
    return v or None


def _id_list(value: Any) -> list[str]:
    """The selected ids of a multi-select — a list (form post) or a comma string."""
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    if value:
        return [s.strip() for s in str(value).split(",") if s.strip()]
    return []


async def prepare_registration(*, identity: dict[str, Any], form: dict[str, Any]) -> dict[str, Any]:
    """Build the registration payload. The form already carries FK ids (the
    dropdowns are id-valued), so no DB resolution is needed here."""
    return build_registration_records(identity=identity, form=form)


def build_registration_records(
    *, identity: dict[str, Any], form: dict[str, Any]
) -> dict[str, Any]:
    """Pure assembly of the DB-shaped records (no I/O), keyed by table name. The
    form's dropdown selections are FK ids, so they map straight onto the seeker /
    profile columns and the preference child rows."""
    seeker_id = _cuid()
    now = _now_iso()
    name = (identity.get("name") or form.get("full_name") or "").strip()
    email = (form.get("email") or "").strip() or None
    phone = (identity.get("customer_id") or "").strip() or None

    education_level = _id(form, "education_level_id")
    course = _id(form, "course_id")
    specialization = _id(form, "specialization_id")
    experience_level = _id(form, "experience_level_id")
    state = _id(form, "state_id")
    district = _id(form, "district_id")
    gender = _enum(form, "gender")
    marital = _enum(form, "marital_status")
    english = _enum(form, "english_proficiency")
    current_status = _enum(form, "current_status")
    year_of_study = _to_int(form.get("current_year_of_study"))
    current_salary = _to_num(form.get("current_salary"))
    expected_salary = _to_num(form.get("expected_salary"))

    skill_ids = _id_list(form.get("skill_ids"))
    location_ids = _id_list(form.get("preferred_location_ids"))
    role_ids = _id_list(form.get("preferred_role_ids"))
    category_ids = _id_list(form.get("preferred_category_ids"))

    seeker = {
        "id": seeker_id,
        "userId": None,
        "fullName": name,
        "email": email,
        "phone": phone,
        "gender": gender,
        "maritalStatus": marital,
        "englishProficiency": english,
        "currentStatus": current_status,
        "currentYearOfStudy": year_of_study,
        "educationLevelId": education_level,
        "courseId": course,
        "specializationId": specialization,
        "experienceLevelId": experience_level,
        "preferredStateId": state,
        "districtId": district,
        "currentSalary": current_salary,
        "expectedSalary": expected_salary,
        "status": "ACTIVE",
        "emailVerified": False,
        "phoneVerified": bool(phone),
        "onboardingDone": True,
        "willingToRelocate": False,
        "registrationSource": "whatsapp",
        "createdAt": now,
        "updatedAt": now,
    }

    # profileCompletion ≈ fraction of the key fields we actually captured.
    key_fields = [
        name, email, phone, education_level, experience_level, district,
        bool(skill_ids), bool(location_ids),
    ]
    completion = round(sum(1 for f in key_fields if f) / len(key_fields) * 100)
    profile = {
        "id": _cuid(),
        "jobSeekerId": seeker_id,
        "relationType": "SELF",
        "fullName": name,
        "email": email,
        "phone": phone,
        "gender": gender,
        "maritalStatus": marital,
        "englishProficiency": english,
        "currentStatus": current_status,
        "currentYearOfStudy": year_of_study,
        "educationLevelId": education_level,
        "courseId": course,
        "specializationId": specialization,
        "experienceLevelId": experience_level,
        "districtId": district,
        "currentSalary": current_salary,
        "expectedSalary": expected_salary,
        "isActive": True,
        "profileCompletion": completion,
        "createdAt": now,
        "updatedAt": now,
    }

    def _children(fk: str, ids: list[str]) -> list[dict[str, Any]]:
        return [
            {"id": _cuid(), "jobSeekerId": seeker_id, fk: i, "createdAt": now}
            for i in ids
        ]

    return {
        "private_job_seekers": seeker,
        "job_seeker_profiles": profile,
        "private_job_seeker_skills": _children("skillId", skill_ids),
        "private_job_seeker_locations": _children("districtId", location_ids),
        "private_job_seeker_preferred_roles": _children("jobRoleId", role_ids),
        "private_job_seeker_categories": _children("categoryId", category_ids),
    }


# ---------------------------------------------------------------------------
# apply-time (progressive top-up)
# ---------------------------------------------------------------------------


def known_apply_fields(registration: dict[str, Any] | None) -> set[str]:
    """Which apply-time fields the profile already has (so we don't re-ask)."""
    profile = (registration or {}).get("job_seeker_profiles") or {}
    known: set[str] = set()
    if profile.get("resume") or profile.get("resumes"):
        known.add("resume")
    if profile.get("expectedSalary") is not None:
        known.add("expected_salary")
    return known


def build_application_record(
    *, registration: dict[str, Any], job: dict[str, Any], answers: dict[str, Any]
) -> dict[str, Any]:
    """The DB-ready ``private_job_applications`` row. ``resume`` is stored on the
    application; ``expected_salary`` belongs on the profile, so it's surfaced
    separately under ``profile_update`` for the caller to merge into the staged
    profile."""
    seeker = registration.get("private_job_seekers") or {}
    profile = registration.get("job_seeker_profiles") or {}
    now = _now_iso()
    application = {
        "id": _cuid(),
        "jobId": job.get("id"),
        "jobSeekerId": seeker.get("id"),
        "profileId": profile.get("id"),
        "resume": answers.get("resume"),
        "coverLetter": answers.get("cover_note"),
        "screeningAnswers": None,
        "status": "PENDING",
        "appliedAt": now,
        "updatedAt": now,
    }
    profile_update: dict[str, Any] = {}
    if answers.get("resume"):
        profile_update["resume"] = answers["resume"]
    if answers.get("expected_salary") is not None:
        profile_update["expectedSalary"] = answers["expected_salary"]
    return {"application": application, "profile_update": profile_update}
