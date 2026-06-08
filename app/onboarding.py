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

from app.db.repositories import LookupRepository

# ---------------------------------------------------------------------------
# Progressive-profiling field plan
# ---------------------------------------------------------------------------
# INITIAL  — collected in the onboarding form (enough to create the seeker +
#            profile and start browsing).
# APPLY    — the gaps we top up at apply-time, asked ONE at a time and only when
#            the profile doesn't already have them. resume lives on the
#            application; expected_salary lives on the profile.
INITIAL_FIELDS = ("full_name", "email", "years_experience", "preferred_role", "location")

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


async def prepare_registration(*, identity: dict[str, Any], form: dict[str, Any]) -> dict[str, Any]:
    """Resolve the form's role/location text to FK ids (read-only DB) and build
    the registration payload."""
    role_text = (form.get("preferred_role") or "").strip()
    loc_text = (form.get("location") or "").strip()
    role = await LookupRepository.find_job_role(role_text) if role_text else None
    district = await LookupRepository.find_district(loc_text) if loc_text else None
    return build_registration_records(identity=identity, form=form, role=role, district=district)


def build_registration_records(
    *,
    identity: dict[str, Any],
    form: dict[str, Any],
    role: dict[str, Any] | None,
    district: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pure assembly of the DB-shaped records (no I/O), keyed by table name.

    ``resolution`` records what each free-text input matched (or didn't), so an
    unmatched role/location is visible when inspecting the staged data.
    """
    seeker_id = _cuid()
    now = _now_iso()
    name = (identity.get("name") or "").strip()
    email = (form.get("email") or "").strip() or None
    phone = (identity.get("customer_id") or "").strip() or None
    experience = _to_int(form.get("years_experience"))

    seeker = {
        "id": seeker_id,
        "userId": None,
        "fullName": name,
        "email": email,
        "phone": phone,
        "experience": experience,
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
    key_fields = [name, email, phone, experience is not None, role, district]
    completion = round(sum(1 for f in key_fields if f) / len(key_fields) * 100)
    profile = {
        "id": _cuid(),
        "jobSeekerId": seeker_id,
        "relationType": "SELF",
        "fullName": name,
        "email": email,
        "phone": phone,
        "experience": experience,
        "isActive": True,
        "profileCompletion": completion,
        "createdAt": now,
        "updatedAt": now,
    }

    preferred_roles = []
    if role:
        preferred_roles.append(
            {"id": _cuid(), "jobSeekerId": seeker_id, "jobRoleId": role["id"], "createdAt": now}
        )
    locations = []
    if district:
        locations.append(
            {"id": _cuid(), "jobSeekerId": seeker_id, "districtId": district["id"], "createdAt": now}
        )

    return {
        "private_job_seekers": seeker,
        "job_seeker_profiles": profile,
        "private_job_seeker_preferred_roles": preferred_roles,
        "private_job_seeker_locations": locations,
        # so unmatched free-text is visible when inspecting the staged record
        "resolution": {
            "preferred_role": {"input": form.get("preferred_role") or "", "matched": role},
            "location": {"input": form.get("location") or "", "matched": district},
        },
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
