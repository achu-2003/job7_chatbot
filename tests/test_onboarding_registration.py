"""The DB-ready registration payload built from a submitted onboarding form
(staged in Redis; no business-DB write yet)."""
import app.onboarding as ob
from app.onboarding import build_registration_records


_IDENTITY = {
    "name": "Asha Rao", "customer_id": "919876543210",
    "conversation_id": "wa_919876543210", "tenant_id": "default",
}


def test_build_registration_records_full():
    form = {"email": "asha@example.com", "years_experience": "3",
            "preferred_role": "Backend Developer", "location": "Chennai"}
    role = {"id": "role_cuid", "name": "Backend Developer", "slug": "backend-developer"}
    district = {"id": "dist_cuid", "name": "Chennai", "slug": "chennai"}
    out = build_registration_records(identity=_IDENTITY, form=form, role=role, district=district)

    seeker = out["private_job_seekers"]
    assert seeker["fullName"] == "Asha Rao"
    assert seeker["email"] == "asha@example.com"
    assert seeker["phone"] == "919876543210"
    assert seeker["experience"] == 3
    assert seeker["registrationSource"] == "whatsapp"
    assert seeker["onboardingDone"] is True
    assert seeker["id"].startswith("c")

    profile = out["job_seeker_profiles"]
    assert profile["jobSeekerId"] == seeker["id"]
    assert profile["relationType"] == "SELF"
    assert profile["profileCompletion"] == 100        # all 6 key fields present

    roles = out["private_job_seeker_preferred_roles"]
    assert len(roles) == 1
    assert roles[0]["jobRoleId"] == "role_cuid" and roles[0]["jobSeekerId"] == seeker["id"]
    locs = out["private_job_seeker_locations"]
    assert len(locs) == 1 and locs[0]["districtId"] == "dist_cuid"
    assert out["resolution"]["preferred_role"]["matched"]["id"] == "role_cuid"


def test_build_registration_unmatched_role_and_location():
    form = {"email": "sam@x.com", "years_experience": "",
            "preferred_role": "Wizard", "location": "Atlantis"}
    out = build_registration_records(identity=_IDENTITY, form=form, role=None, district=None)
    assert out["private_job_seeker_preferred_roles"] == []   # no FK → no child row
    assert out["private_job_seeker_locations"] == []
    assert out["job_seeker_profiles"]["experience"] is None
    # name + email + phone present (3 of 6) → 50%
    assert out["job_seeker_profiles"]["profileCompletion"] == 50
    assert out["resolution"]["location"]["input"] == "Atlantis"
    assert out["resolution"]["location"]["matched"] is None


def test_known_apply_fields():
    assert ob.known_apply_fields(None) == set()
    reg = {"job_seeker_profiles": {"resume": "http://cv", "expectedSalary": 25000}}
    assert ob.known_apply_fields(reg) == {"resume", "expected_salary"}
    assert ob.known_apply_fields({"job_seeker_profiles": {}}) == set()


def test_parse_salary():
    assert ob.parse_salary("25000") == 25000
    assert ob.parse_salary("₹25,000") == 25000
    assert ob.parse_salary("25k") == 25000
    assert ob.parse_salary("3 lakh") == 300000
    assert ob.parse_salary("nope") is None


def test_build_application_record():
    reg = {"private_job_seekers": {"id": "seeker1"},
           "job_seeker_profiles": {"id": "profile1"}}
    job = {"id": "job-db-1", "title": "Backend Developer"}
    built = ob.build_application_record(
        registration=reg, job=job,
        answers={"resume": "http://cv/me", "expected_salary": 30000},
    )
    app = built["application"]
    assert app["jobId"] == "job-db-1" and app["jobSeekerId"] == "seeker1"
    assert app["profileId"] == "profile1" and app["resume"] == "http://cv/me"
    assert app["status"] == "PENDING" and app["id"].startswith("c")
    # expected_salary belongs on the profile, surfaced for the caller to merge
    assert built["profile_update"] == {"resume": "http://cv/me", "expectedSalary": 30000}


async def test_prepare_registration_resolves_fk_ids(monkeypatch):
    async def fake_role(name):
        return {"id": "r1", "name": name, "slug": "x"}

    async def fake_dist(name):
        return {"id": "d1", "name": name, "slug": "y"}

    monkeypatch.setattr(ob.LookupRepository, "find_job_role", fake_role)
    monkeypatch.setattr(ob.LookupRepository, "find_district", fake_dist)

    form = {"email": "a@b.com", "years_experience": "2",
            "preferred_role": "Dev", "location": "Chennai"}
    out = await ob.prepare_registration(identity=_IDENTITY, form=form)
    assert out["private_job_seeker_preferred_roles"][0]["jobRoleId"] == "r1"
    assert out["private_job_seeker_locations"][0]["districtId"] == "d1"
