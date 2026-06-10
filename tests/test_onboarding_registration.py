"""The DB-ready registration payload built from a submitted onboarding form
(staged in Redis; no business-DB write yet)."""
import app.onboarding as ob
from app.onboarding import build_registration_records


_IDENTITY = {
    "name": "Asha Rao", "customer_id": "919876543210",
    "conversation_id": "wa_919876543210", "tenant_id": "default",
}


_FULL_FORM = {
    "full_name": "Asha Rao", "email": "asha@example.com",
    "gender": "female", "marital_status": "single", "date_of_birth": "2000-01-01",
    "state_id": "st1", "district_id": "d1",
    "current_status": "working", "current_year_of_study": "",
    "education_level_id": "edu1", "course_id": "c1", "specialization_id": "sp1",
    "experience_level_id": "x1", "current_salary": "18000", "expected_salary": "25000",
    "english_proficiency": "intermediate",
    "skill_ids": ["sk1", "sk2"], "preferred_location_ids": ["d1", "d2"],
    "preferred_category_ids": ["cat1"], "preferred_role_ids": ["r1", "r2"],
}


def test_build_registration_records_full():
    out = build_registration_records(identity=_IDENTITY, form=_FULL_FORM)

    seeker = out["private_job_seekers"]
    assert seeker["fullName"] == "Asha Rao"
    assert seeker["email"] == "asha@example.com"
    assert seeker["phone"] == "919876543210"
    # enum fields normalised to the UPPER form the DB stores
    assert seeker["gender"] == "FEMALE" and seeker["maritalStatus"] == "SINGLE"
    assert seeker["englishProficiency"] == "INTERMEDIATE" and seeker["currentStatus"] == "WORKING"
    # FK ids map straight through
    assert seeker["educationLevelId"] == "edu1" and seeker["experienceLevelId"] == "x1"
    assert seeker["preferredStateId"] == "st1" and seeker["districtId"] == "d1"
    assert seeker["currentSalary"] == 18000.0 and seeker["expectedSalary"] == 25000.0
    assert seeker["registrationSource"] == "WHATSAPP_BOT" and seeker["onboardingDone"] is True
    assert seeker["id"].startswith("c")

    profile = out["job_seeker_profiles"]
    assert profile["jobSeekerId"] == seeker["id"] and profile["relationType"] == "SELF"
    assert profile["profileCompletion"] == 100        # all key fields present

    sid = seeker["id"]
    assert [r["skillId"] for r in out["private_job_seeker_skills"]] == ["sk1", "sk2"]
    assert [r["districtId"] for r in out["private_job_seeker_locations"]] == ["d1", "d2"]
    assert [r["jobRoleId"] for r in out["private_job_seeker_preferred_roles"]] == ["r1", "r2"]
    assert [r["categoryId"] for r in out["private_job_seeker_categories"]] == ["cat1"]
    assert all(r["jobSeekerId"] == sid for r in out["private_job_seeker_skills"])


def test_build_registration_maps_email_salary_institution_resume():
    form = {
        "full_name": "Asha Rao", "email": "asha@example.com",
        "current_salary": "18000", "institution": "Anna University",
        "resume": "https://drive.google.com/cv", "preferred_location_ids": ["d1"],
    }
    out = build_registration_records(identity=_IDENTITY, form=form)
    seeker, profile = out["private_job_seekers"], out["job_seeker_profiles"]
    assert seeker["email"] == "asha@example.com"
    assert seeker["currentSalary"] == 18000.0
    assert seeker["institution"] == "Anna University"
    assert seeker["resumes"] == ["https://drive.google.com/cv"]
    assert profile["institution"] == "Anna University"
    assert profile["resumes"] == ["https://drive.google.com/cv"]


def test_build_registration_minimal():
    # Only the required fields → empty preference rows, partial completion.
    form = {"full_name": "Sam", "email": "sam@x.com", "preferred_location_ids": ["d9"]}
    out = build_registration_records(identity=_IDENTITY, form=form)
    assert out["private_job_seeker_preferred_roles"] == []
    assert out["private_job_seeker_skills"] == []
    assert out["private_job_seeker_categories"] == []
    assert [r["districtId"] for r in out["private_job_seeker_locations"]] == ["d9"]
    assert out["private_job_seekers"]["currentSalary"] is None
    # name + email + phone + location present (4 of 12 key fields) → 33%
    assert out["job_seeker_profiles"]["profileCompletion"] == 33


async def test_prepare_registration_passes_ids_through():
    # The dropdowns are id-valued, so prepare_registration no longer resolves
    # any text — it just assembles the records from the submitted ids.
    out = await ob.prepare_registration(identity=_IDENTITY, form=_FULL_FORM)
    assert out["private_job_seeker_preferred_roles"][0]["jobRoleId"] == "r1"
    assert out["private_job_seeker_locations"][0]["districtId"] == "d1"


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


