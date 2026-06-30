"""Employer (job-poster) Stage 1-2 — record builder + form rendering (pure)."""
import types
from urllib.parse import urlencode

from app.employer import build_employer_record, build_job_record
from app.api.routes.employer import (
    _expired_html,
    _post_job_html,
    _register_html,
    _success_html,
)

_OPTS = {
    "industries": [{"id": "ind1", "name": "Information Technology"}],
    "designations": [{"id": "des1", "name": "HR Manager"}],
    "states": [{"id": "st1", "name": "Tamil Nadu"}],
    "districts": [{"id": "d1", "name": "Chennai", "parent": "st1"}],
    "categories": [{"id": "cat1", "name": "Engineering"}],
    "experience_levels": [{"id": "x1", "name": "Freshers"}],
}
_JOB_OPTS = {
    "categories": [{"id": "cat1", "name": "Engineering"}],
    "districts": [{"id": "d1", "name": "Chennai"}],
    "states": [{"id": "s1", "name": "Tamil Nadu"}],
}


def test_build_employer_record_maps_private_employers_columns():
    identity = {"customer_id": "919876543210", "name": "Asha"}
    form = {
        "company_name": "Acme Technologies",
        "email": "hr@acme.com",
        "address": "1 MG Road",
        "district_id": "d1",
        "city": "Chennai",
        "pincode": "600001",
        "description": "We build things",
    }
    rec = build_employer_record(identity=identity, form=form)
    emp = rec["private_employers"]
    assert emp["companyName"] == "Acme Technologies"
    assert emp["slug"] == "acme-technologies"
    # industry / company size / contact person / designation were removed
    assert emp["industryId"] is None and emp["designationId"] is None
    assert emp["companySize"] is None and emp["contactPerson"] is None
    assert emp["primaryPhone"] == "919876543210"
    assert emp["districtId"] == "d1"
    # No KYC step — a created profile is immediately ready/verified.
    assert emp["status"] == "APPROVED"
    assert emp["kycStatus"] == "VERIFIED"
    assert emp["phoneVerified"] is True
    assert emp["registrationSource"] == "WHATSAPP_BOT"


def test_build_employer_record_blank_company_still_gets_slug():
    rec = build_employer_record(identity={"customer_id": "91"}, form={"company_name": ""})
    emp = rec["private_employers"]
    assert emp["companyName"] == ""
    assert emp["slug"]                                # falls back to the id


def test_register_form_renders_company_fields():
    out = _register_html("tok123", "919876543210", _OPTS)
    assert 'name="token" value="tok123"' in out
    assert 'action="/employer/register/submit"' in out
    assert 'name="company_name"' in out
    # email / phone / about-company removed (phone is taken from the WhatsApp identity)
    assert 'name="email"' not in out and 'name="primary_phone"' not in out
    assert 'name="description"' not in out
    assert "var DISTRICTS =" in out                          # cascade JS
    # address fields live in a collapsible accordion (expanded by default)
    assert 'class="acc open" id="addrAcc"' in out and 'id="addrHd"' in out
    # Industry / Company size / Contact person / Designation were removed
    assert 'name="industry_id"' not in out and 'name="company_size"' not in out
    assert 'name="contact_person"' not in out and 'name="designation_id"' not in out
    # State + District are REQUIRED (district is the job-location fallback)
    assert 'name="state_id" id="state_id" autocomplete="off" required' in out
    assert 'name="district_id" id="district_id" autocomplete="off" required' in out


def test_register_form_shows_error_banner():
    out = _register_html("t", "91", _OPTS, error="Please fill the required fields.")
    assert "Please fill the required fields." in out


def test_post_job_form_is_a_five_step_wizard():
    out = _post_job_html("tok123", _JOB_OPTS, company_district_id="d1")
    assert 'action="/employer/post-job/submit"' in out
    assert out.count('class="step"') == 5                  # five wizard sections
    assert 'id="jobForm"' in out and 'id="next"' in out and 'id="post"' in out
    # stepper JS: the "&" in the step name must NOT be HTML-escaped here — it's a
    # JS string literal, so &amp; would break the in-script NAMES[cur] comparison.
    assert 'var NAMES = ["Job Details", "Experience & Salary"' in out
    # the experience-step guard detects the step by field presence (robust to name)
    assert "steps[cur].querySelector('[name=experience_type]')" in out
    # Job Location guard: Specific Location requires state + district before Next
    assert "steps[cur].querySelector('[name=job_location_type]')" in out
    assert "Please select the job state." in out and "Please select the job district." in out
    # one field/marker from each remaining section
    for nm in ("title", "category_id", "job_type", "experience_type", "job_location_type",
               "preferred_district_ids", "apply_modes", "contact_whatsapp"):
        assert f'name="{nm}"' in out, nm
    # the Job Category select is populated from the categories lookup
    assert '<select name="category_id"' in out
    assert '<option value="cat1">Engineering</option>' in out
    # the removed sections are gone
    for gone in ("qualification_level", "gender_preference", "english_level",
                 "candidate_distance", "has_security_deposit", "work_start_time",
                 "interview_date", "preferred_languages", "required_assets"):
        assert f'name="{gone}"' not in out, gone


def test_post_job_form_candidate_location_and_apply_methods():
    """Candidate Location Preference carries the quick-select + district chips +
    credit line; Apply Methods defaults to In-App and reveals contact inputs."""
    out = _post_job_html("tok123", _JOB_OPTS, company_district_id="d1")
    for qs in ('data-qs="company"', 'data-qs="nearby"', 'data-qs="all"',
               'data-qs="top"', 'data-qs="custom"'):
        assert qs in out, qs
    assert 'id="pref_add"' in out and 'id="creditInfo"' in out
    assert 'COMPANY_DISTRICT = "d1"' in out and "function quickSelect(" in out
    # Apply Methods: In-App checked by default; phone/whatsapp reveal blocks + JS
    assert 'value="APPLY" id="am_apply" checked' in out
    assert 'id="phoneInput"' in out and 'id="waInput"' in out
    assert "function applyChange()" in out
    # the final "Post Job" (type=submit) is guarded through valid()
    assert "addEventListener('submit', function(e)" in out
    # a chosen Phone Call / WhatsApp requires its number before posting
    assert "Please enter the contact phone number." in out
    assert "Please enter the WhatsApp number." in out
    # an unchosen job location is blocked too
    assert "Please choose a job location." in out
    # candidate location preference requires at least one district
    assert "Please select at least one candidate district." in out


def test_post_job_form_conditional_blocks_no_default_radio():
    """No radio (job type / experience / salary period / location) is pre-selected,
    and the Experience + Location conditional blocks + JS are present."""
    out = _post_job_html("tok123", _JOB_OPTS, company_address="1 MG Rd")
    # radios are unchecked — the only `checked` is the Apply In-App checkbox
    assert out.count(" checked") == 1
    assert 'value="APPLY" id="am_apply" checked' in out
    for block in ('id="expYears"', 'id="internBlock"', 'id="salaryBlock"',
                  'id="stipendInput"', 'id="trainingInput"',
                  'id="locSpecific"', 'id="locCompany"', 'id="locRemote"'):
        assert block in out, block
    assert "function expChange()" in out and "function locChange()" in out
    assert "function locChange()" in out


def test_post_job_form_salary_and_vacancies_required_inline():
    """Min/Max salary + Vacancies are required; errors are shown inline (red
    border + message), never as alert() popups."""
    out = _post_job_html("tok123", _JOB_OPTS, company_address="1 MG Rd")
    assert 'name="salary_min" data-req="1"' in out
    assert 'name="salary_max" data-req="1"' in out
    assert 'name="vacancies" data-req="1"' in out
    # Salary Range (Monthly/Annual) is required whenever the salary block shows
    assert 'Salary Range <span class="req">*</span>' in out
    assert "Please choose Monthly or Annual." in out
    # inline error machinery present; no alert() inside the wizard validator
    assert "function showErr(" in out and "function clearErrs(" in out
    assert "e.className='field-err'" in out
    valid_body = out.split("function valid()")[1].split("next.onclick")[0]
    assert "alert(" not in valid_body


class _FakeRegMemory:
    """Minimal memory for register_submit: identity + idempotent SET-NX + save."""
    def __init__(self, phone="919876543210", tenant="t1"):
        self.phone, self.tenant = phone, tenant
        self.saved = None
        self._seen: set[str] = set()
        self.save_calls = 0

    async def get_employer_identity(self, token):
        return {"tenant_id": self.tenant, "customer_id": self.phone, "name": "Asha"}

    async def get_employer(self, phone, *, tenant_id=None):
        return self.saved

    async def mark_seen(self, key, *, ttl=600):
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    async def clear_seen(self, key):
        self._seen.discard(key)

    async def save_employer(self, phone, record, *, tenant_id=None):
        self.saved = record
        self.save_calls += 1


def _reg_request(mem, **fields):
    body = urlencode({"token": "tok", "company_name": "Acme", "district_id": "d1", **fields}).encode()
    app = types.SimpleNamespace(state=types.SimpleNamespace(memory=mem))
    return types.SimpleNamespace(app=app, body=lambda: _coro(body))


async def _coro(v):
    return v


async def test_register_double_submit_is_idempotent(monkeypatch):
    """A double-tap on the register submit creates ONE profile and pushes the hub
    ONCE (no duplicate company profile). The hub is two bubbles — a Post a Job button
    + a Menu list — so a single registration sends exactly two messages."""
    import app.api.routes.employer as emp
    pushes = []

    async def _push(*a, **k):
        pushes.append(1)
    monkeypatch.setattr("app.whatsapp.delivery.send_message", _push)

    mem = _FakeRegMemory()
    # First submit → creates + pushes the two hub bubbles.
    out1 = await emp.register_submit(_reg_request(mem))
    # Second (double-tap) → already exists → no re-save, no re-push.
    out2 = await emp.register_submit(_reg_request(mem))

    assert mem.save_calls == 1, "profile saved only once"
    assert len(pushes) == 2, "hub pushed only once (two bubbles: Post a Job + Menu)"
    assert "created" in out1.body.decode().lower() and "created" in out2.body.decode().lower()


def test_private_employers_insert_sql_has_enum_casts_and_updated_at():
    """The live INSERT casts the enum columns and sets created/updatedAt."""
    from app.db.repositories import _insert_sql, _prep_row
    rec = build_employer_record(
        identity={"customer_id": "919876543210"},
        form={"company_name": "Acme", "district_id": "d1"})["private_employers"]
    sql = _insert_sql("private_employers", _prep_row("private_employers", rec))
    assert 'CAST(:status AS "EmployerStatus")' in sql
    assert 'CAST(:kycStatus AS "KycStatus")' in sql
    assert '"createdAt"' in sql and '"updatedAt"' in sql and "now(), now()" in sql


async def test_register_writes_to_live_db_when_flag_on(monkeypatch):
    """With EMPLOYER_REGISTER_IN_DB on, the submit also writes to private_employers
    (idempotent, best-effort) and adopts the live id."""
    import app.api.routes.employer as emp
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "employer_register_in_db", True)

    async def _noop_push(*a, **k):
        return None
    monkeypatch.setattr("app.whatsapp.delivery.send_message", _noop_push)

    created = []

    async def fake_create(payload, *, commit=True, welcome_job_credits=0):
        created.append({"company": payload["private_employers"]["companyName"],
                        "welcome": welcome_job_credits})
        return {"committed": True, "inserted": True, "employer_id": "live-emp-1"}
    monkeypatch.setattr(emp.EmployerRepository, "create", staticmethod(fake_create))

    mem = _FakeRegMemory()
    await emp.register_submit(_reg_request(mem))
    assert len(created) == 1 and created[0]["company"] == "Acme"   # written once
    assert created[0]["welcome"] == 1                # welcome job credit granted live
    assert mem.saved["private_employers"]["id"] == "live-emp-1"   # adopted live id


def test_pages_have_brand_and_back_to_chat(monkeypatch):
    """Every employer page carries the Jobs7 brand header + a Back-to-chat link."""
    from app.api.routes.employer import _page
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "whatsapp_business_number", "919876543210")
    out = _page("x", "<p>body</p>")
    assert 'class="brand"' in out and 'class="brand-img"' in out and "/static/logo.png" in out
    assert 'class="backchat"' in out and "https://wa.me/919876543210" in out


def test_native_forms_use_inline_validator():
    """The register form disables native popup bubbles (novalidate) and validates
    inline (red border + message) via the shared attribute-driven script."""
    from app.api.routes.employer import _register_html
    reg = _register_html("t", "919876543210", _JOB_OPTS)
    assert "data-validate" in reg
    assert "form[data-validate]" in reg and "novalidate" in reg
    assert "d.className='field-err'" in reg


def test_post_job_form_salary_and_jobtype_trimmed():
    """Salary range is only Monthly/Annual (no per-day/hour), no negotiable toggle,
    and Job Type is only Full Time / Part Time."""
    out = _post_job_html("tok123", _JOB_OPTS, company_address="1 MG Rd")
    assert "Monthly" in out and "Annual" in out
    assert "Per day" not in out and "Per hour" not in out
    assert 'name="salary_negotiable"' not in out
    assert "Full Time" in out and "Part Time" in out
    assert "Internship" not in out and "Freelance" not in out  # job type trimmed
    assert "1 MG Rd" in out                                     # company address shown


def test_build_job_record_intern_payment():
    rec = build_job_record(employer_id="e1", form={
        "title": "Design Intern", "experience_type": "intern",
        "intern_payment_type": "training_fee", "training_fee": "25000",
        "intern_duration_months": "6",
    }, status="PENDING")
    j = rec["private_jobs"]
    assert j["experienceType"] == "INTERN"
    assert j["internPaymentType"] == "TRAINING_FEE"
    assert j["trainingFee"] == 25000 and j["internDurationMonths"] == 6
    assert j["salaryMin"] is None                          # interns have no salary range


def test_build_job_record_maps_all_private_jobs_columns():
    form = {
        "title": "Senior Welder", "description": "Weld things well.",
        "job_type": "contract", "experience_type": "experienced",
        "experience_min": "2", "experience_max": "5",
        "salary_period": "monthly", "salary_min": "20000", "salary_max": "30000",
        "vacancies": "3",
        "job_location_type": "specific",
        "state_id": "s1", "district_id": "d1", "city": "Chennai", "work_from_home": "",
        "qualification_level": "12th_pass", "gender_preference": "both",
        "marital_status_preference": "any", "english_level": "intermediate",
        "age_min": "18", "age_max": "35",
        "candidate_distance": "15", "willing_to_relocate": "yes",
        "has_security_deposit": "yes", "security_deposit_amt": "300", "security_deposit_reason": "ID",
        "work_start_time": "09:00", "work_end_time": "18:00",
        "interview_date": "2026-06-25", "interview_time": "11:00",
        "category_id": "cat1", "skills": "Welding, Fitting, ",
        "preferred_languages": ["Tamil", "English"], "required_assets": ["Bike", "Aadhar"],
        "apply_modes": ["APPLY", "CALL"], "contact_phone": "9876543210",
        "contact_whatsapp": "9876543210",
    }
    rec = build_job_record(employer_id="emp123", form=form, status="PENDING")
    assert rec["ref"].startswith("JOB-")
    j = rec["private_jobs"]
    assert j["title"] == "Senior Welder" and j["slug"] == "senior-welder"
    assert j["employerId"] == "emp123" and j["categoryId"] == "cat1" and j["districtId"] == "d1"
    assert j["jobType"] == "CONTRACT" and j["workMode"] == "OFFICE"   # OFFICE: not remote
    assert j["experienceType"] == "EXPERIENCED" and j["experienceMin"] == 2 and j["experienceMax"] == 5
    assert j["salaryMin"] == 20000 and j["salaryMax"] == 30000 and j["salaryNegotiable"] is False
    assert j["jobLocationType"] == "SPECIFIC" and j["jobStateId"] == "s1"
    assert j["qualificationLevel"] == "12TH_PASS" and j["genderPreference"] == "BOTH"
    assert j["maritalStatusPreference"] == "ANY" and j["englishLevel"] == "INTERMEDIATE"
    assert j["ageMin"] == 18 and j["ageMax"] == 35
    assert j["candidateRadius"] == 15 and j["candidateRadiusType"] == "15"
    assert j["willingToRelocate"] is True and j["hasSecurityDeposit"] is True
    assert j["securityDepositAmt"] == 300
    assert j["workStartTime"] == "09:00" and j["interviewDate"] == "2026-06-25"
    assert j["skills"] == ["Welding", "Fitting"]
    assert j["preferredLanguages"] == ["Tamil", "English"] and j["requiredAssets"] == ["Bike", "Aadhar"]
    assert j["applyModes"] == ["APPLY", "CALL"] and j["contactWhatsapp"] == "9876543210"
    assert j["status"] == "PENDING"


def test_build_job_record_remote_sets_workmode():
    """A Remote job derives workMode=REMOTE and isWorkFromHome=True from the
    location type (there's no separate work-mode field)."""
    rec = build_job_record(
        employer_id="e1",
        form={"title": "Remote Dev", "job_location_type": "remote"}, status="PENDING")
    j = rec["private_jobs"]
    assert j["jobLocationType"] == "REMOTE"
    assert j["workMode"] == "REMOTE" and j["isWorkFromHome"] is True


def test_build_job_record_distance_anywhere_and_defaults():
    rec = build_job_record(
        employer_id=None,
        form={"title": "Helper", "candidate_distance": "anywhere"}, status="PENDING",
    )
    j = rec["private_jobs"]
    assert j["jobType"] == "FULL_TIME" and j["workMode"] == "OFFICE"
    assert j["salaryPeriod"] == "MONTHLY" and j["vacancies"] == 1
    assert j["candidateRadius"] is None and j["candidateRadiusType"] == "ANYWHERE"
    assert j["applyModes"] == ["APPLY"]                    # default apply mode
    assert j["slug"] == "helper"


def test_success_and_expired_pages():
    assert "wa.me/919876543210" in _success_html("Done!", "x", business_number="919876543210")
    assert "window.close()" in _success_html("Done!", "x")
    assert "expired" in _expired_html().lower()
