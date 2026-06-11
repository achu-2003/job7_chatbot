"""Employer (job-poster) Stage 1-2 — record builder + form rendering (pure)."""
from app.employer import apply_kyc, build_employer_record, build_job_record
from app.api.routes.employer import (
    _expired_html,
    _kyc_html,
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
        "industry_id": "ind1",
        "company_size": "medium",
        "contact_person": "Asha",
        "designation_id": "des1",
        "email": "hr@acme.com",
        "website": "https://acme.com",
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
    assert emp["industryId"] == "ind1"
    assert emp["designationId"] == "des1"
    assert emp["companySize"] == "MEDIUM"            # normalised to the enum
    assert emp["primaryPhone"] == "919876543210"
    assert emp["districtId"] == "d1"
    assert emp["status"] == "PENDING"                # EmployerStatus default
    assert emp["kycStatus"] == "NOT_SUBMITTED"       # KycStatus default
    assert emp["phoneVerified"] is True
    assert emp["registrationSource"] == "WHATSAPP_BOT"


def test_build_employer_record_blank_company_still_gets_slug():
    rec = build_employer_record(identity={"customer_id": "91"}, form={"company_name": ""})
    emp = rec["private_employers"]
    assert emp["companyName"] == ""
    assert emp["slug"]                                # falls back to the id


def test_apply_kyc_auto_verify_flips_status():
    rec = build_employer_record(identity={"customer_id": "91"}, form={"company_name": "Acme"})
    apply_kyc(rec, doc_type="GST", doc_url="https://drive/x", gst="22AAAAA0000A1Z5",
              pan=None, auto_verify=True)
    emp = rec["private_employers"]
    assert emp["kycStatus"] == "VERIFIED"
    assert emp["status"] == "APPROVED"
    assert emp["gstNumber"] == "22AAAAA0000A1Z5"
    assert emp["kycDocumentType"] == "GST"
    assert emp["kycVerifiedAt"]


def test_apply_kyc_manual_review_stays_pending():
    rec = build_employer_record(identity={"customer_id": "91"}, form={"company_name": "Acme"})
    apply_kyc(rec, doc_type="PAN", doc_url="https://drive/x", gst=None, pan="AAAAA0000A",
              auto_verify=False)
    assert rec["private_employers"]["kycStatus"] == "PENDING"
    assert rec["private_employers"]["status"] == "PENDING"


def test_register_form_renders_company_fields():
    out = _register_html("tok123", "919876543210", _OPTS)
    assert 'name="token" value="tok123"' in out
    assert 'action="/employer/register/submit"' in out
    assert 'name="company_name"' in out and 'name="industry_id"' in out
    assert 'name="company_size"' in out and 'name="designation_id"' in out
    assert 'value="919876543210" readonly' in out
    assert "Information Technology" in out and "HR Manager" in out
    assert "STARTUP" in out and "ENTERPRISE" in out          # CompanySize options
    assert "var DISTRICTS =" in out                          # cascade JS
    # State + District are REQUIRED (district is the job-location fallback)
    assert 'name="state_id" id="state_id" autocomplete="off" required' in out
    assert 'name="district_id" id="district_id" autocomplete="off" required' in out


def test_register_form_shows_error_banner():
    out = _register_html("t", "91", _OPTS, error="Please fill the required fields.")
    assert "Please fill the required fields." in out


def test_kyc_form_renders_doc_fields():
    out = _kyc_html("tok123")
    assert 'action="/employer/kyc/submit"' in out
    assert 'name="kyc_document_type"' in out
    assert 'name="gst_number"' in out and 'name="pan_number"' in out
    assert 'name="kyc_document_url"' in out
    assert "GST Certificate" in out


def test_post_job_form_is_a_five_step_wizard():
    out = _post_job_html("tok123", _JOB_OPTS, company_district_id="d1")
    assert 'action="/employer/post-job/submit"' in out
    assert out.count('class="step"') == 5                  # five wizard sections
    assert 'id="jobForm"' in out and 'id="next"' in out and 'id="post"' in out
    assert 'var NAMES = ["Job Details"' in out             # stepper JS
    # one field/marker from each remaining section
    for nm in ("title", "job_type", "experience_type", "job_location_type",
               "preferred_district_ids", "apply_modes", "contact_whatsapp"):
        assert f'name="{nm}"' in out, nm
    # the removed sections are gone
    for gone in ("qualification_level", "gender_preference", "english_level",
                 "candidate_distance", "has_security_deposit", "work_start_time",
                 "interview_date", "category_id", "preferred_languages", "required_assets"):
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
