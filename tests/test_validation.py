"""Job-seeker workflow input validation (app/validation.py)."""
from datetime import date

import pytest

from app import validation as v


@pytest.mark.parametrize("good", ["a@b.com", "first.last@sub.domain.co", "x_y@mail.in"])
def test_valid_email_accepts(good):
    assert v.valid_email(good)


@pytest.mark.parametrize("bad", ["", "a@b", "no-at.com", "a b@c.com", "a@@b.com", "x@y."])
def test_valid_email_rejects(bad):
    assert not v.valid_email(bad)


@pytest.mark.parametrize("good,bad", [("600001", "60001"), ("110011", "012345")])
def test_valid_pincode(good, bad):
    assert v.valid_pincode(good)
    assert not v.valid_pincode(bad)          # 5 digits / leading zero


def test_valid_url():
    assert v.valid_url("https://drive.google.com/file/abc")
    assert v.valid_url("http://x.io/cv.pdf")
    assert not v.valid_url("drive.google.com/abc")     # no scheme
    assert not v.valid_url("ftp://x.com")              # not http(s)
    assert not v.valid_url("just text")


def test_valid_name():
    assert v.valid_name("Prasanth M") and v.valid_name("Mary-Jane O'Neil")
    assert not v.valid_name("") and not v.valid_name("123") and not v.valid_name("@@@")


def test_valid_year():
    assert v.valid_year("2024") and v.valid_year("1999")
    assert not v.valid_year("24") and not v.valid_year("20240")  # wrong length
    assert not v.valid_year("1800") and not v.valid_year("3000") # out of range
    assert not v.valid_year("abcd")


def test_valid_salary():
    assert v.valid_salary("25000") and v.valid_salary("0") and v.valid_salary("18000.5")
    assert not v.valid_salary("-5") and not v.valid_salary("2k") and not v.valid_salary("")


def test_valid_dob_age_window():
    today = date(2026, 6, 12)
    assert v.age_for(date(2000, 1, 1), today=today) == 26
    assert v.valid_dob("2000-01-01")                   # ~26 → ok
    assert not v.valid_dob("2020-01-01")               # ~6 → too young
    assert not v.valid_dob("1900-01-01")               # too old
    assert not v.valid_dob("not-a-date")


def test_valid_resume_filename():
    # only PDF / DOC / DOCX are accepted
    assert v.valid_resume_filename("My_CV.pdf") and v.valid_resume_filename("resume.DOCX")
    assert v.valid_resume_filename("cv.doc")
    # everything else (images, archives, executables) is rejected
    assert not v.valid_resume_filename("song.mp3") and not v.valid_resume_filename("x.exe")
    assert not v.valid_resume_filename("photo.jpg") and not v.valid_resume_filename("scan.png")
    assert not v.valid_resume_filename("notes.rtf") and not v.valid_resume_filename("doc.odt")


def test_validate_registration_requires_name_only():
    # all blank except name → only valid when name is present + plausible
    assert v.validate_registration({"full_name": ""}) == {"full_name": v.validate_registration({"full_name": ""})["full_name"]}
    assert "full_name" in v.validate_registration({})
    assert v.validate_registration({"full_name": "Asha"}) == {}   # name only → clean


def test_validate_registration_flags_bad_formats():
    errs = v.validate_registration({
        "full_name": "Asha", "email": "bad-email", "pincode": "12",
        "year_of_passing": "20", "current_salary": "-3", "date_of_birth": "2021-01-01",
    })
    assert set(errs) == {"email", "pincode", "year_of_passing", "current_salary", "date_of_birth"}


def test_validate_registration_optional_blanks_ok():
    # blank optional fields don't error — only a *provided* bad value does
    assert v.validate_registration({"full_name": "Asha", "email": "", "pincode": ""}) == {}


# --- employer-side validators ----------------------------------------------

def test_valid_phone():
    assert v.valid_phone("9876543210") and v.valid_phone("+91 98765 43210")
    assert v.valid_phone("919876543210")
    assert not v.valid_phone("1234567890")     # starts <6
    assert not v.valid_phone("98765") and not v.valid_phone("notaphone")


def test_valid_gst_and_pan():
    assert v.valid_gst("22AAAAA0000A1Z5")
    assert not v.valid_gst("AAAAA0000A") and not v.valid_gst("22AAAAA0000A1Z")  # short
    assert v.valid_pan("AAAAA0000A") and v.valid_pan("abcde1234f")  # case-insensitive
    assert not v.valid_pan("AAAA0000A") and not v.valid_pan("12345ABCDE")


def test_valid_website():
    assert v.valid_website("acme.com") and v.valid_website("https://sub.acme.co.in/x")
    assert not v.valid_website("acme") and not v.valid_website("just text")


def test_validate_employer():
    assert "company_name" in v.validate_employer({})                    # name required
    assert "district_id" in v.validate_employer({"company_name": "Acme"})  # district required
    ok = {"company_name": "Acme", "district_id": "d1"}
    assert v.validate_employer(ok) == {}                                # minimal valid
    bad = {**ok, "email": "no-at", "website": "bad", "pincode": "12"}
    assert set(v.validate_employer(bad)) == {"email", "website", "pincode"}


def test_validate_kyc():
    assert set(v.validate_kyc({})) == {"kyc_document_type", "kyc_document_url"}  # both required
    ok = {"kyc_document_type": "GST", "kyc_document_url": "https://drive/x"}
    assert v.validate_kyc(ok) == {}
    bad = {**ok, "gst_number": "bad", "pan_number": "bad"}
    assert set(v.validate_kyc(bad)) == {"gst_number", "pan_number"}


# A minimal-but-complete job: title + a (Remote) location (which needs no
# state/district) + one candidate district. Clean base for the conditional tests.
_JOB_OK = {"title": "Welder", "job_location_type": "REMOTE",
           "preferred_district_ids": ["d1"]}


def test_validate_job_post():
    assert "title" in v.validate_job_post({})                          # title required
    assert "job_location_type" in v.validate_job_post({"title": "Welder"})  # location required
    ok = {**_JOB_OK, "salary_min": "20000", "salary_max": "30000",
          "vacancies": "3"}
    assert v.validate_job_post(ok) == {}
    bad = {**_JOB_OK, "salary_min": "30000", "salary_max": "10000",  # max < min
           "vacancies": "abc"}
    errs = v.validate_job_post(bad)
    assert "salary_max" in errs and "vacancies" in errs


def test_validate_job_post_experienced_requires_years():
    base = {**_JOB_OK, "experience_type": "experienced"}
    assert "experience_min" in v.validate_job_post(base)            # years missing
    assert v.validate_job_post({**base, "experience_min": "2", "experience_max": "5"}) == {}


def test_validate_job_post_specific_location_requires_state_district():
    base = {"title": "Welder", "job_location_type": "SPECIFIC", "preferred_district_ids": ["d1"]}
    errs = v.validate_job_post(base)
    assert "state_id" in errs and "district_id" in errs
    assert v.validate_job_post({**base, "state_id": "s1", "district_id": "d1"}) == {}
    # Remote / Company Address don't require a job state/district
    assert v.validate_job_post({**_JOB_OK, "job_location_type": "COMPANY_ADDRESS"}) == {}


def test_validate_job_post_requires_candidate_district():
    # the "Districts *" field must carry at least one district
    base = {"title": "Welder", "job_location_type": "REMOTE"}
    assert "preferred_district_ids" in v.validate_job_post(base)
    assert v.validate_job_post({**base, "preferred_district_ids": ["d1"]}) == {}
    assert "preferred_district_ids" in v.validate_job_post({**base, "preferred_district_ids": []})


def test_validate_job_post_apply_methods_need_numbers():
    # Phone Call selected → a valid contact phone is required.
    assert "contact_phone" in v.validate_job_post({**_JOB_OK, "apply_modes": ["CALL"]})
    assert "contact_phone" in v.validate_job_post({**_JOB_OK, "apply_modes": ["CALL"], "contact_phone": "123"})
    assert v.validate_job_post({**_JOB_OK, "apply_modes": ["CALL"], "contact_phone": "9876543210"}) == {}
    # WhatsApp selected → a valid WhatsApp number is required.
    assert "contact_whatsapp" in v.validate_job_post({**_JOB_OK, "apply_modes": ["WHATSAPP"]})
    assert v.validate_job_post({**_JOB_OK, "apply_modes": ["APPLY", "WHATSAPP"],
                                "contact_whatsapp": "9876543210"}) == {}
    # In-App only → no number needed.
    assert v.validate_job_post({**_JOB_OK, "apply_modes": ["APPLY"]}) == {}


def test_validate_job_post_intern_requires_payment():
    base = {**_JOB_OK, "title": "Intern", "experience_type": "intern"}
    assert "intern_payment_type" in v.validate_job_post(base)       # type missing
    # company pays → stipend required
    assert "intern_stipend" in v.validate_job_post({**base, "intern_payment_type": "STIPEND"})
    assert v.validate_job_post({**base, "intern_payment_type": "STIPEND", "intern_stipend": "10000"}) == {}
    # intern pays → training fee required
    assert "training_fee" in v.validate_job_post({**base, "intern_payment_type": "TRAINING_FEE"})
    assert v.validate_job_post({**base, "intern_payment_type": "TRAINING_FEE", "training_fee": "25000"}) == {}


def test_validate_job_post_any_experience_no_year_requirement():
    # Any/Fresher don't require years or intern fields
    assert v.validate_job_post({**_JOB_OK, "experience_type": "any"}) == {}
    assert v.validate_job_post({**_JOB_OK, "experience_type": "fresher"}) == {}
