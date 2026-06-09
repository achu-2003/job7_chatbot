"""Pure builders for the tappable job-browse flow (no I/O)."""
from app.agent import jobflow


def _job(ref, title, location, smin=None, smax=None, etype="full_time"):
    return {
        "job_ref": ref, "title": title, "location": location,
        "salary_min": smin, "salary_max": smax, "employment_type": etype,
    }


def test_salary_to_lpa():
    assert jobflow.salary_to_lpa(1_000_000, 1_000_000) == "10 LPA"
    assert jobflow.salary_to_lpa(1_500_000, 3_000_000) == "15–30 LPA"
    assert jobflow.salary_to_lpa(None, 1_250_000) == "12.5 LPA"
    assert jobflow.salary_to_lpa(None, None) is None
    assert jobflow.salary_to_lpa(0, 0) is None


def test_salary_display_monthly_vs_annual():
    # Live board stores MONTHLY figures (under ₹1L) → shown as ₹/month, not LPA.
    assert jobflow.salary_display(20000, 50000) == "₹20,000–50,000/month"
    assert jobflow.salary_display(None, 35000) == "₹35,000/month"
    # Annual figures (≥ ₹1L) → LPA.
    assert jobflow.salary_display(1_000_000, 1_000_000) == "10 LPA"
    assert jobflow.salary_display(1_500_000, 3_000_000) == "15–30 LPA"
    assert jobflow.salary_display(None, None) is None


def test_same_role_family():
    assert jobflow.same_role_family("Python Developer", "Senior Python Developer")
    assert jobflow.same_role_family("Frontend Developer", "Frontend Engineer")
    # distinct distinctive words → not the same role
    assert not jobflow.same_role_family("Frontend Developer", "Backend Developer")
    assert not jobflow.same_role_family("QA Engineer", "Python Developer")


def test_infer_wfh():
    assert jobflow.infer_wfh("Remote (India)")
    assert jobflow.infer_wfh("Work From Home")
    assert not jobflow.infer_wfh("Chennai")
    assert not jobflow.infer_wfh(None)


def test_job_card_text_renders_available_fields_and_wfh():
    card = jobflow.job_card_text(
        _job("py-dev-3011", "Remote Python Developer", "Remote (India)", 1_000_000, 1_000_000),
        idx=1,
    )
    assert card.startswith("1️⃣ Remote Python Developer")
    assert "🏠 Work From Home" in card
    assert "💰 10 LPA" in card
    assert "🆔 py-dev-3011" in card
    # no experience data → no experience line
    assert "Years" not in card


def test_job_card_text_shows_location_when_not_remote():
    card = jobflow.job_card_text(_job("r1", "QA Engineer", "Chennai", 800_000, 800_000))
    assert "📍 Chennai" in card
    assert "🏠" not in card


def test_job_card_text_renders_rich_details():
    job = {
        "job_ref": "ssm-tower-office", "title": "Office Staff", "location": "Chennai",
        "work_mode": "OFFICE", "employment_type": "full_time",
        "salary_min": 20000, "salary_max": 45000, "vacancies": 10,
        "experience_min": 0, "experience_max": None,
        "qualification_level": "10TH_ABOVE", "english_level": "BASIC",
        "age_min": 18, "age_max": 35,
        "description": "DIRECT JOINING ONLY\n\n\nCOMPANY NAME: SSM PRIVATE LIMITED\n\nLOCATION: Chennai Guindy",
    }
    card = jobflow.job_card_text(job, idx=1)
    assert card.startswith("1️⃣ Office Staff")
    assert "📍 Chennai · Office" in card
    assert "💼 Full-time" in card
    assert "🧰 Any experience" in card
    assert "🎓 10th & above" in card
    assert "🗣️ English: Basic" in card
    assert "🎂 Age 18–35 yrs" in card
    assert "💰 ₹20,000–45,000/month" in card
    assert "👥 10 vacancies" in card
    assert "🆔 ssm-tower-office" in card
    assert "COMPANY NAME: SSM PRIVATE LIMITED" in card     # description included
    assert "\n\n\n" not in card                            # blank-line runs collapsed


def test_job_card_text_experience_range_and_caps_description():
    job = {"job_ref": "r", "title": "Dev", "experience_min": 2, "experience_max": 5,
           "description": "word " * 600}
    card = jobflow.job_card_text(job)
    assert "🧰 2–5 yrs experience" in card
    assert len(card) <= 1000                               # capped to WhatsApp body limit


def test_role_list_paginates_with_more_row():
    jobs = [_job(f"r{i}", f"Role {i}", "Chennai") for i in range(15)]
    payload, _ = jobflow.role_list_message(jobs, category="IT", offset=0)
    rows = payload["interactive"]["action"]["sections"][0]["rows"]
    assert len(rows) == 10                       # 9 roles + 1 "More roles" row
    assert rows[-1]["id"] == "more:IT:9"
    assert rows[0]["id"] == "job:r0"
    # last page (offset 9) → remaining 6 roles, no more-row
    payload2, _ = jobflow.role_list_message(jobs, category="IT", offset=9)
    rows2 = payload2["interactive"]["action"]["sections"][0]["rows"]
    assert len(rows2) == 6
    assert all(not r["id"].startswith("more:") for r in rows2)


def test_recommend_list_uses_view_ids():
    jobs = [_job("r1", "QA Engineer", "Chennai"), _job("r2", "Tester", "Remote (India)")]
    payload, text = jobflow.recommend_list_message(jobs, name="Asha")
    rows = payload["interactive"]["action"]["sections"][0]["rows"]
    assert [r["id"] for r in rows] == ["view:r1", "view:r2"]
    assert "Asha" in text


def test_job_cards_have_apply_save_share_buttons():
    jobs = [_job("a1", "Dev", "Pune", 1_000_000, 1_000_000)]
    messages, text = jobflow.job_cards(jobs)
    assert len(messages) == 1
    buttons = messages[0]["interactive"]["action"]["buttons"]
    ids = [b["reply"]["id"] for b in buttons]
    titles = [b["reply"]["title"] for b in buttons]
    assert ids == ["apply:a1", "save:a1", "share:a1"]
    assert titles == ["Apply", "Save", "Share"]
    assert "Dev" in text


def test_job_cards_capped():
    jobs = [_job(f"x{i}", f"Dev {i}", "Pune") for i in range(8)]
    messages, text = jobflow.job_cards(jobs, limit=5)
    assert len(messages) == 5
    assert "3 more" in text


def test_split_id():
    assert jobflow.split_id("job:abc-123") == ("job", "abc-123")
    assert jobflow.split_id("loc:Chennai") == ("loc", "Chennai")
    assert jobflow.split_id("more:IT:9") == ("more", "IT:9")
    assert jobflow.split_id(None) == ("", "")
    assert jobflow.split_id("plainword") == ("", "plainword")
