"""Onboarding 9-step wizard HTML rendering (pure helpers — no Redis needed)."""
from app.api.routes.onboard import _expired_html, _form_html, _success_html

# Minimal option lists (the real ones come from the DB) for the pure renderer.
_OPTS = {
    "states": [{"id": "st1", "name": "Tamil Nadu"}],
    "districts": [{"id": "d1", "name": "Chennai", "parent": "st1"}],
    "education_levels": [{"id": "e1", "name": "Bachelor Degree"}],
    "courses": [{"id": "c1", "name": "B.Tech"}],
    "specializations": [{"id": "sp1", "name": "Computer Science", "parent": "c1"}],
    "experience_levels": [{"id": "x1", "name": "Freshers"}],
    "skills": [{"id": "sk1", "name": "Python"}],
    "roles": [{"id": "r1", "name": "Software Developer"}],
    "categories": [{"id": "cat1", "name": "Information Technology"}],
    "other_states": [{"id": "os1", "name": "Karnataka (Bangalore)"}],
    "languages": [{"id": "l1", "name": "Tamil"}, {"id": "l2", "name": "English"}],
}


def test_form_renders_nine_step_wizard():
    out = _form_html("tok123", "Prasanth", "919876543210", _OPTS)
    assert 'name="token" value="tok123"' in out
    assert 'action="/onboard/submit"' in out
    assert "Prasanth" in out
    assert out.count('class="step"') == 9                  # nine wizard steps
    assert out.count("<span>") == 9                        # nine progress dots
    # mobile shown read-only, prefilled
    assert 'value="919876543210" readonly' in out
    # chip groups (single/multi → hidden inputs at runtime)
    for nm in ("current_status", "gender", "marital_status", "work_mode",
               "expected_salary", "job_types", "interested_in_abroad"):
        assert f'data-name="{nm}"' in out, nm
    # date of birth combiner + cascading selects
    assert 'id="date_of_birth"' in out and 'id="dob_d"' in out
    assert 'name="state_id"' in out and 'name="district_id"' in out
    assert 'name="education_level_id"' in out and 'name="specialization_id"' in out
    # token pickers (incl. the new other-states one)
    for tid in ("ts_skills", "ts_locations", "ts_categories", "ts_roles", "ts_other_states"):
        assert f'id="{tid}"' in out, tid
    # languages step → speak/write checkboxes
    assert 'name="lang_speak" value="l1"' in out and 'name="lang_write" value="l2"' in out
    # embedded option data + stepper JS
    assert "const OTHER_STATES =" in out and "Software Developer" in out
    assert "function valid(" in out and 'id="nextBtn"' in out
    # contact + education + salary inputs
    assert 'name="email"' in out
    assert 'name="resume"' in out
    assert 'name="institution"' in out
    assert 'name="current_salary"' in out


def test_form_marks_required_fields():
    out = _form_html("t", "Asha", "91", _OPTS)
    # required containers carry data-req (validated client-side on Next)
    assert 'data-name="gender" data-req="1"' in out
    assert "<div id=\"ts_skills\" data-req></div>" in out


def test_form_shows_error_when_given():
    out = _form_html("tok123", "Prasanth", "91", _OPTS, error="Something went wrong.")
    assert "Something went wrong." in out


def test_form_escapes_name_to_prevent_injection():
    out = _form_html("t", "<script>alert(1)</script>", "91", _OPTS)
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out


def test_success_and_expired_pages_render():
    assert "registration successful" in _success_html("Prasanth").lower()
    assert "Prasanth" in _success_html("Prasanth")
    assert "expired" in _expired_html().lower()


def test_success_page_has_back_to_chat_button():
    # With a business number → a wa.me deep link (pre-filled so one tap returns
    # to the chat); without → a plain Close button.
    out = _success_html("Prasanth", business_number="919876543210")
    assert "https://wa.me/919876543210" in out
    assert "?text=" not in out          # no pre-filled message — menu is already pushed
    assert "Back to chat" in out
    plain = _success_html("Prasanth")
    assert "window.close()" in plain
    assert "wa.me" not in plain
