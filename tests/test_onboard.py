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
    # token pickers (preferred locations now state-first: pick states → districts)
    for tid in ("ts_skills", "ts_pref_states", "ts_locations", "ts_categories", "ts_roles"):
        assert f'id="{tid}"' in out, tid
    # languages step → speak/write checkboxes
    assert 'name="lang_speak" value="l1"' in out and 'name="lang_write" value="l2"' in out
    # embedded option data + stepper JS
    assert "const OTHER_STATES =" in out and "Software Developer" in out
    assert "function valid(" in out and 'id="nextBtn"' in out
    # contact + education + salary inputs
    assert 'name="email"' in out
    assert 'type="file" name="resume"' in out             # resume is a file picker
    assert 'enctype="multipart/form-data"' in out         # so the form is multipart
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


class _FakeUpload:
    """Minimal stand-in for a Starlette UploadFile (filename + async read)."""
    def __init__(self, filename, data):
        self.filename = filename
        self._data = data

    async def read(self):
        return self._data


async def test_resume_upload_saves_file_and_returns_url(tmp_path, monkeypatch):
    import app.api.routes.onboard as ob
    from app.config import get_settings

    monkeypatch.setattr(ob, "_RESUME_DIR", tmp_path)
    monkeypatch.setattr(get_settings(), "public_base_url", "https://x.test")
    url = await ob._save_resume_upload(_FakeUpload("My Résumé.pdf", b"%PDF-1.4 data"), "tok")
    assert url.startswith("https://x.test/uploads/resumes/") and url.endswith(".pdf")
    files = list(tmp_path.iterdir())
    assert len(files) == 1 and files[0].read_bytes() == b"%PDF-1.4 data"


async def test_resume_upload_ignores_non_file():
    import app.api.routes.onboard as ob
    assert await ob._save_resume_upload("https://a-pasted-link", "tok") == ""   # plain str
    assert await ob._save_resume_upload(None, "tok") == ""                      # nothing


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
