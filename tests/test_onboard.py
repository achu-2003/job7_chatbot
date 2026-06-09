"""Onboarding form HTML rendering (pure helpers — no Redis needed)."""
from app.api.routes.onboard import _expired_html, _form_html, _success_html

# Minimal option lists (the real ones come from the DB) for the pure renderer.
_OPTS = {
    "states": [{"id": "st1", "name": "Tamil Nadu"}],
    "districts": [{"id": "d1", "name": "Chennai"}],
    "education_levels": [{"id": "e1", "name": "Bachelor Degree"}],
    "courses": [{"id": "c1", "name": "B.Tech"}],
    "specializations": [{"id": "sp1", "name": "Computer Science"}],
    "experience_levels": [{"id": "x1", "name": "Freshers"}],
    "skills": [{"id": "sk1", "name": "Python"}],
    "roles": [{"id": "r1", "name": "Software Developer"}],
    "categories": [{"id": "cat1", "name": "Information Technology"}],
}


def test_form_renders_token_and_fields():
    out = _form_html("tok123", "Prasanth", _OPTS)
    assert 'name="token" value="tok123"' in out
    assert 'name="full_name"' in out
    assert 'name="email"' in out
    # the new Step 1-5 fields + dropdown options
    assert 'name="gender"' in out and 'name="state_id"' in out
    assert 'name="education_level_id"' in out and 'name="experience_level_id"' in out
    # skills / locations / categories / roles are type-to-search token pickers
    # (chips), fed from JS data arrays — not native multi-selects
    assert '<div id="ts_skills">' in out and '<div id="ts_locations">' in out
    assert '<div id="ts_roles">' in out and '<div id="ts_categories">' in out
    assert "function tokenSelect(" in out
    assert 'tokenSelect(document.getElementById("ts_roles"), "preferred_role_ids"' in out
    assert "const ROLES =" in out and "Software Developer" in out   # role data embedded
    assert "multiple>" not in out and "multiple required>" not in out  # no native multiselects
    # cascading: district/specialization are filled by JS from the parent choice,
    # so they start empty (no duplicate flat lists baked in)
    assert "Select a state first" in out and "Select a course first" in out
    assert "const DISTRICTS =" in out and 'cascade("state_id", "district_id"' in out
    assert 'action="/onboard/submit"' in out
    assert "Prasanth" in out


def test_form_shows_error_when_given():
    out = _form_html("tok123", "Prasanth", _OPTS, error="Please enter a valid email address.")
    assert "valid email" in out


def test_form_escapes_name_to_prevent_injection():
    out = _form_html("t", "<script>alert(1)</script>", _OPTS)
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out


def test_success_and_expired_pages_render():
    assert "successfully" in _success_html("Prasanth").lower()
    assert "Prasanth" in _success_html("Prasanth")
    assert "expired" in _expired_html().lower()


def test_success_page_has_back_to_chat_button():
    # With a business number → a wa.me deep link (pre-filled so one tap returns
    # to the chat); without → a plain Close button.
    out = _success_html("Prasanth", business_number="919876543210")
    assert "https://wa.me/919876543210?text=Hi" in out
    assert "Back to chat" in out
    plain = _success_html("Prasanth")
    assert "window.close()" in plain
    assert "wa.me" not in plain
