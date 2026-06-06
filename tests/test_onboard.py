"""Onboarding form HTML rendering (pure helpers — no Redis needed)."""
from app.api.routes.onboard import _expired_html, _form_html, _success_html


def test_form_renders_token_and_fields():
    out = _form_html("tok123", "Prasanth")
    assert 'name="token" value="tok123"' in out
    assert 'name="email"' in out
    assert 'name="years_experience"' in out
    assert 'name="preferred_role"' in out
    assert 'name="location"' in out
    assert 'action="/onboard/submit"' in out
    assert "Prasanth" in out


def test_form_shows_error_when_given():
    out = _form_html("tok123", "Prasanth", error="Please enter a valid email address.")
    assert "valid email" in out


def test_form_escapes_name_to_prevent_injection():
    out = _form_html("t", "<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out


def test_success_and_expired_pages_render():
    assert "all set" in _success_html("Prasanth").lower()
    assert "Prasanth" in _success_html("Prasanth")
    assert "expired" in _expired_html().lower()
