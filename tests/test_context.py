"""is_followup — distinguishes a question about the pinned product from a new one."""
from app.agent.context import is_followup, toon_context


def test_pronoun_references_are_followups():
    assert is_followup("can u give some details about this") is True
    assert is_followup("how much is it") is True
    assert is_followup("is that in stock?") is True


def test_bare_facet_questions_are_followups():
    assert is_followup("what colours are available") is True
    assert is_followup("sizes?") is True
    assert is_followup("price") is True


def test_affirmations_and_continuations_are_followups():
    # the screenshot bug: "yes" must resolve to the current product, not an old order
    assert is_followup("yes") is True
    assert is_followup("ok") is True
    assert is_followup("sure") is True
    assert is_followup("show more") is True
    assert is_followup("same in black") is True
    assert is_followup("order it") is True


def test_naming_a_new_product_is_not_a_followup():
    assert is_followup("show me red sarees") is False
    assert is_followup("details about the lehenga") is False   # names a product
    assert is_followup("do you have blouses") is False


def test_greeting_is_not_a_followup():
    assert is_followup("hi") is False
    assert is_followup("hello there") is False


def test_application_status_renders_in_context():
    # Regression: get_application_status results must reach the responder as a
    # clean table, not a truncated blob — otherwise the bot ignores them and
    # wrongly asks for name/email on a "show my application" request.
    results = [{
        "tool": "get_application_status",
        "result": {"found": True, "applications": [
            {"job_title": "Kt developer", "status": "REJECTED",
             "created_at": "2026-03-21T11:02:53"},
        ]},
    }]
    ctx = toon_context(results)
    assert "Kt developer" in ctx
    assert "REJECTED" in ctx
    assert "applications" in ctx


def test_search_jobs_renders_in_context_with_availability():
    results = [{"tool": "search_jobs", "result": [
        {"title": "Office Staff", "availability": "open", "salary_min": 20000},
        {"title": "Old Role", "availability": "expired"},
    ]}]
    ctx = toon_context(results)
    assert "Office Staff" in ctx and "Old Role" in ctx
    assert "expired" in ctx
