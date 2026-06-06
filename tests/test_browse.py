"""Deterministic category matcher + listing formatter."""
from app.agent.browse import build_job_list_text, match_category

_CATS = [
    "Information Technology", "Other", "Sales & Marketing", "Administration",
    "Customer Service", "Healthcare", "Banking & Insurance",
]


def test_acronym_resolves_to_full_category():
    assert match_category("show the IT jobs", _CATS) == "Information Technology"
    assert match_category("list all 18 IT jobs", _CATS) == "Information Technology"


def test_full_name_and_word_match():
    assert match_category("Sales & Marketing", _CATS) == "Sales & Marketing"
    assert match_category("i want sales jobs", _CATS) == "Sales & Marketing"
    assert match_category("healthcare", _CATS) == "Healthcare"


def test_prefix_match():
    assert match_category("admin jobs", _CATS) == "Administration"


def test_non_category_returns_none():
    assert match_category("show me python developer roles", _CATS) is None
    assert match_category("what is the salary", _CATS) is None


def test_listing_includes_every_job_and_header():
    jobs = [{"title": f"Role {i}", "location": "Chennai", "job_ref": f"r{i}"} for i in range(15)]
    text = build_job_list_text(jobs, "Sales & Marketing")
    assert "all 15 Sales & Marketing roles" in text
    for i in range(15):
        assert f"Role {i}" in text
    assert "[r7]" in text and "Chennai" in text


def test_listing_without_location_has_no_trailing_dash():
    text = build_job_list_text([{"title": "Sales Head", "job_ref": "sales-head"}], "Sales & Marketing")
    assert "• Sales Head  [sales-head]" in text
    assert "Sales Head —" not in text          # no empty "— " when location is missing
