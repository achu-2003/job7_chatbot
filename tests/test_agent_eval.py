"""Eval-style tests for the agent: grounding + routing on canned tool data."""
from __future__ import annotations

import pytest

from app.chatbot.validator import HallucinationValidator


def test_eval_job_search_grounded():
    v = HallucinationValidator()
    rows = [{"title": "Senior Backend Engineer", "job_ref": "JOB-AB1001",
             "salary_min": 2500000, "salary_max": 4000000}]
    ok = v.validate(
        "Senior Backend Engineer (JOB-AB1001), ₹2500000-4000000.",
        sql_rows=rows, vector_hits=[],
    )
    assert ok.valid

    # A job reference that was never returned must be flagged.
    bad = v.validate(
        "Check out JOB-ZZ9999 — great role.",
        sql_rows=rows, vector_hits=[],
    )
    assert not bad.valid


def test_eval_application_status():
    v = HallucinationValidator()
    # get_application_status returns a nested {"application": {...}} shape —
    # the validator walks one level deep to allow the ref.
    rows = [{"found": True, "application": {"app_ref": "APP-CD5678", "status": "SCREENING"}}]
    ok = v.validate(
        "Your application APP-CD5678 is in screening.",
        sql_rows=rows, vector_hits=[],
    )
    assert ok.valid

    bad = v.validate(
        "Your application APP-FAKE11 is in screening.",
        sql_rows=rows, vector_hits=[],
    )
    assert not bad.valid


def test_eval_submit_application_ref_grounded():
    v = HallucinationValidator()
    # submit_application returns the ref under "application_ref".
    rows = [{"submitted": True, "application_ref": "APP-EF9012", "job_ref": "JOB-AB1001"}]
    ok = v.validate(
        "Applied! Your reference is APP-EF9012.",
        sql_rows=rows, vector_hits=[],
    )
    assert ok.valid


def test_eval_faq_grounded():
    v = HallucinationValidator()
    hits = [{"metadata": {"title": "Eligibility Policy"}}]
    ok = v.validate(
        'Our "Eligibility Policy" covers who can apply.',
        sql_rows=[], vector_hits=hits,
    )
    assert ok.valid


def test_eval_faq_grounded_negative():
    v = HallucinationValidator()
    hits = [{"metadata": {"title": "Eligibility Policy"}}]
    bad = v.validate(
        'Our "Relocation Policy" pays for your move.',
        sql_rows=[], vector_hits=hits,
    )
    assert not bad.valid
