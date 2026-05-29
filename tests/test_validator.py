from app.chatbot.validator import HallucinationValidator


def test_validator_passes_when_salary_in_context():
    v = HallucinationValidator()
    sql_rows = [{"title": "Senior Backend Engineer", "salary_min": 2500000, "salary_max": 4000000}]
    res = v.validate(
        'Our "Senior Backend Engineer" role pays up to ₹4000000.',
        sql_rows=sql_rows,
        vector_hits=[],
    )
    assert res.valid, res.offending


def test_validator_flags_unsupported_salary():
    v = HallucinationValidator()
    sql_rows = [{"title": "Senior Backend Engineer", "salary_min": 2500000, "salary_max": 4000000}]
    res = v.validate(
        "This role pays ₹9999999.",
        sql_rows=sql_rows,
        vector_hits=[],
    )
    assert not res.valid
    assert any("unsupported_salary" in o for o in (res.offending or []))


def test_validator_flags_unsupported_job_ref():
    v = HallucinationValidator()
    res = v.validate(
        "Check out JOB-ZZ9999 — a great fit.",
        sql_rows=[{"job_ref": "JOB-AB1001"}],
        vector_hits=[],
    )
    assert not res.valid
    assert any("unsupported_job_ref" in o for o in (res.offending or []))


def test_validator_flags_unsupported_app_ref():
    v = HallucinationValidator()
    res = v.validate(
        "Your application APP-FAKE11 is in screening.",
        sql_rows=[{"app_ref": "APP-CD5678"}],
        vector_hits=[],
    )
    assert not res.valid
    assert any("unsupported_app_ref" in o for o in (res.offending or []))


def test_validator_allows_nested_app_ref():
    # get_application_status returns {"application": {...}} — the validator
    # walks one level deep, so a ref nested under it counts as grounded.
    v = HallucinationValidator()
    res = v.validate(
        "Your application APP-CD5678 is in screening.",
        sql_rows=[{"found": True, "application": {"app_ref": "APP-CD5678"}}],
        vector_hits=[],
    )
    assert res.valid, res.offending


def test_validator_allows_response_without_numbers():
    v = HallucinationValidator()
    res = v.validate(
        "You can apply to as many open roles as you like.",
        sql_rows=[],
        vector_hits=[],
    )
    assert res.valid
