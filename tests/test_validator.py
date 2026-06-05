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


def test_validator_passes_salary_shorthand():
    # The model renders a grounded annual figure (1_500_000) as Indian
    # shorthand. Each form must scale back to the real number and pass, instead
    # of reading as a bare 15/30 and being flagged as a hallucination.
    v = HallucinationValidator()
    sql_rows = [{"title": "Enterprise Account Executive",
                 "salary_min": 1500000, "salary_max": 3000000}]
    for reply in (
        "Enterprise Account Executive — Mumbai — ₹15 LPA. Want to know more?",
        "Pays ₹15 lakh to ₹30 lakh per year.",
        "Salary is ₹15L–₹30L.",
    ):
        res = v.validate(reply, sql_rows=sql_rows, vector_hits=[])
        assert res.valid, (reply, res.offending)


def test_validator_still_flags_bogus_shorthand():
    # Unit scaling must not become a blanket pass — a figure that doesn't match
    # any grounded salary even after scaling is still a hallucination.
    v = HallucinationValidator()
    sql_rows = [{"salary_min": 1500000, "salary_max": 3000000}]
    res = v.validate("This pays ₹99 lakh.", sql_rows=sql_rows, vector_hits=[])
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
