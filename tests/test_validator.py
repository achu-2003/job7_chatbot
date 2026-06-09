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


def test_validator_flags_fabricated_job_count():
    # The classic bug: "We have 69 open jobs ..." with no overview/search result
    # backing it. With allowed_counts provided (empty → nothing grounded), the
    # count is a fabrication.
    v = HallucinationValidator()
    res = v.validate(
        "We have 69 open jobs — Sales (15), IT (18), Admin (3). Which area interests you?",
        sql_rows=[], vector_hits=[], allowed_counts=set(),
    )
    assert not res.valid
    assert any("unsupported_job_count:69" in o for o in (res.offending or []))
    assert any("unsupported_category_count:15" in o for o in (res.offending or []))


def test_validator_allows_grounded_job_count():
    # When the turn really fetched the overview, the total + category counts are
    # grounded and pass.
    v = HallucinationValidator()
    res = v.validate(
        "We have 4 open jobs — Technician (2), Admin (1), IT (1). Which area interests you?",
        sql_rows=[], vector_hits=[], allowed_counts={4, 2, 1},
    )
    assert res.valid, res.offending


def test_validator_count_check_disabled_without_allowed_counts():
    # Back-compat: callers that don't compute counts (allowed_counts=None) skip
    # the count check entirely — a count phrase is not flagged.
    v = HallucinationValidator()
    res = v.validate(
        "We have 69 open jobs right now.",
        sql_rows=[], vector_hits=[],
    )
    assert res.valid


def test_validator_allows_count_the_candidate_used():
    # Echoing a number the candidate themselves mentioned is not a fabrication.
    v = HallucinationValidator()
    res = v.validate(
        "Sure — here are 3 roles for you.",
        sql_rows=[], vector_hits=[], allowed_counts=set(),
        customer_query="show me 3 roles",
    )
    assert res.valid, res.offending
