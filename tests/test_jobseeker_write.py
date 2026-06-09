"""The transactional registration-insert builder (pure SQL assembly — no DB)."""
from datetime import datetime

from app.db.repositories import _insert_sql, _prep_row


def test_prep_row_converts_dob_and_drops_server_timestamps():
    r = _prep_row("private_job_seekers", {
        "id": "c1", "fullName": "X", "dateOfBirth": "2000-05-12",
        "createdAt": "2026-06-09T00:00:00+00:00", "updatedAt": "2026-06-09T00:00:00+00:00",
    })
    assert isinstance(r["dateOfBirth"], datetime)
    assert r["dateOfBirth"].year == 2000 and r["dateOfBirth"].month == 5
    assert "createdAt" not in r and "updatedAt" not in r   # SQL sets now()


def test_prep_row_drops_empty_jobtypes_keeps_filled():
    assert "jobTypes" not in _prep_row("job_seeker_profiles", {"id": "c1", "jobTypes": []})
    kept = _prep_row("job_seeker_profiles", {"id": "c1", "jobTypes": ["FULL_TIME"]})
    assert kept["jobTypes"] == ["FULL_TIME"]


def test_prep_row_bad_dob_becomes_none():
    assert _prep_row("private_job_seekers", {"id": "c1", "dateOfBirth": "not-a-date"})["dateOfBirth"] is None


def test_insert_sql_casts_enums_and_sets_timestamps():
    sql = _insert_sql("private_job_seekers", {"id": "c1", "status": "ACTIVE"})
    assert 'INSERT INTO private_job_seekers' in sql
    assert 'CAST(:status AS "JobSeekerStatus")' in sql       # enum cast
    assert '"createdAt"' in sql and '"updatedAt"' in sql and "now(), now()" in sql


def test_insert_sql_profile_relationtype_cast():
    sql = _insert_sql("job_seeker_profiles", {"id": "c1", "relationType": "SELF"})
    assert 'CAST(:relationType AS "RelationType")' in sql


def test_insert_sql_child_table_has_createdAt_only_no_casts():
    sql = _insert_sql("private_job_seeker_skills", {"id": "c1", "skillId": "s1"})
    assert '"createdAt"' in sql and '"updatedAt"' not in sql   # children have no updatedAt
    assert "CAST(" not in sql and sql.endswith("now())")
