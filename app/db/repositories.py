"""Repositories for the job-application domain.

Two trust tiers (see docs/AGENT_ARCHITECTURE.md):

* :class:`JobRepository` — READ-ONLY catalog (jobs + departments). Mirrors the
  old ``ProductRepository``: parameterised search, ``get_by_ids`` for
  authoritative hydration, and ``iter_active_for_embedding`` for reindex.
* :class:`CandidateRepository` / :class:`ApplicationRepository` — candidate-owned
  READ + scoped WRITE. Writes go through ``session_scope`` (commit on success,
  rollback on error) and are idempotent: re-submitting the same
  (tenant, candidate, job) returns the existing application instead of creating
  a duplicate.

All queries are parameterised; nothing the LLM produces ever reaches SQL as raw
text. Identity (tenant + the caller's phone) is injected by the tool layer, so a
candidate can only ever read or write their own records.
"""
from __future__ import annotations

import re
import time
import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from app.config import get_settings
from app.core.exceptions import UnsafeSQLError
from app.core.metrics import SQL_LATENCY
from app.core.tenancy import get_current_tenant_id
from app.db.session import session_scope


def _tenant_clause(table_alias: str, params: dict[str, Any], tenant_id: str | None) -> str:
    """Append a tenant filter when DB isolation is enabled.

    When ``enforce_db_tenant_isolation`` is False, returns "" and leaves params
    untouched. When True, the targeted table must carry a ``tenant_id`` column
    (all job-domain tables do).
    """
    if not get_settings().enforce_db_tenant_isolation:
        return ""
    tid = tenant_id or get_current_tenant_id()
    params["__tenant_id"] = tid
    return f" AND {table_alias}.tenant_id = :__tenant_id"


# ---------------------------------------------------------------
# SQL safety (defensive — currently no AI-generated SQL is used)
# ---------------------------------------------------------------

_FORBIDDEN_TOKENS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|"
    r"COPY|VACUUM|ATTACH|MERGE|CALL|DO)\b",
    re.IGNORECASE,
)
_MULTI_STMT = re.compile(r";\s*\S")


def assert_safe_select(sql: str) -> None:
    stripped = sql.strip().rstrip(";")
    if not stripped.lower().startswith("select"):
        raise UnsafeSQLError("only SELECT statements are allowed")
    if _MULTI_STMT.search(stripped):
        raise UnsafeSQLError("multiple statements are not allowed")
    if _FORBIDDEN_TOKENS.search(stripped):
        raise UnsafeSQLError("forbidden SQL keyword detected")


# ---------------------------------------------------------------
# Filter mapping
# ---------------------------------------------------------------

_EMPLOYMENT_MAP = {
    "full time": "full_time", "fulltime": "full_time", "full-time": "full_time",
    "permanent": "full_time",
    "part time": "part_time", "parttime": "part_time", "part-time": "part_time",
    "contract": "contract", "contractor": "contract", "freelance": "contract",
    "intern": "intern", "internship": "intern",
}


def _map_employment_type(value: str | None) -> str | None:
    if not value:
        return None
    return _EMPLOYMENT_MAP.get(value.strip().lower())


# ---------------------------------------------------------------
# Jobs (read-only catalog)
# ---------------------------------------------------------------


class JobRepository:
    """Read-only search against jobs + departments.

    WHERE clauses are appended dynamically based on which filters are present,
    so we never evaluate dead ``IS NULL`` branches and avoid bind-vs-cast
    ambiguity.
    """

    _SELECT = (
        "SELECT "
        "  j.id, j.job_ref, j.title, j.description, j.location, "
        "  j.employment_type, j.seniority, j.salary_min, j.salary_max, "
        "  j.salary_currency, j.skills, j.status, "
        "  d.name AS department_name "
        "FROM jobs j "
        "LEFT JOIN departments d ON d.id = j.department_id "
    )

    @staticmethod
    async def search(
        *,
        query_terms: str | None = None,   # ignored — semantic recall is the vector layer's job
        department: str | None = None,
        location: str | None = None,
        employment_type: str | None = None,
        seniority: str | None = None,
        max_salary: Decimal | float | None = None,
        min_salary: Decimal | float | None = None,
        limit: int = 10,
        tenant_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        where: list[str] = ["j.status = 'OPEN'"]
        params: dict[str, Any] = {"limit": limit}
        tenant_sql = _tenant_clause("j", params, tenant_id)
        if tenant_sql:
            where.append(tenant_sql.lstrip(" AND "))

        if department is not None:
            where.append("LOWER(d.name) LIKE LOWER(:dept_like)")
            params["dept_like"] = f"%{department}%"
        if location is not None:
            where.append("LOWER(j.location) LIKE LOWER(:loc_like)")
            params["loc_like"] = f"%{location}%"
        mapped_type = _map_employment_type(employment_type)
        if mapped_type is not None:
            where.append("j.employment_type = :etype")
            params["etype"] = mapped_type
        if seniority is not None:
            where.append("LOWER(j.seniority) = LOWER(:seniority)")
            params["seniority"] = seniority
        # Salary overlap: keep a job if its band could satisfy the candidate's
        # bound (NULL salary fields are treated as "unspecified", not excluded).
        if min_salary is not None:
            where.append("(j.salary_max IS NULL OR j.salary_max >= :min_salary)")
            params["min_salary"] = float(min_salary)
        if max_salary is not None:
            where.append("(j.salary_min IS NULL OR j.salary_min <= :max_salary)")
            params["max_salary"] = float(max_salary)

        sql = (
            JobRepository._SELECT
            + f"WHERE {' AND '.join(where)} "
            + "ORDER BY j.created_at DESC LIMIT :limit"
        )

        start = time.perf_counter()
        async with session_scope() as session:
            res = await session.execute(text(sql), params)
            rows = [dict(r._mapping) for r in res]
        SQL_LATENCY.labels(op="job_search").observe(time.perf_counter() - start)
        return rows, sql

    @staticmethod
    async def get_by_ids(
        ids: list[str],
        *,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not ids:
            return []
        params: dict[str, Any] = {"ids": ids}
        tenant_sql = _tenant_clause("j", params, tenant_id)
        sql = text(
            JobRepository._SELECT
            + f"WHERE j.id = ANY(:ids) AND j.status = 'OPEN'{tenant_sql}"
        )
        start = time.perf_counter()
        async with session_scope() as session:
            res = await session.execute(sql, params)
            rank = {pid: i for i, pid in enumerate(ids)}
            rows = sorted(
                (dict(r._mapping) for r in res),
                key=lambda r: rank.get(str(r["id"]), 10_000),
            )
        SQL_LATENCY.labels(op="job_lookup").observe(time.perf_counter() - start)
        return rows

    @staticmethod
    async def get_by_ref(
        job_ref: str,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Resolve a human-facing JOB-XXXX reference to its row (for apply)."""
        params: dict[str, Any] = {"ref": job_ref.upper()}
        tenant_sql = _tenant_clause("j", params, tenant_id)
        sql = text(
            JobRepository._SELECT
            + f"WHERE UPPER(j.job_ref) = :ref AND j.status = 'OPEN'{tenant_sql}"
        )
        async with session_scope() as session:
            res = await session.execute(sql, params)
            row = res.first()
        return dict(row._mapping) if row else None

    @staticmethod
    async def iter_active_for_embedding(
        *,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Pull all OPEN jobs for the reindex flow."""
        params: dict[str, Any] = {}
        tenant_sql = _tenant_clause("j", params, tenant_id)
        sql = text(
            JobRepository._SELECT
            + f"WHERE j.status = 'OPEN'{tenant_sql}"
        )
        async with session_scope() as session:
            res = await session.execute(sql, params)
            return [dict(r._mapping) for r in res]


# ---------------------------------------------------------------
# Candidates (read + scoped write) — built up conversationally
# ---------------------------------------------------------------


class CandidateRepository:
    @staticmethod
    async def upsert(
        *,
        tenant_id: str,
        phone: str,
        full_name: str | None = None,
        email: str | None = None,
        years_experience: float | None = None,
        current_role: str | None = None,
        resume_url: str | None = None,
    ) -> dict[str, Any]:
        """Create or update the candidate keyed by (tenant, phone). Only
        non-NULL fields overwrite — partial details captured over several turns
        accumulate rather than clobbering each other (COALESCE keeps the old
        value when a new one isn't supplied)."""
        sql = text(
            """
            INSERT INTO candidates
                (tenant_id, phone, full_name, email, years_experience,
                 current_role, resume_url)
            VALUES (:tid, :phone, :name, :email, :yoe, :role, :resume)
            ON CONFLICT (tenant_id, phone) DO UPDATE SET
                full_name        = COALESCE(EXCLUDED.full_name, candidates.full_name),
                email            = COALESCE(EXCLUDED.email, candidates.email),
                years_experience = COALESCE(EXCLUDED.years_experience, candidates.years_experience),
                current_role     = COALESCE(EXCLUDED.current_role, candidates.current_role),
                resume_url       = COALESCE(EXCLUDED.resume_url, candidates.resume_url),
                updated_at       = now()
            RETURNING id, tenant_id, phone, full_name, email, years_experience,
                      current_role, resume_url
            """
        )
        params = {
            "tid": tenant_id, "phone": phone, "name": full_name, "email": email,
            "yoe": years_experience, "role": current_role, "resume": resume_url,
        }
        async with session_scope() as session:
            res = await session.execute(sql, params)
            return dict(res.first()._mapping)

    @staticmethod
    async def get(*, tenant_id: str, phone: str) -> dict[str, Any] | None:
        async with session_scope() as session:
            res = await session.execute(
                text(
                    "SELECT id, tenant_id, phone, full_name, email, "
                    "years_experience, current_role, resume_url "
                    "FROM candidates WHERE tenant_id = :tid AND phone = :phone"
                ),
                {"tid": tenant_id, "phone": phone},
            )
            row = res.first()
            return dict(row._mapping) if row else None


# ---------------------------------------------------------------
# Applications (read + scoped, idempotent write)
# ---------------------------------------------------------------


def _new_app_ref() -> str:
    return f"APP-{uuid.uuid4().hex[:6].upper()}"


class ApplicationRepository:
    @staticmethod
    async def submit(
        *,
        tenant_id: str,
        candidate_id: str,
        job_id: str,
        cover_note: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Create an application, idempotently.

        The (tenant, candidate, job) uniqueness constraint makes a retried
        submit a no-op: ON CONFLICT we return the existing row. Returns
        ``(application, created)`` where ``created`` is False if it already
        existed — so the agent can say "you've already applied" honestly.
        """
        sql = text(
            """
            INSERT INTO applications
                (tenant_id, app_ref, candidate_id, job_id, cover_note)
            VALUES (:tid, :ref, :cid, :jid, :note)
            ON CONFLICT (tenant_id, candidate_id, job_id) DO UPDATE
                SET updated_at = now()
            RETURNING id, app_ref, status, cover_note, created_at,
                      (xmax = 0) AS created
            """
        )
        params = {
            "tid": tenant_id, "ref": _new_app_ref(),
            "cid": candidate_id, "jid": job_id, "note": cover_note,
        }
        start = time.perf_counter()
        async with session_scope() as session:
            res = await session.execute(sql, params)
            row = dict(res.first()._mapping)
        SQL_LATENCY.labels(op="application_submit").observe(time.perf_counter() - start)
        created = bool(row.pop("created", False))
        return row, created

    @staticmethod
    async def get_by_ref(
        app_ref: str,
        *,
        tenant_id: str,
        candidate_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Look up one application by its APP-XXXX reference. When
        ``candidate_id`` is given, the application must belong to that candidate
        — used to scope a candidate to their own applications."""
        params: dict[str, Any] = {"tid": tenant_id, "ref": app_ref.upper()}
        scope = ""
        if candidate_id is not None:
            scope = " AND a.candidate_id = :cid"
            params["cid"] = candidate_id
        sql = text(
            f"""
            SELECT
                a.app_ref, a.status, a.cover_note, a.created_at, a.updated_at,
                j.job_ref, j.title AS job_title, j.location, d.name AS department_name
            FROM applications a
            JOIN jobs j ON j.id = a.job_id
            LEFT JOIN departments d ON d.id = j.department_id
            WHERE a.tenant_id = :tid AND UPPER(a.app_ref) = :ref{scope}
            """
        )
        start = time.perf_counter()
        async with session_scope() as session:
            res = await session.execute(sql, params)
            row = res.first()
        SQL_LATENCY.labels(op="application_lookup").observe(time.perf_counter() - start)
        return dict(row._mapping) if row else None

    @staticmethod
    async def latest_for_candidate(
        candidate_id: str,
        *,
        tenant_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        sql = text(
            """
            SELECT
                a.app_ref, a.status, a.created_at,
                j.job_ref, j.title AS job_title, j.location
            FROM applications a
            JOIN jobs j ON j.id = a.job_id
            WHERE a.tenant_id = :tid AND a.candidate_id = :cid
            ORDER BY a.created_at DESC
            LIMIT :limit
            """
        )
        params = {"tid": tenant_id, "cid": candidate_id, "limit": limit}
        start = time.perf_counter()
        async with session_scope() as session:
            res = await session.execute(sql, params)
            rows = [dict(r._mapping) for r in res]
        SQL_LATENCY.labels(op="application_history").observe(time.perf_counter() - start)
        return rows
