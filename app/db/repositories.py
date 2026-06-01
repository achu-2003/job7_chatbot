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

# Maps a candidate's free-text employment type onto the jobs7uat ``JobType``
# enum: FULL_TIME | PART_TIME | CONTRACT | INTERNSHIP | FREELANCE.
_EMPLOYMENT_MAP = {
    "full time": "FULL_TIME", "fulltime": "FULL_TIME", "full-time": "FULL_TIME",
    "permanent": "FULL_TIME",
    "part time": "PART_TIME", "parttime": "PART_TIME", "part-time": "PART_TIME",
    "contract": "CONTRACT", "contractor": "CONTRACT",
    "freelance": "FREELANCE", "freelancer": "FREELANCE",
    "intern": "INTERNSHIP", "internship": "INTERNSHIP",
}


def _map_employment_type(value: str | None) -> str | None:
    if not value:
        return None
    return _EMPLOYMENT_MAP.get(value.strip().lower())


# ---------------------------------------------------------------
# Jobs (read-only catalog)
# ---------------------------------------------------------------


class JobRepository:
    """Read-only search against the live ``jobs7uat`` job board.

    The real job data lives in the ``private_*`` table family (``private_jobs``,
    ``private_job_categories``, ``private_companies``) — NOT the empty plain
    ``jobs`` table. We read those real columns but ALIAS them back to the key
    names the rest of the agent expects (``job_ref`` ← ``slug``,
    ``department_name`` ← category name, ``location`` ← ``locationDetails``,
    ``salary_min`` ← ``salaryMin`` …), so nothing downstream (``_compact_job``,
    the validator, the responder) had to change.

    "Shown to candidates" = every PUBLISHED posting: anything that isn't a
    half-written ``DRAFT`` and isn't soft-deleted. Expired/closed/suspended jobs
    ARE shown (the reply labels their status, e.g. "(expired)") so candidates see
    the full board and know which are still open. The candidate-facing reference
    is the human-readable ``slug``.

    WHERE clauses are appended dynamically based on which filters are present,
    so we never evaluate dead ``IS NULL`` branches and avoid bind-vs-cast
    ambiguity.
    """

    # Shown = published (not DRAFT) and not soft-deleted. Expired/closed/etc. are
    # included on purpose and labelled in the reply.
    _LIVE = "j.status <> 'DRAFT' AND j.\"deletedAt\" IS NULL"

    _SELECT = (
        'SELECT '
        '  j.id, j.slug AS job_ref, j.title, j.description, '
        '  j."locationDetails" AS location, '
        '  j."jobType"::text AS employment_type, '
        '  NULL::text AS seniority, '
        '  j."salaryMin" AS salary_min, j."salaryMax" AS salary_max, '
        "  'INR' AS salary_currency, "
        '  j.skills, j.status::text AS status, '
        '  cat.name AS department_name '
        'FROM private_jobs j '
        'LEFT JOIN private_job_categories cat ON cat.id = j."categoryId" '
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
        where: list[str] = [JobRepository._LIVE]
        params: dict[str, Any] = {"limit": limit}
        tenant_sql = _tenant_clause("j", params, tenant_id)
        if tenant_sql:
            where.append(tenant_sql.lstrip(" AND "))

        if department is not None:
            where.append("LOWER(cat.name) LIKE LOWER(:dept_like)")
            params["dept_like"] = f"%{department}%"
        if location is not None:
            where.append('LOWER(j."locationDetails") LIKE LOWER(:loc_like)')
            params["loc_like"] = f"%{location}%"
        mapped_type = _map_employment_type(employment_type)
        if mapped_type is not None:
            where.append('j."jobType" = CAST(:etype AS "PrivateJobType")')
            params["etype"] = mapped_type
        # private_jobs has no seniority/experienceLevel column — experience is a
        # numeric min/max range, so we don't filter on a seniority label here.
        # Salary overlap: keep a job if its band could satisfy the candidate's
        # bound (NULL salary fields are treated as "unspecified", not excluded).
        if min_salary is not None:
            where.append('(j."salaryMax" IS NULL OR j."salaryMax" >= :min_salary)')
            params["min_salary"] = float(min_salary)
        if max_salary is not None:
            where.append('(j."salaryMin" IS NULL OR j."salaryMin" <= :max_salary)')
            params["max_salary"] = float(max_salary)

        sql = (
            JobRepository._SELECT
            + f"WHERE {' AND '.join(where)} "
            + 'ORDER BY j."createdAt" DESC LIMIT :limit'
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
            + f"WHERE j.id = ANY(:ids) AND {JobRepository._LIVE}{tenant_sql}"
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
        """Resolve a candidate-facing job reference (the ``slug``) to its row,
        for apply. Slugs are lowercase, so match case-insensitively and also
        accept the raw ``id`` as a fallback (in case the model echoes it)."""
        params: dict[str, Any] = {"ref": job_ref.strip(), "ref_l": job_ref.strip().lower()}
        tenant_sql = _tenant_clause("j", params, tenant_id)
        sql = text(
            JobRepository._SELECT
            + f"WHERE (LOWER(j.slug) = :ref_l OR j.id = :ref) "
            + f"AND {JobRepository._LIVE}{tenant_sql}"
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
        """Pull all live (ACTIVE/APPROVED) jobs for the reindex flow."""
        params: dict[str, Any] = {}
        tenant_sql = _tenant_clause("j", params, tenant_id)
        sql = text(
            JobRepository._SELECT
            + f"WHERE {JobRepository._LIVE}{tenant_sql}"
        )
        async with session_scope() as session:
            res = await session.execute(sql, params)
            return [dict(r._mapping) for r in res]

    @staticmethod
    async def category_summary(
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Total open-job count + counts per category, for answering "list all
        jobs" with a count and a category menu instead of a wall of postings."""
        params: dict[str, Any] = {}
        tenant_sql = _tenant_clause("j", params, tenant_id)
        sql = text(
            'SELECT COALESCE(cat.name, \'Other\') AS category, COUNT(*) AS n '
            'FROM private_jobs j '
            'LEFT JOIN private_job_categories cat ON cat.id = j."categoryId" '
            f'WHERE {JobRepository._LIVE}{tenant_sql} '
            'GROUP BY cat.name ORDER BY n DESC'
        )
        async with session_scope() as session:
            res = await session.execute(sql, params)
            rows = [dict(r._mapping) for r in res]
        total = sum(r["n"] for r in rows)
        return {"total": total, "categories": rows}


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
        """DISABLED on jobs7uat. Candidates are ``users`` rows in the live job
        board; creating one from a WhatsApp chat (role/status/password, dedup,
        notifications) needs a design we haven't built, so the apply flow is
        turned off in ``submit_application_core``. This method is intentionally
        unreachable — raise loudly if anything calls it, rather than run the old
        SQL below against a ``candidates`` table that doesn't exist here."""
        raise NotImplementedError(
            "candidate writes are disabled on jobs7uat; apply via portal/handoff"
        )
        sql = text(  # noqa: F841  (kept as a reference for re-enabling apply)
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
        """Find the candidate by phone in ``private_job_seekers`` (the job-board's
        job-seeker records). Aliased to the keys the agent expects so the rest of
        the code is schema-agnostic."""
        async with session_scope() as session:
            res = await session.execute(
                text(
                    'SELECT js.id, js.phone, js."fullName" AS full_name, js.email '
                    'FROM private_job_seekers js '
                    'WHERE js.phone = :phone '
                    'LIMIT 1'
                ),
                {"phone": phone},
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
        """DISABLED on jobs7uat — see ``CandidateRepository.upsert``. Writing an
        application means inserting into the live job board's ``applications``
        table (FK to a ``users`` candidate), which the apply flow intentionally
        doesn't do yet. Raise loudly rather than run the old SQL below, whose
        columns (``app_ref``, ``tenant_id``, ``candidate_id``) don't exist here.
        """
        raise NotImplementedError(
            "application writes are disabled on jobs7uat; apply via portal/handoff"
        )
        sql = text(  # noqa: F841  (kept as a reference for re-enabling apply)
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
        params: dict[str, Any] = {"ref": app_ref.strip()}
        scope = ""
        if candidate_id is not None:
            scope = ' AND a."jobSeekerId" = :cid'
            params["cid"] = candidate_id
        sql = text(
            f"""
            SELECT
                a.id AS app_ref, a.status::text AS status,
                a."coverLetter" AS cover_note,
                a."appliedAt" AS created_at, a."updatedAt" AS updated_at,
                j.slug AS job_ref, j.title AS job_title,
                j."locationDetails" AS location, cat.name AS department_name
            FROM private_job_applications a
            JOIN private_jobs j ON j.id = a."jobId"
            LEFT JOIN private_job_categories cat ON cat.id = j."categoryId"
            WHERE a.id = :ref{scope}
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
                a.id AS app_ref, a.status::text AS status,
                a."appliedAt" AS created_at,
                j.slug AS job_ref, j.title AS job_title,
                j."locationDetails" AS location
            FROM private_job_applications a
            JOIN private_jobs j ON j.id = a."jobId"
            WHERE a."jobSeekerId" = :cid
            ORDER BY a."appliedAt" DESC
            LIMIT :limit
            """
        )
        params = {"cid": candidate_id, "limit": limit}
        start = time.perf_counter()
        async with session_scope() as session:
            res = await session.execute(sql, params)
            rows = [dict(r._mapping) for r in res]
        SQL_LATENCY.labels(op="application_history").observe(time.perf_counter() - start)
        return rows
