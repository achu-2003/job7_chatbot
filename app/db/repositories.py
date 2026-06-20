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

import json
import re
import time
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from app.config import get_settings
from app.core.exceptions import UnsafeSQLError
from app.core.logging import get_logger
from app.core.metrics import SQL_LATENCY
from app.core.tenancy import get_current_tenant_id
from app.db.session import get_engine, session_scope

log = get_logger("repositories")


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


def _best_tier_title_matches(
    words: list[str], rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Keep only the rows that match the MOST distinct query words.

    A typed query like "flutter developer" otherwise returns every job that
    merely shares the generic word "developer" (Frontend Developer, Software
    Developer …). We score each row by how many distinct query words appear in
    its title / category / skills, then return only the best-scoring tier — so a
    real "Flutter Developer" (2 words) wins and the generic 1-word matches drop.
    Single-word queries ("welder") naturally keep every match (all score 1).
    """
    if not rows:
        return []

    def _hits(row: dict[str, Any]) -> int:
        hay = " ".join(
            [
                str(row.get("title") or ""),
                str(row.get("department_name") or ""),
                " ".join(str(s) for s in (row.get("skills") or [])),
            ]
        ).lower()
        return sum(1 for w in words if w in hay)

    scored = [(_hits(r), r) for r in rows]
    best = max(s for s, _ in scored)
    return [r for s, r in scored if s == best][:limit]


class JobRepository:
    """Read-only search against the live ``jobs7uat`` job board.

    The real job data lives in the ``private_*`` table family (``private_jobs``,
    ``private_job_categories``, ``private_companies``) — NOT the empty plain
    ``jobs`` table. We read those real columns but ALIAS them back to the key
    names the rest of the agent expects (``job_ref`` ← ``slug``,
    ``department_name`` ← category name, ``location`` ← ``locationDetails``,
    ``salary_min`` ← ``salaryMin`` …), so nothing downstream (``_compact_job``,
    the validator, the responder) had to change.

    "Shown to candidates" = only genuinely OPEN postings: status ``LIVE`` or
    ``APPROVED`` and not soft-deleted. Expired/closed/suspended/pending/draft jobs
    are NOT shown — so the counts, category menu, search, and recommendations all
    reflect live openings a candidate can actually apply to. The candidate-facing
    reference is the human-readable ``slug``.

    WHERE clauses are appended dynamically based on which filters are present,
    so we never evaluate dead ``IS NULL`` branches and avoid bind-vs-cast
    ambiguity.
    """

    # Shown = open openings only: LIVE or APPROVED, and not soft-deleted.
    # Expired/closed/suspended/pending/draft are excluded (not "open jobs").
    _LIVE = "j.status IN ('LIVE', 'APPROVED') AND j.\"deletedAt\" IS NULL"

    _SELECT = (
        'SELECT '
        '  j.id, j.slug AS job_ref, j.title, j.description, '
        '  j."locationDetails" AS location, '
        '  j."jobType"::text AS employment_type, '
        '  j."workMode"::text AS work_mode, '
        '  NULL::text AS seniority, '
        '  j."salaryMin" AS salary_min, j."salaryMax" AS salary_max, '
        '  j."salaryPeriod"::text AS salary_period, '
        "  'INR' AS salary_currency, "
        '  j."experienceMin" AS experience_min, j."experienceMax" AS experience_max, '
        '  j.vacancies, j."qualificationLevel" AS qualification_level, '
        '  j.qualifications, j."englishLevel" AS english_level, '
        '  j."ageMin" AS age_min, j."ageMax" AS age_max, j.benefits, '
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
    async def search_by_title(
        query: str, *, limit: int = 8, tenant_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Deterministic free-text job search for a typed query. Splits the query
        into WORDS and scores each live job by how many DISTINCT words hit its
        TITLE, a required SKILL, or its CATEGORY — then returns only the best-
        matching tier. So "welder" → Welder jobs, "python" → jobs needing Python,
        and "flutter developer" → Flutter Developer (2 words) WITHOUT dragging in
        every generic "… Developer" role (1 word). Ranked: exact-phrase title
        first, then most words matched, then a title-word hit, then recency."""
        words = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(w) >= 2]
        if not words:
            return []
        params: dict[str, Any] = {"phrase": f"%{' '.join(words)}%"}
        tenant_sql = _tenant_clause("j", params, tenant_id)
        # Per-word match flags so we can rank by the COUNT of distinct query words
        # a job matches (title / category / skills) — not just "any word hits".
        hit_terms: list[str] = []
        title_terms: list[str] = []
        for i, w in enumerate(words):
            key = f"w{i}"
            params[key] = f"%{w}%"
            hit_terms.append(
                f"(CASE WHEN j.title ILIKE :{key} OR cat.name ILIKE :{key} "
                f"OR EXISTS (SELECT 1 FROM unnest(j.skills) sk WHERE sk ILIKE :{key}) "
                "THEN 1 ELSE 0 END)"
            )
            title_terms.append(f"(CASE WHEN j.title ILIKE :{key} THEN 1 ELSE 0 END)")
        word_hits = "(" + " + ".join(hit_terms) + ")"
        title_hits = "(" + " + ".join(title_terms) + ")"
        phrase_hit = "(CASE WHEN j.title ILIKE :phrase THEN 1 ELSE 0 END)"
        # Fetch a generous window ordered best-first, then keep only the top tier
        # in Python (so the cut is on distinct-word count, not a hard SQL LIMIT).
        params["window"] = max(limit * 5, 25)
        sql = text(
            JobRepository._SELECT
            + f"WHERE {JobRepository._LIVE}{tenant_sql} AND {word_hits} > 0 "
            + f"ORDER BY {phrase_hit} DESC, {word_hits} DESC, {title_hits} DESC, "
            + 'j."createdAt" DESC '
            + "LIMIT :window"
        )
        start = time.perf_counter()
        async with session_scope() as session:
            rows = [dict(r._mapping) for r in (await session.execute(sql, params)).fetchall()]
        SQL_LATENCY.labels(op="job_title_search").observe(time.perf_counter() - start)
        return _best_tier_title_matches(words, rows, limit)

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

    @staticmethod
    async def recommend_by_skills(
        *, skill_names: list[str], limit: int = 8, tenant_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Open jobs whose required ``skills`` overlap the candidate's skills,
        ranked by how many skills match (most matches first). Case-insensitive.
        Returns [] when the candidate has no skills or nothing overlaps."""
        names = [s.strip().lower() for s in (skill_names or []) if s and s.strip()]
        if not names:
            return []
        params: dict[str, Any] = {"skills": names, "limit": limit}
        tenant_sql = _tenant_clause("j", params, tenant_id)
        overlap = "(SELECT count(*) FROM unnest(j.skills) sk WHERE lower(sk) = ANY(:skills))"
        sql = text(
            JobRepository._SELECT
            + f"WHERE {JobRepository._LIVE}{tenant_sql} "
            + "AND EXISTS (SELECT 1 FROM unnest(j.skills) sk WHERE lower(sk) = ANY(:skills)) "
            + f'ORDER BY {overlap} DESC, j."createdAt" DESC '
            + "LIMIT :limit"
        )
        async with session_scope() as session:
            rows = (await session.execute(sql, params)).fetchall()
        return [dict(r._mapping) for r in rows]


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
                    # Match on the last 10 digits: the job board stores bare local
                    # numbers (e.g. 9872003072) while WhatsApp delivers them with a
                    # country code (e.g. 919872003072). Strip non-digits on both
                    # sides, then compare the trailing 10 so either form matches.
                    "WHERE right(regexp_replace(js.phone, '\\D', '', 'g'), 10) "
                    "    = right(regexp_replace(:phone, '\\D', '', 'g'), 10) "
                    'LIMIT 1'
                ),
                {"phone": phone},
            )
            row = res.first()
            return dict(row._mapping) if row else None

    @staticmethod
    async def skill_names(*, phone: str) -> list[str]:
        """The candidate's skill NAMES (from ``private_job_seeker_skills`` →
        ``private_skills``), matched by phone. Used to recommend jobs whose
        required skills overlap. Empty list when the number isn't registered or
        has no skills on file."""
        sql = text(
            'SELECT DISTINCT sk.name FROM private_job_seekers js '
            'JOIN private_job_seeker_skills jss ON jss."jobSeekerId" = js.id '
            'JOIN private_skills sk ON sk.id = jss."skillId" '
            "WHERE right(regexp_replace(js.phone, '\\D', '', 'g'), 10) "
            "    = right(regexp_replace(:phone, '\\D', '', 'g'), 10)"
        )
        async with session_scope() as session:
            rows = (await session.execute(sql, {"phone": phone})).fetchall()
        return [r._mapping["name"] for r in rows]

    @staticmethod
    async def application_ids(*, phone: str) -> dict[str, str] | None:
        """The LIVE ``(jobSeekerId, profileId)`` for a phone — needed to write an
        application with FK-valid ids. The Redis-staged registration may carry
        placeholder cuids (or ids from a since-superseded registration), so the
        application must reference the real job-board rows. None if not registered.
        """
        sql = text(
            'SELECT js.id AS "jobSeekerId", p.id AS "profileId" '
            'FROM private_job_seekers js '
            'LEFT JOIN job_seeker_profiles p ON p."jobSeekerId" = js.id '
            "WHERE right(regexp_replace(js.phone, '\\D', '', 'g'), 10) "
            "    = right(regexp_replace(:phone, '\\D', '', 'g'), 10) "
            'ORDER BY p."createdAt" NULLS LAST LIMIT 1'
        )
        async with session_scope() as session:
            row = (await session.execute(sql, {"phone": phone})).first()
        return dict(row._mapping) if row else None


# ---------------------------------------------------------------
# Applications (read + scoped, idempotent write)
# ---------------------------------------------------------------


def _new_app_ref() -> str:
    return f"APP-{uuid.uuid4().hex[:6].upper()}"


class ApplicationRepository:
    @staticmethod
    async def employer_daily_cap_reached(job_id: str | None) -> bool:
        """True when the job's employer is on an ACTIVE plan whose ``dailyApplyCap``
        has already been reached TODAY (across all their jobs) — so a new applicant
        should be turned away until tomorrow. No job / no active sub / NULL cap →
        False (unlimited). Fails SAFE to False (a DB blip never blocks an apply)."""
        if not job_id:
            return False
        try:
            async with session_scope() as session:
                row = (await session.execute(text(
                    'SELECT j."employerId" AS emp, s."dailyApplyCap" AS cap '
                    'FROM private_jobs j '
                    'LEFT JOIN subscriptions s ON s."employerId" = j."employerId" '
                    '  AND s.status = CAST(:ac AS "SubscriptionStatus") AND s."endDate" > now() '
                    'WHERE j.id = :j ORDER BY s."endDate" DESC NULLS LAST LIMIT 1'),
                    {"j": job_id, "ac": "ACTIVE"})).first()
                if not row:
                    return False
                m = row._mapping
                if m["cap"] is None:                  # no active plan / unlimited cap
                    return False
                used = (await session.execute(text(
                    'SELECT count(*) FROM private_job_applications a '
                    'JOIN private_jobs j2 ON j2.id = a."jobId" '
                    'WHERE j2."employerId" = :e '
                    "AND a.\"appliedAt\" >= date_trunc('day', now())"),
                    {"e": m["emp"]})).scalar()
                return int(used or 0) >= int(m["cap"])
        except Exception as exc:  # noqa: BLE001 — never block an apply on a DB error
            log.warning("daily_cap_check_failed", error=str(exc)[:200])
            return False

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
    async def create(payload: dict[str, Any], *, commit: bool = True) -> dict[str, Any]:
        """Write the candidate's application to the live job board
        (``private_job_applications``).

        IDEMPOTENT: the table has a UNIQUE ``(jobId, profileId)``, so re-applying
        to the same job is a NO-OP (``ON CONFLICT DO NOTHING``) — the first
        application and any employer status change on it are preserved, and a
        double-tap never errors or duplicates. ``commit=False`` is a DRY RUN
        (insert + rollback, exercising every type / enum / FK constraint).
        ``status`` / ``appliedAt`` have DB defaults; ``updatedAt`` is set to now().
        """
        app = payload.get("application", payload)
        sa = app.get("screeningAnswers")
        params = {
            "id": app["id"],
            "jobId": app.get("jobId"),
            "jobSeekerId": app.get("jobSeekerId"),
            "profileId": app.get("profileId"),
            "resume": app.get("resume"),
            "coverLetter": app.get("coverLetter"),
            "screeningAnswers": json.dumps(sa) if isinstance(sa, (dict, list)) else None,
            "status": (app.get("status") or "PENDING"),
        }
        sql = text(
            'INSERT INTO private_job_applications '
            '("id","jobId","jobSeekerId","profileId","resume","coverLetter",'
            '"screeningAnswers","status","appliedAt","updatedAt") '
            "VALUES (:id,:jobId,:jobSeekerId,:profileId,:resume,:coverLetter,"
            'CAST(:screeningAnswers AS jsonb),CAST(:status AS "Status"), now(), now()) '
            'ON CONFLICT ("jobId","profileId") DO NOTHING '
            "RETURNING id"
        )
        start = time.perf_counter()
        async with get_engine().connect() as conn:
            trans = await conn.begin()
            try:
                res = await conn.execute(sql, params)
                inserted = res.first() is not None      # None when the conflict skipped it
                await (trans.commit() if commit else trans.rollback())
            except Exception:
                await trans.rollback()
                raise
        SQL_LATENCY.labels(op="application_create").observe(time.perf_counter() - start)
        return {"committed": commit, "application_id": app["id"], "inserted": inserted}

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
                a."viewedAt" AS viewed_at, a."shortlistedAt" AS shortlisted_at,
                j.slug AS job_ref, j.title AS job_title,
                COALESCE(d.name, j."locationDetails") AS location,
                c.name AS company,
                j."salaryMin" AS salary_min, j."salaryMax" AS salary_max,
                j."salaryPeriod"::text AS salary_period,
                j."jobType"::text AS job_type
            FROM private_job_applications a
            JOIN private_jobs j ON j.id = a."jobId"
            LEFT JOIN private_companies c ON c.id = j."companyId"
            LEFT JOIN districts d ON d.id = j."districtId"
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


# ---------------------------------------------------------------
# Lookups (read-only reference data) — resolve a candidate's free-text
# preference to the FK id the profile child tables need.
# ---------------------------------------------------------------


class LookupRepository:
    """Resolve free-text role/location to the reference ids used by
    ``private_job_seeker_preferred_roles.jobRoleId`` / ``..._locations.districtId``.

    Match precedence: exact name → exact slug → contains. Read-only; these are
    global reference tables (no tenant column)."""

    @staticmethod
    async def find_job_role(name: str) -> dict[str, Any] | None:
        q = (name or "").strip()
        if not q:
            return None
        sql = text(
            "SELECT id, name, slug FROM private_job_roles "
            'WHERE "isActive" = TRUE AND ('
            "  LOWER(TRIM(name)) = LOWER(:q) OR LOWER(slug) = LOWER(:slug) "
            "  OR name ILIKE :like) "
            "ORDER BY CASE WHEN LOWER(TRIM(name)) = LOWER(:q) THEN 0 "
            "              WHEN LOWER(slug) = LOWER(:slug) THEN 1 ELSE 2 END, "
            '         "displayOrder" NULLS LAST '
            "LIMIT 1"
        )
        params = {"q": q, "slug": q.lower().replace(" ", "-"), "like": f"%{q}%"}
        async with session_scope() as session:
            row = (await session.execute(sql, params)).first()
        return dict(row._mapping) if row else None

    @staticmethod
    async def find_district(name: str) -> dict[str, Any] | None:
        q = (name or "").strip()
        if not q:
            return None
        sql = text(
            "SELECT id, name, slug FROM districts "
            "WHERE LOWER(TRIM(name)) = LOWER(:q) OR LOWER(slug) = LOWER(:slug) "
            "   OR name ILIKE :like "
            "ORDER BY CASE WHEN LOWER(TRIM(name)) = LOWER(:q) THEN 0 "
            "              WHEN LOWER(slug) = LOWER(:slug) THEN 1 ELSE 2 END "
            "LIMIT 1"
        )
        params = {"q": q, "slug": q.lower().replace(" ", "-"), "like": f"%{q}%"}
        async with session_scope() as session:
            row = (await session.execute(sql, params)).first()
        return dict(row._mapping) if row else None

    # ---- option lists for the onboarding form -----------------------------
    # (table, filter on isActive?, ORDER BY, parent-id column). Read-only global
    # reference data populating the form's dropdowns; each option's value is its
    # id. ``parent`` (when set) carries the FK to its parent option so the form
    # can CASCADE — districts → their state, specializations → their course — so
    # the dependent list is filtered (and de-duplicated) by the parent choice.
    _OPTION_SOURCES = {
        "education_levels": ("private_education_levels", True, '"displayOrder" NULLS LAST, name', None),
        "courses":          ("private_courses", True, '"displayOrder" NULLS LAST, name', None),
        "specializations":  ("private_specializations", True, '"displayOrder" NULLS LAST, name', "courseId"),
        "experience_levels":("private_experience_levels", True, '"displayOrder" NULLS LAST', None),
        "skills":           ("private_skills", True, "name", None),
        "roles":            ("private_job_roles", True, '"displayOrder" NULLS LAST, name', None),
        "categories":       ("private_job_categories", False, "name", None),
        "states":           ("states", False, "name", None),
        "districts":        ("districts", False, "name", "stateId"),
        "other_states":     ("private_other_state_masters", True, '"displayOrder" NULLS LAST, name', None),
        "languages":        ("languages", True, '"displayOrder" NULLS LAST, name', None),
        # employer-side reference data (Stage 1 registration form)
        "industries":       ("private_industry_masters", True, '"displayOrder" NULLS LAST, name', None),
        "designations":     ("private_employer_designations", True, '"displayOrder" NULLS LAST, name', None),
    }

    @staticmethod
    async def options(kind: str) -> list[dict[str, Any]]:
        """``[{"id", "name"[, "parent"]}]`` for a form dropdown. ``parent`` is the
        FK to the parent option for cascading lists. Returns [] (never raises) on
        a miss so a single bad lookup can't break the whole form."""
        src = LookupRepository._OPTION_SOURCES.get(kind)
        if not src:
            return []
        table, has_active, order, parent = src
        sel = "id, name" + (f', "{parent}" AS parent' if parent else "")
        where = 'WHERE "isActive" = TRUE ' if has_active else ""
        sql = text(f"SELECT {sel} FROM {table} {where}ORDER BY {order}")
        try:
            async with session_scope() as session:
                rows = (await session.execute(sql)).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                m = r._mapping
                item = {"id": m["id"], "name": m["name"]}
                if parent:
                    item["parent"] = m["parent"]
                out.append(item)
            return out
        except Exception as exc:  # noqa: BLE001 — a lookup miss must not break the form
            log.warning("lookup_options_failed", kind=kind, error=str(exc)[:160])
            return []

    @staticmethod
    async def name_for(kind: str, id_: str | None) -> str | None:
        """The display name for a selected option id (for back-compat fields like
        the recommendation engine's preferred_role/location text). None on miss."""
        src = LookupRepository._OPTION_SOURCES.get(kind)
        if not id_ or not src:
            return None
        try:
            async with session_scope() as session:
                row = (await session.execute(
                    text(f"SELECT name FROM {src[0]} WHERE id = :id"), {"id": id_}
                )).first()
            return row._mapping["name"] if row else None
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------
# Job-seeker registration WRITE (live job board) — flag-gated
# ---------------------------------------------------------------

# Tables with an updatedAt column (set to now() on insert alongside createdAt).
_WITH_UPDATED_AT = {"private_job_seekers", "job_seeker_profiles", "private_jobs", "private_employers"}
# Columns that are Postgres ENUMs → need an explicit CAST on insert.
_ENUM_CASTS = {
    "private_job_seekers": {"status": '"JobSeekerStatus"'},
    "job_seeker_profiles": {"relationType": '"RelationType"'},
    "private_jobs": {
        "jobType": '"PrivateJobType"', "status": '"PrivateJobStatus"',
        "workMode": '"WorkMode"', "salaryPeriod": '"SalaryPeriod"',
    },
    "private_employers": {
        "status": '"EmployerStatus"', "kycStatus": '"KycStatus"',
        "companySize": '"CompanySize"',
    },
}
# Columns whose staged 'YYYY-MM-DD[ ...]' string must become a real datetime
# before insert (asyncpg rejects a bare string for a timestamp column).
_TS_STRING_COLS = ("dateOfBirth", "interviewDate", "expiresAt", "featuredUntil")
# Preference child tables, inserted after the seeker + profile (FK order).
_CHILD_TABLES = (
    "private_job_seeker_skills", "private_job_seeker_locations",
    "private_job_seeker_preferred_roles", "private_job_seeker_categories",
    "profile_skills", "profile_categories", "profile_preferred_roles",
    "profile_other_states", "profile_languages",
)


def _prep_row(table: str, row: dict[str, Any]) -> dict[str, Any]:
    """Make a staged row insertable: drop the server-set timestamps (the SQL uses
    now()), coerce date/timestamp strings to datetimes (asyncpg rejects a bare
    string for a timestamp), and drop EMPTY array columns so the column default
    applies (an empty array has no inferable element type)."""
    r = dict(row)
    r.pop("createdAt", None)
    r.pop("updatedAt", None)
    for col in _TS_STRING_COLS:
        v = r.get(col)
        if isinstance(v, str) and v:
            try:
                r[col] = datetime.strptime(v[:10], "%Y-%m-%d")
            except ValueError:
                r[col] = None
    # Drop any empty list/tuple so the column's array default (e.g. text[]) applies.
    for k in list(r):
        if isinstance(r[k], (list, tuple)) and not r[k]:
            r.pop(k, None)
    return r


def _insert_sql(table: str, row: dict[str, Any]) -> str:
    casts = _ENUM_CASTS.get(table, {})
    cols = list(row.keys())
    ts = ["createdAt"] + (["updatedAt"] if table in _WITH_UPDATED_AT else [])
    col_sql = ", ".join([f'"{c}"' for c in cols] + [f'"{t}"' for t in ts])
    val_sql = ", ".join(
        [f"CAST(:{c} AS {casts[c]})" if c in casts else f":{c}" for c in cols]
        + ["now()"] * len(ts)
    )
    return f"INSERT INTO {table} ({col_sql}) VALUES ({val_sql})"


class JobSeekerRepository:
    """WRITE path: register a candidate on the live job board.

    Inserts ``private_job_seekers`` → ``job_seeker_profiles`` → every preference
    child row in ONE transaction. ``commit=False`` is a DRY RUN — every statement
    runs against the DB (so all type / FK / enum constraints are exercised) and is
    then rolled back, writing nothing.
    """

    @staticmethod
    async def create(payload: dict[str, Any], *, commit: bool = True) -> dict[str, Any]:
        seeker = _prep_row("private_job_seekers", payload["private_job_seekers"])
        profile = _prep_row("job_seeker_profiles", payload["job_seeker_profiles"])
        counts: dict[str, int] = {}
        start = time.perf_counter()
        async with get_engine().connect() as conn:
            trans = await conn.begin()
            try:
                await conn.execute(text(_insert_sql("private_job_seekers", seeker)), seeker)
                await conn.execute(text(_insert_sql("job_seeker_profiles", profile)), profile)
                for table in _CHILD_TABLES:
                    rows = payload.get(table) or []
                    for row in rows:
                        r = _prep_row(table, row)
                        await conn.execute(text(_insert_sql(table, r)), r)
                    counts[table] = len(rows)
                await (trans.commit() if commit else trans.rollback())
            except Exception:
                await trans.rollback()
                raise
        SQL_LATENCY.labels(op="jobseeker_register").observe(time.perf_counter() - start)
        return {
            "committed": commit,
            "seeker_id": seeker["id"],
            "profile_id": profile["id"],
            "children": counts,
        }

    @staticmethod
    async def list_candidates(*, limit: int = 8) -> list[dict[str, Any]]:
        """Active job-seekers for the employer 'view candidates' flow. Returns the
        FULL rows (incl. aggregated ``skills`` + preferred ``roles``); the caller
        masks them by entitlement tier (name + years of experience only until the
        employer has paid)."""
        sql = text(
            """
            SELECT js.id, js."fullName" AS full_name, js.email, js.phone, js.city,
                   el.name AS experience_level, d.name AS district,
                   array_agg(DISTINCT sk.name) FILTER (WHERE sk.name IS NOT NULL) AS skills,
                   array_agg(DISTINCT jr.name) FILTER (WHERE jr.name IS NOT NULL) AS roles
            FROM private_job_seekers js
            LEFT JOIN private_experience_levels el ON el.id = js."experienceLevelId"
            LEFT JOIN districts d ON d.id = js."districtId"
            LEFT JOIN private_job_seeker_skills jss ON jss."jobSeekerId" = js.id
            LEFT JOIN private_skills sk ON sk.id = jss."skillId"
            LEFT JOIN private_job_seeker_preferred_roles jsr ON jsr."jobSeekerId" = js.id
            LEFT JOIN private_job_roles jr ON jr.id = jsr."jobRoleId"
            WHERE js.status = 'ACTIVE' AND js."fullName" IS NOT NULL
            GROUP BY js.id, js."fullName", js.email, js.phone, js.city,
                     el.name, d.name, js."createdAt"
            ORDER BY js."createdAt" DESC NULLS LAST
            LIMIT :limit
            """
        )
        try:
            async with session_scope() as session:
                rows = (await session.execute(sql, {"limit": limit})).fetchall()
            return [dict(r._mapping) for r in rows]
        except Exception as exc:  # noqa: BLE001 — a lookup miss must not break the flow
            log.warning("list_candidates_failed", error=str(exc)[:200])
            return []

    @staticmethod
    async def search_candidates(*, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Active job-seekers whose SKILL, preferred ROLE, or CATEGORY matches the
        employer's free-text query. The query is split into WORDS and a candidate
        matches if ANY skill/role/category contains ANY word — so "python
        developer" finds someone with the *Python* skill (and a Developer role),
        and "python django" finds someone with either. Ranked by how many of the
        searched skills the candidate has. Same row shape as ``list_candidates``;
        the caller masks by entitlement tier."""
        q = (query or "").strip()
        if not q:
            return []
        # Tokenise into words → "%word%" patterns; match any against any field.
        terms = [t for t in re.split(r"\s+", q.lower()) if len(t) >= 2] or [q.lower()]
        patterns = [f"%{t}%" for t in terms]
        sql = text(
            """
            WITH matched AS (
                SELECT js.id,
                       count(DISTINCT sk.name)
                           FILTER (WHERE sk.name ILIKE ANY(:patterns)) AS skill_hits,
                       bool_or(jr.name ILIKE ANY(:patterns)) AS role_hit,
                       bool_or(cat.name ILIKE ANY(:patterns)) AS cat_hit
                FROM private_job_seekers js
                LEFT JOIN private_job_seeker_skills jss ON jss."jobSeekerId" = js.id
                LEFT JOIN private_skills sk ON sk.id = jss."skillId"
                LEFT JOIN private_job_seeker_preferred_roles jsr ON jsr."jobSeekerId" = js.id
                LEFT JOIN private_job_roles jr ON jr.id = jsr."jobRoleId"
                LEFT JOIN private_job_seeker_categories jsc ON jsc."jobSeekerId" = js.id
                LEFT JOIN private_job_categories cat ON cat.id = jsc."categoryId"
                WHERE js.status = 'ACTIVE' AND js."fullName" IS NOT NULL
                GROUP BY js.id
                HAVING count(DISTINCT sk.name) FILTER (WHERE sk.name ILIKE ANY(:patterns)) > 0
                    OR bool_or(jr.name ILIKE ANY(:patterns))
                    OR bool_or(cat.name ILIKE ANY(:patterns))
            )
            SELECT js.id, js."fullName" AS full_name, js.email, js.phone, js.city,
                   el.name AS experience_level, d.name AS district,
                   array_agg(DISTINCT sk2.name) FILTER (WHERE sk2.name IS NOT NULL) AS skills,
                   array_agg(DISTINCT jr2.name) FILTER (WHERE jr2.name IS NOT NULL) AS roles
            FROM private_job_seekers js
            JOIN matched m ON m.id = js.id
            LEFT JOIN private_experience_levels el ON el.id = js."experienceLevelId"
            LEFT JOIN districts d ON d.id = js."districtId"
            LEFT JOIN private_job_seeker_skills jss2 ON jss2."jobSeekerId" = js.id
            LEFT JOIN private_skills sk2 ON sk2.id = jss2."skillId"
            LEFT JOIN private_job_seeker_preferred_roles jsr2 ON jsr2."jobSeekerId" = js.id
            LEFT JOIN private_job_roles jr2 ON jr2.id = jsr2."jobRoleId"
            GROUP BY js.id, js."fullName", js.email, js.phone, js.city,
                     el.name, d.name, js."createdAt", m.skill_hits, m.role_hit
            ORDER BY m.skill_hits DESC, m.role_hit DESC, js."createdAt" DESC NULLS LAST
            LIMIT :limit
            """
        )
        try:
            async with session_scope() as session:
                rows = (await session.execute(
                    sql, {"patterns": patterns, "limit": limit}
                )).fetchall()
            return [dict(r._mapping) for r in rows]
        except Exception as exc:  # noqa: BLE001 — a lookup miss must not break the flow
            log.warning("search_candidates_failed", error=str(exc)[:200])
            return []


# ---------------------------------------------------------------
# Job posting WRITE (live job board) — flag-gated, with dry-run
# ---------------------------------------------------------------


class JobPostRepository:
    """WRITE path: post a job to the live job board (``private_jobs``).

    ``commit=False`` is a DRY RUN — the INSERT runs against the DB (so every
    type / enum / FK / NOT-NULL constraint is exercised) and is then rolled back,
    writing nothing. Used to verify a staged ``private_jobs`` record is storable.
    """

    @staticmethod
    async def create(payload: dict[str, Any], *, commit: bool = True) -> dict[str, Any]:
        job = _prep_row("private_jobs", payload["private_jobs"])
        start = time.perf_counter()
        async with get_engine().connect() as conn:
            trans = await conn.begin()
            try:
                await conn.execute(text(_insert_sql("private_jobs", job)), job)
                await (trans.commit() if commit else trans.rollback())
            except Exception:
                await trans.rollback()
                raise
        SQL_LATENCY.labels(op="job_post").observe(time.perf_counter() - start)
        return {"committed": commit, "job_id": job["id"]}

    @staticmethod
    async def list_for_employer(employer_id: str | None, *, limit: int = 20) -> list[dict[str, Any]]:
        """The employer's own posted jobs from live ``private_jobs`` (newest first),
        for the 'My Jobs' display. Read-only."""
        if not employer_id:
            return []
        sql = text(
            'SELECT id, title, slug, status::text AS status, vacancies, '
            '"jobType"::text AS "jobType", "workMode"::text AS "workMode", '
            '"jobLocationType", "locationDetails", "salaryMin", "salaryMax", '
            '"salaryPeriod"::text AS "salaryPeriod", "experienceType", '
            '"experienceMin", "experienceMax", "internStipend", "trainingFee", '
            '"creditsUsed", "expiresAt", "createdAt" '
            'FROM private_jobs WHERE "employerId" = :e AND "deletedAt" IS NULL '
            'ORDER BY "createdAt" DESC LIMIT :lim')
        try:
            async with session_scope() as session:
                rows = (await session.execute(sql, {"e": employer_id, "lim": limit})).fetchall()
        except Exception as exc:  # noqa: BLE001 — a read miss must not break the turn
            log.warning("list_for_employer_failed", error=str(exc)[:200])
            return []
        return [dict(r._mapping) for r in rows]

    @staticmethod
    async def count_active_for_employer(employer_id: str | None) -> int:
        """How many job 'slots' the employer currently occupies — non-deleted,
        non-expired jobs that aren't closed/rejected. Drives subscription
        ``maxActiveJobs`` enforcement (a free slot is available while this is below
        the plan's cap). Fails safe to 0 on error."""
        if not employer_id:
            return 0
        sql = text(
            'SELECT count(*) FROM private_jobs WHERE "employerId" = :e '
            'AND "deletedAt" IS NULL '
            'AND ("expiresAt" IS NULL OR "expiresAt" > now()) '
            "AND status::text NOT IN ('CLOSED', 'EXPIRED', 'REJECTED')")
        try:
            async with session_scope() as session:
                return int((await session.execute(sql, {"e": employer_id})).scalar() or 0)
        except Exception as exc:  # noqa: BLE001 — a read miss must not break posting
            log.warning("count_active_jobs_failed", error=str(exc)[:200])
            return 0


class EmployerRepository:
    """WRITE path: register a company on the live job board (``private_employers``).

    ``commit=False`` is a DRY RUN — the INSERT runs against the DB (so every
    type / enum / FK / NOT-NULL constraint is exercised) and is then rolled back,
    writing nothing. Idempotent on the phone: a row with the same trailing-10-digit
    ``primaryPhone`` already present → no second insert (returns existing id)."""

    @staticmethod
    async def exists_by_phone(phone: str, *, tenant_id: str | None = None) -> str | None:
        """The id of an existing employer with this phone (last-10 match), else None."""
        if not phone:
            return None
        sql = text(
            'SELECT id FROM private_employers '
            'WHERE right(regexp_replace("primaryPhone", \'\\D\', \'\', \'g\'), 10) '
            '    = right(regexp_replace(:phone, \'\\D\', \'\', \'g\'), 10) '
            'LIMIT 1'
        )
        async with session_scope() as session:
            row = (await session.execute(sql, {"phone": phone})).first()
        return str(row._mapping["id"]) if row else None

    @staticmethod
    async def create(
        payload: dict[str, Any], *, commit: bool = True, welcome_job_credits: int = 0,
    ) -> dict[str, Any]:
        """Insert the company into ``private_employers`` and — when
        ``welcome_job_credits > 0`` — create its ``credit_wallets`` row seeded with
        the free welcome credit + a ``CREDIT_WELCOME_SIGNUP`` ``credit_ledger`` row,
        all in ONE transaction (so the live wallet exists from registration).
        Idempotent on the phone (no duplicate employer/wallet)."""
        emp = _prep_row("private_employers", payload["private_employers"])
        phone = emp.get("primaryPhone") or ""
        # Idempotency: never insert a second row for a phone that already exists.
        if commit:
            existing = await EmployerRepository.exists_by_phone(phone)
            if existing:
                return {"committed": False, "inserted": False, "employer_id": existing}
        welcome = max(0, int(welcome_job_credits))
        start = time.perf_counter()
        async with get_engine().connect() as conn:
            trans = await conn.begin()
            try:
                await conn.execute(text(_insert_sql("private_employers", emp)), emp)
                if welcome > 0:
                    wid = uuid.uuid4().hex
                    await conn.execute(text(
                        'INSERT INTO credit_wallets (id, "employerId", "jobCredits", '
                        '"unlockCredits", "boostCredits", "updatedAt") '
                        'VALUES (:id, :e, :jc, 0, 0, now())'),
                        {"id": wid, "e": emp["id"], "jc": welcome})
                    await conn.execute(text(
                        'INSERT INTO credit_ledger (id, "walletId", "creditType", action, '
                        'amount, balance, "referenceType", "referenceId", description, "createdAt") '
                        'VALUES (:id, :w, CAST(:ct AS "CreditType"), CAST(:ac AS "LedgerAction"), '
                        ':amt, :bal, :rt, :rid, :d, now())'),
                        {"id": uuid.uuid4().hex, "w": wid, "ct": "JOB_POST",
                         "ac": "CREDIT_WELCOME_SIGNUP", "amt": welcome, "bal": welcome,
                         "rt": "signup", "rid": emp["id"],
                         "d": (f"Congratulations! You have received {welcome} FREE job "
                               f"credit{'s' if welcome != 1 else ''} to post your first job.")})
                await (trans.commit() if commit else trans.rollback())
            except Exception:
                await trans.rollback()
                raise
        SQL_LATENCY.labels(op="employer_register").observe(time.perf_counter() - start)
        return {"committed": commit, "inserted": commit, "employer_id": emp["id"]}


# ---------------------------------------------------------------
# Employer credit wallet — READ-ONLY (live billing balance)
# ---------------------------------------------------------------


class CreditWalletRepository:
    """READ-ONLY access to the live ``credit_wallets`` balance.

    The employer flow debits a Redis-mirrored wallet (test harness), but the
    *starting* balance ('Have' on the Activate screen) is the employer's real
    job-credit balance from the live billing table. Read-only: we never write to
    credit_wallets / credit_ledger here.
    """

    @staticmethod
    async def job_credit_balance(employer_id: str | None) -> int | None:
        """The employer's live ``jobCredits`` balance, or None if they have no
        wallet (e.g. a Redis-only test employer whose id isn't in the DB)."""
        bals = await CreditWalletRepository.balances(employer_id)
        return bals["job"] if bals else None

    @staticmethod
    async def balances(employer_id: str | None) -> dict[str, int] | None:
        """All three live balances {job, unlock, boost}, or None if no wallet."""
        if not employer_id:
            return None
        sql = text('SELECT "jobCredits", "unlockCredits", "boostCredits" '
                   'FROM credit_wallets WHERE "employerId" = :eid LIMIT 1')
        try:
            async with session_scope() as session:
                row = (await session.execute(sql, {"eid": employer_id})).first()
        except Exception as exc:  # noqa: BLE001 — a billing-read miss must not break the flow
            log.warning("wallet_balances_failed", error=str(exc)[:200])
            return None
        if not row:
            return None
        m = row._mapping
        return {"job": int(m["jobCredits"]), "unlock": int(m["unlockCredits"]),
                "boost": int(m["boostCredits"])}

    @staticmethod
    async def ledger(employer_id: str | None, *, limit: int = 12) -> list[dict[str, Any]]:
        """The employer's live credit_ledger transactions (newest first), shaped
        like the Redis ledger so the Credits & Wallet page renders unchanged."""
        if not employer_id:
            return []
        sql = text(
            'SELECT l."creditType"::text AS ct, l.action::text AS act, l.amount, '
            'l.balance, l.description, l."createdAt" '
            'FROM credit_ledger l JOIN credit_wallets w ON w.id = l."walletId" '
            'WHERE w."employerId" = :e ORDER BY l."createdAt" DESC LIMIT :lim')
        try:
            async with session_scope() as session:
                rows = (await session.execute(sql, {"e": employer_id, "lim": limit})).fetchall()
        except Exception as exc:  # noqa: BLE001 — a read miss must not break the page
            log.warning("wallet_ledger_failed", error=str(exc)[:200])
            return []
        out: list[dict[str, Any]] = []
        for r in rows:
            m = r._mapping
            created = m["createdAt"]
            out.append({
                "creditType": m["ct"],
                "action": "debit" if str(m["act"]).startswith("DEBIT") else "credit",
                "amount": int(m["amount"]), "balance": int(m["balance"]),
                "description": m["description"] or "",
                "createdAt": int(created.timestamp()) if created else 0,
            })
        return out

    @staticmethod
    async def post_job_with_billing(
        *, employer_id: str, job_payload: dict[str, Any], need: int,
        commit: bool = True,
    ) -> dict[str, Any]:
        """ONE transaction: insert the job into ``private_jobs`` AND record the
        credit consumption — ATOMICALLY debit ``credit_wallets.jobCredits``
        (``jobCredits = GREATEST(0, jobCredits - need)`` — the DB is the source of
        truth, never a SET to a cached value) and write a ``DEBIT_JOB_POST``
        ``credit_ledger`` row referencing the job.

        Idempotent on the job: if the ``private_jobs`` row already exists it's a
        no-op (so a plan-covered post, which writes no debit row, is idempotent
        too). ``need=0`` (plan-covered) inserts the job and debits nothing.
        ``commit=False`` is a DRY RUN. Returns the new authoritative ``balances``."""
        job = _prep_row("private_jobs", job_payload)
        job_id = job["id"]
        need = int(need)
        start = time.perf_counter()
        async with get_engine().connect() as conn:
            trans = await conn.begin()
            try:
                done = (await conn.execute(
                    text('SELECT 1 FROM private_jobs WHERE id = :j LIMIT 1'),
                    {"j": job_id})).first()
                if done:
                    await trans.rollback()
                    return {"committed": False, "reason": "already_posted", "job_id": job_id}
                # Atomic debit (floored at 0). need=0 → no-op increment, current balance.
                wid, nb = await CreditWalletRepository._apply_delta(
                    conn, employer_id, {"job": -need})
                # Insert the job. Record a DEBIT_JOB_POST ledger row ONLY when credits
                # were actually charged — a subscription-covered post (need=0) writes
                # the job but no debit row.
                await conn.execute(text(_insert_sql("private_jobs", job)), job)
                if need > 0:
                    await conn.execute(text(
                        'INSERT INTO credit_ledger (id, "walletId", "creditType", action, amount, '
                        'balance, "referenceType", "referenceId", description, "createdAt") '
                        'VALUES (:id, :w, CAST(:ct AS "CreditType"), CAST(:ac AS "LedgerAction"), '
                        ':amt, :bal, :rt, :rid, :d, now())'),
                        {"id": uuid.uuid4().hex, "w": wid, "ct": "JOB_POST", "ac": "DEBIT_JOB_POST",
                         "amt": need, "bal": nb["job"], "rt": "job_activation", "rid": job_id,
                         "d": f"Job posted ({need} credit{'s' if need != 1 else ''})"})
                await (trans.commit() if commit else trans.rollback())
            except Exception:
                await trans.rollback()
                raise
        SQL_LATENCY.labels(op="job_post_billing").observe(time.perf_counter() - start)
        return {"committed": commit, "job_id": job_id, "credits": need, "balances": nb}

    _BUCKETS = (("job", "jobCredits", "JOB_POST"),
                ("unlock", "unlockCredits", "UNLOCK"),
                ("boost", "boostCredits", "BOOST"))

    @staticmethod
    async def _apply_delta(conn, employer_id: str, delta: dict[str, int]) -> tuple[str, dict[str, int]]:
        """ATOMICALLY apply a per-bucket delta to ``credit_wallets`` inside an EXISTING
        transaction (``conn``), get-or-creating the wallet. Uses
        ``col = GREATEST(0, col + :delta)`` so the DB itself is the source of truth —
        a concurrent/external writer is added on top, never overwritten. Returns
        ``(wallet_id, new_balances)`` where new_balances is the authoritative post-op
        ``{job, unlock, boost}`` read back from the DB."""
        d = {k: int(delta.get(k, 0)) for k in ("job", "unlock", "boost")}
        wid = (await conn.execute(
            text('SELECT id FROM credit_wallets WHERE "employerId" = :e LIMIT 1'),
            {"e": employer_id})).scalar()
        if not wid:
            # New wallet starts at 0, so the opening balance IS the (non-negative) delta.
            wid = uuid.uuid4().hex
            seed = {k: max(0, v) for k, v in d.items()}
            await conn.execute(text(
                'INSERT INTO credit_wallets (id, "employerId", "jobCredits", '
                '"unlockCredits", "boostCredits", "updatedAt") '
                'VALUES (:id, :e, :j, :u, :b, now())'),
                {"id": wid, "e": employer_id, "j": seed["job"],
                 "u": seed["unlock"], "b": seed["boost"]})
            return wid, seed
        row = (await conn.execute(text(
            'UPDATE credit_wallets SET '
            '"jobCredits" = GREATEST(0, "jobCredits" + :dj), '
            '"unlockCredits" = GREATEST(0, "unlockCredits" + :du), '
            '"boostCredits" = GREATEST(0, "boostCredits" + :db), '
            '"updatedAt" = now() WHERE id = :w '
            'RETURNING "jobCredits" AS job, "unlockCredits" AS unlock, "boostCredits" AS boost'),
            {"dj": d["job"], "du": d["unlock"], "db": d["boost"], "w": wid})).first()
        m = row._mapping
        return wid, {"job": int(m["job"]), "unlock": int(m["unlock"]), "boost": int(m["boost"])}

    @staticmethod
    async def record_purchase(
        *, employer_id: str, grants: dict[str, int],
        price: float, payment_id: str, bundle: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Record a live credit PURCHASE by ATOMICALLY INCREMENTING ``credit_wallets``
        (``col = col + grant`` in the DB — never a SET to a cached value, so a
        concurrent/external writer is never clobbered), write a ``credit_ledger``
        CREDIT row per bucket with the resulting DB balance, and — for a bundle — a
        ``bundle_purchases`` row.

        ``bundle`` → ``CREDIT_BUNDLE`` / 'bundle_purchase'; else ``CREDIT_INDIVIDUAL``
        / 'individual_purchase'. Idempotent on ``payment_id``. Returns the new
        authoritative ``balances``. ``commit=False`` is a DRY RUN."""
        grants = {k: int(grants.get(k, 0)) for k in ("job", "unlock", "boost")}
        if not any(v > 0 for v in grants.values()):
            return {"committed": False, "reason": "nothing_to_grant"}
        action = "CREDIT_BUNDLE" if bundle else "CREDIT_INDIVIDUAL"
        ref_type = "bundle_purchase" if bundle else "individual_purchase"
        ref_id = payment_id or uuid.uuid4().hex
        start = time.perf_counter()
        async with get_engine().connect() as conn:
            trans = await conn.begin()
            try:
                if payment_id:                                  # idempotency on the payment id
                    seen = (await conn.execute(
                        text('SELECT 1 FROM credit_ledger WHERE "referenceId" = :r LIMIT 1'),
                        {"r": payment_id})).first()
                    if seen:
                        await trans.rollback()
                        return {"committed": False, "reason": "already_recorded"}
                wid, nb = await CreditWalletRepository._apply_delta(conn, employer_id, grants)
                if bundle:
                    await conn.execute(text(
                        'INSERT INTO bundle_purchases (id, "employerId", "bundleId", '
                        '"priceAtPurchase", "jobCreditsAdded", "unlockCreditsAdded", '
                        '"boostCreditsAdded", "expiresAt") VALUES (:id, :e, :bid, :p, :j, :u, :b, '
                        'now() + make_interval(days => :days))'),
                        {"id": uuid.uuid4().hex, "e": employer_id, "bid": bundle["id"],
                         "p": float(price), "j": grants["job"], "u": grants["unlock"],
                         "b": grants["boost"], "days": int(bundle.get("validityDays") or 365)})
                for key, _field, ctype in CreditWalletRepository._BUCKETS:
                    if grants[key] > 0:
                        await conn.execute(text(
                            'INSERT INTO credit_ledger (id, "walletId", "creditType", action, '
                            'amount, balance, "referenceType", "referenceId", description, "createdAt") '
                            'VALUES (:id, :w, CAST(:ct AS "CreditType"), CAST(:ac AS "LedgerAction"), '
                            ':amt, :bal, :rt, :rid, :d, now())'),
                            {"id": uuid.uuid4().hex, "w": wid, "ct": ctype, "ac": action,
                             "amt": grants[key], "bal": nb[key], "rt": ref_type, "rid": ref_id,
                             "d": (f"Purchased {bundle['name']} bundle" if bundle
                                   else f"Purchased {grants[key]} {ctype.lower()} credit"
                                        f"{'s' if grants[key] != 1 else ''}")})
                await (trans.commit() if commit else trans.rollback())
            except Exception:
                await trans.rollback()
                raise
        SQL_LATENCY.labels(op="credit_purchase").observe(time.perf_counter() - start)
        return {"committed": commit, "granted": grants, "balances": nb}


class CreditBundleRepository:
    """READ-ONLY access to the live ``credit_bundles`` catalog (Starter/Growth/
    Pro/Business) — the 'Bundles' tab of Buy Credits. One-time credit grants
    (job + unlock + boost); the price here is authoritative for the order."""

    @staticmethod
    async def list_active() -> list[dict[str, Any]]:
        sql = text(
            'SELECT id, "bundleType", name, slug, price, "originalPrice", '
            '       "jobCredits", "unlockCredits", "boostCredits", "validityDays" '
            'FROM credit_bundles WHERE "isActive" = TRUE '
            'ORDER BY "displayOrder" NULLS LAST, price'
        )
        try:
            async with session_scope() as session:
                rows = (await session.execute(sql)).fetchall()
        except Exception as exc:  # noqa: BLE001 — catalog read must not break the flow
            log.warning("credit_bundles_failed", error=str(exc)[:200])
            return []
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r._mapping)
            for k in ("price", "originalPrice"):
                if isinstance(d.get(k), Decimal):
                    d[k] = float(d[k])
            out.append(d)
        return out

    @staticmethod
    async def get(bundle_id: str) -> dict[str, Any] | None:
        if not bundle_id:
            return None
        for b in await CreditBundleRepository.list_active():
            if str(b.get("id")) == str(bundle_id):
                return b
        return None


# ---------------------------------------------------------------
# Subscription plans — READ-ONLY (the employer plan catalog)
# ---------------------------------------------------------------


class SubscriptionPlanRepository:
    """READ-ONLY access to the live ``subscription_plans`` catalog (Free/Starter/
    Growth/Pro/Business). Drives the employer 'Upgrade Plan' page; the price here
    is authoritative for the Razorpay order amount (never trust a client price)."""

    @staticmethod
    async def list_active() -> list[dict[str, Any]]:
        sql = text(
            'SELECT id, type, name, "nameTamil", slug, price, "billingCycle", '
            '       "maxActiveJobs", "maxLocationsPerJob", "dailyApplyCap", '
            '       "monthlyCredits", "monthlyBoosts", features '
            'FROM subscription_plans WHERE "isActive" = TRUE '
            'ORDER BY "displayOrder" NULLS LAST, price'
        )
        try:
            async with session_scope() as session:
                rows = (await session.execute(sql)).fetchall()
        except Exception as exc:  # noqa: BLE001 — catalog read must not break the flow
            log.warning("subscription_plans_failed", error=str(exc)[:200])
            return []
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r._mapping)
            if isinstance(d.get("price"), Decimal):
                d["price"] = float(d["price"])
            out.append(d)
        return out

    @staticmethod
    async def get(plan_id: str) -> dict[str, Any] | None:
        if not plan_id:
            return None
        for p in await SubscriptionPlanRepository.list_active():
            if str(p.get("id")) == str(plan_id):
                return p
        return None


class PaymentRepository:
    """Durable Razorpay order/payment records in ``private_payments``.

    A PAYMENT_PENDING row (with the full order context in the ``metadata`` jsonb)
    is written when an order is CREATED, then flipped to SUCCESS on verify. This
    is what stops a Redis loss from stranding a paid order: the verify callback
    can reconcile the order context (grant / plan / validity / token …) from the
    DB when the Redis copy is gone. ``amount`` is stored in RUPEES (matching the
    existing rows). Idempotent on ``razorpayOrderId``; ``mark_success`` only
    transitions PAYMENT_PENDING → SUCCESS so a replayed verify is detectable.
    """

    @staticmethod
    async def create_order(
        *, employer_id: str | None, amount: float, payment_type: str,
        order_id: str, metadata: dict[str, Any], description: str = "",
        commit: bool = True,
    ) -> dict[str, Any]:
        """Write the PENDING payment row. No-op (idempotent) if a row for this
        ``order_id`` already exists. ``commit=False`` is a DRY RUN."""
        if not (employer_id and order_id):
            return {"created": False, "reason": "missing_employer_or_order"}
        start = time.perf_counter()
        async with get_engine().connect() as conn:
            trans = await conn.begin()
            try:
                exists = (await conn.execute(
                    text('SELECT id FROM private_payments WHERE "razorpayOrderId" = :o LIMIT 1'),
                    {"o": order_id})).scalar()
                if exists:
                    await trans.rollback()
                    return {"created": False, "reason": "already_exists", "id": exists}
                pid = "c" + uuid.uuid4().hex[:24]
                await conn.execute(text(
                    'INSERT INTO private_payments (id, "employerId", amount, currency, '
                    '"paymentType", "paymentGateway", "razorpayOrderId", status, metadata, '
                    'description, "createdAt", "updatedAt") '
                    'VALUES (:id, :e, :amt, :cur, :pt, :gw, :o, '
                    'CAST(:st AS "PaymentStatus"), CAST(:md AS jsonb), :d, now(), now())'),
                    {"id": pid, "e": employer_id, "amt": float(amount), "cur": "INR",
                     "pt": payment_type, "gw": "RAZORPAY", "o": order_id,
                     "st": "PAYMENT_PENDING", "md": json.dumps(metadata or {}, default=str),
                     "d": description})
                await (trans.commit() if commit else trans.rollback())
            except Exception:
                await trans.rollback()
                raise
        SQL_LATENCY.labels(op="payment_create").observe(time.perf_counter() - start)
        return {"created": commit, "id": pid}

    @staticmethod
    async def get_context(order_id: str | None) -> dict[str, Any] | None:
        """The staged order context (metadata, employer_id, amount, status) for
        reconciliation when Redis lost the order. None on miss / read error."""
        if not order_id:
            return None
        try:
            async with session_scope() as session:
                r = (await session.execute(text(
                    'SELECT metadata, "employerId", amount, status::text AS status '
                    'FROM private_payments WHERE "razorpayOrderId" = :o LIMIT 1'),
                    {"o": order_id})).first()
        except Exception as exc:  # noqa: BLE001 — a read miss must not break verify
            log.warning("payment_context_failed", error=str(exc)[:200])
            return None
        if not r:
            return None
        m = r._mapping
        md = m["metadata"]
        if isinstance(md, str):                       # asyncpg may hand back jsonb as str
            try:
                md = json.loads(md)
            except json.JSONDecodeError:
                md = {}
        return {
            "metadata": md or {}, "employer_id": m["employerId"],
            "amount": float(m["amount"]) if m["amount"] is not None else None,
            "status": m["status"],
        }

    @staticmethod
    async def id_for_order(order_id: str | None) -> str | None:
        """The ``private_payments.id`` for a Razorpay order — used as the FK that
        ``subscriptions.paymentId`` references. None on miss."""
        if not order_id:
            return None
        try:
            async with session_scope() as session:
                return (await session.execute(text(
                    'SELECT id FROM private_payments WHERE "razorpayOrderId" = :o LIMIT 1'),
                    {"o": order_id})).scalar()
        except Exception as exc:  # noqa: BLE001 — a read miss must not break verify
            log.warning("payment_id_for_order_failed", error=str(exc)[:200])
            return None

    @staticmethod
    async def mark_success(
        *, order_id: str, payment_id: str, signature: str = "", commit: bool = True,
    ) -> dict[str, Any]:
        """Transition PAYMENT_PENDING → SUCCESS for this order. Idempotent: a
        replayed verify finds no pending row and reports ``already=True`` (so the
        caller can skip double-granting). ``commit=False`` is a DRY RUN."""
        if not order_id:
            return {"updated": False, "already": False}
        async with get_engine().connect() as conn:
            trans = await conn.begin()
            try:
                res = await conn.execute(text(
                    'UPDATE private_payments SET status = CAST(:st AS "PaymentStatus"), '
                    '"razorpayPaymentId" = :pid, "razorpaySignature" = :sig, '
                    '"paidAt" = now(), "updatedAt" = now() '
                    'WHERE "razorpayOrderId" = :o '
                    'AND status = CAST(:pend AS "PaymentStatus")'),
                    {"st": "SUCCESS", "pid": payment_id or None, "sig": signature or None,
                     "o": order_id, "pend": "PAYMENT_PENDING"})
                updated = (res.rowcount or 0) > 0
                await (trans.commit() if (commit and updated) else trans.rollback())
            except Exception:
                await trans.rollback()
                raise
        return {"updated": bool(updated and commit), "already": not updated}


class SubscriptionRepository:
    """Durable employer subscription records in ``subscriptions`` + mirror of the
    plan's monthly credit grant.

    On activation, in ONE transaction: supersede the employer's prior ACTIVE
    subscription (one active at a time), INSERT the new ACTIVE row (``endDate`` =
    now + plan validity, ``paymentId`` → the ``private_payments`` row), then mirror
    the plan's monthly credits to ``credit_wallets`` (SET unlock/boost to the Redis
    post-grant balances) + a ``CREDIT_PLAN`` ``credit_ledger`` row. Idempotent: on
    ``payment_row_id`` when paid, else on one ACTIVE row per employer+plan (free).
    ``commit=False`` is a DRY RUN.
    """

    @staticmethod
    async def get_entitlement(employer_id: str | None) -> dict[str, Any]:
        """The employer's CURRENT plan entitlement, with LAZY EXPIRY: if the active
        subscription's ``endDate`` has passed it is flipped to EXPIRED and treated
        as inactive (no scheduler needed). Returns ``{active, plan_id, sub_id,
        max_active_jobs, max_locations_per_job, end_date}``; ``active=False`` (all
        zeros) when there's no live subscription. Fails safe to inactive on error
        (so a DB blip charges credits rather than wrongly granting a free post)."""
        inactive = {"active": False, "plan_id": None, "plan_name": None, "sub_id": None,
                    "max_active_jobs": 0, "max_locations_per_job": 0,
                    "daily_apply_cap": None, "end_date": None, "just_expired": False}
        if not employer_id:
            return inactive
        try:
            async with get_engine().connect() as conn:
                trans = await conn.begin()
                try:
                    row = (await conn.execute(text(
                        'SELECT s.id, s."planId", s."maxActiveJobs", s."maxLocationsPerJob", '
                        's."dailyApplyCap", s."endDate", (s."endDate" < now()) AS expired, '
                        'p.name AS plan_name '
                        'FROM subscriptions s LEFT JOIN subscription_plans p ON p.id = s."planId" '
                        'WHERE s."employerId" = :e '
                        'AND s.status = CAST(:st AS "SubscriptionStatus") '
                        'ORDER BY s."endDate" DESC LIMIT 1'),
                        {"e": employer_id, "st": "ACTIVE"})).first()
                    if not row:
                        await trans.rollback()
                        return inactive
                    m = row._mapping
                    if m["expired"]:
                        # Lazy expiry — flip to EXPIRED. The status-guarded UPDATE makes
                        # the "just expired" signal fire EXACTLY ONCE (a concurrent caller
                        # finds it already EXPIRED and gets no row), so an expiry
                        # notification is sent only once.
                        flipped = (await conn.execute(text(
                            'UPDATE subscriptions SET status = CAST(:ex AS "SubscriptionStatus"), '
                            '"updatedAt" = now() WHERE id = :i '
                            'AND status = CAST(:ac AS "SubscriptionStatus") RETURNING id'),
                            {"ex": "EXPIRED", "ac": "ACTIVE", "i": m["id"]})).first()
                        await trans.commit()
                        return {**inactive, "plan_name": m["plan_name"],
                                "just_expired": bool(flipped)}
                    await trans.rollback()
                    return {
                        "active": True, "plan_id": m["planId"], "plan_name": m["plan_name"],
                        "sub_id": m["id"],
                        "max_active_jobs": int(m["maxActiveJobs"] or 0),
                        "max_locations_per_job": int(m["maxLocationsPerJob"] or 0),
                        "daily_apply_cap": (int(m["dailyApplyCap"]) if m["dailyApplyCap"] is not None else None),
                        "end_date": m["endDate"], "just_expired": False,
                    }
                except Exception:
                    await trans.rollback()
                    raise
        except Exception as exc:  # noqa: BLE001 — fail safe: no entitlement on error
            log.warning("get_entitlement_failed", error=str(exc)[:200])
            return inactive

    @staticmethod
    async def cancel(employer_id: str | None, *, commit: bool = True) -> dict[str, Any]:
        """Cancel the employer's ACTIVE subscription (status → CANCELLED, autoRenew
        off). Benefits stop immediately; get_entitlement / claim_due_renewal only
        target ACTIVE subs, so no further free posts or monthly re-grants. Idempotent
        (no ACTIVE sub → nothing cancelled). Returns ``{cancelled, plan_name}``."""
        if not employer_id:
            return {"cancelled": False, "reason": "no_employer"}
        async with get_engine().connect() as conn:
            trans = await conn.begin()
            try:
                row = (await conn.execute(text(
                    'UPDATE subscriptions s SET status = CAST(:c AS "SubscriptionStatus"), '
                    '"autoRenew" = false, "updatedAt" = now() '
                    'FROM subscription_plans p '
                    'WHERE p.id = s."planId" AND s."employerId" = :e '
                    'AND s.status = CAST(:ac AS "SubscriptionStatus") '
                    'RETURNING p.name AS plan_name'),
                    {"c": "CANCELLED", "ac": "ACTIVE", "e": employer_id})).first()
                await (trans.commit() if (commit and row) else trans.rollback())
                if not row:
                    return {"cancelled": False, "reason": "no_active_subscription"}
                return {"cancelled": bool(commit), "plan_name": row._mapping["plan_name"]}
            except Exception:
                await trans.rollback()
                raise

    @staticmethod
    async def activate(
        *, employer_id: str | None, plan: dict[str, Any], days: int,
        monthly_credits: int, monthly_boosts: int,
        payment_row_id: str | None = None, commit: bool = True,
    ) -> dict[str, Any]:
        if not (employer_id and plan and plan.get("id")):
            return {"created": False, "reason": "missing_employer_or_plan"}
        plan_id = plan["id"]
        start = time.perf_counter()
        async with get_engine().connect() as conn:
            trans = await conn.begin()
            try:
                # Idempotency: a paid sub is keyed on its payment row; a free sub on
                # an existing ACTIVE row for the same employer+plan.
                if payment_row_id:
                    dup = (await conn.execute(text(
                        'SELECT id FROM subscriptions WHERE "paymentId" = :p LIMIT 1'),
                        {"p": payment_row_id})).scalar()
                else:
                    dup = (await conn.execute(text(
                        'SELECT id FROM subscriptions WHERE "employerId" = :e AND "planId" = :pl '
                        'AND status = CAST(:st AS "SubscriptionStatus") LIMIT 1'),
                        {"e": employer_id, "pl": plan_id, "st": "ACTIVE"})).scalar()
                if dup:
                    await trans.rollback()
                    return {"created": False, "reason": "already_active", "id": dup}
                # One active subscription per employer — retire any prior ACTIVE one.
                await conn.execute(text(
                    'UPDATE subscriptions SET status = CAST(:ex AS "SubscriptionStatus"), '
                    '"updatedAt" = now() WHERE "employerId" = :e '
                    'AND status = CAST(:ac AS "SubscriptionStatus")'),
                    {"ex": "EXPIRED", "ac": "ACTIVE", "e": employer_id})
                sub_id = "c" + uuid.uuid4().hex[:24]
                await conn.execute(text(
                    'INSERT INTO subscriptions (id, "employerId", "planId", status, '
                    '"startDate", "endDate", "maxActiveJobs", "maxLocationsPerJob", '
                    '"dailyApplyCap", "paymentId", "autoRenew", "lastCreditGrantDate", '
                    '"creditsGrantedThisMonth", "boostsGrantedThisMonth", "createdAt", "updatedAt") '
                    'VALUES (:id, :e, :pl, CAST(:st AS "SubscriptionStatus"), now(), '
                    'now() + make_interval(days => :days), :maj, :mlp, :cap, :pid, false, now(), '
                    ':cg, :bg, now(), now())'),
                    {"id": sub_id, "e": employer_id, "pl": plan_id, "st": "ACTIVE",
                     "days": int(days), "maj": int(plan.get("maxActiveJobs") or 0),
                     "mlp": int(plan.get("maxLocationsPerJob") or 0),
                     "cap": plan.get("dailyApplyCap"), "pid": payment_row_id,
                     "cg": int(monthly_credits), "bg": int(monthly_boosts)})
                # Atomically grant the plan's monthly credits to the live wallet + ledger.
                wid, nb = await CreditWalletRepository._apply_delta(
                    conn, employer_id, {"unlock": int(monthly_credits), "boost": int(monthly_boosts)})
                ref = payment_row_id or f"sub_{sub_id}"
                plan_name = plan.get("name") or "Plan"
                for amount, ctype, bal in (
                    (int(monthly_credits), "UNLOCK", nb["unlock"]),
                    (int(monthly_boosts), "BOOST", nb["boost"]),
                ):
                    if amount > 0:
                        await conn.execute(text(
                            'INSERT INTO credit_ledger (id, "walletId", "creditType", action, '
                            'amount, balance, "referenceType", "referenceId", description, "createdAt") '
                            'VALUES (:id, :w, CAST(:ct AS "CreditType"), CAST(:ac AS "LedgerAction"), '
                            ':amt, :bal, :rt, :rid, :d, now())'),
                            {"id": uuid.uuid4().hex, "w": wid, "ct": ctype, "ac": "CREDIT_PLAN",
                             "amt": amount, "bal": bal, "rt": "subscription", "rid": ref,
                             "d": f"{plan_name} plan — monthly {ctype.lower()} credits"})
                await (trans.commit() if commit else trans.rollback())
            except Exception:
                await trans.rollback()
                raise
        SQL_LATENCY.labels(op="subscription_activate").observe(time.perf_counter() - start)
        return {"created": commit, "id": sub_id}

    @staticmethod
    async def claim_due_renewal(employer_id: str | None, *, cycle_days: int = 30) -> dict[str, Any] | None:
        """ATOMICALLY claim a monthly re-grant for the employer's active subscription
        if a full ``cycle_days`` has elapsed since ``lastCreditGrantDate`` (and the
        sub is ACTIVE + not past ``endDate``). The single guarded UPDATE is the lock:
        a concurrent caller sees the just-bumped date and the guard fails, so credits
        are never granted twice in a cycle. Returns ``{sub_id, monthly_credits,
        monthly_boosts}`` to grant, or None when nothing is due. The caller then
        grants to Redis + mirrors via ``record_renewal_grant``."""
        if not employer_id:
            return None
        async with get_engine().connect() as conn:
            trans = await conn.begin()
            try:
                row = (await conn.execute(text(
                    'UPDATE subscriptions s SET "lastCreditGrantDate" = now(), '
                    '"creditsGrantedThisMonth" = s."creditsGrantedThisMonth" + p."monthlyCredits", '
                    '"boostsGrantedThisMonth" = s."boostsGrantedThisMonth" + p."monthlyBoosts", '
                    '"updatedAt" = now() '
                    'FROM subscription_plans p '
                    'WHERE s."planId" = p.id AND s."employerId" = :e '
                    'AND s.status = CAST(:ac AS "SubscriptionStatus") AND s."endDate" > now() '
                    'AND s."lastCreditGrantDate" <= now() - make_interval(days => :cd) '
                    'RETURNING s.id AS sub_id, p."monthlyCredits" AS mc, p."monthlyBoosts" AS mb'),
                    {"e": employer_id, "ac": "ACTIVE", "cd": int(cycle_days)})).first()
                await (trans.commit() if row else trans.rollback())
            except Exception:
                await trans.rollback()
                raise
        if not row:
            return None
        m = row._mapping
        return {"sub_id": m["sub_id"], "monthly_credits": int(m["mc"] or 0),
                "monthly_boosts": int(m["mb"] or 0)}

    @staticmethod
    async def record_renewal_grant(
        *, employer_id: str, sub_id: str, monthly_credits: int, monthly_boosts: int,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Mirror a claimed monthly re-grant to the live wallet by ATOMICALLY
        incrementing unlock/boost + a ``CREDIT_PLAN`` ``credit_ledger`` row."""
        async with get_engine().connect() as conn:
            trans = await conn.begin()
            try:
                wid, nb = await CreditWalletRepository._apply_delta(
                    conn, employer_id, {"unlock": int(monthly_credits), "boost": int(monthly_boosts)})
                ref = f"sub_renew_{sub_id}"
                for amount, ctype, bal in (
                    (int(monthly_credits), "UNLOCK", nb["unlock"]),
                    (int(monthly_boosts), "BOOST", nb["boost"]),
                ):
                    if amount > 0:
                        await conn.execute(text(
                            'INSERT INTO credit_ledger (id, "walletId", "creditType", action, '
                            'amount, balance, "referenceType", "referenceId", description, "createdAt") '
                            'VALUES (:id, :w, CAST(:ct AS "CreditType"), CAST(:ac AS "LedgerAction"), '
                            ':amt, :bal, :rt, :rid, :d, now())'),
                            {"id": uuid.uuid4().hex, "w": wid, "ct": ctype, "ac": "CREDIT_PLAN",
                             "amt": amount, "bal": bal, "rt": "subscription_renewal", "rid": ref,
                             "d": f"Plan monthly {ctype.lower()} credits (renewal)"})
                await (trans.commit() if commit else trans.rollback())
            except Exception:
                await trans.rollback()
                raise
        return {"recorded": commit}


class SavedJobRepository:
    """Durable seeker bookmarks in ``private_saved_jobs`` — so a saved job survives a
    Redis loss and syncs with the main app's saved list. Idempotent on
    (jobSeekerId, jobId): re-saving the same job is a no-op (the table has no unique
    constraint, so the check is explicit). ``commit=False`` is a DRY RUN."""

    @staticmethod
    async def create(
        *, job_seeker_id: str | None, job_id: str | None, commit: bool = True,
    ) -> dict[str, Any]:
        if not (job_seeker_id and job_id):
            return {"created": False, "reason": "missing_ids"}
        async with get_engine().connect() as conn:
            trans = await conn.begin()
            try:
                dup = (await conn.execute(text(
                    'SELECT id FROM private_saved_jobs '
                    'WHERE "jobSeekerId" = :s AND "jobId" = :j LIMIT 1'),
                    {"s": job_seeker_id, "j": job_id})).scalar()
                if dup:
                    await trans.rollback()
                    return {"created": False, "reason": "already_saved", "id": dup}
                sid = "c" + uuid.uuid4().hex[:24]
                await conn.execute(text(
                    'INSERT INTO private_saved_jobs (id, "jobSeekerId", "jobId", "savedAt") '
                    'VALUES (:id, :s, :j, now())'),
                    {"id": sid, "s": job_seeker_id, "j": job_id})
                await (trans.commit() if commit else trans.rollback())
            except Exception:
                await trans.rollback()
                raise
        return {"created": commit, "id": sid}
