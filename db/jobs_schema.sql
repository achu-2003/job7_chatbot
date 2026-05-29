-- Job-application domain schema (the "business" tables for the recruiting agent).
--
-- Three tiers of trust mirror docs/AGENT_ARCHITECTURE.md:
--   * jobs / departments      -> READ-ONLY catalog (like products/categories were)
--   * candidates / applications -> candidate-owned, READ + scoped WRITE
--   * agent_* (see app/memory/schema.sql) -> agent memory, unchanged
--
-- Apply with:  python scripts/init_jobs_db.py   (or psql -f db/jobs_schema.sql)
-- Idempotent: every statement is CREATE ... IF NOT EXISTS, safe to re-run.

-- ---------------------------------------------------------------------------
-- Catalog (read-only from the agent's perspective; recruiters own writes)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS departments (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   TEXT NOT NULL,
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, name)
);

-- An open (or closed) role a candidate can apply to.
CREATE TABLE IF NOT EXISTS jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL,
    job_ref         TEXT NOT NULL,                       -- human-facing id, e.g. JOB-AB1234
    title           TEXT NOT NULL,
    department_id   UUID REFERENCES departments(id) ON DELETE SET NULL,
    location        TEXT,                                -- "Bengaluru" | "Remote (India)"
    employment_type TEXT,                                -- full_time | part_time | contract | intern
    seniority       TEXT,                                -- junior | mid | senior | lead
    salary_min      NUMERIC(12,2),
    salary_max      NUMERIC(12,2),
    salary_currency TEXT NOT NULL DEFAULT 'INR',
    skills          TEXT[] NOT NULL DEFAULT '{}',        -- ["python","fastapi"]
    description     TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'OPEN',        -- OPEN | CLOSED | DRAFT
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, job_ref)
);
CREATE INDEX IF NOT EXISTS jobs_tenant_status_idx ON jobs (tenant_id, status);

-- ---------------------------------------------------------------------------
-- Candidate-owned (read + scoped write)
-- ---------------------------------------------------------------------------

-- One person, keyed by their WhatsApp phone (digits). Built up conversationally.
CREATE TABLE IF NOT EXISTS candidates (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL,
    phone           TEXT NOT NULL,                       -- WhatsApp sender (digits)
    full_name       TEXT,
    email           TEXT,
    years_experience NUMERIC(4,1),
    current_role    TEXT,
    resume_url      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, phone)
);

-- A candidate's application to a job.
CREATE TABLE IF NOT EXISTS applications (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL,
    app_ref         TEXT NOT NULL,                       -- human-facing id, e.g. APP-CD5678
    candidate_id    UUID NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    job_id          UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    status          TEXT NOT NULL DEFAULT 'SUBMITTED',   -- SUBMITTED | SCREENING | INTERVIEW | OFFER | REJECTED | WITHDRAWN
    cover_note      TEXT,
    -- Idempotency: (tenant, candidate, job) is unique, so a retried submit
    -- never creates a duplicate application. The submit repo upserts on this.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, app_ref),
    UNIQUE (tenant_id, candidate_id, job_id)
);
CREATE INDEX IF NOT EXISTS applications_candidate_idx ON applications (tenant_id, candidate_id, created_at DESC);
