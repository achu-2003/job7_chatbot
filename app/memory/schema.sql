-- Agent-owned, read-write memory tables. SEPARATE from the read-only,
-- Prisma-owned business tables (products/orders/...). The agent never writes to
-- business data. Apply with: python -m scripts.init_agent_db  (or via Alembic).
--
-- Apply with:  ./.venv/bin/python scripts/init_agent_db.py
-- Idempotent: every statement is CREATE ... IF NOT EXISTS, safe to re-run.

-- A resumable conversation thread with one customer.
CREATE TABLE IF NOT EXISTS agent_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL,
    customer_id     TEXT NOT NULL,                  -- WhatsApp phone (digits)
    conversation_id TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active', -- active | dormant | closed
    rolling_summary TEXT NOT NULL DEFAULT '',
    last_active_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, conversation_id)
);

-- Goal stack: what the customer is trying to achieve.
CREATE TABLE IF NOT EXISTS agent_goals (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
    tenant_id   TEXT NOT NULL,
    description TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',     -- active | blocked | done | abandoned
    priority    INT  NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_goals_session_idx ON agent_goals (session_id, status);

-- Episodic memory: a compact record of each turn / notable event.
CREATE TABLE IF NOT EXISTS agent_episodes (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
    tenant_id   TEXT NOT NULL,
    role        TEXT NOT NULL,                      -- user | assistant | system
    summary     TEXT NOT NULL,                      -- 1-2 line "what happened"
    tool_calls  JSONB NOT NULL DEFAULT '[]',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_episodes_session_idx ON agent_episodes (session_id, created_at);

-- Deferred / unfinished work to resume or run proactively.
CREATE TABLE IF NOT EXISTS agent_pending_actions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
    goal_id     UUID REFERENCES agent_goals(id) ON DELETE SET NULL,
    tenant_id   TEXT NOT NULL,
    kind        TEXT NOT NULL,                      -- followup | reminder | retry_tool
    payload     JSONB NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'pending',    -- pending | done | cancelled
    run_after   TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_pending_due_idx ON agent_pending_actions (status, run_after);

-- Structured semantic facts (the queryable half of semantic memory).
CREATE TABLE IF NOT EXISTS customer_facts (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    key         TEXT NOT NULL,                      -- name | size | budget | favourite_category
    value       TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.7,
    source      TEXT,                               -- which turn/tool asserted it
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, customer_id, key)
);
