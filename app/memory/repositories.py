"""Read-WRITE repositories for the agent-owned ``agent_*`` tables.

Deliberately separate from ``app.db.repositories`` (which is read-only against
the Prisma-owned business tables). Writes go through ``session_scope`` which
commits on success / rolls back on error. Every query is tenant-scoped and
parameterised.

``ensure_schema()`` applies ``schema.sql`` (idempotent CREATE ... IF NOT EXISTS),
so a fresh deploy can bootstrap the tables without a separate migration tool;
Alembic remains an option for change management later.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.core.logging import get_logger
from app.db.session import session_scope

log = get_logger("agent_memory")

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


async def ensure_schema() -> None:
    """Create the agent_* tables if they don't exist. Safe to call repeatedly."""
    statements = [s.strip() for s in _SCHEMA_PATH.read_text().split(";") if s.strip()]
    async with session_scope() as session:
        for stmt in statements:
            await session.execute(text(stmt))
    log.info("agent_schema_ensured", statements=len(statements))


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------


class AgentSessionRepository:
    @staticmethod
    async def get(*, tenant_id: str, conversation_id: str) -> dict[str, Any] | None:
        """Read the session WITHOUT touching last_active_at — so the caller can
        measure the idle gap (resume-after-hours detection) before bumping it."""
        async with session_scope() as session:
            res = await session.execute(
                text(
                    "SELECT id, tenant_id, customer_id, conversation_id, status, "
                    "rolling_summary, last_active_at, created_at FROM agent_sessions "
                    "WHERE tenant_id = :tid AND conversation_id = :cid"
                ),
                {"tid": tenant_id, "cid": conversation_id},
            )
            row = res.first()
            return dict(row._mapping) if row else None

    @staticmethod
    async def touch(session_id: str) -> None:
        async with session_scope() as session:
            await session.execute(
                text("UPDATE agent_sessions SET last_active_at = now() WHERE id = :id"),
                {"id": session_id},
            )

    @staticmethod
    async def get_or_create(
        *, tenant_id: str, customer_id: str, conversation_id: str
    ) -> dict[str, Any]:
        """Upsert the session for this conversation and bump ``last_active_at``.
        Returns the (possibly pre-existing) row — caller reads ``rolling_summary``
        and ``last_active_at`` to detect a resumed-after-gap session."""
        sql = text(
            """
            INSERT INTO agent_sessions (tenant_id, customer_id, conversation_id)
            VALUES (:tenant_id, :customer_id, :conversation_id)
            ON CONFLICT (tenant_id, conversation_id)
            DO UPDATE SET last_active_at = now()
            RETURNING id, tenant_id, customer_id, conversation_id, status,
                      rolling_summary, last_active_at, created_at
            """
        )
        async with session_scope() as session:
            res = await session.execute(
                sql,
                {
                    "tenant_id": tenant_id,
                    "customer_id": customer_id,
                    "conversation_id": conversation_id,
                },
            )
            return dict(res.first()._mapping)

    @staticmethod
    async def update_summary(session_id: str, summary: str) -> None:
        async with session_scope() as session:
            await session.execute(
                text("UPDATE agent_sessions SET rolling_summary = :s WHERE id = :id"),
                {"s": summary, "id": session_id},
            )

    @staticmethod
    async def set_status(session_id: str, status: str) -> None:
        async with session_scope() as session:
            await session.execute(
                text("UPDATE agent_sessions SET status = :st WHERE id = :id"),
                {"st": status, "id": session_id},
            )

    @staticmethod
    async def close(*, tenant_id: str, conversation_id: str) -> None:
        """Close the session and wipe its rolling summary (a fresh next session).
        Scoped to one (tenant, conversation) — never touches other customers."""
        async with session_scope() as session:
            await session.execute(
                text(
                    "UPDATE agent_sessions SET status = 'closed', rolling_summary = '' "
                    "WHERE tenant_id = :tid AND conversation_id = :cid"
                ),
                {"tid": tenant_id, "cid": conversation_id},
            )


# ---------------------------------------------------------------------------
# goals
# ---------------------------------------------------------------------------


class GoalRepository:
    @staticmethod
    async def add(
        *, session_id: str, tenant_id: str, description: str, priority: int = 0
    ) -> dict[str, Any]:
        sql = text(
            """
            INSERT INTO agent_goals (session_id, tenant_id, description, priority)
            VALUES (:session_id, :tenant_id, :description, :priority)
            RETURNING id, description, status, priority, created_at
            """
        )
        async with session_scope() as session:
            res = await session.execute(
                sql,
                {
                    "session_id": session_id,
                    "tenant_id": tenant_id,
                    "description": description,
                    "priority": priority,
                },
            )
            return dict(res.first()._mapping)

    @staticmethod
    async def list_active(session_id: str) -> list[dict[str, Any]]:
        async with session_scope() as session:
            res = await session.execute(
                text(
                    "SELECT id, description, status, priority, created_at "
                    "FROM agent_goals WHERE session_id = :sid AND status = 'active' "
                    "ORDER BY priority DESC, created_at"
                ),
                {"sid": session_id},
            )
            return [dict(r._mapping) for r in res]

    @staticmethod
    async def set_status(goal_id: str, status: str) -> None:
        async with session_scope() as session:
            await session.execute(
                text(
                    "UPDATE agent_goals SET status = :st, updated_at = now() "
                    "WHERE id = :id"
                ),
                {"st": status, "id": goal_id},
            )


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------


class EpisodeRepository:
    @staticmethod
    async def add(
        *,
        session_id: str,
        tenant_id: str,
        role: str,
        summary: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        async with session_scope() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO agent_episodes (session_id, tenant_id, role, summary, tool_calls)
                    VALUES (:sid, :tid, :role, :summary, CAST(:tc AS JSONB))
                    """
                ),
                {
                    "sid": session_id,
                    "tid": tenant_id,
                    "role": role,
                    "summary": summary,
                    "tc": json.dumps(tool_calls or []),
                },
            )

    @staticmethod
    async def recent(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        async with session_scope() as session:
            res = await session.execute(
                text(
                    "SELECT role, summary, tool_calls, created_at FROM agent_episodes "
                    "WHERE session_id = :sid ORDER BY created_at DESC LIMIT :lim"
                ),
                {"sid": session_id, "lim": limit},
            )
            rows = [dict(r._mapping) for r in res]
        rows.reverse()  # chronological for the caller
        return rows


# ---------------------------------------------------------------------------
# pending actions
# ---------------------------------------------------------------------------


class PendingActionRepository:
    @staticmethod
    async def add(
        *,
        session_id: str,
        tenant_id: str,
        kind: str,
        payload: dict[str, Any],
        run_after: Any | None = None,
        goal_id: str | None = None,
    ) -> dict[str, Any]:
        sql = text(
            """
            INSERT INTO agent_pending_actions
                (session_id, tenant_id, goal_id, kind, payload, run_after)
            VALUES (:sid, :tid, :gid, :kind, CAST(:payload AS JSONB), :run_after)
            RETURNING id, kind, status, run_after, created_at
            """
        )
        async with session_scope() as session:
            res = await session.execute(
                sql,
                {
                    "sid": session_id,
                    "tid": tenant_id,
                    "gid": goal_id,
                    "kind": kind,
                    "payload": json.dumps(payload),
                    "run_after": run_after,
                },
            )
            return dict(res.first()._mapping)

    @staticmethod
    async def list_pending(session_id: str) -> list[dict[str, Any]]:
        async with session_scope() as session:
            res = await session.execute(
                text(
                    "SELECT id, kind, payload, run_after, created_at "
                    "FROM agent_pending_actions "
                    "WHERE session_id = :sid AND status = 'pending' "
                    "ORDER BY created_at"
                ),
                {"sid": session_id},
            )
            return [dict(r._mapping) for r in res]

    @staticmethod
    async def due(tenant_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Pending actions whose ``run_after`` has passed — drained by the
        scheduler worker (Phase 4)."""
        async with session_scope() as session:
            res = await session.execute(
                text(
                    "SELECT id, session_id, kind, payload FROM agent_pending_actions "
                    "WHERE tenant_id = :tid AND status = 'pending' "
                    "AND (run_after IS NULL OR run_after <= now()) "
                    "ORDER BY run_after NULLS FIRST LIMIT :lim"
                ),
                {"tid": tenant_id, "lim": limit},
            )
            return [dict(r._mapping) for r in res]

    @staticmethod
    async def mark(action_id: str, status: str) -> None:
        async with session_scope() as session:
            await session.execute(
                text("UPDATE agent_pending_actions SET status = :st WHERE id = :id"),
                {"st": status, "id": action_id},
            )


# ---------------------------------------------------------------------------
# customer facts (structured semantic memory)
# ---------------------------------------------------------------------------


class CustomerFactRepository:
    @staticmethod
    async def upsert(
        *,
        tenant_id: str,
        customer_id: str,
        key: str,
        value: str,
        confidence: float = 0.7,
        source: str | None = None,
    ) -> None:
        sql = text(
            """
            INSERT INTO customer_facts
                (tenant_id, customer_id, key, value, confidence, source)
            VALUES (:tid, :cid, :key, :value, :conf, :source)
            ON CONFLICT (tenant_id, customer_id, key)
            DO UPDATE SET value = EXCLUDED.value,
                          confidence = EXCLUDED.confidence,
                          source = EXCLUDED.source,
                          updated_at = now()
            """
        )
        async with session_scope() as session:
            await session.execute(
                sql,
                {
                    "tid": tenant_id,
                    "cid": customer_id,
                    "key": key,
                    "value": value,
                    "conf": confidence,
                    "source": source,
                },
            )

    @staticmethod
    async def all_for(tenant_id: str, customer_id: str) -> dict[str, str]:
        async with session_scope() as session:
            res = await session.execute(
                text(
                    "SELECT key, value FROM customer_facts "
                    "WHERE tenant_id = :tid AND customer_id = :cid"
                ),
                {"tid": tenant_id, "cid": customer_id},
            )
            return {r._mapping["key"]: r._mapping["value"] for r in res}
