"""Bootstrap the agent-owned ``agent_*`` memory tables.

Run once per environment (idempotent):

    ./.venv/bin/python scripts/init_agent_db.py

Only touches the agent's own tables — never the read-only business schema.
"""
from __future__ import annotations

import asyncio

from app.db.session import close_engine, init_engine
from app.memory.repositories import ensure_schema


async def _main() -> None:
    await init_engine()
    try:
        await ensure_schema()
        print("✅ agent_* tables ensured.")
    finally:
        await close_engine()


if __name__ == "__main__":
    asyncio.run(_main())
