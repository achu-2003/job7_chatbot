"""Bootstrap the job-application business schema (jobs/candidates/applications).

Run once per environment (idempotent):

    ./.venv/bin/python scripts/init_jobs_db.py [--seed]

Creates the recruiting catalog + candidate-owned tables defined in
``db/jobs_schema.sql``. With ``--seed`` it also loads ``db/jobs_seed.sql``
(sample departments + open roles) for local testing.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

from app.db.session import close_engine, init_engine, session_scope

_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA = _ROOT / "db" / "jobs_schema.sql"
_SEED = _ROOT / "db" / "jobs_seed.sql"


async def _apply(path: Path) -> int:
    statements = [s.strip() for s in path.read_text(encoding="utf-8").split(";") if s.strip()]
    async with session_scope() as session:
        for stmt in statements:
            await session.execute(text(stmt))
    return len(statements)


async def _main(seed: bool) -> None:
    await init_engine()
    try:
        n = await _apply(_SCHEMA)
        print(f"✅ jobs schema ensured ({n} statements).")
        if seed:
            n = await _apply(_SEED)
            print(f"🌱 seed data loaded ({n} statements).")
    finally:
        await close_engine()


if __name__ == "__main__":
    asyncio.run(_main(seed="--seed" in sys.argv))
