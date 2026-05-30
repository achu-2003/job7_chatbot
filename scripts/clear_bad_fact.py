"""One-off cleanup: remove the corrupted identity fact for one customer.

Early buggy versions stored junk like ``full_name = 'Searching Python Developer
Job'``. The reset-customer endpoint keeps customer_facts, so this clears the bad
name fact directly. Scoped to a single phone — touches nothing else.

    python scripts/clear_bad_fact.py 919042147220
    python scripts/clear_bad_fact.py 919042147220 --all   # wipe ALL facts for them
"""
from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text

from app.db.session import close_engine, init_engine, session_scope


async def main(phone: str, wipe_all: bool) -> None:
    digits = "".join(c for c in phone if c.isdigit())
    await init_engine()
    try:
        async with session_scope() as s:
            before = (
                await s.execute(
                    text("SELECT key, value FROM customer_facts WHERE customer_id = :c"),
                    {"c": digits},
                )
            ).fetchall()
            print("facts before:", {k: v for k, v in before})

            if wipe_all:
                await s.execute(
                    text("DELETE FROM customer_facts WHERE customer_id = :c"),
                    {"c": digits},
                )
                print("deleted ALL facts for", digits)
            else:
                # Only drop a full_name that clearly isn't a name (contains
                # 'job'/'search'/'developer' — leftover junk, not a real name).
                res = await s.execute(
                    text(
                        "DELETE FROM customer_facts "
                        "WHERE customer_id = :c AND key = 'full_name' "
                        "AND (LOWER(value) LIKE '%job%' OR LOWER(value) LIKE '%search%' "
                        "     OR LOWER(value) LIKE '%developer%')"
                    ),
                    {"c": digits},
                )
                print("deleted bad full_name rows:", res.rowcount)
    finally:
        await close_engine()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/clear_bad_fact.py <phone> [--all]")
        raise SystemExit(1)
    asyncio.run(main(sys.argv[1], "--all" in sys.argv))
