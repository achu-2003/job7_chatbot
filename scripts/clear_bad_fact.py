"""One-off cleanup: remove a corrupted identity fact for one customer.

Early buggy versions stored junk like ``full_name = 'Searching Python Developer
Job'``, and typos typed into the name prompt (e.g. ``'instrested'``) also stick.
The reset-customer endpoint keeps customer_facts, so this clears the bad name
fact directly. Scoped to a single phone — touches nothing else.

    python scripts/clear_bad_fact.py 919042147220             # drop implausible names
    python scripts/clear_bad_fact.py 919042147220 --clear-name # drop full_name (keep email)
    python scripts/clear_bad_fact.py 919042147220 --all        # wipe ALL facts for them
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make `app` importable no matter where this script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.agent.identity import is_plausible_name
from app.db.session import close_engine, init_engine, session_scope


async def main(phone: str, *, wipe_all: bool, clear_name: bool) -> None:
    digits = "".join(c for c in phone if c.isdigit())
    await init_engine()
    try:
        async with session_scope() as s:
            before = {
                k: v
                for k, v in (
                    await s.execute(
                        text("SELECT key, value FROM customer_facts WHERE customer_id = :c"),
                        {"c": digits},
                    )
                ).fetchall()
            }
            print("facts before:", before)

            if wipe_all:
                await s.execute(
                    text("DELETE FROM customer_facts WHERE customer_id = :c"),
                    {"c": digits},
                )
                print("deleted ALL facts for", digits)
            elif clear_name:
                # Drop the full_name fact unconditionally (keeps email), so the
                # next message re-onboards the name. Use this for a typo'd name
                # like "instrested" that LOOKS like a real name.
                res = await s.execute(
                    text(
                        "DELETE FROM customer_facts "
                        "WHERE customer_id = :c AND key = 'full_name'"
                    ),
                    {"c": digits},
                )
                print("deleted full_name rows:", res.rowcount)
            else:
                # Drop a stored full_name that isn't actually name-shaped (legacy
                # junk like 'Searching Python Developer Job'). A real name is left
                # alone — use --clear-name to force-remove one.
                stored = before.get("full_name")
                if stored and not is_plausible_name(stored):
                    res = await s.execute(
                        text(
                            "DELETE FROM customer_facts "
                            "WHERE customer_id = :c AND key = 'full_name'"
                        ),
                        {"c": digits},
                    )
                    print("deleted implausible full_name:", repr(stored), "rows:", res.rowcount)
                else:
                    print("nothing to delete (full_name looks like a real name):", repr(stored))
                    print("→ use --clear-name to force-remove it, or --all to wipe everything.")
    finally:
        await close_engine()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/clear_bad_fact.py <phone> [--clear-name | --all]")
        raise SystemExit(1)
    asyncio.run(
        main(sys.argv[1], wipe_all="--all" in sys.argv, clear_name="--clear-name" in sys.argv)
    )
