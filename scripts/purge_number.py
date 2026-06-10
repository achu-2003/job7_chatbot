"""Purge ONE phone number's test data - Redis (+ customer_facts), and optionally
the live job board. Strictly scoped to the given number (last-10-digit match) so
no other person's data is ever touched.

    python scripts/purge_number.py 919042177457
        -> Redis conversation/profile keys + customer_facts for this number.

    python scripts/purge_number.py 919042177457 --db
        -> the above PLUS the live job board rows the BOT created
          (private_job_seekers / private_employers where registrationSource
          starts with 'whatsapp'). FK cascades remove the profile + all children.

    python scripts/purge_number.py 919042177457 --db --force-db
        -> also remove live rows even if NOT bot-created. Use with care - this can
          delete a real seeker/employer that happens to share the number.

    python scripts/purge_number.py 919042177457 --dry-run
        -> show exactly what WOULD be removed, delete nothing.

Note: Redis is live conversation memory - if the number keeps messaging the bot,
its `history` / `onboard_token` keys are recreated by each new inbound message.
Run this after the number has stopped sending messages for a lasting clean slate.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make `app` importable no matter where this script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.chatbot.memory import ConversationMemory
from app.db.session import close_engine, init_engine, session_scope


def _last10(phone: str) -> str:
    digits = "".join(c for c in phone if c.isdigit())
    return digits[-10:] if len(digits) >= 10 else digits


async def _purge_redis(needle: str, *, dry_run: bool) -> list[str]:
    """All Redis keys mentioning the number, plus the reverse form-token keys
    (which are keyed by a random token, not the number)."""
    m = ConversationMemory()
    await m.connect()
    r = m._redis
    keys: set[str] = set()
    async for k in r.scan_iter(match=f"*{needle}*", count=1000):
        keys.add(k)
    # Resolve reverse token maps from the forward keys we just found.
    for k in list(keys):
        if k.endswith(":onboard_token"):
            v = await r.get(k)
            if v:
                keys.add(f"onboard:token:{v}")
        elif ":employer:" in k and k.endswith(":token"):
            v = await r.get(k)
            if v:
                keys.add(f"employer:token:{v}")
    ordered = sorted(keys)
    print(f"\nRedis - {len(ordered)} key(s):")
    for k in ordered:
        print("   ", k)
    if ordered and not dry_run:
        await r.delete(*ordered)
        print(f"  -> deleted {len(ordered)} key(s)")
    await m.close()
    return ordered


async def _purge_facts(needle: str, *, dry_run: bool) -> int:
    """customer_facts rows (name / email / lane) for the number - Postgres."""
    # Match on the trailing 10 digits: customer_facts.customer_id is stored as the
    # full WhatsApp number (e.g. 919042177457), so an exact last-10 match misses it.
    where = "right(regexp_replace(customer_id, '[^0-9]', '', 'g'), 10) = :n"
    async with session_scope() as s:
        rows = (await s.execute(
            text(f"SELECT customer_id, key, value FROM customer_facts WHERE {where}"),
            {"n": needle},
        )).fetchall()
        print(f"\ncustomer_facts - {len(rows)} row(s):")
        for r in rows:
            m = r._mapping
            print(f"    [{m['customer_id']}] {m['key']} = {m['value']!r}")
        if rows and not dry_run:
            res = await s.execute(
                text(f"DELETE FROM customer_facts WHERE {where}"),
                {"n": needle},
            )
            print(f"  -> deleted {res.rowcount} row(s)")
        return len(rows)


async def _purge_jobboard(needle: str, *, force: bool, dry_run: bool) -> None:
    """Live job board: private_job_seekers + private_employers for the number.
    By default only BOT-created rows (registrationSource ILIKE 'whatsapp%'); the
    FK cascades delete the profile + every child row automatically."""
    src = "" if force else "AND \"registrationSource\" ILIKE 'whatsapp%' "
    async with session_scope() as s:
        seekers = (await s.execute(text(
            'SELECT id, "fullName", phone, "registrationSource" FROM private_job_seekers '
            "WHERE right(regexp_replace(phone, '[^0-9]', '', 'g'), 10) = :n " + src
        ), {"n": needle})).fetchall()
        emps = (await s.execute(text(
            'SELECT id, "companyName", "primaryPhone", "registrationSource" FROM private_employers '
            "WHERE right(regexp_replace(\"primaryPhone\", '[^0-9]', '', 'g'), 10) = :n " + src
        ), {"n": needle})).fetchall()

        print(f"\nprivate_job_seekers - {len(seekers)} row(s)"
              f"{' (bot-created only)' if not force else ' (ALL, --force-db)'}:")
        for r in seekers:
            m = r._mapping
            print(f"    {m['id']}  {m['fullName']!r}  {m['phone']!r}  src={m['registrationSource']!r}")
        print(f"private_employers - {len(emps)} row(s):")
        for r in emps:
            m = r._mapping
            print(f"    {m['id']}  {m['companyName']!r}  {m['primaryPhone']!r}  src={m['registrationSource']!r}")

        if dry_run:
            return
        sids = [r._mapping["id"] for r in seekers]
        eids = [r._mapping["id"] for r in emps]
        if sids:
            await s.execute(text("DELETE FROM private_job_seekers WHERE id = ANY(:ids)"), {"ids": sids})
            print(f"  -> deleted {len(sids)} seeker row(s) (+ cascaded children)")
        if eids:
            await s.execute(text("DELETE FROM private_employers WHERE id = ANY(:ids)"), {"ids": eids})
            print(f"  -> deleted {len(eids)} employer row(s) (+ cascaded children)")


async def main(phone: str, *, with_db: bool, force_db: bool, dry_run: bool) -> None:
    needle = _last10(phone)
    if len(needle) < 10:
        print(f"refusing: {phone!r} has fewer than 10 digits - give a full number.")
        raise SystemExit(2)
    mode = "DRY RUN - nothing will be deleted" if dry_run else "PURGE"
    print(f"=== {mode} for number ending {needle} ===")
    await init_engine()
    try:
        await _purge_redis(needle, dry_run=dry_run)
        await _purge_facts(needle, dry_run=dry_run)
        if with_db:
            await _purge_jobboard(needle, force=force_db, dry_run=dry_run)
        else:
            print("\n(live job board skipped - pass --db to also purge "
                  "private_job_seekers / private_employers)")
    finally:
        await close_engine()
    print("\nDone.")


if __name__ == "__main__":
    args = sys.argv[1:]
    nums = [a for a in args if not a.startswith("-")]
    if not nums:
        print("usage: python scripts/purge_number.py <phone> [--db] [--force-db] [--dry-run]")
        raise SystemExit(1)
    asyncio.run(main(
        nums[0],
        with_db="--db" in args,
        force_db="--force-db" in args,
        dry_run="--dry-run" in args,
    ))
