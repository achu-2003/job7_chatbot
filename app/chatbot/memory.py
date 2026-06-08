"""Multi-turn conversational memory backed by Redis.

A conversation is identified by ``(tenant_id, conversation_id)``. Two tenants
sending the same conversation_id never share state — Redis keys are prefixed
with ``t:{tenant_id}:`` so isolation is enforced at the key level (not in
application code, which is easy to forget).

The list contains the last N message turns serialised as JSON. TTL bounds
memory cost.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import redis.asyncio as redis

from app.config import get_settings
from app.core.logging import get_logger
from app.core.tenancy import get_current_tenant_id

log = get_logger("memory")

_MAX_TURNS = 12


class ConversationMemory:
    def __init__(self) -> None:
        self._redis: redis.Redis | None = None
        self._ttl = get_settings().redis_memory_ttl_seconds

    async def connect(self) -> None:
        self._redis = redis.from_url(
            get_settings().redis_url, decode_responses=True
        )
        await self._redis.ping()
        log.info("memory_ready")

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def mark_seen(self, key: str, *, ttl: int = 600) -> bool:
        """Atomically record an inbound message id as processed (idempotency).

        Returns True the FIRST time (the caller should process this message) and
        False if the key was already seen within ``ttl`` — i.e. a Meta webhook
        retry to skip. Not tenant-scoped: WhatsApp message ids are globally
        unique.
        """
        assert self._redis is not None
        ok = await self._redis.set(f"seen:{key}", "1", nx=True, ex=ttl)
        return bool(ok)

    # ------------------------------------------------------------------
    # key builders — always tenant-scoped
    # ------------------------------------------------------------------

    def _key(self, conv_id: str, *, tenant_id: str | None = None) -> str:
        tid = tenant_id or get_current_tenant_id()
        return f"t:{tid}:conv:{conv_id}:history"

    def _last_product_key(self, conv_id: str, *, tenant_id: str | None = None) -> str:
        tid = tenant_id or get_current_tenant_id()
        return f"t:{tid}:conv:{conv_id}:last_product"

    def _browse_key(self, conv_id: str, *, tenant_id: str | None = None) -> str:
        tid = tenant_id or get_current_tenant_id()
        return f"t:{tid}:conv:{conv_id}:browse"

    def _saved_key(self, conv_id: str, *, tenant_id: str | None = None) -> str:
        tid = tenant_id or get_current_tenant_id()
        return f"t:{tid}:conv:{conv_id}:saved"

    def _applied_key(self, conv_id: str, *, tenant_id: str | None = None) -> str:
        tid = tenant_id or get_current_tenant_id()
        return f"t:{tid}:conv:{conv_id}:applied"

    # ------------------------------------------------------------------
    # history
    # ------------------------------------------------------------------

    async def append(
        self,
        conv_id: str,
        role: str,
        content: str,
        *,
        tenant_id: str | None = None,
    ) -> None:
        assert self._redis is not None
        key = self._key(conv_id, tenant_id=tenant_id)
        payload = json.dumps({"role": role, "content": content})
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, payload)
            pipe.ltrim(key, -_MAX_TURNS, -1)
            pipe.expire(key, self._ttl)
            await pipe.execute()

    async def history(
        self,
        conv_id: str,
        *,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        assert self._redis is not None
        raw = await self._redis.lrange(self._key(conv_id, tenant_id=tenant_id), 0, -1)
        out: list[dict[str, Any]] = []
        for item in raw:
            try:
                out.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return out

    async def clear(
        self,
        conv_id: str,
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Clear this conversation's short-term history, pinned product AND the
        in-progress job-browse selection. All keys are tenant+conversation
        scoped, so only this one customer is reset. (Saved/applied jobs are kept
        on purpose — they're the candidate's durable list.)"""
        assert self._redis is not None
        await self._redis.delete(
            self._key(conv_id, tenant_id=tenant_id),
            self._last_product_key(conv_id, tenant_id=tenant_id),
            self._browse_key(conv_id, tenant_id=tenant_id),
        )

    # ------------------------------------------------------------------
    # cached last-product (for facet follow-ups)
    # ------------------------------------------------------------------

    async def set_last_product(
        self,
        conv_id: str,
        product_id: str,
        doc: str,
        *,
        tenant_id: str | None = None,
    ) -> None:
        assert self._redis is not None
        payload = json.dumps({"product_id": product_id, "doc": doc})
        await self._redis.setex(
            self._last_product_key(conv_id, tenant_id=tenant_id), self._ttl, payload
        )

    async def get_last_product(
        self,
        conv_id: str,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, str] | None:
        assert self._redis is not None
        raw = await self._redis.get(self._last_product_key(conv_id, tenant_id=tenant_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # job-browse state (the multi-step category → role → location flow)
    # ------------------------------------------------------------------

    async def set_browse_state(
        self, conv_id: str, state: dict[str, Any], *, tenant_id: str | None = None
    ) -> None:
        """Remember where the candidate is in the tappable job-browse flow
        (selected category/role, page offset). TTL-bounded like the rest of the
        conversation's short-term memory."""
        assert self._redis is not None
        await self._redis.setex(
            self._browse_key(conv_id, tenant_id=tenant_id), self._ttl, json.dumps(state)
        )

    async def get_browse_state(
        self, conv_id: str, *, tenant_id: str | None = None
    ) -> dict[str, Any] | None:
        assert self._redis is not None
        raw = await self._redis.get(self._browse_key(conv_id, tenant_id=tenant_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # apply-time top-up (the in-progress "collect resume/salary" Q&A)
    # ------------------------------------------------------------------

    def _apply_key(self, conv_id: str, *, tenant_id: str | None = None) -> str:
        tid = tenant_id or get_current_tenant_id()
        return f"t:{tid}:conv:{conv_id}:apply"

    async def set_apply_state(
        self, conv_id: str, state: dict[str, Any], *, tenant_id: str | None = None
    ) -> None:
        assert self._redis is not None
        await self._redis.setex(
            self._apply_key(conv_id, tenant_id=tenant_id), self._ttl, json.dumps(state)
        )

    async def get_apply_state(
        self, conv_id: str, *, tenant_id: str | None = None
    ) -> dict[str, Any] | None:
        assert self._redis is not None
        raw = await self._redis.get(self._apply_key(conv_id, tenant_id=tenant_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def clear_apply_state(self, conv_id: str, *, tenant_id: str | None = None) -> None:
        assert self._redis is not None
        await self._redis.delete(self._apply_key(conv_id, tenant_id=tenant_id))

    # ------------------------------------------------------------------
    # saved jobs / applied-interest (the candidate's durable lists, Redis only)
    # ------------------------------------------------------------------

    async def _add_to_set(
        self, key: str, ref: str, job: dict[str, Any], *, ttl_days: int = 90
    ) -> None:
        assert self._redis is not None
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hset(key, ref, json.dumps(job, default=str))
            pipe.expire(key, ttl_days * 86400)
            await pipe.execute()

    async def save_job(
        self, conv_id: str, ref: str, job: dict[str, Any], *, tenant_id: str | None = None
    ) -> None:
        """Add a job to the candidate's saved list (keyed by its reference, so
        re-saving is idempotent)."""
        await self._add_to_set(
            self._saved_key(conv_id, tenant_id=tenant_id), ref, job
        )

    async def record_interest(
        self, conv_id: str, ref: str, job: dict[str, Any], *, tenant_id: str | None = None
    ) -> None:
        """Record that the candidate tapped Apply on a job. No business-DB write —
        a recruiter follows up — so we just keep the interest in Redis."""
        await self._add_to_set(
            self._applied_key(conv_id, tenant_id=tenant_id), ref, job
        )

    def _applications_key(self, conv_id: str, *, tenant_id: str | None = None) -> str:
        tid = tenant_id or get_current_tenant_id()
        return f"t:{tid}:conv:{conv_id}:applications"

    async def save_application(
        self, conv_id: str, ref: str, record: dict[str, Any], *, tenant_id: str | None = None
    ) -> None:
        """Stage a DB-ready ``private_job_applications`` record in Redis (keyed by
        job ref so re-applying overwrites). No business-DB write yet."""
        await self._add_to_set(
            self._applications_key(conv_id, tenant_id=tenant_id), ref, record
        )

    async def get_applications(
        self, conv_id: str, *, tenant_id: str | None = None
    ) -> dict[str, Any]:
        """All staged applications for this conversation, keyed by job ref."""
        assert self._redis is not None
        raw = await self._redis.hgetall(self._applications_key(conv_id, tenant_id=tenant_id))
        out: dict[str, Any] = {}
        for k, v in (raw or {}).items():
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                continue
        return out

    # ------------------------------------------------------------------
    # live feed (recent turns for the monitor UI) — per tenant
    # ------------------------------------------------------------------

    def _feed_key(self, tenant_id: str | None = None) -> str:
        tid = tenant_id or get_current_tenant_id()
        return f"t:{tid}:livefeed"

    async def push_feed(
        self, record: dict[str, Any], *, tenant_id: str | None = None, cap: int = 100
    ) -> None:
        """Append a turn record to this tenant's live feed (capped ring buffer)."""
        assert self._redis is not None
        key = self._feed_key(tenant_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, json.dumps(record, default=str))
            pipe.ltrim(key, -cap, -1)
            pipe.expire(key, self._ttl)
            await pipe.execute()

    async def recent_feed(
        self, *, tenant_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        assert self._redis is not None
        raw = await self._redis.lrange(self._feed_key(tenant_id), -limit, -1)
        out: list[dict[str, Any]] = []
        for item in raw:
            try:
                out.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return out

    # ------------------------------------------------------------------
    # onboarding form (new-candidate profile captured via a self-hosted
    # web form — stored in Redis ONLY, never the business DB)
    # ------------------------------------------------------------------
    #
    # Three keys per onboarding:
    #   t:{tid}:conv:{conv}:onboard        → the submitted form data (the gate)
    #   t:{tid}:conv:{conv}:onboard_token  → this conversation's current token
    #   onboard:token:{token}              → token → {tenant, customer, conv,
    #                                         name}  (NOT tenant-prefixed: the
    #                                         form POST only carries the token)

    def _onboard_key(self, conv_id: str, *, tenant_id: str | None = None) -> str:
        tid = tenant_id or get_current_tenant_id()
        return f"t:{tid}:conv:{conv_id}:onboard"

    def _onboard_token_fwd_key(self, conv_id: str, *, tenant_id: str | None = None) -> str:
        tid = tenant_id or get_current_tenant_id()
        return f"t:{tid}:conv:{conv_id}:onboard_token"

    @staticmethod
    def _onboard_token_key(token: str) -> str:
        return f"onboard:token:{token}"

    async def get_onboarding(
        self, conv_id: str, *, tenant_id: str | None = None
    ) -> dict[str, Any] | None:
        """The submitted onboarding form for this conversation, or None if the
        candidate hasn't completed it yet (the 'is onboarded?' gate)."""
        assert self._redis is not None
        raw = await self._redis.get(self._onboard_key(conv_id, tenant_id=tenant_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def mark_onboarding_welcomed(
        self, conv_id: str, *, tenant_id: str | None = None
    ) -> None:
        """Flag the submitted form as acknowledged in chat, so the one-time
        'profile complete' success message is sent exactly once (the next turn
        proceeds normally). Preserves the record's remaining TTL."""
        assert self._redis is not None
        key = self._onboard_key(conv_id, tenant_id=tenant_id)
        raw = await self._redis.get(key)
        if not raw:
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        data["welcomed"] = True
        ttl = await self._redis.ttl(key)
        if ttl and ttl > 0:
            await self._redis.setex(key, ttl, json.dumps(data))
        else:
            await self._redis.set(key, json.dumps(data))

    async def ensure_onboarding_token(
        self, conv_id: str, *, tenant_id: str, customer_id: str, name: str | None
    ) -> str:
        """Return this conversation's form token, minting one (and the reverse
        token→identity map the form POST resolves) on first use. Reused across
        turns so the candidate keeps getting the same link; TTL/name refreshed."""
        assert self._redis is not None
        ttl = get_settings().onboarding_ttl_seconds
        identity = json.dumps({
            "tenant_id": tenant_id, "customer_id": customer_id,
            "conversation_id": conv_id, "name": name or "",
        })
        fwd = self._onboard_token_fwd_key(conv_id, tenant_id=tenant_id)
        token = await self._redis.get(fwd)
        if not token:
            token = uuid.uuid4().hex
            await self._redis.setex(fwd, ttl, token)
        # (Re)write the reverse map so the token resolves and keeps a fresh TTL.
        await self._redis.setex(self._onboard_token_key(token), ttl, identity)
        await self._redis.expire(fwd, ttl)
        return token

    def _registration_key(self, conv_id: str, *, tenant_id: str | None = None) -> str:
        tid = tenant_id or get_current_tenant_id()
        return f"t:{tid}:conv:{conv_id}:registration"

    async def save_registration(
        self, conv_id: str, payload: dict[str, Any], *, tenant_id: str | None = None
    ) -> None:
        """Stage the DB-ready registration payload (seeker + profile + child rows)
        in Redis — NOT the business DB — so the data shape can be verified before
        real INSERTs are turned on."""
        assert self._redis is not None
        await self._redis.setex(
            self._registration_key(conv_id, tenant_id=tenant_id),
            get_settings().onboarding_ttl_seconds,
            json.dumps(payload, default=str),
        )
        log.info("registration_staged", conversation_id=conv_id)

    async def get_registration(
        self, conv_id: str, *, tenant_id: str | None = None
    ) -> dict[str, Any] | None:
        assert self._redis is not None
        raw = await self._redis.get(self._registration_key(conv_id, tenant_id=tenant_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def update_registration_profile(
        self, conv_id: str, fields: dict[str, Any], *, tenant_id: str | None = None
    ) -> None:
        """Merge fields (e.g. resume, expectedSalary collected at apply-time) into
        the staged registration's profile, so later applies don't re-ask."""
        reg = await self.get_registration(conv_id, tenant_id=tenant_id)
        if not reg:
            return
        profile = reg.get("job_seeker_profiles") or {}
        profile.update(fields)
        reg["job_seeker_profiles"] = profile
        await self.save_registration(conv_id, reg, tenant_id=tenant_id)

    async def get_onboarding_identity(self, token: str) -> dict[str, Any] | None:
        """Resolve a form token back to its candidate identity (for the form
        page + submission). None if the token is unknown or expired."""
        assert self._redis is not None
        raw = await self._redis.get(self._onboard_token_key(token))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def save_onboarding(
        self, token: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Store a form submission against the token's conversation (Redis only).
        Returns the resolved identity, or None when the token is invalid."""
        assert self._redis is not None
        identity = await self.get_onboarding_identity(token)
        if not identity:
            return None
        record = {
            "name": identity.get("name", ""),
            "email": (data.get("email") or "").strip(),
            "years_experience": (data.get("years_experience") or "").strip(),
            # Preferred role drives the "Recommended Jobs" menu button; older
            # records (before the form split) only carry the combined `location`.
            "preferred_role": (data.get("preferred_role") or "").strip(),
            "location": (data.get("location") or "").strip(),
            "submitted_at": int(time.time()),
        }
        await self._redis.setex(
            self._onboard_key(identity["conversation_id"], tenant_id=identity["tenant_id"]),
            get_settings().onboarding_ttl_seconds,
            json.dumps(record),
        )
        log.info("onboarding_saved", conversation_id=identity["conversation_id"])
        return identity
