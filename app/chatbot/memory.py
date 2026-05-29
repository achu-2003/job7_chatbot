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
        """Clear this conversation's short-term history AND pinned product. Both
        keys are tenant+conversation scoped, so only this one customer is reset."""
        assert self._redis is not None
        await self._redis.delete(
            self._key(conv_id, tenant_id=tenant_id),
            self._last_product_key(conv_id, tenant_id=tenant_id),
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
