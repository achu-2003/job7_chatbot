"""Simple Redis list as a durable FIFO queue.

We avoid RQ/Celery to keep the worker memory footprint tiny — a single list +
BLPOP is enough for embedding sync. Idempotency is handled by UPSERT in the
vector store.
"""
from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis

from app.config import get_settings
from app.core.logging import get_logger

log = get_logger("queue")


class EmbeddingQueue:
    def __init__(self) -> None:
        self._redis: redis.Redis | None = None
        self._key = get_settings().redis_queue_key

    async def connect(self) -> None:
        self._redis = redis.from_url(get_settings().redis_url, decode_responses=True)
        await self._redis.ping()
        log.info("queue_ready", key=self._key)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def enqueue(self, event: dict[str, Any]) -> None:
        assert self._redis is not None
        await self._redis.rpush(self._key, json.dumps(event))

    async def dequeue(self, timeout: int = 5) -> dict[str, Any] | None:
        assert self._redis is not None
        res = await self._redis.blpop([self._key], timeout=timeout)
        if not res:
            return None
        _key, payload = res
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            log.warning("queue_bad_payload", payload=payload)
            return None

    async def depth(self) -> int:
        assert self._redis is not None
        return int(await self._redis.llen(self._key))
