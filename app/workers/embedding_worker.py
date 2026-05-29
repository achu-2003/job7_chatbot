"""Generic embedding worker.

Queue contract — every job is one of:

    {"action": "upsert", "collection": "<name>", "tenant_id": "<id>",
     "id": "<id>", "document": "<text>", "metadata": {...}}

    {"action": "delete", "collection": "<name>", "tenant_id": "<id>",
     "ids": ["<id>", ...]}

    {"action": "upsert_image", "tenant_id": "<id>",
     "product_id": "<id>", "image_url": "https://...",
     "metadata": {"title": "...", "category": "..."}}

``tenant_id`` is required on every job — Qdrant points are tenant-scoped and
deleting/upserting without it would silently target the wrong namespace. The
reindex endpoints are the only producers. Upserts are idempotent — re-running
reindex any time is safe.
"""
from __future__ import annotations

import asyncio
import signal
from typing import Any

import httpx

from app.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.metrics import EMBED_QUEUE_DEPTH, ERROR_COUNTER
from app.db.session import close_engine, init_engine
from app.vector.store import VectorStore
from app.workers.queue import EmbeddingQueue

configure_logging()
log = get_logger("embedding_worker")


class Worker:
    """Embedding sync worker. Can run standalone or bundled into the API."""

    def __init__(
        self,
        *,
        vector: VectorStore | None = None,
        queue: EmbeddingQueue | None = None,
        owns_resources: bool = True,
    ) -> None:
        self._vector = vector or VectorStore()
        self._queue = queue or EmbeddingQueue()
        self._owns_resources = owns_resources
        self._stop = asyncio.Event()

    async def run(self, *, install_signal_handlers: bool = True) -> None:
        if self._owns_resources:
            await init_engine()
            await self._queue.connect()
            await self._vector.connect()

        if install_signal_handlers:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, self._stop.set)
                except NotImplementedError:
                    pass

        try:
            await self._consume()
        finally:
            if self._owns_resources:
                await self._vector.close()
                await self._queue.close()
                await close_engine()

    async def start_in_background(self) -> asyncio.Task:
        return asyncio.create_task(self.run(install_signal_handlers=False),
                                   name="embedding_worker")

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------

    async def _consume(self) -> None:
        while not self._stop.is_set():
            try:
                event = await self._queue.dequeue(timeout=2)
                if event is None:
                    continue
                depth = await self._queue.depth()
                EMBED_QUEUE_DEPTH.observe(depth)
                await self._process(event)
            except Exception as exc:  # noqa: BLE001
                ERROR_COUNTER.labels(layer="embedding_worker").inc()
                log.exception("worker_process_error", error=str(exc))

    async def _process(self, event: dict[str, Any]) -> None:
        action = event.get("action")
        tenant_id = event.get("tenant_id")
        # `collection` is required for text upsert/delete but NOT for
        # upsert_image (the image collection name is resolved internally by
        # VectorStore.upsert_images).
        if not action or not tenant_id:
            log.warning("invalid_event", event=event)
            return

        if action == "upsert":
            collection = event.get("collection")
            ident = event.get("id")
            doc = event.get("document")
            if not collection or not ident or not doc:
                log.warning("invalid_upsert", event=event)
                return
            await self._vector.upsert(
                collection,
                ids=[ident],
                documents=[doc],
                metadatas=[event.get("metadata") or {}],
                tenant_id=tenant_id,
            )
        elif action == "delete":
            collection = event.get("collection")
            ids = event.get("ids") or []
            if collection and ids:
                await self._vector.delete(collection, ids, tenant_id=tenant_id)
        elif action == "upsert_image":
            product_id = event.get("product_id")
            image_url = event.get("image_url")
            if not product_id or not image_url:
                log.warning("invalid_upsert_image", event=event)
                return
            image_bytes = await self._download_image(image_url)
            if image_bytes is None:
                return
            await self._vector.upsert_images(
                product_ids=[product_id],
                image_bytes=[image_bytes],
                metadatas=[event.get("metadata") or {}],
                tenant_id=tenant_id,
            )
        else:
            log.warning("unknown_action", action=action)

    async def _download_image(self, url: str) -> bytes | None:
        """Fetch product image bytes. Returns None on any failure — image
        indexing is best-effort, a single bad URL must not poison the
        worker."""
        timeout = get_settings().image_download_timeout_seconds
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
                resp = await c.get(url)
            if resp.status_code != 200:
                log.warning(
                    "image_download_non_200", url=url, status=resp.status_code,
                )
                return None
            content_type = resp.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                log.warning(
                    "image_download_wrong_type", url=url, content_type=content_type,
                )
                return None
            # CLIP/PIL can't rasterise vector formats — skip cleanly rather
            # than letting PIL raise "cannot identify image file" downstream
            # (which would inflate the worker error counter). Common with
            # placeholder.co SVGs on seed/demo products.
            if "svg" in content_type:
                log.info("image_skip_unsupported", url=url, content_type=content_type)
                return None
            return resp.content
        except httpx.HTTPError as exc:
            log.warning("image_download_failed", url=url, error=str(exc))
            return None


def main() -> None:
    worker = Worker()
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
