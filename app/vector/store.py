"""Qdrant-backed semantic search service.

Multi-tenant by construction: every point carries ``tenant_id`` in payload
and every query enforces a must-match filter on it. Two tenants with the
same raw product id can never collide (deterministic UUID5 point id of
``"{tenant_id}:{raw_id}"``).

Hybrid search (``hybrid_search_enabled=True``, default):
    * Collections are created with two NAMED vector slots —
      ``dense`` (e.g. multilingual-e5 384-dim, cosine) and
      ``sparse`` (BM25-like keyword vector).
    * Upserts write both slots.
    * Reads run Qdrant ``query_points`` with two ``prefetch`` clauses and
      Reciprocal Rank Fusion. This recovers exact title/SKU matches that
      pure dense retrieval misses while keeping semantic recall.

Single-vector mode (``hybrid_search_enabled=False``):
    Collections are created with one unnamed vector slot — same shape as
    before the hybrid upgrade. Kept for the simpler deploy path.

Transactional data (price, stock, payment_status) NEVER lives here — that
lives in Postgres. Vectors are for similarity, not source-of-truth.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest

from app.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import VECTOR_LATENCY
from app.core.tenancy import get_current_tenant_id
from app.vector.embeddings import (
    EmbeddingClient,
    ImageEmbeddingClient,
    SparseEmbeddingClient,
)

log = get_logger("vector_store")

# Stable namespace so UUID5(tenant:raw) is deterministic across processes,
# reindexes, and deployments. Do not change — orphans every existing point.
_POINT_NS = uuid.UUID("8c5d2b9e-44a6-4f7c-ae0f-3f3b2d4c1e90")

# Named vector slots used in hybrid mode.
_DENSE = "dense"
_SPARSE = "sparse"


def _point_id(tenant_id: str, raw_id: str) -> str:
    return str(uuid.uuid5(_POINT_NS, f"{tenant_id}:{raw_id}"))


class VectorStore:
    """Async facade over Qdrant. Always tenant-scoped."""

    def __init__(self) -> None:
        self._client: AsyncQdrantClient | None = None
        self._embed = EmbeddingClient()
        self._sparse = SparseEmbeddingClient()
        self._image = ImageEmbeddingClient()
        self._connect_lock = asyncio.Lock()
        self._ready = False
        self._hybrid = True  # set from settings on connect()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._ready:
                return
            s = get_settings()
            self._hybrid = s.hybrid_search_enabled

            if s.vector_mode == "embedded":
                import os
                os.makedirs(s.qdrant_persist_dir, exist_ok=True)
                self._client = AsyncQdrantClient(path=s.qdrant_persist_dir)
            else:
                self._client = AsyncQdrantClient(
                    host=s.qdrant_host,
                    port=s.qdrant_port,
                    grpc_port=s.qdrant_grpc_port,
                    prefer_grpc=s.qdrant_use_grpc,
                    api_key=s.qdrant_api_key or None,
                    https=False,
                )

            await self._ensure_collections()
            self._ready = True
            log.info(
                "vector_store_ready",
                mode=s.vector_mode,
                hybrid=self._hybrid,
                collections=[
                    s.vector_collection_products,
                    s.vector_collection_faq,
                    s.vector_collection_policies,
                ],
                vector_size=s.local_embed_dimensions,
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._ready = False

    async def _ensure_collections(self) -> None:
        """Create domain collections (idempotent) and verify their schema
        matches the current ``hybrid_search_enabled`` mode.

        On mismatch (e.g. you flipped hybrid mode on/off, or upgraded from
        the pre-hybrid version), the existing collection has the wrong
        vector shape and every query would crash. Behaviour:

            embedded mode  → drop and recreate (dev data is disposable)
            http mode      → raise RuntimeError with the fix command

        Payload indexes are skipped in embedded mode because the local
        Qdrant doesn't support them and only logs spammy warnings.
        """
        assert self._client is not None
        s = get_settings()
        existing = {c.name for c in (await self._client.get_collections()).collections}

        # Image collection — separate from the text ones because CLIP is
        # 512-dim while text is 384-dim. Always single dense vector (no
        # sparse, since BM25 doesn't make sense for image bytes). Created
        # via the same idempotent + schema-mismatch-recreate path below.
        img_name = s.vector_collection_product_images
        if img_name in existing:
            if not await self._image_schema_matches(img_name):
                if s.vector_mode == "embedded":
                    log.warning(
                        "image_collection_schema_mismatch_recreating",
                        collection=img_name,
                    )
                    await self._client.delete_collection(img_name)
                    existing.discard(img_name)
                else:
                    raise RuntimeError(
                        f"Qdrant image collection {img_name!r} has the wrong "
                        f"vector size. Delete it and re-run /reindex/images."
                    )
        if img_name not in existing:
            await self._client.create_collection(
                collection_name=img_name,
                vectors_config=rest.VectorParams(
                    size=s.image_embed_dimensions,
                    distance=rest.Distance.COSINE,
                ),
            )
            log.info(
                "qdrant_image_collection_created",
                collection=img_name, size=s.image_embed_dimensions,
            )

        for name in (
            s.vector_collection_products,
            s.vector_collection_faq,
            s.vector_collection_policies,
            s.vector_collection_customer_memory,
        ):
            if name in existing:
                if not await self._schema_matches(name):
                    if s.vector_mode == "embedded":
                        log.warning(
                            "collection_schema_mismatch_recreating",
                            collection=name, hybrid=self._hybrid,
                        )
                        await self._client.delete_collection(name)
                        existing.discard(name)
                    else:
                        raise RuntimeError(
                            f"Qdrant collection {name!r} has a schema that "
                            f"doesn't match hybrid_search_enabled={self._hybrid}. "
                            f"Either flip HYBRID_SEARCH_ENABLED back, or delete "
                            f"the collection (POST /collections/{name}/snapshots "
                            f"first if you need a backup) and re-run reindex."
                        )

            if name not in existing:
                if self._hybrid:
                    await self._client.create_collection(
                        collection_name=name,
                        vectors_config={
                            _DENSE: rest.VectorParams(
                                size=s.local_embed_dimensions,
                                distance=rest.Distance.COSINE,
                            ),
                        },
                        sparse_vectors_config={
                            _SPARSE: rest.SparseVectorParams(
                                index=rest.SparseIndexParams(on_disk=False),
                            ),
                        },
                    )
                else:
                    await self._client.create_collection(
                        collection_name=name,
                        vectors_config=rest.VectorParams(
                            size=s.local_embed_dimensions,
                            distance=rest.Distance.COSINE,
                        ),
                    )
                log.info(
                    "qdrant_collection_created",
                    collection=name, hybrid=self._hybrid,
                )

            # Local/embedded Qdrant doesn't support payload indexes — it
            # logs a noisy stderr warning each time we try. Skip there;
            # full-scan filtering is fine at our scale.
            if s.vector_mode == "embedded":
                continue
            try:
                await self._client.create_payload_index(
                    collection_name=name,
                    field_name="tenant_id",
                    field_schema=rest.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("payload_index_exists_or_failed", collection=name, error=str(exc))

    async def _schema_matches(self, name: str) -> bool:
        """Return True if the existing collection's vector config matches
        the current ``self._hybrid`` mode."""
        assert self._client is not None
        info = await self._client.get_collection(name)
        params = info.config.params
        actual_vectors = getattr(params, "vectors", None)
        actual_sparse = getattr(params, "sparse_vectors", None) or {}
        is_named = isinstance(actual_vectors, dict)
        has_dense_slot = is_named and _DENSE in actual_vectors
        has_sparse_slot = isinstance(actual_sparse, dict) and _SPARSE in actual_sparse
        if self._hybrid:
            return has_dense_slot and has_sparse_slot
        return not is_named  # single unnamed vector → legacy mode

    async def _image_schema_matches(self, name: str) -> bool:
        """Return True if the image collection's vector size matches the
        configured CLIP dimension. Caught early so a model swap (e.g.
        ViT-B-32 → ViT-L-14) doesn't silently produce broken searches."""
        assert self._client is not None
        s = get_settings()
        info = await self._client.get_collection(name)
        params = info.config.params
        actual = getattr(params, "vectors", None)
        if isinstance(actual, dict):
            # Image collection should be a single unnamed vector.
            return False
        size = getattr(actual, "size", None)
        return size == s.image_embed_dimensions

    # ------------------------------------------------------------------
    # write side (called from worker)
    # ------------------------------------------------------------------

    async def upsert(
        self,
        collection: str,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        tenant_id: str | None = None,
    ) -> None:
        if not ids:
            return
        assert self._client is not None, "VectorStore.connect() not called"
        tid = tenant_id or get_current_tenant_id()
        docs = list(documents)

        # Dense + (optionally) sparse, both in parallel.
        if self._hybrid:
            dense, sparse = await asyncio.gather(
                self._embed.embed(docs, mode="passage"),
                self._sparse.embed(docs, mode="passage"),
            )
        else:
            dense = await self._embed.embed(docs, mode="passage")
            sparse = []

        metadatas = list(metadatas or [{} for _ in ids])
        if len(metadatas) != len(ids):
            raise ValueError("metadatas must align with ids when provided")

        points: list[rest.PointStruct] = []
        for i, (raw_id, doc, dvec, meta) in enumerate(
            zip(ids, docs, dense, metadatas)
        ):
            payload: dict[str, Any] = {
                "tenant_id": tid,
                "raw_id": str(raw_id),
                "document": doc,
                **(meta or {}),
            }
            if self._hybrid:
                indices, values = sparse[i]
                vector: Any = {
                    _DENSE: dvec,
                    _SPARSE: rest.SparseVector(indices=indices, values=values),
                }
            else:
                vector = dvec
            points.append(
                rest.PointStruct(
                    id=_point_id(tid, str(raw_id)),
                    vector=vector,
                    payload=payload,
                )
            )

        await self._client.upsert(collection_name=collection, points=points, wait=False)
        log.info(
            "vector_upsert",
            collection=collection, tenant_id=tid, n=len(points), hybrid=self._hybrid,
        )

    async def upsert_images(
        self,
        *,
        product_ids: Sequence[str],
        image_bytes: Sequence[bytes],
        metadatas: Sequence[dict[str, Any]] | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """Embed product images with CLIP and store in the image collection.
        One point per product (the first/primary image is what we index)."""
        if not product_ids:
            return
        assert self._client is not None
        s = get_settings()
        tid = tenant_id or get_current_tenant_id()
        vectors = await self._image.embed_bytes(list(image_bytes))
        metadatas = list(metadatas or [{} for _ in product_ids])

        points = [
            rest.PointStruct(
                id=_point_id(tid, str(pid)),
                vector=vec,
                payload={
                    "tenant_id": tid,
                    "raw_id": str(pid),
                    **(meta or {}),
                },
            )
            for pid, vec, meta in zip(product_ids, vectors, metadatas)
        ]
        await self._client.upsert(
            collection_name=s.vector_collection_product_images,
            points=points,
            wait=False,
        )
        log.info(
            "image_upsert",
            collection=s.vector_collection_product_images,
            tenant_id=tid, n=len(points),
        )

    async def count(
        self, collection: str, *, tenant_id: str | None = None
    ) -> int:
        """Number of points in a collection for the given tenant. Used by the
        admin status endpoint to confirm indexing actually wrote data."""
        assert self._client is not None
        tid = tenant_id or get_current_tenant_id()
        result = await self._client.count(
            collection_name=collection,
            count_filter=_build_filter(tenant_id=tid, where=None),
            exact=True,
        )
        return int(result.count)

    async def query_image(
        self,
        image_bytes: bytes,
        *,
        top_k: int = 5,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """CLIP-embed the inbound image and search the product_images
        collection. Returns Chroma-shaped dicts with cosine score so the
        caller can apply ``image_match_threshold``."""
        assert self._client is not None
        s = get_settings()
        tid = tenant_id or get_current_tenant_id()
        vectors = await self._image.embed_bytes([image_bytes])
        if not vectors:
            return []
        flt = _build_filter(tenant_id=tid, where=None)
        start = time.perf_counter()
        hits = await self._client.search(
            collection_name=s.vector_collection_product_images,
            query_vector=vectors[0],
            query_filter=flt,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        VECTOR_LATENCY.labels(
            collection=s.vector_collection_product_images
        ).observe(time.perf_counter() - start)

        out: list[dict[str, Any]] = []
        for h in hits:
            payload = dict(h.payload or {})
            raw_id = payload.pop("raw_id", str(h.id))
            payload.pop("tenant_id", None)
            score = float(h.score)
            out.append({
                "id": raw_id,
                "metadata": payload,
                "score": score,
                "distance": max(0.0, 1.0 - score),
            })
        return out

    async def delete(
        self,
        collection: str,
        ids: Sequence[str],
        *,
        tenant_id: str | None = None,
    ) -> None:
        if not ids:
            return
        assert self._client is not None
        tid = tenant_id or get_current_tenant_id()
        point_ids = [_point_id(tid, str(i)) for i in ids]
        await self._client.delete(
            collection_name=collection,
            points_selector=rest.PointIdsList(points=point_ids),
            wait=False,
        )
        log.info("vector_delete", collection=collection, tenant_id=tid, n=len(point_ids))

    # ------------------------------------------------------------------
    # read side (called from chat path)
    # ------------------------------------------------------------------

    async def query(
        self,
        collection: str,
        text: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Tenant-scoped search. Returns Chroma-shaped dicts so call sites
        upstream of this module don't change.

        In hybrid mode, dense + sparse pre-fetches are fused via Reciprocal
        Rank Fusion server-side. ``where`` adds AND filters alongside the
        mandatory tenant_id must-match."""
        assert self._client is not None
        s = get_settings()
        top_k = top_k or s.vector_top_k
        tid = tenant_id or get_current_tenant_id()
        flt = _build_filter(tenant_id=tid, where=where)

        start = time.perf_counter()
        if self._hybrid:
            dense_q, sparse_q = await asyncio.gather(
                self._embed.embed([text], mode="query"),
                self._sparse.embed([text], mode="query"),
            )
            if not dense_q:
                return []
            sparse_indices, sparse_values = sparse_q[0] if sparse_q else ([], [])
            prefetch_limit = max(s.hybrid_prefetch_limit, top_k)
            response = await self._client.query_points(
                collection_name=collection,
                prefetch=[
                    rest.Prefetch(
                        query=dense_q[0],
                        using=_DENSE,
                        filter=flt,
                        limit=prefetch_limit,
                    ),
                    rest.Prefetch(
                        query=rest.SparseVector(
                            indices=sparse_indices, values=sparse_values,
                        ),
                        using=_SPARSE,
                        filter=flt,
                        limit=prefetch_limit,
                    ),
                ],
                query=rest.FusionQuery(fusion=rest.Fusion.RRF),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            hits = response.points
        else:
            embeddings = await self._embed.embed([text], mode="query")
            if not embeddings:
                return []
            hits = await self._client.search(
                collection_name=collection,
                query_vector=embeddings[0],
                query_filter=flt,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        VECTOR_LATENCY.labels(collection=collection).observe(time.perf_counter() - start)

        out: list[dict[str, Any]] = []
        for h in hits:
            payload = dict(h.payload or {})
            raw_id = payload.pop("raw_id", str(h.id))
            document = payload.pop("document", "")
            payload.pop("tenant_id", None)
            score = float(getattr(h, "score", 0.0) or 0.0)
            # RRF score is unbounded positive; cosine score is in [-1,1]. We
            # only use distance for the cached-product follow-up heuristic
            # (distance > 0.6 → weak match), which is meaningful for cosine.
            # For RRF we surface score=score and synthesise a coarse
            # distance so the downstream heuristic still functions.
            distance = max(0.0, 1.0 - score) if score <= 1.0 else 0.0
            out.append({
                "id": raw_id,
                "document": document,
                "metadata": payload,
                "distance": distance,
                "score": score,
            })
        return out


# ---------------------------------------------------------------------------
# filter builder
# ---------------------------------------------------------------------------


def _build_filter(*, tenant_id: str, where: dict[str, Any] | None) -> rest.Filter:
    must: list[rest.FieldCondition] = [
        rest.FieldCondition(
            key="tenant_id",
            match=rest.MatchValue(value=tenant_id),
        )
    ]
    if where:
        for k, v in where.items():
            if isinstance(v, (list, tuple, set)):
                must.append(
                    rest.FieldCondition(key=k, match=rest.MatchAny(any=list(v)))
                )
            else:
                must.append(
                    rest.FieldCondition(key=k, match=rest.MatchValue(value=v))
                )
    return rest.Filter(must=must)
