"""Semantic memory — durable knowledge about a customer.

Two halves:
* **free-text** recall via the Qdrant ``customer_memory`` collection
  (`remember` / `recall`), tenant- and customer-scoped;
* **structured** facts via the ``customer_facts`` table (name, size, budget…).

Reuses the existing :class:`VectorStore` (`upsert`/`query` with a per-customer
payload filter) — no new vector primitives needed.
"""
from __future__ import annotations

import uuid
from typing import Any

from app.config import get_settings
from app.core.logging import get_logger
from app.memory.repositories import CustomerFactRepository
from app.vector.store import VectorStore

log = get_logger("semantic_memory")


class SemanticMemory:
    def __init__(self, vector: VectorStore) -> None:
        self.vector = vector

    # ---- free-text (vector) -----------------------------------------

    async def remember(
        self, text: str, *, tenant_id: str, customer_id: str, kind: str = "note"
    ) -> None:
        """Store a durable free-text memory (an exchange, a stated preference…)."""
        if not text.strip():
            return
        await self.vector.upsert(
            get_settings().vector_collection_customer_memory,
            ids=[uuid.uuid4().hex],
            documents=[text],
            metadatas=[{"customer_id": customer_id, "kind": kind}],
            tenant_id=tenant_id,
        )

    async def recall(
        self, query: str, *, tenant_id: str, customer_id: str, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Semantically recall this customer's prior memories relevant to the
        current message. Scoped to the customer via a payload filter."""
        if not query.strip():
            return []
        hits = await self.vector.query(
            get_settings().vector_collection_customer_memory,
            query,
            where={"customer_id": customer_id},
            tenant_id=tenant_id,
            top_k=top_k,
        )
        return [
            {
                "text": h.get("document", ""),
                "kind": (h.get("metadata") or {}).get("kind"),
            }
            for h in hits
        ]

    # ---- structured facts -------------------------------------------

    async def remember_fact(
        self,
        *,
        tenant_id: str,
        customer_id: str,
        key: str,
        value: str,
        confidence: float = 0.7,
        source: str | None = None,
    ) -> None:
        await CustomerFactRepository.upsert(
            tenant_id=tenant_id,
            customer_id=customer_id,
            key=key,
            value=value,
            confidence=confidence,
            source=source,
        )

    async def facts(self, *, tenant_id: str, customer_id: str) -> dict[str, str]:
        return await CustomerFactRepository.all_for(tenant_id, customer_id)
