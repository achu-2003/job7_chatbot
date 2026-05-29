"""Admin endpoints.

POST /reindex
    Re-embeds the calling tenant's jobs + FAQs + policies. The tenant is
    identified by ``X-API-Key``; jobs are stamped with the resolved
    ``tenant_id`` so the embedding worker writes to a tenant-isolated subset
    of the shared Qdrant collections.

GET /reindex/status
    Returns current queue depth (global; the queue is not partitioned).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import ORJSONResponse

from app.api.deps import TenantDep, get_queue
from app.config import get_settings
from app.core.tenancy import Tenant
from app.db.repositories import JobRepository
from app.memory.gateway import MemoryGateway
from app.workers.queue import EmbeddingQueue

router = APIRouter(default_response_class=ORJSONResponse)

CONTENT_DIR = Path(__file__).resolve().parents[3] / "content"


def _job_document(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    skills = ", ".join(row.get("skills") or [])
    parts = [
        f"TITLE: {row['title']}",
        f"DEPARTMENT: {row.get('department_name') or 'unknown'}",
        f"LOCATION: {row.get('location') or '-'}",
        f"EMPLOYMENT TYPE: {row.get('employment_type') or '-'}",
        f"SENIORITY: {row.get('seniority') or '-'}",
        f"SKILLS: {skills or '-'}",
        f"DESCRIPTION: {row.get('description') or ''}",
    ]
    document = "\n".join(parts)
    metadata = {
        "job_id": row["id"],
        "job_ref": row.get("job_ref"),
        "title": row["title"],
        "department": row.get("department_name"),
        "location": row.get("location"),
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}
    return document, metadata


def _faq_document(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    doc = f"Q: {item['question']}\nA: {item['answer']}"
    return doc, {"tags": ", ".join(item.get("tags", []))}


def _policy_document(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    doc = f"{item['title']}\n{item['body']}"
    return doc, {"category": item.get("category", "")}


def _load_json(name: str) -> list[dict[str, Any]]:
    path = CONTENT_DIR / name
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/reindex", summary="Re-embed jobs, FAQs and policies for the calling tenant")
async def reindex(
    tenant: Tenant = TenantDep,
    queue: EmbeddingQueue = Depends(get_queue),
) -> dict[str, Any]:
    s = get_settings()
    counts = {"jobs": 0, "faqs": 0, "policies": 0}

    # ---- jobs from live DB
    rows = await JobRepository.iter_active_for_embedding(tenant_id=tenant.id)
    for row in rows:
        document, metadata = _job_document(row)
        await queue.enqueue(
            {
                "action": "upsert",
                "collection": s.vector_collection_products,
                "tenant_id": tenant.id,
                "id": row["id"],
                "document": document,
                "metadata": metadata,
            }
        )
    counts["jobs"] = len(rows)

    # ---- FAQs from JSON
    for item in _load_json("faqs.json"):
        document, metadata = _faq_document(item)
        await queue.enqueue(
            {
                "action": "upsert",
                "collection": s.vector_collection_faq,
                "tenant_id": tenant.id,
                "id": item["id"],
                "document": document,
                "metadata": metadata,
            }
        )
        counts["faqs"] += 1

    # ---- Policies from JSON
    for item in _load_json("policies.json"):
        document, metadata = _policy_document(item)
        await queue.enqueue(
            {
                "action": "upsert",
                "collection": s.vector_collection_policies,
                "tenant_id": tenant.id,
                "id": item["id"],
                "document": document,
                "metadata": metadata,
            }
        )
        counts["policies"] += 1

    return {"tenant_id": tenant.id, **counts}


@router.get("/reindex/status")
async def reindex_status(
    request: Request,
    tenant: Tenant = TenantDep,
    queue: EmbeddingQueue = Depends(get_queue),
) -> dict[str, Any]:
    """Queue depth + per-collection point counts for the calling tenant.
    Use the counts to confirm indexing actually wrote vectors (e.g. the jobs
    collection > 0 means job search is ready)."""
    s = get_settings()
    vector = request.app.state.vector
    depth = await queue.depth()
    counts: dict[str, int] = {}
    for name in (
        s.vector_collection_products,
        s.vector_collection_faq,
        s.vector_collection_policies,
    ):
        try:
            counts[name] = await vector.count(name, tenant_id=tenant.id)
        except Exception as exc:  # noqa: BLE001
            counts[name] = -1  # collection missing / not yet created
    return {"tenant_id": tenant.id, "queue_depth": depth, "counts": counts}


@router.get("/diag/vector")
async def diag_vector(
    q: str,
    request: Request,
    tenant: Tenant = TenantDep,
) -> dict[str, Any]:
    """Diagnostic: run a vector search against the products collection and
    return the raw top hits + scores. Useful for debugging retrieval-quality
    issues without grepping logs."""
    s = get_settings()
    vector = request.app.state.vector
    hits = await vector.query(
        s.vector_collection_products, q, top_k=10, tenant_id=tenant.id,
    )
    return {
        "tenant_id": tenant.id,
        "collection": s.vector_collection_products,
        "embed_model": s.local_embed_model,
        "hybrid": s.hybrid_search_enabled,
        "query": q,
        "n_hits": len(hits),
        "hits": [
            {
                "id": h.get("id"),
                "score": h.get("score"),
                "distance": h.get("distance"),
                "title": (h.get("metadata") or {}).get("title"),
                "category": (h.get("metadata") or {}).get("category"),
                "doc": (h.get("document") or "")[:120],
            }
            for h in hits
        ],
    }


@router.get("/live-feed", summary="Recent WhatsApp/chat turns for the live monitor UI")
async def live_feed(
    request: Request,
    tenant: Tenant = TenantDep,
    limit: int = 50,
) -> dict[str, Any]:
    """Most recent turns (client info + received message + reply + flow trace +
    memory) for this tenant — the Streamlit monitor polls this."""
    turns = await request.app.state.memory.recent_feed(
        tenant_id=tenant.id, limit=max(1, min(limit, 200)),
    )
    return {"tenant_id": tenant.id, "count": len(turns), "turns": turns}


@router.post("/reset-customer", summary="Refresh one customer's conversation memory by phone")
async def reset_customer(
    phone: str,
    request: Request,
    tenant: Tenant = TenantDep,
) -> dict[str, Any]:
    """Clear a SINGLE customer's short-term history + pinned product + session
    summary (durable customer_facts are kept). The customer is identified by
    phone — the same key the WhatsApp webhook uses (``wa_<digits>``). Strictly
    scoped to this (tenant, customer); never affects anyone else's memory.

        POST /api/v1/admin/reset-customer?phone=917540031625
    """
    digits = "".join(c for c in phone if c.isdigit())
    if not digits:
        return ORJSONResponse({"error": "phone must contain digits"}, status_code=400)
    conv_id = f"wa_{digits}"
    gateway = MemoryGateway(
        vector=request.app.state.vector, short_term=request.app.state.memory,
    )
    await gateway.reset_session(tenant_id=tenant.id, conversation_id=conv_id)
    return {"status": "reset", "tenant_id": tenant.id, "conversation_id": conv_id}
