"""Shared FastAPI dependencies (state-bound singletons created at startup).

This module also owns the per-request tenant resolution used by every route
under /api/v1/*. The dependency reads ``X-API-Key`` (or accepts the configured
default when ``multi_tenant_required`` is False), maps it to a Tenant, and
binds the tenant_id to:

    * ``current_tenant_id`` ContextVar (read by VectorStore / memory / repos)
    * the structlog contextvars store (so log lines carry tenant_id)
"""
from __future__ import annotations

import structlog
from fastapi import Depends, Header, HTTPException, Request, status

from app.chatbot.graph import ChatGraphRunner
from app.chatbot.memory import ConversationMemory
from app.config import get_settings
from app.core.logging import get_logger
from app.core.tenancy import Tenant, current_tenant_id, get_registry
from app.vector.store import VectorStore
from app.workers.queue import EmbeddingQueue

log = get_logger("auth")


def get_vector(request: Request) -> VectorStore:
    return request.app.state.vector


def get_memory(request: Request) -> ConversationMemory:
    return request.app.state.memory


def get_queue(request: Request) -> EmbeddingQueue:
    return request.app.state.queue


def get_graph(request: Request) -> ChatGraphRunner:
    return request.app.state.graph_runner


async def require_tenant(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Tenant:
    """Resolve the calling tenant. Returns a Tenant object and binds the
    tenant_id to the request context + structlog so downstream code never has
    to thread the value manually.

    Behaviour:
        * ``multi_tenant_required=True`` → X-API-Key is mandatory and must
          match a registered key. 401 on missing/unknown.
        * ``multi_tenant_required=False`` → if X-API-Key is present it is
          honored; otherwise the default tenant is used. Backwards-compatible
          with single-tenant deployments.
    """
    s = get_settings()
    registry = get_registry()

    tenant = registry.resolve_by_api_key(x_api_key)

    if tenant is None:
        if s.multi_tenant_required:
            log.warning("auth_missing_or_invalid_key", provided=bool(x_api_key))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid X-API-Key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        tenant = registry.get(s.default_tenant_id)

    current_tenant_id.set(tenant.id)
    structlog.contextvars.bind_contextvars(tenant_id=tenant.id)
    return tenant


def bind_tenant(tenant_id: str) -> Tenant:
    """Programmatic tenant binding for non-HTTP entry points (workers, the
    WhatsApp webhook after we map the phone_number_id to a tenant). Returns
    the Tenant so callers can read its name; sets the ContextVar + log
    binding."""
    registry = get_registry()
    tenant = registry.get(tenant_id)
    current_tenant_id.set(tenant.id)
    structlog.contextvars.bind_contextvars(tenant_id=tenant.id)
    return tenant


TenantDep = Depends(require_tenant)
