"""Tenant identity for the request lifecycle.

A tenant is a logical isolation boundary. Every API request resolves to exactly
one tenant via its ``X-API-Key`` header (or the configured default when
``multi_tenant_required`` is False). The resolved ``tenant_id`` is then
propagated to:

    * Qdrant payload filters (must-match on ``tenant_id``)
    * Redis conversation-memory keys (prefixed with ``t:{tenant_id}:``)
    * Postgres repository queries (when ``enforce_db_tenant_isolation`` is True)
    * Structured-log records (bound as ``tenant_id``)

The single source of truth for "which tenant is this code running for" is the
``current_tenant_id`` ContextVar, set by the auth dependency at the API edge
and inherited by every awaitable spawned inside that request.
"""
from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings
from app.core.logging import get_logger

log = get_logger("tenancy")


# ---------------------------------------------------------------------------
# context var
# ---------------------------------------------------------------------------

current_tenant_id: ContextVar[str | None] = ContextVar("tenant_id", default=None)


def get_current_tenant_id() -> str:
    """Return the bound tenant_id or the configured default. Never returns
    None — every code path downstream of auth has a concrete tenant string."""
    tid = current_tenant_id.get()
    if tid:
        return tid
    return get_settings().default_tenant_id


# ---------------------------------------------------------------------------
# tenant model + registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tenant:
    id: str
    name: str


class TenantRegistry:
    """Reads tenant_api_keys / tenant_names from settings on first use.

    The JSON env vars are parsed once and cached. If you rotate keys, restart
    the process — this is intentional (avoids races on a hot path)."""

    def __init__(self, api_key_map: dict[str, str], name_map: dict[str, str]) -> None:
        self._by_key = api_key_map
        self._names = name_map

    def resolve_by_api_key(self, api_key: str | None) -> Tenant | None:
        if not api_key:
            return None
        tid = self._by_key.get(api_key.strip())
        if not tid:
            return None
        return Tenant(id=tid, name=self._names.get(tid, tid))

    def get(self, tenant_id: str) -> Tenant:
        return Tenant(id=tenant_id, name=self._names.get(tenant_id, tenant_id))

    @property
    def known_ids(self) -> list[str]:
        return sorted({*self._by_key.values()})


@lru_cache(maxsize=1)
def get_registry() -> TenantRegistry:
    s = get_settings()
    api_key_map = _parse_json_map(s.tenant_api_keys, field="tenant_api_keys")
    name_map = _parse_json_map(s.tenant_names, field="tenant_names")
    # Ensure default tenant is always resolvable even with no registered keys.
    if s.default_tenant_id not in name_map:
        name_map[s.default_tenant_id] = s.default_tenant_id
    log.info(
        "tenant_registry_loaded",
        tenants=sorted({*api_key_map.values(), s.default_tenant_id}),
        keys=len(api_key_map),
        multi_tenant_required=s.multi_tenant_required,
    )
    return TenantRegistry(api_key_map=api_key_map, name_map=name_map)


def _parse_json_map(raw: str, *, field: str) -> dict[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("tenant_config_parse_failed", field=field, error=str(exc))
        return {}
    if not isinstance(parsed, dict):
        log.warning("tenant_config_not_object", field=field)
        return {}
    return {str(k): str(v) for k, v in parsed.items()}
