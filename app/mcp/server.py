"""Standalone MCP server exposing the recruiting tools to external MCP clients.

Wraps the same core capabilities the in-process agent uses (see
``app.mcp.tools``) as FastMCP tools, so Claude Desktop / any MCP client can
search jobs, submit and check applications, and read policies/FAQs with the same
tenant-scoped, identity-injected guarantees.

Run:  python -m app.mcp.server
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from app.mcp.tools import (
    get_application_status_core,
    request_human_handoff_core,
    search_docs_core,
    search_jobs_core,
    submit_application_core,
)
from app.chatbot.escalation import EscalationService
from app.config import get_settings
from app.vector.store import VectorStore

settings = get_settings()
mcp = FastMCP("recruiting-agent")

_vector: VectorStore | None = None
_escalation = EscalationService()


async def _vec() -> VectorStore:
    global _vector
    if _vector is None:
        _vector = VectorStore()
        await _vector.connect()
    return _vector


@mcp.tool()
async def search_jobs(
    query: str,
    tenant_id: str,
    location: str | None = None,
    employment_type: str | None = None,
    min_salary: float | None = None,
    max_salary: float | None = None,
) -> list[dict[str, Any]]:
    """Search open job postings."""
    return await search_jobs_core(
        await _vec(),
        tenant_id=tenant_id,
        query=query,
        location=location,
        employment_type=employment_type,
        min_salary=min_salary,
        max_salary=max_salary,
    )


@mcp.tool()
async def submit_application(
    job_ref: str,
    tenant_id: str,
    phone: str,
    full_name: str,
    email: str,
    years_experience: float | None = None,
    cover_note: str | None = None,
) -> dict[str, Any]:
    """Submit a candidate's application to a job (idempotent)."""
    return await submit_application_core(
        tenant_id=tenant_id,
        phone=phone,
        job_ref=job_ref,
        full_name=full_name,
        email=email,
        years_experience=years_experience,
        cover_note=cover_note,
    )


@mcp.tool()
async def get_application_status(
    phone: str, tenant_id: str, application_ref: str | None = None,
) -> dict[str, Any]:
    """Look up a candidate's application(s) by their phone."""
    return await get_application_status_core(
        tenant_id=tenant_id, phone=phone, application_ref=application_ref,
    )


@mcp.tool()
async def search_policies(query: str, tenant_id: str) -> list[dict[str, Any]]:
    """Search hiring policy documents."""
    return await search_docs_core(
        await _vec(), tenant_id=tenant_id, collection=settings.vector_collection_policies, query=query,
    )


@mcp.tool()
async def search_faq(query: str, tenant_id: str) -> list[dict[str, Any]]:
    """Search FAQ documents."""
    return await search_docs_core(
        await _vec(), tenant_id=tenant_id, collection=settings.vector_collection_faq, query=query,
    )


@mcp.tool()
async def request_human_handoff(
    reason: str, summary: str, tenant_id: str, severity: str = "normal",
) -> dict[str, Any]:
    """Escalate to a human recruiter."""
    return await request_human_handoff_core(
        _escalation, reason=reason, summary=summary, severity=severity,
        context={"tenant_id": tenant_id},
    )


if __name__ == "__main__":
    mcp.run()
