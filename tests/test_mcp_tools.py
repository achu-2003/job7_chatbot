"""MCP tool registry: schema shape, dispatch, and identity-injection safety."""
from __future__ import annotations

from typing import Any

import pytest

from app.db.repositories import (
    ApplicationRepository,
    CandidateRepository,
    JobRepository,
)
from app.memory.repositories import PendingActionRepository
from app.mcp.tools import ToolContext, ToolRegistry


class _FakeVector:
    """Stub VectorStore.query returning fixed job hits."""

    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self._hits = hits if hits is not None else [
            {"id": "j1", "metadata": {"job_id": "j1", "title": "Senior Backend Engineer"}},
        ]
        self.raise_on_query = False

    async def query(self, collection, text, *, tenant_id=None, top_k=None, where=None):
        if self.raise_on_query:
            raise RuntimeError("qdrant down")
        return self._hits


def _ctx(phone: str | None = "919876543210") -> ToolContext:
    return ToolContext(tenant_id="t1", customer_external_id=phone, request_id="REQ1")


def _registry(vector=None) -> ToolRegistry:
    return ToolRegistry(vector=vector or _FakeVector())


# --- schema -----------------------------------------------------------------


def test_schemas_are_openai_shaped():
    reg = _registry()
    schemas = reg.openai_schemas()
    assert {s["function"]["name"] for s in schemas} == set(reg.names())
    for s in schemas:
        assert s["type"] == "function"
        assert "parameters" in s["function"]


def test_application_tools_never_expose_identity():
    """The model must not be able to set tenant_id or another candidate's phone:
    those keys must be absent from every application tool's argument schema."""
    reg = _registry()
    by_name = {s["function"]["name"]: s["function"]["parameters"] for s in reg.openai_schemas()}
    for tool in ("get_application_status", "submit_application"):
        props = by_name[tool].get("properties", {})
        assert "phone" not in props
        assert "candidate_ref" not in props
        assert "tenant_id" not in props


# --- dispatch ---------------------------------------------------------------


async def test_search_jobs_returns_compact_rows(monkeypatch):
    async def fake_get_by_ids(ids, *, tenant_id=None):
        return [{
            "id": "j1", "job_ref": "JOB-AB1001", "title": "Senior Backend Engineer",
            "department_name": "Engineering", "location": "Bengaluru",
            "employment_type": "full_time", "seniority": "senior",
            "salary_min": 2500000, "salary_max": 4000000, "salary_currency": "INR",
            "skills": ["python", "fastapi"], "status": "LIVE",
        }]

    monkeypatch.setattr(JobRepository, "get_by_ids", staticmethod(fake_get_by_ids))
    out = await _registry().dispatch("search_jobs", {"query": "backend"}, _ctx())
    assert out == [{
        "id": "j1", "job_ref": "JOB-AB1001", "title": "Senior Backend Engineer",
        "department": "Engineering", "location": "Bengaluru",
        "employment_type": "full_time", "seniority": "senior",
        "salary_min": 2500000.0, "salary_max": 4000000.0, "salary_currency": "INR",
        "skills": ["python", "fastapi"], "availability": "open",
    }]


async def test_search_jobs_labels_expired_availability(monkeypatch):
    # Non-open jobs are shown but flagged so the bot won't tell a candidate to
    # apply to a dead listing.
    async def fake_get_by_ids(ids, *, tenant_id=None):
        return [{"id": "j2", "job_ref": "old-role", "title": "Old Role",
                 "status": "EXPIRED", "salary_min": 1000, "salary_max": 2000}]

    monkeypatch.setattr(JobRepository, "get_by_ids", staticmethod(fake_get_by_ids))
    out = await _registry().dispatch("search_jobs", {"query": "old"}, _ctx())
    assert out[0]["availability"] == "expired"


async def test_search_jobs_hits_without_metadata_job_id(monkeypatch):
    # Regression: real vector hits may carry the job id only as `id` (no
    # metadata.job_id). The old filter required metadata.job_id and silently
    # dropped every hit → "no jobs found" even though search matched. Now the
    # hit's own id is used as a fallback.
    async def fake_get_by_ids(ids, *, tenant_id=None):
        assert ids == ["job-xyz"]  # the hit id was used
        return [{
            "id": "job-xyz", "job_ref": "office-staff", "title": "Office Staff",
            "department_name": "Administration", "location": None,
            "employment_type": "FULL_TIME", "seniority": None,
            "salary_min": 20000, "salary_max": 45000, "salary_currency": "INR",
            "skills": [],
        }]

    monkeypatch.setattr(JobRepository, "get_by_ids", staticmethod(fake_get_by_ids))
    # hit has NO metadata.job_id — just id + a stray metadata field
    vector = _FakeVector(hits=[{"id": "job-xyz", "metadata": {"title": "Office Staff"}}])
    out = await _registry(vector).dispatch("search_jobs", {"query": "office"}, _ctx())
    assert len(out) == 1 and out[0]["title"] == "Office Staff"


async def test_search_jobs_facet_filter_excludes_mismatch(monkeypatch):
    async def fake_get_by_ids(ids, *, tenant_id=None):
        return [{
            "id": "j1", "job_ref": "JOB-AB1001", "title": "Senior Backend Engineer",
            "location": "Bengaluru", "employment_type": "full_time",
            "salary_min": 2500000, "salary_max": 4000000,
        }]

    monkeypatch.setattr(JobRepository, "get_by_ids", staticmethod(fake_get_by_ids))
    out = await _registry().dispatch(
        "search_jobs", {"query": "engineer", "location": "Mumbai"}, _ctx(),
    )
    assert out == []  # the only hit is Bengaluru, candidate asked for Mumbai


async def test_get_application_status_scopes_to_caller(monkeypatch):
    seen: dict[str, Any] = {}

    async def fake_candidate_get(*, tenant_id=None, phone=None):
        seen["phone"] = phone
        return {"id": "c1"}

    async def fake_latest(candidate_id, *, tenant_id=None, limit=5):
        seen["candidate_id"] = candidate_id
        return [{"app_ref": "APP-CD5678", "status": "SCREENING", "job_title": "Backend Eng"}]

    monkeypatch.setattr(CandidateRepository, "get", staticmethod(fake_candidate_get))
    monkeypatch.setattr(ApplicationRepository, "latest_for_candidate", staticmethod(fake_latest))
    out = await _registry().dispatch("get_application_status", {}, _ctx("919876543210"))
    assert out["found"] is True
    assert out["applications"][0]["app_ref"] == "APP-CD5678"
    assert seen["candidate_id"] == "c1"
    assert seen["phone"] in {"919876543210", "9876543210"}


async def test_get_application_status_no_candidate(monkeypatch):
    async def fake_candidate_get(*, tenant_id=None, phone=None):
        return None

    monkeypatch.setattr(CandidateRepository, "get", staticmethod(fake_candidate_get))
    out = await _registry().dispatch("get_application_status", {}, _ctx())
    assert out["found"] is False


async def test_application_status_requires_identity():
    out = await _registry().dispatch("get_application_status", {}, _ctx(phone=None))
    assert "error" in out


async def test_submit_application_returns_app_link_without_asking_for_details(monkeypatch):
    # Applying is finished in the Jobs7 app: an identified candidate confirming a
    # real role gets the app link back (no name/email gate — those are on file),
    # and the live job board's write repos are never touched.
    async def fake_get_by_ref(ref, *, tenant_id=None):
        return {"id": "j1", "job_ref": "senior-backend-engineer",
                "title": "Senior Backend Engineer"}

    monkeypatch.setattr(JobRepository, "get_by_ref", staticmethod(fake_get_by_ref))
    out = await _registry().dispatch(
        "submit_application", {"job_ref": "senior-backend-engineer"}, _ctx(),
    )
    assert out["submitted"] is False
    assert out["apply_via_app"] is True
    assert out["job_title"] == "Senior Backend Engineer"
    assert out["app_url"].startswith("https://play.google.com/store/apps/details")
    assert "jobs7" in out["message"].lower()


async def test_submit_application_requires_identity():
    out = await _registry().dispatch(
        "submit_application", {"job_ref": "JOB-AB1001"}, _ctx(phone=None),
    )
    assert out["error"] == "no candidate identity on this channel"


async def test_submit_application_unknown_job(monkeypatch):
    async def fake_get_by_ref(ref, *, tenant_id=None):
        return None

    monkeypatch.setattr(JobRepository, "get_by_ref", staticmethod(fake_get_by_ref))
    out = await _registry().dispatch(
        "submit_application", {"job_ref": "JOB-ZZ9999"}, _ctx(),
    )
    assert out["error"] == "job_not_found"


async def test_handoff_dispatches_ticket():
    reg = _registry()
    calls: list[Any] = []

    async def fake_dispatch(ticket, context):
        calls.append((ticket, context))

    reg.escalation.dispatch = fake_dispatch  # type: ignore[assignment]
    out = await reg.dispatch(
        "request_human_handoff",
        {"reason": "withdraw application", "summary": "wants to withdraw"},
        _ctx(),
    )
    assert out["handed_off"] is True
    assert calls and calls[0][0].reason == "withdraw application"
    assert calls[0][1]["candidate"] == "919876543210"


async def test_schedule_followup_requires_a_session():
    out = await _registry().dispatch(
        "schedule_followup", {"minutes": 60, "message": "hi"}, _ctx()
    )
    assert "error" in out


async def test_schedule_followup_writes_pending_action(monkeypatch):
    captured: dict = {}

    async def fake_add(**kw):
        captured.update(kw)
        return {"id": "pa1"}

    monkeypatch.setattr(PendingActionRepository, "add", staticmethod(fake_add))
    ctx = ToolContext(tenant_id="t1", customer_external_id="9198",
                      request_id="r", session_id="sess-1")
    out = await _registry().dispatch(
        "schedule_followup", {"minutes": 30, "message": "Update on your application!"}, ctx
    )
    assert out == {"scheduled": True, "in_minutes": 30}
    assert captured["session_id"] == "sess-1"
    assert captured["tenant_id"] == "t1"
    assert captured["kind"] == "followup"
    assert captured["run_after"] is not None


async def test_unknown_tool_returns_error():
    out = await _registry().dispatch("delete_everything", {}, _ctx())
    assert "error" in out and "unknown tool" in out["error"]


async def test_dispatch_never_raises_on_backend_failure():
    vector = _FakeVector()
    vector.raise_on_query = True
    out = await _registry(vector).dispatch("search_jobs", {"query": "x"}, _ctx())
    assert out["error"] == "tool execution failed"
