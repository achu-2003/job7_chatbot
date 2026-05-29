"""Smoke test the FastAPI app boots and exposes /health/live without external services.

Heavier integration tests should run inside docker-compose against real
Postgres/Redis/Chroma.
"""
import asyncio

from fastapi.testclient import TestClient


def test_health_live_no_deps(monkeypatch):
    # patch out the startup hooks that need live services
    from app import main as app_module

    async def _noop():
        return None

    monkeypatch.setattr(app_module, "init_engine", _noop)
    monkeypatch.setattr(app_module, "close_engine", _noop)
    # the agent memory bootstrap also needs the DB — stub it for the deps-free boot
    monkeypatch.setattr(app_module, "ensure_schema", _noop)

    class _Stub:
        async def connect(self):
            return None

        async def close(self):
            return None

    monkeypatch.setattr(app_module, "VectorStore", _Stub)
    monkeypatch.setattr(app_module, "ConversationMemory", _Stub)
    monkeypatch.setattr(app_module, "EmbeddingQueue", _Stub)

    with TestClient(app_module.app) as client:
        r = client.get("/health/live")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}
