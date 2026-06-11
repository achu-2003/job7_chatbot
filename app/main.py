"""FastAPI entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from app.api.routes import admin, chat, employer, health, onboard, whatsapp
from app.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware
from app.db.session import close_engine, init_engine
from app.agent.runtime import AgentRuntime
from app.chatbot.graph import ChatGraphRunner
from app.chatbot.memory import ConversationMemory
from app.memory.repositories import ensure_schema
from app.vector.store import VectorStore
from app.workers.agent_scheduler import AgentScheduler
from app.workers.embedding_worker import Worker as EmbeddingWorker
from app.workers.queue import EmbeddingQueue

settings = get_settings()
configure_logging()
log = get_logger("app")


def _base_url_unreachable(url: str) -> bool:
    """True when the onboarding base URL is loopback — such links are neither
    tappable (no TLD) nor reachable from a phone over WhatsApp."""
    import re

    return bool(re.search(r"://(localhost|127\.0\.0\.1|0\.0\.0\.0)\b", url or "", re.I))


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "startup",
        env=settings.app_env,
        version="1.0.0",
        vector_mode=settings.vector_mode,
        worker_in_api=settings.run_worker_in_api,
    )
    if _base_url_unreachable(settings.public_base_url):
        log.warning(
            "public_base_url_not_reachable",
            url=settings.public_base_url,
            hint="Onboarding form links won't be tappable or openable on WhatsApp. "
            "Set PUBLIC_BASE_URL to a public https URL (e.g. an ngrok/cloudflared "
            "tunnel) for anywhere, or http://<your-LAN-IP>:8000 for same-WiFi testing.",
        )
    await init_engine()
    app.state.vector = VectorStore()
    app.state.memory = ConversationMemory()
    app.state.queue = EmbeddingQueue()
    await app.state.vector.connect()
    await app.state.memory.connect()
    await app.state.queue.connect()
    # Production AgentRuntime (planner/memory/reflection + paced delivery) unless
    # rolled back to the legacy ChatGraphRunner via AGENT_RUNTIME_ENABLED=false.
    if settings.agent_runtime_enabled:
        await ensure_schema()  # idempotent: create agent_* memory tables if absent
        app.state.graph_runner = AgentRuntime(
            vector=app.state.vector, memory=app.state.memory,
        )
        log.info("serving_with", engine="agent_runtime")
    else:
        app.state.graph_runner = ChatGraphRunner(
            vector=app.state.vector, memory=app.state.memory,
        )
        log.info("serving_with", engine="chat_graph_runner")

    app.state.worker = None
    app.state.worker_task = None
    if settings.run_worker_in_api:
        app.state.worker = EmbeddingWorker(
            vector=app.state.vector,
            queue=app.state.queue,
            owns_resources=False,
        )
        app.state.worker_task = await app.state.worker.start_in_background()
        log.info("embedding_worker_bundled")

    # Proactive follow-up scheduler (drains scheduled WhatsApp follow-ups).
    app.state.scheduler = None
    if settings.agent_runtime_enabled and settings.agent_scheduler_enabled:
        app.state.scheduler = AgentScheduler()
        await app.state.scheduler.start_in_background()
        log.info("agent_scheduler_bundled")

    try:
        yield
    finally:
        if app.state.scheduler is not None:
            app.state.scheduler.stop()
        if app.state.worker is not None:
            app.state.worker.stop()
        if app.state.worker_task is not None:
            try:
                await app.state.worker_task
            except Exception:  # noqa: BLE001
                pass
        await app.state.queue.close()
        await app.state.memory.close()
        await app.state.vector.close()
        await close_engine()
        log.info("shutdown")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=Response,  # routes set their own JSONResponse
)
app.add_middleware(RequestContextMiddleware)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(whatsapp.router, prefix="/api/whatsapp", tags=["whatsapp"])
app.include_router(onboard.router, prefix="/onboard", tags=["onboard"])
app.include_router(employer.router, prefix="/employer", tags=["employer"])

# Serve uploaded resumes (read-only) from local disk. Created on startup so the
# mount never fails on a fresh checkout.
import os  # noqa: E402

from starlette.staticfiles import StaticFiles  # noqa: E402

os.makedirs("uploads/resumes", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.get("/metrics")
async def metrics() -> Response:
    if not settings.prometheus_enabled:
        return Response(status_code=404)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
