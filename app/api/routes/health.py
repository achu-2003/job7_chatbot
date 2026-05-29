"""Liveness + readiness probes."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import ORJSONResponse
from sqlalchemy import text

from app.api.deps import get_memory, get_vector
from app.chatbot.memory import ConversationMemory
from app.db.session import session_scope
from app.schemas.domain import HealthStatus
from app.vector.store import VectorStore

router = APIRouter(default_response_class=ORJSONResponse)


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", response_model=HealthStatus)
async def ready(
    memory: ConversationMemory = Depends(get_memory),
    vector: VectorStore = Depends(get_vector),
) -> HealthStatus:
    db_ok = redis_ok = vec_ok = False
    try:
        async with session_scope() as s:
            await s.execute(text("SELECT 1"))
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    try:
        await memory._redis.ping()  # type: ignore[union-attr]
        redis_ok = True
    except Exception:  # noqa: BLE001
        redis_ok = False
    try:
        vec_ok = bool(vector._collections)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        vec_ok = False

    status = "ok" if db_ok and redis_ok and vec_ok else "degraded"
    return HealthStatus(status=status, db=db_ok, redis=redis_ok, vector=vec_ok)
