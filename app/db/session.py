"""Async SQLAlchemy engine + session factory."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


async def init_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        return
    settings = get_settings()
    # The DB is a SHARED Postgres cluster with a low max_connections, so keep a
    # small, well-behaved pool: at most pool_size+max_overflow = 5 connections
    # per process. pool_recycle drops connections after 15 min (so a crashed/
    # reloaded process can't leave them idle forever), pool_pre_ping skips dead
    # ones, and application_name tags our connections so they're identifiable in
    # pg_stat_activity (the earlier 'too many clients' pile-up was untagged).
    _engine = create_async_engine(
        settings.database_url,
        pool_size=3,
        max_overflow=2,
        pool_timeout=30,
        pool_recycle=900,
        pool_pre_ping=True,
        connect_args={"server_settings": {"application_name": "job7_chatbot"}},
        future=True,
    )
    _sessionmaker = async_sessionmaker(
        _engine, expire_on_commit=False, class_=AsyncSession
    )


async def close_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    if _sessionmaker is None:
        await init_engine()
    assert _sessionmaker is not None
    async with _sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_engine() -> AsyncEngine:
    assert _engine is not None, "engine not initialised"
    return _engine
