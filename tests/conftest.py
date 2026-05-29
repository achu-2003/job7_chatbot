"""Test environment: force unit-test-safe settings BEFORE any app import.

We need to override the live `.env` so tests don't try to connect to the
production Postgres or start the embedding worker.
"""
import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://x:y@localhost:5432/x")
os.environ.setdefault("LLM_API_KEY", "sk-test")
os.environ["RUN_WORKER_IN_API"] = "false"
os.environ["VECTOR_MODE"] = "embedded"
os.environ["JSON_LOGS"] = "false"
os.environ["REDIS_URL"] = "redis://127.0.0.1:6379/0"
