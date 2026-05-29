"""Legacy chat graph runner — RETIRED for the job-application migration.

The original ``ChatGraphRunner`` was the pre-agent e-commerce pipeline (it read
the catalog via ``ProductRepository`` and customer orders via
``OrderRepository``). It has been superseded by
``app.agent.runtime.AgentRuntime`` (the production planner/memory/reflection
agent), which is the only runner wired to the live routes — the chat and
WhatsApp endpoints call ``graph_runner.handle(...)``, which this legacy runner
never implemented.

During the migration to the job-application domain the e-commerce repositories
this runner depended on were replaced, so the old body no longer applies. Rather
than carry dead e-commerce code, the runner is kept only as an import-safe stub
so ``AGENT_RUNTIME_ENABLED=false`` fails loudly with a clear message instead of
silently serving the wrong domain.

To bring back a non-agent fallback, reimplement ``handle()`` here against the
job schema (``JobRepository`` / ``ApplicationRepository``).
"""
from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

log = get_logger("chat_graph_runner")


class ChatGraphRunner:
    """Retired legacy runner. Kept import-safe; raises if actually used."""

    def __init__(self, *, vector: Any, memory: Any, llm: Any = None) -> None:
        self.vector = vector
        self.memory = memory
        self.llm = llm
        log.warning(
            "legacy_runner_instantiated",
            note="ChatGraphRunner is retired; set AGENT_RUNTIME_ENABLED=true to use AgentRuntime.",
        )

    async def handle(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "ChatGraphRunner is retired in the job-application build. "
            "Set AGENT_RUNTIME_ENABLED=true (default) to use AgentRuntime, or "
            "reimplement ChatGraphRunner.handle() against the job schema."
        )
