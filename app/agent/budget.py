"""Per-turn budget — the guardrail that makes the reasoning loop *bounded*.

The planner ⇄ executor ⇄ reflection loop must never spin forever or blow the
WhatsApp/webhook latency window. Every iteration ``tick()``s the budget; the
graph checks ``exhausted()`` and, when true, forces the responder to answer with
whatever has been gathered (graceful, never a 500).

Time is measured with ``time.monotonic()`` (immune to wall-clock jumps). All
methods accept an injectable ``now`` so they're trivially unit-testable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.config import get_settings


@dataclass
class Budget:
    max_loops: int
    deadline: float        # monotonic timestamp after which we must stop
    loops: int = 0

    @classmethod
    def start(
        cls, *, max_loops: int, deadline_seconds: float, now: float | None = None
    ) -> "Budget":
        base = time.monotonic() if now is None else now
        return cls(max_loops=max_loops, deadline=base + deadline_seconds)

    @classmethod
    def from_settings(cls, now: float | None = None) -> "Budget":
        s = get_settings()
        return cls.start(
            max_loops=s.agent_max_loops,
            deadline_seconds=s.agent_deadline_seconds,
            now=now,
        )

    def tick(self) -> None:
        self.loops += 1

    def exhausted(self, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        return self.loops >= self.max_loops or current >= self.deadline

    def remaining_loops(self) -> int:
        return max(0, self.max_loops - self.loops)

    def time_left(self, now: float | None = None) -> float:
        current = time.monotonic() if now is None else now
        return max(0.0, self.deadline - current)
