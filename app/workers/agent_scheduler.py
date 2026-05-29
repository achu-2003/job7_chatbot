"""Proactive follow-up scheduler.

Drains due ``agent_pending_actions`` (scheduled by the ``schedule_followup``
tool) and sends them to the customer on WhatsApp — the bit that makes the agent
feel proactive ("your size is back in stock") rather than purely reactive.

Scope note: WhatsApp only allows business-initiated messages within 24h of the
customer's last message; beyond that you need an approved template. This worker
sends plain text (fine inside the window) — wire a template path for older
follow-ups when you add templates. OFF by default (``agent_scheduler_enabled``).

``run_once`` is the testable unit; ``start_in_background`` runs it on a poll
loop, mirroring the embedding worker's lifecycle so main.py manages it the same
way.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from app.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import AGENT_FOLLOWUPS
from app.memory.repositories import PendingActionRepository
from app.whatsapp import delivery as wa_delivery

log = get_logger("agent_scheduler")


class AgentScheduler:
    def __init__(self, *, tenant_id: str | None = None, poll_seconds: int | None = None) -> None:
        s = get_settings()
        self.settings = s
        self.tenant_id = tenant_id or s.whatsapp_tenant_id or s.default_tenant_id
        self.poll_seconds = poll_seconds or s.agent_scheduler_poll_seconds
        self._stop = False
        self._task: asyncio.Task | None = None

    async def run_once(self) -> int:
        """Send every due follow-up for this tenant. Returns how many were sent."""
        due = await PendingActionRepository.due(self.tenant_id)
        sent = 0
        for action in due:
            payload = _as_dict(action.get("payload"))
            customer = payload.get("customer_id")
            message = payload.get("message")
            try:
                if customer and message:
                    await wa_delivery.deliver(
                        self.settings, customer, [{"text": message, "typing_ms": 0}],
                    )
                    AGENT_FOLLOWUPS.labels(event="sent").inc()
                    sent += 1
                await PendingActionRepository.mark(action["id"], "done")
            except Exception as exc:  # noqa: BLE001 — one bad action shouldn't stall the rest
                log.warning("pending_action_failed", id=str(action.get("id")), error=str(exc)[:200])
        if due:
            log.info("scheduler_drained", due=len(due), sent=sent, tenant_id=self.tenant_id)
        return sent

    async def _loop(self) -> None:
        log.info("agent_scheduler_started", tenant_id=self.tenant_id, poll_seconds=self.poll_seconds)
        while not self._stop:
            try:
                await self.run_once()
            except Exception as exc:  # noqa: BLE001
                log.warning("scheduler_tick_error", error=str(exc)[:200])
            await asyncio.sleep(self.poll_seconds)

    async def start_in_background(self) -> asyncio.Task:
        self._task = asyncio.create_task(self._loop(), name="agent_scheduler")
        return self._task

    def stop(self) -> None:
        self._stop = True


def _as_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}
    return {}
