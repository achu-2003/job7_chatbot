"""Human escalation workflow.

Builds the canned escalation payload and records the event. In production
this would post to a CRM/Slack/Zendesk via webhook — that integration is
isolated here behind ``EscalationService.dispatch`` so it can be swapped
without touching the orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger

log = get_logger("escalation")


@dataclass
class EscalationTicket:
    reason: str
    customer_message: str
    severity: str = "normal"  # normal | high
    suggested_response: str = (
        "I'll connect you with one of our human agents who can help further. "
        "Your conversation has been forwarded and someone will reach out shortly."
    )


class EscalationService:
    async def dispatch(self, ticket: EscalationTicket, context: dict[str, Any]) -> None:
        # Plug your CRM/Slack/email integration here.
        log.warning(
            "escalation_triggered",
            reason=ticket.reason,
            severity=ticket.severity,
            customer_query=ticket.customer_message,
            **context,
        )
