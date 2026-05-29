"""Pydantic request/response schemas for the chat API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, max_length=64)
    customer_external_id: str | None = Field(default=None, max_length=64)
    channel: Literal["whatsapp", "web", "mobile", "other"] = "web"


class RoutingInfo(BaseModel):
    sql_used: bool
    vector_used: bool
    llm_used: bool
    escalate: bool
    reason: str


class ChatResponse(BaseModel):
    request_id: str
    conversation_id: str
    intent: str | None
    response: str
    routing: RoutingInfo | None
    validation: str | None
    escalated: bool
    latency_ms: int
    # debug/observability (used by the Streamlit flow UI; None on other channels)
    trace: list | None = None       # ordered [label, value] events for this turn
    memory: dict | None = None      # memory snapshot (focus product, facts, …)
    steps: dict | None = None       # plan + tool calls (args/results) + reflections
