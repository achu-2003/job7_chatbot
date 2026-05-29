"""Prometheus metrics. Scraped at /metrics."""
from __future__ import annotations

from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "chatbot_http_requests_total",
    "HTTP requests grouped by method/path/status",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "chatbot_http_request_latency_seconds",
    "End-to-end HTTP request latency",
    ["method", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

SQL_LATENCY = Histogram(
    "chatbot_sql_latency_seconds",
    "PostgreSQL query latency",
    ["op"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)

VECTOR_LATENCY = Histogram(
    "chatbot_vector_latency_seconds",
    "Vector retrieval latency",
    ["collection"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)

LLM_TOKENS = Counter(
    "chatbot_llm_tokens_total",
    "LLM tokens consumed (provider-agnostic)",
    ["model", "kind"],  # kind = prompt | completion
)

LLM_LATENCY = Histogram(
    "chatbot_llm_latency_seconds",
    "LLM/embed call latency (provider-agnostic)",
    ["model", "purpose"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
)

INTENT_COUNTER = Counter(
    "chatbot_intent_total",
    "Detected intents",
    ["intent"],
)

ROUTING_COUNTER = Counter(
    "chatbot_routing_total",
    "Retrieval routing decisions",
    ["sql_used", "vector_used"],
)

HALLUCINATION_COUNTER = Counter(
    "chatbot_validation_total",
    "Hallucination validation outcomes",
    ["result"],  # VALID | REGENERATED | ESCALATED
)

ERROR_COUNTER = Counter(
    "chatbot_errors_total",
    "Unhandled errors by layer",
    ["layer"],
)

EMBED_QUEUE_DEPTH = Histogram(
    "chatbot_embed_queue_depth",
    "Embedding queue depth observed by worker",
    buckets=(0, 1, 5, 10, 50, 100, 500, 1000),
)

# ---- production agent runtime ----

AGENT_LOOPS = Histogram(
    "chatbot_agent_loops",
    "Planner→executor→reflection loops per turn",
    buckets=(1, 2, 3, 4, 5),
)
AGENT_TOOL_CALLS = Counter(
    "chatbot_agent_tool_calls_total",
    "MCP tool dispatches by tool and result",
    ["tool", "result"],  # result = ok | error
)
AGENT_REFLECTIONS = Counter(
    "chatbot_agent_reflections_total",
    "Reflection verdicts",
    ["verdict"],  # finish | replan
)
AGENT_FOLLOWUPS = Counter(
    "chatbot_agent_followups_total",
    "Proactive follow-ups",
    ["event"],  # scheduled | sent
)
AGENT_PROMPT_CALLS = Counter(
    "chatbot_agent_prompt_calls_total",
    "LLM prompt invocations by prompt and version (rollout/rollback tracking)",
    ["prompt", "version"],
)
WEBHOOK_DEDUPE = Counter(
    "chatbot_webhook_dedupe_total",
    "Inbound WhatsApp webhook idempotency outcomes",
    ["result"],  # new | duplicate
)
