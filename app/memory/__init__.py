"""Agent memory layer.

Four tiers (see ``docs/AGENT_ARCHITECTURE.md`` §6):

* working   — the live ``AgentState`` scratchpad (in process, one turn)
* short-term — recent role-tagged turns in Redis (``app.chatbot.memory``)
* episodic  — "what happened" rows in Postgres (``agent_episodes``)
* semantic  — durable facts: Qdrant ``customer_memory`` + Postgres ``customer_facts``

This package owns the **read-write** ``agent_*`` tables. They are deliberately
separate from the read-only, Prisma-owned business tables (products/orders): the
agent never writes to business data.
"""
