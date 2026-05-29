"""Production WhatsApp AI agent.

A stateful planner → executor → reflection agent with four-tier memory, dynamic
MCP tool routing, and human-like conversational pacing. See
``docs/AGENT_ARCHITECTURE.md`` for the full design and phased roadmap.

Phase 0 (this commit) lands the foundation only: the agent state model
(:mod:`app.agent.state`) and the per-turn budget (:mod:`app.agent.budget`). The
graph runtime and nodes arrive in later phases; the live bot is still served by
``app.chatbot.graph.ChatGraphRunner`` until the runtime reaches parity.
"""
