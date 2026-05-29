"""LangGraph node implementations for the agent runtime.

Each node is a small async function that takes the ``AgentState`` plus its
explicit dependencies (gateway, llm, …). ``AgentRuntime`` binds the deps and
exposes them to the graph as coroutine methods. Keeping the logic in functions
(not methods) makes each node independently unit-testable with fakes.
"""
