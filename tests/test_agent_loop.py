"""MCPAgent loop: tool dispatch, message threading, and the iteration cap."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.chatbot import agent as agent_module
from app.chatbot.agent import MCPAgent
from app.mcp.tools import ToolContext


# --- fakes ------------------------------------------------------------------


def _msg(content: str | None = None, tool_calls: list[Any] | None = None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


class _FakeLLM:
    """Replays a scripted list of assistant messages, recording each call."""

    def __init__(self, scripted: list[Any]) -> None:
        self.scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    async def chat_with_tools(self, *, purpose, messages, tools, tool_choice="auto",
                              model=None, temperature=0.2, max_tokens=600):
        self.calls.append({"messages": list(messages), "tool_choice": tool_choice})
        return self.scripted.pop(0), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class _StubRegistry:
    def __init__(self, results: dict[str, Any] | None = None) -> None:
        self.dispatched: list[tuple[str, dict, ToolContext]] = []
        self._results = results or {}

    def openai_schemas(self):
        return [{
            "type": "function",
            "function": {
                "name": "get_order_status", "description": "d",
                "parameters": {
                    "type": "object",
                    "properties": {"order_number": {"type": "string"}},
                    "required": ["order_number"],
                },
            },
        }]

    def names(self):
        return ["get_order_status"]

    async def dispatch(self, name, args, ctx):
        self.dispatched.append((name, args, ctx))
        return self._results.get(name, {"ok": True})


def _patch_settings(monkeypatch, *, max_iter=4, model=""):
    monkeypatch.setattr(
        agent_module, "get_settings",
        lambda: SimpleNamespace(
            mcp_agent_max_iterations=max_iter, mcp_agent_model=model,
            llm_model_chat="gpt-4o-mini",
        ),
    )


def _ctx():
    return ToolContext(tenant_id="t1", customer_external_id="919876543210", request_id="r1")


# --- tests ------------------------------------------------------------------


async def test_agent_calls_tool_then_answers(monkeypatch):
    _patch_settings(monkeypatch)
    llm = _FakeLLM([
        _msg(tool_calls=[_tool_call("c1", "get_order_status", '{"order_number": "SS-AB12345678"}')]),
        _msg(content="Your order has shipped! 📦"),
    ])
    registry = _StubRegistry(results={
        "get_order_status": {"found": True, "order": {"order_number": "SS-AB12345678", "order_status": "SHIPPED"}},
    })
    agent = MCPAgent(llm=llm, registry=registry)

    result = await agent.run(
        customer_query="where is my order SS-AB12345678?", history=[], ctx=_ctx(),
    )

    assert result.response == "Your order has shipped! 📦"
    assert result.tool_calls == 1
    # tool dispatched with parsed args + the injected (not model-set) identity
    name, args, ctx = registry.dispatched[0]
    assert name == "get_order_status"
    assert args == {"order_number": "SS-AB12345678"}
    assert ctx.customer_external_id == "919876543210"
    # the tool result was threaded back into the 2nd LLM call
    second_call_roles = [m.get("role") for m in llm.calls[1]["messages"]]
    assert "tool" in second_call_roles
    assert llm.calls[0]["tool_choice"] == "auto"


async def test_agent_surfaces_products_for_rendering(monkeypatch):
    _patch_settings(monkeypatch)
    llm = _FakeLLM([
        _msg(tool_calls=[_tool_call("c1", "search_products", '{"query": "red saree"}')]),
        _msg(content="Here are some lovely red sarees! 🌹"),
    ])
    hits = [{"id": "p1", "title": "Red Silk Saree", "price": 1899}]
    registry = _StubRegistry(results={"search_products": hits})
    agent = MCPAgent(llm=llm, registry=registry)

    result = await agent.run(customer_query="red saree", history=[], ctx=_ctx())

    assert result.response == "Here are some lovely red sarees! 🌹"
    assert result.products == hits  # surfaced for the graph to render as a card/list


async def test_cached_product_is_seeded_into_context(monkeypatch):
    _patch_settings(monkeypatch, max_iter=1)
    llm = _FakeLLM([_msg(content="It comes in red and blue.")])
    agent = MCPAgent(llm=llm, registry=_StubRegistry())

    await agent.run(
        customer_query="what colors?", history=[], ctx=_ctx(),
        cached_product={"product_id": "p1", "doc": "Red Silk Saree — Sarees, ₹1899"},
    )

    # the recently-discussed product was put in the user message
    user_msg = llm.calls[0]["messages"][-1]["content"]
    assert "RECENTLY DISCUSSED PRODUCT" in user_msg
    assert "Red Silk Saree" in user_msg


async def test_last_step_disables_tools(monkeypatch):
    _patch_settings(monkeypatch, max_iter=1)
    llm = _FakeLLM([_msg(content="Here is a direct answer.")])
    agent = MCPAgent(llm=llm, registry=_StubRegistry())

    result = await agent.run(customer_query="hi", history=[], ctx=_ctx())

    assert result.response == "Here is a direct answer."
    assert result.tool_calls == 0
    assert llm.calls[0]["tool_choice"] == "none"  # only step is the last → forced answer


async def test_bad_tool_arguments_degrade_to_empty(monkeypatch):
    _patch_settings(monkeypatch)
    llm = _FakeLLM([
        _msg(tool_calls=[_tool_call("c1", "get_order_status", "not-json{")]),
        _msg(content="done"),
    ])
    registry = _StubRegistry()
    agent = MCPAgent(llm=llm, registry=registry)

    result = await agent.run(customer_query="?", history=[], ctx=_ctx())

    assert result.response == "done"
    assert registry.dispatched[0][1] == {}  # malformed JSON → {}


async def test_llm_failure_degrades_gracefully(monkeypatch):
    """A provider 429 / outage must not crash the turn — return a polite ask-again."""
    _patch_settings(monkeypatch)

    class _BoomLLM:
        async def chat_with_tools(self, **kwargs):
            raise RuntimeError("Error code: 429 - rate_limit_exceeded")

    agent = MCPAgent(llm=_BoomLLM(), registry=_StubRegistry())
    result = await agent.run(customer_query="show me sarees", history=[], ctx=_ctx())

    assert result.response  # non-empty, no exception
    assert "again" in result.response.lower()
    assert result.products == []


async def test_empty_final_answer_gets_fallback(monkeypatch):
    _patch_settings(monkeypatch, max_iter=1)
    llm = _FakeLLM([_msg(content=None)])
    agent = MCPAgent(llm=llm, registry=_StubRegistry())

    result = await agent.run(customer_query="?", history=[], ctx=_ctx())

    assert result.response  # never empty
    assert "couldn't" in result.response.lower()
