"""Planner normalises search-tool args so a missing/misnamed `query` never
reaches the handler as a KeyError (regression for tool_error 'query')."""
from app.agent.nodes.planner import plan


class _FakeLLM:
    def __init__(self, obj):
        self._obj = obj

    async def json_chat(self, *, purpose, messages, **k):
        return self._obj, {}


class _FakeRegistry:
    def names(self):
        return {"search_jobs", "search_policies", "search_faq", "submit_application"}

    def openai_schemas(self):
        return [{"function": {"name": n, "description": n}} for n in self.names()]


def _state(text="show me python jobs"):
    return {"inbound_text": text, "loop_count": 0, "goals": [{"id": "g", "description": "d",
            "status": "active", "priority": 0, "created_at": 0.0}]}


async def _plan(obj, text="show me python jobs"):
    return await plan(_state(text), llm=_FakeLLM(obj),
                      registry=_FakeRegistry(), memory_context=None)


async def test_missing_query_falls_back_to_inbound_text():
    out = await _plan({"goal": "find jobs", "direct_answer": False,
                       "steps": [{"tool": "search_jobs", "args": {}}]})
    assert out["plan"][0]["args"]["query"] == "show me python jobs"


async def test_aliased_query_is_normalised():
    out = await _plan({"goal": "find jobs", "direct_answer": False,
                       "steps": [{"tool": "search_jobs", "args": {"q": "python"}}]})
    assert out["plan"][0]["args"]["query"] == "python"


async def test_blank_query_falls_back():
    out = await _plan({"goal": "find jobs", "direct_answer": False,
                       "steps": [{"tool": "search_jobs", "args": {"query": "   "}}]},
                      text="data roles")
    assert out["plan"][0]["args"]["query"] == "data roles"


async def test_good_query_is_kept():
    out = await _plan({"goal": "find jobs", "direct_answer": False,
                       "steps": [{"tool": "search_jobs", "args": {"query": "remote backend"}}]})
    assert out["plan"][0]["args"]["query"] == "remote backend"
