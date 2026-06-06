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


async def test_apply_confirmation_on_pinned_role_routes_to_submit_application():
    # "yes" on the pinned role → submit_application with the job_ref pulled from
    # the current-job doc, no LLM plan (the shortcut fires first).
    state = _state("yes")
    state["cached_product"] = {"doc": "Tester — IT — Chennai — ref tester-1775205995605"}
    out = await plan(state, llm=_FakeLLM({}), registry=_FakeRegistry(), memory_context=None)
    assert len(out["plan"]) == 1
    step = out["plan"][0]
    assert step["tool"] == "submit_application"
    assert step["args"] == {"job_ref": "tester-1775205995605"}


async def test_non_apply_followup_stays_direct_not_apply():
    # A non-apply follow-up ("tell me more") about the pinned role must NOT be
    # treated as an apply — it stays a direct answer (no tool step).
    state = _state("tell me more")
    state["cached_product"] = {"doc": "Tester — IT — Chennai — ref tester-1775205995605"}
    out = await plan(state, llm=_FakeLLM({}), registry=_FakeRegistry(), memory_context=None)
    assert out["plan"] == []
