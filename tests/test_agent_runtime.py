"""AgentRuntime — Phase 2 planner→executor→reflection loop (LLM + memory mocked)."""
from types import SimpleNamespace

from app.agent.runtime import AgentRuntime


class _FakeVector:
    async def query(self, *a, **k):
        return []


class _FakeLLM:
    """Scripted JSON plans/reflections + a canned final reply."""

    def __init__(self, *, plans, reflects=None, reply="OK 😊"):
        self._plans = list(plans)
        self._reflects = list(reflects or [])
        self.reply = reply
        self.json_calls: list[str] = []
        self.chat_calls: list[str] = []

    async def json_chat(self, *, purpose, messages, **k):
        self.json_calls.append(purpose)
        if purpose == "agent_plan":
            return (self._plans.pop(0) if self._plans else {}), {}
        if purpose == "agent_reflect":
            nxt = self._reflects.pop(0) if self._reflects else {"goal_satisfied": True, "next": "finish"}
            return nxt, {}
        return {}, {}

    async def chat(self, *, purpose, messages, **k):
        self.chat_calls.append(purpose)
        return self.reply, {}


def _runtime() -> AgentRuntime:
    return AgentRuntime(vector=_FakeVector(), memory=SimpleNamespace())


def _stub_memory(rt: AgentRuntime) -> None:
    async def fake_load(**kw):
        return {
            "session_id": "s1",
            "session_status": "active",
            "short_term": [],
            "rolling_summary": "",
            "customer_facts": {},
            "semantic_hits": [],
            "pending_actions": [],
            # pre-existing goal so the planner skips the DB write in tests
            "goals": [{"id": "g1", "description": "help the customer",
                       "status": "active", "priority": 0, "created_at": 0.0}],
            "last_active_at": None,
        }

    async def fake_persist(**kw):
        return None

    async def fake_set_focus(**kw):
        return None

    rt.gateway.load = fake_load            # type: ignore[assignment]
    rt.gateway.persist = fake_persist      # type: ignore[assignment]
    rt.gateway.set_focus_product = fake_set_focus  # type: ignore[assignment]
    rt.gateway.should_summarize = lambda st: False  # type: ignore[assignment]


async def _handle(rt: AgentRuntime, message: str = "i need a saree"):
    return await rt.handle(
        request_id="r1", conversation_id="wa_91", customer_query=message,
        customer_external_id="91", channel="whatsapp", tenant_id="t",
    )


def test_graph_has_reasoning_pipeline():
    rt = _runtime()
    assert sorted(rt._graph.get_graph().nodes) == [
        "__end__", "__start__", "execute", "greeting_response", "humanize",
        "load_context", "persist", "planner", "reflect", "responder", "summarize",
    ]


async def test_greeting_short_circuits_the_llm():
    rt = _runtime()
    _stub_memory(rt)
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    assert "looking for" in out["response"].lower()      # fixed greeting
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []   # 0 LLM


def test_planner_context_is_lean_no_summary_pollution():
    rt = _runtime()
    state = {
        "inbound_text": "details about blouse",          # names a product → not a follow-up
        "cached_product": {"doc": "Saree — ₹999", "product_id": "p1"},
        "rolling_summary": "earlier talked about a bridal lehenga",
        "semantic_hits": [{"text": "lehenga maroon size L"}],
        "customer_facts": {"name": "Asha"},
    }
    planner_ctx = rt._memory_context(state, for_planner=True) or ""
    full_ctx = rt._memory_context(state, for_planner=False) or ""
    assert "lehenga" not in planner_ctx.lower()   # planner won't pollute the search
    assert "Asha" in planner_ctx                   # facts kept
    assert "lehenga" in full_ctx.lower()           # responder still has full context


async def test_farewell_resets_session():
    rt = _runtime()
    _stub_memory(rt)
    reset_calls: list = []

    async def fake_reset(**kw):
        reset_calls.append(kw)

    rt.gateway.reset_session = fake_reset  # type: ignore[assignment]
    rt.llm = _FakeLLM(plans=[], reply="x")
    out = await _handle(rt, "that's all")
    assert "anytime" in out["response"].lower()
    assert rt.llm.json_calls == []                       # 0 LLM
    assert reset_calls and reset_calls[0]["conversation_id"] == "wa_91"


async def test_handle_returns_delivery_plan_and_compat_keys():
    rt = _runtime()
    _stub_memory(rt)
    rt.llm = _FakeLLM(plans=[{"goal": "greet", "direct_answer": True, "steps": []}],
                      reply="Hey! How can I help today? 😊")
    out = await _handle(rt)
    # ChatResponse-compatible keys present (so the HTTP chat endpoint works too)
    assert out["validation"] == "AGENT"
    assert out["routing"] is None and out["escalated"] is False
    # paced bubbles produced
    plan = out["delivery_plan"]
    assert plan and all("typing_ms" in b and b["text"] for b in plan)


async def test_direct_answer_skips_tools():
    rt = _runtime()
    _stub_memory(rt)
    rt.llm = _FakeLLM(plans=[{"goal": "greet", "direct_answer": True, "steps": []}],
                      reply="Hey there! 👋")
    out = await _handle(rt)
    assert out["response"] == "Hey there! 👋"
    assert rt.llm.json_calls == ["agent_plan"]      # no reflect
    assert rt.llm.chat_calls == ["agent_respond"]


async def test_tool_path_plans_executes_reflects_responds():
    rt = _runtime()
    _stub_memory(rt)
    rt.llm = _FakeLLM(
        plans=[{"goal": "find python jobs", "direct_answer": False,
                "steps": [{"tool": "search_jobs", "args": {"query": "python developer"}}]}],
        reflects=[{"goal_satisfied": True, "next": "finish"}],
        reply="Here are some open Python roles!",
    )
    dispatched: list[tuple] = []

    async def fake_dispatch(name, args, ctx):
        dispatched.append((name, args))
        return [{"id": "p1", "job_ref": "JOB-AB1001", "title": "Python Developer"}]

    rt.tool_registry.dispatch = fake_dispatch  # type: ignore[assignment]

    out = await _handle(rt)
    assert out["response"].startswith("Here are")
    assert dispatched[0][0] == "search_jobs"
    # token saver: one successful tool step skips the reflection call
    assert rt.llm.json_calls == ["agent_plan"]
    assert rt.llm.chat_calls == ["agent_respond"]


async def test_replan_is_bounded_by_budget():
    from app.config import get_settings

    rt = _runtime()
    _stub_memory(rt)
    # A multi-step plan (so reflection runs); reflection always wants to replan
    # → only the loop budget stops it.
    rt.llm = _FakeLLM(
        plans=[{"goal": "g", "direct_answer": False, "steps": [
            {"tool": "search_jobs", "args": {"query": "x"}},
            {"tool": "search_faq", "args": {"query": "y"}},
        ]}] * 10,
        reflects=[{"goal_satisfied": False, "next": "replan"}] * 10,
        reply="Best I can do for now!",
    )

    async def fake_dispatch(name, args, ctx):
        return []

    rt.tool_registry.dispatch = fake_dispatch  # type: ignore[assignment]

    out = await _handle(rt)
    assert out["response"] == "Best I can do for now!"
    # capped at agent_max_loops — never spins
    assert rt.llm.json_calls.count("agent_plan") == get_settings().agent_max_loops
    assert rt.llm.chat_calls == ["agent_respond"]
