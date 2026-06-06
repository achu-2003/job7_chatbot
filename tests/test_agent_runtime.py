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


def _stub_memory(
    rt: AgentRuntime,
    *,
    facts: dict | None = None,
    candidate: dict | None = None,
    onboarded: bool = True,
    welcomed: bool = True,
    browse: dict | None = None,
) -> None:
    # Default to a fully-onboarded sender so the identity gate is a no-op and the
    # reasoning tests below exercise the normal path. Onboarding tests pass
    # facts={} / onboarded=False (and optionally a candidate) to drive the gate.
    if facts is None:
        facts = {"full_name": "Asha", "email": "asha@example.com"}

    async def fake_load(**kw):
        return {
            "session_id": "s1",
            "session_status": "active",
            "short_term": [],
            "rolling_summary": "",
            "customer_facts": dict(facts),
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

    async def fake_lookup(**kw):
        return candidate

    async def fake_onboarding(**kw):
        # The submitted form (the gate). None → not onboarded yet. ``welcomed``
        # marks whether we've already sent the in-chat success message.
        return {"email": "x@y.com", "welcomed": welcomed} if onboarded else None

    async def fake_onboarding_token(**kw):
        return "tok123"

    async def fake_mark_welcomed(**kw):
        return None

    async def fake_browse(**kw):
        return browse

    rt.gateway.load = fake_load            # type: ignore[assignment]
    rt.gateway.persist = fake_persist      # type: ignore[assignment]
    rt.gateway.set_focus_product = fake_set_focus  # type: ignore[assignment]
    rt.gateway.should_summarize = lambda st: False  # type: ignore[assignment]
    rt.gateway.onboarding = fake_onboarding          # type: ignore[assignment]
    rt.gateway.onboarding_token = fake_onboarding_token  # type: ignore[assignment]
    rt.gateway.mark_onboarding_welcomed = fake_mark_welcomed  # type: ignore[assignment]
    rt._candidate_lookup = fake_lookup     # type: ignore[assignment]
    rt._category_browse = fake_browse      # type: ignore[assignment]


async def _handle(rt: AgentRuntime, message: str = "i need a saree"):
    return await rt.handle(
        request_id="r1", conversation_id="wa_91", customer_query=message,
        customer_external_id="91", channel="whatsapp", tenant_id="t",
    )


def test_graph_has_reasoning_pipeline():
    rt = _runtime()
    assert sorted(rt._graph.get_graph().nodes) == [
        "__end__", "__start__", "browse", "execute", "greeting_response", "humanize",
        "identify", "load_context", "onboarding_response", "persist", "planner",
        "reflect", "responder", "summarize",
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


async def test_unknown_number_is_asked_for_name():
    """A number with no job-board record and no captured name is asked to
    introduce itself — with zero LLM calls — instead of being helped."""
    rt = _runtime()
    _stub_memory(rt, facts={}, onboarded=False)   # nothing on file, no DB candidate
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    assert "full name" in out["response"].lower()
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []   # 0 LLM


async def test_name_known_but_form_not_submitted_gets_form_link(monkeypatch):
    """Name captured, form not yet submitted → the gate hands over the form link
    (and keeps gating job help), with zero LLM calls. With a non-https base URL
    the link is inline text, no cta button. Pin the base URL so the test doesn't
    depend on the ambient .env (which may set an https ngrok URL)."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "public_base_url", "http://localhost:8000")
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Achuthan E"}, onboarded=False)
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "tell me about jobs")
    assert "form" in out["response"].lower()
    assert "/onboard/form?token=tok123" in out["response"]   # tokenised link
    assert "achuthan" in out["response"].lower()             # addressed by name
    assert out.get("whatsapp_interactive") is None           # http base → no cta
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


async def test_form_link_sent_as_cta_button_when_https(monkeypatch):
    """With an https public base URL, the form link is sent as a tappable
    cta_url 'Open form' button, and the inline-link text remains as fallback."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "public_base_url", "https://abc.ngrok-free.app")
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Partha"}, onboarded=False)
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "list jobs")

    cta = out["whatsapp_interactive"]
    assert cta is not None
    interactive = cta["interactive"]
    assert interactive["type"] == "cta_url"
    params = interactive["action"]["parameters"]
    assert params["display_text"] == "Open form"
    assert params["url"] == "https://abc.ngrok-free.app/onboard/form?token=tok123"
    # fallback text still carries the link inline
    assert "token=tok123" in out["response"]
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


async def test_form_submission_completes_onboarding():
    """Once the form is submitted (present in Redis), the gate opens and the
    sender is handled normally — greeted by name here, with 0 LLM."""
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Achuthan E"}, onboarded=True)
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    assert "achuthan" in out["response"].lower()    # past the gate → greeted
    assert "looking for" in out["response"].lower()
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


async def test_form_just_submitted_shows_success_message():
    """The first turn after the form is submitted gets a one-time 'profile
    complete' success message (0 LLM), then later turns proceed normally."""
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Achuthan E"}, onboarded=True, welcomed=False)
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    assert "all set" in out["response"].lower()
    assert "profile" in out["response"].lower()
    assert "achuthan" in out["response"].lower()
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


async def test_known_user_greeted_by_name():
    """A registered sender saying 'hi' is greeted by their first name (from the
    job board, seeded into facts by identify) — still 0 LLM."""
    rt = _runtime()
    _stub_memory(
        rt, facts={},
        candidate={"id": 1, "full_name": "Reg User", "email": "reg@x.com"},
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    assert "reg" in out["response"].lower()              # greeted by name
    assert "looking for" in out["response"].lower()
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


async def test_registered_jobseeker_skips_onboarding():
    """A phone found in the job board is treated as known — straight to the agent,
    no name/email questions."""
    rt = _runtime()
    _stub_memory(
        rt, facts={},
        candidate={"id": 1, "full_name": "Reg User", "email": "reg@x.com"},
    )
    rt.llm = _FakeLLM(plans=[{"goal": "greet", "direct_answer": True, "steps": []}],
                      reply="Here are some roles!")
    out = await _handle(rt, "show me python jobs")
    assert out["response"] == "Here are some roles!"   # normal agent path


async def test_category_browse_short_circuits_to_full_listing():
    """A message that names a category lists every job in it deterministically —
    no planner, no LLM — and is flagged for single-bubble delivery."""
    rt = _runtime()
    jobs = [{"job_ref": f"r{i}", "title": f"IT Role {i}", "location": "Chennai"}
            for i in range(18)]
    _stub_memory(rt, browse={"category": "Information Technology", "jobs": jobs})
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "show the IT jobs")
    assert "all 18 Information Technology roles" in out["response"]
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []   # 0 LLM
    assert len(out["delivery_plan"]) == 1                         # one bubble, not truncated


async def test_non_category_message_falls_through_to_agent():
    """When the message names no category, browse returns nothing and the normal
    planner path runs."""
    rt = _runtime()
    _stub_memory(rt, browse=None)
    rt.llm = _FakeLLM(plans=[{"goal": "greet", "direct_answer": True, "steps": []}],
                      reply="Found some roles!")
    out = await _handle(rt, "python developer roles")
    assert out["response"] == "Found some roles!"
    assert rt.llm.json_calls == ["agent_plan"]


async def test_apply_confirmation_hands_over_app_link_with_cta(monkeypatch):
    """A known candidate saying 'yes' on the pinned role is handed the Jobs7 app
    link as a tappable cta button — deterministically, with 0 LLM calls (the
    planner shortcut → submit_application → deterministic responder)."""
    from app.config import get_settings

    monkeypatch.setattr(
        get_settings(), "jobs7_app_url",
        "https://play.google.com/store/apps/details?id=com.jobs7",
    )
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Achuthan E"})

    # Pin a current role (as the focus-pin would after the candidate picked it).
    async def fake_load(**kw):
        return {
            "session_id": "s1", "session_status": "active", "short_term": [],
            "rolling_summary": "", "customer_facts": {"full_name": "Achuthan E"},
            "semantic_hits": [], "pending_actions": [],
            "goals": [{"id": "g1", "description": "d", "status": "active",
                       "priority": 0, "created_at": 0.0}],
            "last_active_at": None,
            "cached_product": {"product_id": "j1",
                               "doc": "Tester — IT — Chennai — ref tester-1775205995605"},
        }

    rt.gateway.load = fake_load  # type: ignore[assignment]

    async def fake_dispatch(name, args, ctx):
        assert name == "submit_application"
        return {"apply_via_app": True, "job_title": "Tester",
                "job_ref": "tester-1775205995605",
                "app_url": "https://play.google.com/store/apps/details?id=com.jobs7"}

    rt.tool_registry.dispatch = fake_dispatch  # type: ignore[assignment]
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")

    out = await _handle(rt, "yes")
    assert "Tester" in out["response"]
    assert "id=com.jobs7" in out["response"]
    assert "email" not in out["response"].lower()
    cta = out["whatsapp_interactive"]["interactive"]
    assert cta["action"]["parameters"]["display_text"] == "Open in Jobs7"
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []   # 0 LLM


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
