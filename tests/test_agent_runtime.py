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
    overview: dict | None = None,
    recommend: list | None = None,
    browse_state: dict | None = None,
    job_lookup: dict | None = None,
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

    async def fake_overview(**kw):
        return overview or {"total_open_jobs": 0, "categories": []}

    async def fake_recommend(*a, **kw):
        return recommend or []

    saved_calls: list = []
    applied_calls: list = []

    async def fake_browse_state(**kw):
        return browse_state

    async def fake_set_browse_state(**kw):
        return None

    async def fake_save_job(**kw):
        saved_calls.append(kw)

    async def fake_record_interest(**kw):
        applied_calls.append(kw)

    async def fake_job_lookup(**kw):
        return job_lookup

    rt.gateway.load = fake_load            # type: ignore[assignment]
    rt.gateway.persist = fake_persist      # type: ignore[assignment]
    rt.gateway.set_focus_product = fake_set_focus  # type: ignore[assignment]
    rt.gateway.should_summarize = lambda st: False  # type: ignore[assignment]
    rt.gateway.onboarding = fake_onboarding          # type: ignore[assignment]
    rt.gateway.onboarding_token = fake_onboarding_token  # type: ignore[assignment]
    rt.gateway.mark_onboarding_welcomed = fake_mark_welcomed  # type: ignore[assignment]
    rt.gateway.browse_state = fake_browse_state       # type: ignore[assignment]
    rt.gateway.set_browse_state = fake_set_browse_state  # type: ignore[assignment]
    rt.gateway.save_job = fake_save_job               # type: ignore[assignment]
    rt.gateway.record_interest = fake_record_interest  # type: ignore[assignment]
    rt._candidate_lookup = fake_lookup     # type: ignore[assignment]
    rt._category_browse = fake_browse      # type: ignore[assignment]
    rt._jobs_overview = fake_overview      # type: ignore[assignment]
    rt._recommend = fake_recommend         # type: ignore[assignment]
    rt._job_lookup = fake_job_lookup       # type: ignore[assignment]
    # expose action-call logs for assertions
    rt._test_saved = saved_calls           # type: ignore[attr-defined]
    rt._test_applied = applied_calls       # type: ignore[attr-defined]


async def _handle(rt: AgentRuntime, message: str = "i need a saree", *, interactive_id: str | None = None):
    return await rt.handle(
        request_id="r1", conversation_id="wa_91", customer_query=message,
        customer_external_id="91", channel="whatsapp", tenant_id="t",
        interactive_id=interactive_id,
    )


def test_graph_has_reasoning_pipeline():
    rt = _runtime()
    assert sorted(rt._graph.get_graph().nodes) == [
        "__end__", "__start__", "browse", "execute", "greeting_response", "humanize",
        "identify", "load_context", "menu", "onboarding_response", "persist", "planner",
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


async def test_form_just_submitted_shows_welcome_and_menu():
    """The first turn after the form is submitted gets a one-time 'welcome back'
    message WITH the quick-reply menu (0 LLM); later turns proceed normally."""
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Achuthan E"}, onboarded=True, welcomed=False)
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    assert "welcome back" in out["response"].lower()
    assert "achuthan" in out["response"].lower()
    assert "looking for" in out["response"].lower()
    # the 3 menu buttons ride along
    titles = [b["reply"]["title"]
              for b in out["whatsapp_interactive"]["interactive"]["action"]["buttons"]]
    assert titles == ["Job Search", "Application Status", "Recommended Jobs"]
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


async def test_greeting_offers_quick_reply_menu():
    """The 'hi' greeting carries the three quick-reply buttons (Job Search,
    Application Status, Recommended Jobs) as a WhatsApp interactive payload."""
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Achuthan E"})
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    cta = out["whatsapp_interactive"]["interactive"]
    assert cta["type"] == "button"
    titles = [b["reply"]["title"] for b in cta["action"]["buttons"]]
    assert titles == ["Job Search", "Application Status", "Recommended Jobs"]
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []   # 0 LLM


async def test_job_search_button_shows_category_list():
    """Tapping 'Job Search' (title comes back as text) returns a tappable list of
    job categories — deterministically, 0 LLM."""
    rt = _runtime()
    _stub_memory(rt, overview={
        "total_open_jobs": 69,
        "categories": [{"category": "Information Technology", "count": 18},
                       {"category": "Sales & Marketing", "count": 15},
                       {"category": "Administration", "count": 3}],
    })
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Job Search")
    interactive = out["whatsapp_interactive"]["interactive"]
    assert interactive["type"] == "list"
    rows = interactive["action"]["sections"][0]["rows"]
    assert [r["title"] for r in rows] == [
        "Information Technology", "Sales & Marketing", "Administration"]
    assert "69 open jobs" in out["response"]                # text fallback
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []   # 0 LLM
    assert len(out["delivery_plan"]) == 1                   # single bubble


async def test_recommended_jobs_button_lists_profile_matches():
    """Tapping 'Recommended Jobs' returns a TAPPABLE list of profile-matched roles
    (each row id = view:<ref>, so a tap shows that one job) — deterministically."""
    rt = _runtime()
    _stub_memory(
        rt, facts={"full_name": "Achuthan E"},
        recommend=[{"job_ref": "r1", "title": "QA Engineer", "location": "Chennai"},
                   {"job_ref": "r2", "title": "Tester", "location": "Remote"}],
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Recommended Jobs")
    interactive = out["whatsapp_interactive"]["interactive"]
    assert interactive["type"] == "list"
    ids = [r["id"] for r in interactive["action"]["sections"][0]["rows"]]
    assert ids == ["view:r1", "view:r2"]
    assert "Based on your profile" in out["response"]            # text fallback
    assert "Achuthan" in out["response"]
    assert "QA Engineer" in out["response"] and "Tester" in out["response"]
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []   # 0 LLM


async def test_tap_recommended_job_shows_single_card():
    """Tapping a recommended role shows THAT specific job's card (one card with
    Apply/Save/Share) — not the whole role family."""
    rt = _runtime()
    _stub_memory(rt, job_lookup={
        "job_ref": "r1", "title": "QA Engineer", "location": "Chennai",
        "salary_min": 25000, "salary_max": 40000, "employment_type": "full_time"})
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "QA Engineer", interactive_id="view:r1")
    cards = out["whatsapp_messages"]
    assert len(cards) == 1
    ids = [b["reply"]["id"] for b in cards[0]["interactive"]["action"]["buttons"]]
    assert ids == ["apply:r1", "save:r1", "share:r1"]
    assert "QA Engineer" in out["response"]
    assert "₹25,000–40,000/month" in out["response"]
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


async def test_application_status_button_falls_through_to_pipeline():
    """'Application Status' isn't a menu short-circuit — it flows to the normal
    planner path (which routes to get_application_status)."""
    rt = _runtime()
    _stub_memory(rt)
    rt.llm = _FakeLLM(
        plans=[{"goal": "status", "direct_answer": False,
                "steps": [{"tool": "get_application_status", "args": {}}]}],
        reply="(ignored — app-status is formatted deterministically)",
    )

    async def fake_dispatch(name, args, ctx):
        assert name == "get_application_status"
        return {"found": False}

    rt.tool_registry.dispatch = fake_dispatch  # type: ignore[assignment]
    out = await _handle(rt, "Application Status")
    assert "don't have any applications" in out["response"].lower()
    assert rt.llm.json_calls == ["agent_plan"]              # went through the planner


async def test_category_browse_shows_tappable_role_list():
    """A message that names a category returns a TAPPABLE role list (paged 10 at
    a time) — deterministically, no planner, no LLM. With 18 roles the first page
    shows 9 + a 'More roles' pager row."""
    rt = _runtime()
    jobs = [{"job_ref": f"r{i}", "title": f"IT Role {i}", "location": "Chennai"}
            for i in range(18)]
    _stub_memory(rt, browse={"category": "Information Technology", "jobs": jobs})
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "show the IT jobs")
    interactive = out["whatsapp_interactive"]["interactive"]
    assert interactive["type"] == "list"
    rows = interactive["action"]["sections"][0]["rows"]
    assert len(rows) == 10                       # 9 roles + "More roles"
    assert rows[0]["id"] == "job:r0"
    assert rows[-1]["id"] == "more:Information Technology:9"
    assert "18 Information Technology roles" in out["response"]    # text fallback
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []
    assert len(out["delivery_plan"]) == 1                          # single bubble


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


# ---- interactive job-browse flow (category → role → location → cards) -------

_FLOW_JOBS = [
    {"job_ref": "r1", "title": "Python Developer", "location": "Chennai",
     "salary_min": 1_000_000, "salary_max": 1_000_000, "employment_type": "full_time"},
    {"job_ref": "r2", "title": "Python Developer", "location": "Bengaluru",
     "salary_min": 1_200_000, "salary_max": 1_500_000, "employment_type": "full_time"},
    {"job_ref": "r3", "title": "QA Engineer", "location": "Remote (India)",
     "salary_min": 800_000, "salary_max": 800_000, "employment_type": "full_time"},
]


async def test_tap_category_lists_roles_as_buttons():
    """Tapping a category row (structured id) returns a tappable role list."""
    rt = _runtime()
    _stub_memory(rt, browse={"category": "Information Technology", "jobs": _FLOW_JOBS})
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Information Technology", interactive_id="category:Information Technology")
    interactive = out["whatsapp_interactive"]["interactive"]
    assert interactive["type"] == "list"
    ids = [r["id"] for r in interactive["action"]["sections"][0]["rows"]]
    assert ids == ["job:r1", "job:r2", "job:r3"]
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


async def test_tap_role_shows_job_cards_directly():
    """Tapping a role goes STRAIGHT to the detail cards (no location step) — all
    postings of that role, each with Apply/Save/Share."""
    rt = _runtime()
    _stub_memory(
        rt,
        browse={"category": "Information Technology", "jobs": _FLOW_JOBS},
        browse_state={"stage": "roles", "category": "Information Technology"},
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Python Developer", interactive_id="job:r1")
    assert out.get("whatsapp_interactive") is None               # no location list
    cards = out["whatsapp_messages"]
    assert len(cards) == 2                                        # both Python Developer postings
    ids = [b["reply"]["id"] for b in cards[0]["interactive"]["action"]["buttons"]]
    assert ids == ["apply:r1", "save:r1", "share:r1"]
    assert "Python Developer openings" in out["response"]
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


async def test_tap_role_shows_all_matching_openings():
    """A role with 3 open postings → 3 cards (the user's 'backend developer' case);
    other roles in the category are NOT mixed in."""
    jobs = [
        {"job_ref": "b1", "title": "Backend Developer", "location": "Chennai",
         "salary_min": 30000, "salary_max": 60000, "employment_type": "full_time"},
        {"job_ref": "b2", "title": "Senior Backend Developer", "location": "Bengaluru"},
        {"job_ref": "b3", "title": "Backend Developer", "location": "Remote (India)"},
        {"job_ref": "f1", "title": "Frontend Developer", "location": "Chennai"},
        {"job_ref": "q1", "title": "QA Engineer", "location": "Pune"},
    ]
    rt = _runtime()
    _stub_memory(
        rt,
        browse={"category": "Information Technology", "jobs": jobs},
        browse_state={"stage": "roles", "category": "Information Technology"},
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Backend Developer", interactive_id="job:b1")
    assert len(out["whatsapp_messages"]) == 3                     # 3 backend roles, not frontend/QA
    assert "3 Backend Developer openings" in out["response"]
    assert "₹30,000–60,000/month" in out["response"]             # monthly salary, not LPA


async def test_tap_apply_records_interest():
    """Apply records the candidate's interest in Redis and confirms — no LLM."""
    rt = _runtime()
    _stub_memory(rt, job_lookup={"job_ref": "r1", "title": "Python Developer",
                                 "location": "Chennai"})
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Apply", interactive_id="apply:r1")
    assert rt._test_applied and rt._test_applied[0]["ref"] == "r1"   # type: ignore[attr-defined]
    assert "Python Developer" in out["response"]
    assert "interest" in out["response"].lower()
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


async def test_tap_save_adds_to_saved_list():
    """Save adds the job to the saved list and confirms."""
    rt = _runtime()
    _stub_memory(rt, job_lookup={"job_ref": "r2", "title": "Python Developer",
                                 "location": "Bengaluru"})
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Save", interactive_id="save:r2")
    assert rt._test_saved and rt._test_saved[0]["ref"] == "r2"       # type: ignore[attr-defined]
    assert "saved" in out["response"].lower()
