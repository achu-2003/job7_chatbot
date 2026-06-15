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


_UNSET = object()


def _stub_memory(
    rt: AgentRuntime,
    *,
    facts: dict | None = None,
    candidate: object = _UNSET,
    onboarded: bool = True,
    welcomed: bool = True,
    browse: dict | None = None,
    overview: dict | None = None,
    recommend: list | None = None,
    candidate_skills: list | None = None,
    skill_jobs: list | None = None,
    browse_state: dict | None = None,
    job_lookup: dict | None = None,
    apply_state: dict | None = None,
    registration: dict | None = None,
    title_jobs: list | None = None,
    employer: dict | None = None,
    candidates: list | None = None,
    search_results: list | None = None,
    application_ids: dict | None = None,
) -> None:
    # Default to a fully-onboarded sender so the identity gate is a no-op and the
    # reasoning tests below exercise the normal path. Onboarding tests pass
    # facts={} / onboarded=False (and optionally a candidate) to drive the gate.
    if facts is None:
        facts = {"full_name": "Asha", "email": "asha@example.com"}

    # "Known" is now decided ONLY by the job-board DB lookup. So an unspecified
    # candidate defaults to a real DB record when ``onboarded`` (the established
    # user the reasoning tests assume), and to None when not (onboarding tests).
    # Pass candidate=... explicitly to override, or candidate=None to force a new
    # number even with onboarded=True.
    if candidate is _UNSET:
        candidate = (
            {"id": 1, "full_name": facts.get("full_name") or "Asha",
             "email": facts.get("email") or "known@example.com"}
            if onboarded else None
        )

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

    async def fake_candidate_skills(**kw):
        return candidate_skills or []

    async def fake_recommend_by_skills(**kw):
        return skill_jobs or []

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
    app_calls: list = []
    apply_state_saves: list = []

    async def fake_apply_state(**kw):
        return apply_state

    async def fake_set_apply_state(**kw):
        apply_state_saves.append(kw.get("state"))

    async def fake_clear_apply_state(**kw):
        return None

    async def fake_registration(**kw):
        return registration

    async def fake_update_reg_profile(**kw):
        return None

    async def fake_save_application(**kw):
        app_calls.append(kw)

    async def fake_apply_token(**kw):
        return "atok123"

    rt.gateway.apply_state = fake_apply_state               # type: ignore[assignment]
    rt.gateway.set_apply_state = fake_set_apply_state       # type: ignore[assignment]
    rt.gateway.clear_apply_state = fake_clear_apply_state   # type: ignore[assignment]
    rt.gateway.apply_token = fake_apply_token               # type: ignore[assignment]
    rt.gateway.registration = fake_registration            # type: ignore[assignment]
    rt.gateway.update_registration_profile = fake_update_reg_profile  # type: ignore[assignment]
    rt.gateway.save_application = fake_save_application     # type: ignore[assignment]
    fact_calls: list = []

    async def fake_remember_fact(**kw):
        fact_calls.append(kw)

    rt.gateway.semantic.remember_fact = fake_remember_fact  # type: ignore[attr-defined]
    rt._test_facts = fact_calls            # type: ignore[attr-defined]
    rt._candidate_lookup = fake_lookup     # type: ignore[assignment]
    rt._category_browse = fake_browse      # type: ignore[assignment]
    rt._jobs_overview = fake_overview      # type: ignore[assignment]
    rt._recommend = fake_recommend         # type: ignore[assignment]
    rt._candidate_skills = fake_candidate_skills        # type: ignore[assignment]
    rt._recommend_by_skills = fake_recommend_by_skills  # type: ignore[assignment]
    rt._job_lookup = fake_job_lookup       # type: ignore[assignment]

    async def fake_title_search(**kw):
        return title_jobs or []

    rt._title_search = fake_title_search   # type: ignore[assignment]

    # ---- employer (job-poster) lane stubs ----
    employer_state = {"rec": employer}     # mutable so a simulated payment sticks

    async def fake_employer(**kw):
        return employer_state["rec"]

    async def fake_employer_token(**kw):
        return "etok123"

    async def fake_update_employer(*, fields, **kw):
        rec = dict(employer_state["rec"] or {})
        rec.update(fields)
        employer_state["rec"] = rec
        return rec

    async def fake_list_candidates(**kw):
        return candidates or []

    async def fake_search_candidates(*, query, **kw):
        return search_results if search_results is not None else (candidates or [])

    # ---- live application write (private_job_applications) ----
    app_db_calls: list = []

    async def fake_application_ids(**kw):
        return application_ids

    async def fake_create_application(payload, **kw):
        app_db_calls.append(payload)
        return {"committed": True, "application_id": payload["application"]["id"], "inserted": True}

    rt.gateway.employer = fake_employer            # type: ignore[assignment]
    rt.gateway.employer_token = fake_employer_token  # type: ignore[assignment]
    rt.gateway.update_employer = fake_update_employer  # type: ignore[assignment]
    rt._list_candidates = fake_list_candidates     # type: ignore[assignment]
    rt._search_candidates = fake_search_candidates  # type: ignore[assignment]
    rt._application_ids = fake_application_ids      # type: ignore[assignment]
    rt._create_application = fake_create_application  # type: ignore[assignment]
    rt._test_app_db = app_db_calls                 # type: ignore[attr-defined]

    # expose action-call logs for assertions
    rt._test_saved = saved_calls           # type: ignore[attr-defined]
    rt._test_applied = applied_calls       # type: ignore[attr-defined]
    rt._test_applications = app_calls      # type: ignore[attr-defined]
    rt._test_apply_saves = apply_state_saves  # type: ignore[attr-defined]


async def _handle(
    rt: AgentRuntime, message: str = "i need a saree", *,
    interactive_id: str | None = None, attachment: dict | None = None,
):
    return await rt.handle(
        request_id="r1", conversation_id="wa_91", customer_query=message,
        customer_external_id="91", channel="whatsapp", tenant_id="t",
        interactive_id=interactive_id, attachment=attachment,
    )


def test_graph_has_reasoning_pipeline():
    rt = _runtime()
    assert sorted(rt._graph.get_graph().nodes) == [
        "__end__", "__start__", "browse", "creator_response", "execute",
        "greeting_response", "humanize", "identify", "load_context", "menu",
        "onboarding_response", "persist", "planner", "reflect", "responder",
        "role_select", "summarize",
    ]


async def test_greeting_offers_role_choice():
    """A 'hi' from a sender with NO chosen lane yet opens the Job Seeker / Job
    Creator choice (two reply buttons) — deterministically, 0 LLM."""
    rt = _runtime()
    _stub_memory(rt)                                  # default facts have no "lane"
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    cta = out["whatsapp_interactive"]["interactive"]
    assert cta["type"] == "button"
    titles = [b["reply"]["title"] for b in cta["action"]["buttons"]]
    assert titles == ["Job Seeker", "Employer"]
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []   # 0 LLM


async def test_lane_choice_is_remembered_on_tap():
    """Tapping 'Job Seeker' shows the seeker hub AND persists the lane so it's
    never asked again."""
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Asha"})    # no lane yet
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Job Seeker", interactive_id="role:seeker")
    titles = [b["reply"]["title"] for b in out["whatsapp_interactive"]["interactive"]["action"]["buttons"]]
    assert titles == ["Job Search", "Application Status", "Recommended Jobs"]
    # lane was written to customer_facts
    assert any(c.get("key") == "lane" and c.get("value") == "seeker"
               for c in rt._test_facts)              # type: ignore[attr-defined]


async def test_remembered_seeker_lane_skips_the_question():
    """A returning Job Seeker saying 'hi' goes straight to the seeker hub — the
    lane question is NOT shown again."""
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Asha", "lane": "seeker"})
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    titles = [b["reply"]["title"] for b in out["whatsapp_interactive"]["interactive"]["action"]["buttons"]]
    assert titles == ["Job Search", "Application Status", "Recommended Jobs"]
    assert "Job Seeker" not in titles                 # not re-asked
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


async def test_typing_switch_reopens_lane_choice():
    """Typing 'switch' re-opens the Job Seeker / Employer lane choice (the filter),
    overriding the remembered lane — 0 LLM."""
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Asha", "lane": "seeker"})
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "switch")
    cta = out["whatsapp_interactive"]["interactive"]
    assert cta["type"] == "button"
    titles = [b["reply"]["title"] for b in cta["action"]["buttons"]]
    assert titles == ["Job Seeker", "Employer"]
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


async def test_remembered_creator_lane_goes_to_creator_flow(monkeypatch):
    """A returning employer with NO staged profile is taken straight to the
    employer registration form — the lane question is not re-asked."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "public_base_url", "https://abc.ngrok-free.app")
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Asha", "lane": "creator"}, employer=None)
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    assert "register" in out["response"].lower()
    cta = out["whatsapp_interactive"]["interactive"]
    assert cta["type"] == "cta_url"
    assert "/employer/register?token=etok123" in cta["action"]["parameters"]["url"]
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


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


async def test_unknown_number_is_asked_lane_first():
    """A brand-new number's FIRST message opens the Job Seeker / Employer choice
    (lane is always asked before any onboarding) — with zero LLM calls."""
    rt = _runtime()
    _stub_memory(rt, facts={}, onboarded=False)   # nothing on file, no DB candidate
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "i need a job")
    titles = [b["reply"]["title"]
              for b in out["whatsapp_interactive"]["interactive"]["action"]["buttons"]]
    assert titles == ["Job Seeker", "Employer"]
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []   # 0 LLM


async def test_seeker_lane_hands_over_form_link(monkeypatch):
    """A new number on the Job Seeker lane (not yet submitted) gets the form link
    straight away — no in-chat name question — with zero LLM calls. With a non-
    https base URL the link is inline text, no cta button."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "public_base_url", "http://localhost:8000")
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Achuthan E", "lane": "seeker"}, onboarded=False)
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
    _stub_memory(rt, facts={"full_name": "Partha", "lane": "seeker"}, onboarded=False)
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


async def test_db_known_user_gets_lane_choice():
    """A number found in the job-board DB is known → greeted by name with the
    Job Seeker / Job Creator lane choice, with 0 LLM (no onboarding)."""
    rt = _runtime()
    _stub_memory(rt, candidate={"id": 7, "full_name": "Achuthan E", "email": "a@x.com"})
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    assert "achuthan" in out["response"].lower()    # past the gate → greeted by name
    titles = [b["reply"]["title"]
              for b in out["whatsapp_interactive"]["interactive"]["action"]["buttons"]]
    assert titles == ["Job Seeker", "Employer"]   # lane choice, not the name/form ask
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


async def test_seeker_not_in_db_is_re_onboarded():
    """DB is the SINGLE source of truth: a seeker with a staged Redis form but NOT
    found in the job-board DB is onboarded again (handed the form link) — a Redis
    form alone never counts as registered, so a failed/disabled DB write surfaces
    instead of being masked."""
    rt = _runtime()
    _stub_memory(
        rt, facts={"full_name": "Asha", "lane": "seeker"}, candidate=None, onboarded=True,
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Job Search")
    body = out["response"].lower()
    assert "/onboard/form" in body or "setting up your profile" in body   # re-onboarded
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


async def test_new_number_hi_offers_lane_choice():
    """A number NOT in our DB saying 'hi' is offered the Job Seeker / Employer
    choice first (lane is asked before onboarding)."""
    rt = _runtime()
    _stub_memory(rt, facts={}, onboarded=False)   # no DB candidate, no form
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    titles = [b["reply"]["title"]
              for b in out["whatsapp_interactive"]["interactive"]["action"]["buttons"]]
    assert titles == ["Job Seeker", "Employer"]
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


async def test_known_user_greeted_by_name():
    """A registered sender saying 'hi' is greeted by their first name (from the
    job board, seeded into facts by identify) on the lane-choice prompt — 0 LLM."""
    rt = _runtime()
    _stub_memory(
        rt, facts={},
        candidate={"id": 1, "full_name": "Reg User", "email": "reg@x.com"},
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    assert "reg" in out["response"].lower()              # greeted by name
    titles = [b["reply"]["title"]
              for b in out["whatsapp_interactive"]["interactive"]["action"]["buttons"]]
    assert titles == ["Job Seeker", "Employer"]
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


async def test_seeker_tap_offers_quick_reply_menu():
    """Tapping 'Job Seeker' (role:seeker) resumes the candidate flow: a known
    sender gets the three quick-reply buttons (Job Search, Application Status,
    Recommended Jobs) as a WhatsApp interactive payload — 0 LLM."""
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Achuthan E"})
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Job Seeker", interactive_id="role:seeker")
    cta = out["whatsapp_interactive"]["interactive"]
    assert cta["type"] == "button"
    titles = [b["reply"]["title"] for b in cta["action"]["buttons"]]
    assert titles == ["Job Search", "Application Status", "Recommended Jobs"]
    assert "looking for" in out["response"].lower()
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []   # 0 LLM


async def test_creator_tap_starts_employer_registration(monkeypatch):
    """Tapping 'Employer' (role:creator) with no staged profile hands over the
    company registration form — deterministically, 0 LLM."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "public_base_url", "https://abc.ngrok-free.app")
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Achuthan E"}, employer=None)
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Employer", interactive_id="role:creator")
    assert "register" in out["response"].lower()
    cta = out["whatsapp_interactive"]["interactive"]
    assert cta["type"] == "cta_url"
    assert "/employer/register?token=etok123" in cta["action"]["parameters"]["url"]
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []   # 0 LLM


# --- employer flow: Stages 2-5 -------------------------------------------

def _employer(*, kyc="VERIFIED", paid=False, jobs=None):
    return {
        "private_employers": {"companyName": "Acme Technologies", "kycStatus": kyc},
        "paid": paid,
        "jobs": jobs or [],
    }


async def test_registered_employer_unverified_gets_kyc_gate(monkeypatch):
    """A registered employer whose KYC isn't VERIFIED is sent to the KYC form."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "public_base_url", "https://abc.ngrok-free.app")
    rt = _runtime()
    _stub_memory(rt, facts={"full_name": "Asha", "lane": "creator"},
                 employer=_employer(kyc="NOT_SUBMITTED"))
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    assert "verify" in out["response"].lower()
    assert "/employer/kyc?token=etok123" in out["whatsapp_interactive"]["interactive"]["action"]["parameters"]["url"]


async def test_kyc_under_review_message():
    """KYC PENDING (submitted, awaiting) → an 'under review' note, no form."""
    rt = _runtime()
    _stub_memory(rt, facts={"lane": "creator"}, employer=_employer(kyc="PENDING"))
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    assert "under review" in out["response"].lower()
    assert out.get("whatsapp_interactive") is None


async def test_verified_employer_sees_menu():
    """A verified employer greeting gets the employer hub (3 buttons)."""
    rt = _runtime()
    _stub_memory(rt, facts={"lane": "creator"}, employer=_employer())
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "hi")
    titles = [b["reply"]["title"] for b in out["whatsapp_interactive"]["interactive"]["action"]["buttons"]]
    assert titles == ["Post a Job", "View Candidates", "My Jobs"]
    assert "acme" in out["response"].lower()


async def test_post_job_tap_hands_over_form(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "public_base_url", "https://abc.ngrok-free.app")
    rt = _runtime()
    _stub_memory(rt, facts={"lane": "creator"}, employer=_employer())
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Post a Job", interactive_id="emp:post")
    assert "/employer/post-job?token=etok123" in out["whatsapp_interactive"]["interactive"]["action"]["parameters"]["url"]


async def test_view_candidates_masked_until_paid():
    """Verified-but-unpaid employer sees name + experience only, plus an Unlock
    button — contact details are NOT in the payload. (Employer is NOT a seeker in
    the DB — onboarded=False — so this also guards the onboarding-form leak.)"""
    rt = _runtime()
    _stub_memory(
        rt, facts={"lane": "creator"}, onboarded=False, employer=_employer(paid=False),
        candidates=[{"full_name": "Rahul", "experience_level": "2-3 years",
                     "phone": "9990001111", "email": "rahul@x.com", "city": "Chennai"}],
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "View Candidates", interactive_id="emp:candidates")
    body = out["response"]
    assert "Rahul" in body and "2-3 years" in body
    assert "9990001111" not in body and "rahul@x.com" not in body   # masked
    titles = [b["reply"]["title"] for b in out["whatsapp_interactive"]["interactive"]["action"]["buttons"]]
    assert any("Unlock" in t for t in titles)


async def test_payment_unlocks_full_candidate_details():
    """Tapping Confirm Payment sets the entitlement and reveals contact details —
    a TEXT-only reply, so it must NOT leak the seeker onboarding form/button."""
    rt = _runtime()
    _stub_memory(
        rt, facts={"lane": "creator"}, onboarded=False, employer=_employer(paid=False),
        candidates=[{"full_name": "Rahul", "experience_level": "2-3 years",
                     "phone": "9990001111", "email": "rahul@x.com", "city": "Chennai"}],
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Confirm Payment", interactive_id="emp:pay")
    body = out["response"]
    assert "payment successful" in body.lower()
    assert "9990001111" in body and "rahul@x.com" in body   # full details now
    assert "form" not in body.lower()                        # NOT the seeker form
    # the leaked seeker-form cta button must be cleared on a text-only reply
    assert out.get("whatsapp_interactive") is None


async def test_employer_not_in_seeker_db_never_gets_onboarding_form():
    """Regression: a verified employer NOT in the seeker DB tapping View
    Candidates gets candidate details, never the seeker onboarding form/cta."""
    rt = _runtime()
    _stub_memory(
        rt, facts={"lane": "creator"}, onboarded=False, employer=_employer(paid=True),
        candidates=[{"full_name": "Rahul", "experience_level": "2-3 years",
                     "phone": "9990001111", "email": "rahul@x.com"}],
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "View Candidates", interactive_id="emp:candidates")
    assert "Rahul" in out["response"] and "9990001111" in out["response"]
    assert "/onboard/form" not in (out["response"] or "")
    assert out.get("whatsapp_interactive") is None           # no leaked Open-form cta


async def test_employer_text_search_by_skill_masked():
    """A verified-but-unpaid employer typing a skill/role gets matching candidates,
    masked (name + experience), routed through the skill/role search."""
    rt = _runtime()
    _stub_memory(
        rt, facts={"lane": "creator"}, onboarded=False, employer=_employer(paid=False),
        search_results=[{"full_name": "Vikram", "experience_level": "3-4 years",
                         "phone": "9991112222", "email": "vik@x.com",
                         "skills": ["Welding", "Fitting"]}],
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "welder")
    body = out["response"]
    assert "Vikram" in body and "welder" in body.lower()       # matched + query echoed
    assert "9991112222" not in body and "vik@x.com" not in body  # masked
    titles = [b["reply"]["title"] for b in out["whatsapp_interactive"]["interactive"]["action"]["buttons"]]
    assert any("Unlock" in t for t in titles)


async def test_employer_text_search_full_when_paid():
    """A PAID employer searching by skill sees full details (incl. skills)."""
    rt = _runtime()
    _stub_memory(
        rt, facts={"lane": "creator"}, onboarded=False, employer=_employer(paid=True),
        search_results=[{"full_name": "Vikram", "experience_level": "3-4 years",
                         "phone": "9991112222", "email": "vik@x.com",
                         "roles": ["Welder", "Fitter"], "skills": ["Welding", "Fitting"]}],
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "welding")
    body = out["response"]
    assert "Vikram" in body and "9991112222" in body
    assert "Welder" in body and "Welding" in body              # role + skills shown
    assert out.get("whatsapp_interactive") is None             # text-only, no leak


async def test_employer_search_no_match_nudges_to_menu():
    """A skill/role query with no matches nudges back to the menu, not an error."""
    rt = _runtime()
    _stub_memory(rt, facts={"lane": "creator"}, onboarded=False,
                 employer=_employer(paid=True), search_results=[])
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "astronaut")
    assert "no candidates found" in out["response"].lower()
    titles = [b["reply"]["title"] for b in out["whatsapp_interactive"]["interactive"]["action"]["buttons"]]
    assert titles == ["Post a Job", "View Candidates", "My Jobs"]


async def test_unverified_employer_cannot_view_candidates():
    """An emp:candidates tap before KYC verification falls back to the KYC gate —
    candidate data is never served."""
    rt = _runtime()
    _stub_memory(rt, facts={"lane": "creator"}, employer=_employer(kyc="NOT_SUBMITTED"),
                 candidates=[{"full_name": "Rahul", "experience_level": "2-3 years"}])
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "View Candidates", interactive_id="emp:candidates")
    assert "verify" in out["response"].lower()
    assert "Rahul" not in out["response"]


async def test_new_number_seeker_tap_hands_over_form(monkeypatch):
    """An unknown number that taps 'Job Seeker' is handed the onboarding form link
    straight away (the form collects the name) — 0 LLM."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "public_base_url", "https://abc.ngrok-free.app")
    rt = _runtime()
    _stub_memory(rt, facts={}, onboarded=False)
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Job Seeker", interactive_id="role:seeker")
    assert "form" in out["response"].lower()
    assert "/onboard/form?token=tok123" in out["whatsapp_interactive"]["interactive"]["action"]["parameters"]["url"]
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


async def test_recommended_jobs_prefers_skill_match():
    """'Recommended Jobs' first recommends jobs whose skills overlap the
    candidate's skills (skill match wins over the role/location fallback)."""
    rt = _runtime()
    _stub_memory(
        rt, facts={"full_name": "Achuthan E"},
        candidate_skills=["Python", "Django", "MySQL"],
        skill_jobs=[{"job_ref": "j1", "title": "Software Developer", "location": "Chennai"},
                    {"job_ref": "j2", "title": "Frontend Developer", "location": "Coimbatore"}],
        recommend=[{"job_ref": "x9", "title": "SHOULD NOT APPEAR", "location": "Nowhere"}],
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Recommended Jobs")
    interactive = out["whatsapp_interactive"]["interactive"]
    ids = [r["id"] for r in interactive["action"]["sections"][0]["rows"]]
    assert ids == ["view:j1", "view:j2"]                          # skill-matched jobs
    assert "Software Developer" in out["response"]
    assert "SHOULD NOT APPEAR" not in out["response"]             # role fallback NOT used
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []


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


async def test_typed_specific_role_shows_matching_job_cards():
    """Typing a specific role that's NOT a category ('Welder job') runs a
    deterministic title search and shows the matching job cards — no LLM."""
    rt = _runtime()
    welders = [
        {"job_ref": "w1", "title": "Welder", "location": "Chennai",
         "salary_min": 20000, "salary_max": 30000, "employment_type": "full_time"},
        {"job_ref": "w2", "title": "Senior Welder", "location": "Coimbatore"},
    ]
    _stub_memory(rt, browse=None, title_jobs=welders)   # not a category; title search hits
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Welder job")
    cards = out["whatsapp_messages"]
    assert len(cards) == 2
    assert cards[0]["interactive"]["action"]["buttons"][0]["reply"]["id"] == "apply:w1"
    assert "matching" in out["response"].lower() and "Welder" in out["response"]
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []   # 0 LLM


async def test_non_category_no_title_match_falls_through_to_agent():
    """When the message names no category AND no job title matches, browse returns
    nothing and the normal planner path runs."""
    rt = _runtime()
    _stub_memory(rt, browse=None, title_jobs=[])         # category miss + title miss
    rt.llm = _FakeLLM(plans=[{"goal": "greet", "direct_answer": True, "steps": []}],
                      reply="Found some roles!")
    out = await _handle(rt, "tell me a joke")
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


# ---- apply-time progressive top-up ------------------------------------------

_REG = {"private_job_seekers": {"id": "seeker1"}, "job_seeker_profiles": {"id": "profile1"}}
_JOB = {"id": "job-db-1", "job_ref": "r1", "title": "Backend Developer"}


async def test_apply_writes_application_to_live_db_with_resume(monkeypatch):
    """When live writes are on, finalizing an apply persists the application —
    incl. the uploaded resume — to private_job_applications, using the REAL live
    jobSeekerId/profileId (resolved by phone, not the staged-registration ids)."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "register_in_db", True)
    rt = _runtime()
    _stub_memory(
        rt, registration=_REG, job_lookup=_JOB,
        application_ids={"jobSeekerId": "live-seeker-9", "profileId": "live-profile-9"},
        apply_state={"job_ref": "r1", "job_title": "Backend Developer", "job": _JOB,
                     "pending": ["resume"], "answers": {}},
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "https://cv/me.pdf")              # resume link finalizes
    assert rt._test_app_db, "application not written to live DB"   # type: ignore[attr-defined]
    app = rt._test_app_db[0]["application"]                        # type: ignore[attr-defined]
    assert app["jobId"] == "job-db-1"
    assert app["jobSeekerId"] == "live-seeker-9" and app["profileId"] == "live-profile-9"
    assert app["resume"] == "https://cv/me.pdf" and app["status"] == "PENDING"
    assert "applied" in out["response"].lower()


async def test_apply_resume_rejects_non_url_text():
    """Typing non-link text on the resume step is re-asked, not accepted as a
    resume (validation) — the application is not finalized."""
    rt = _runtime()
    _stub_memory(
        rt, registration=_REG, job_lookup=_JOB,
        apply_state={"job_ref": "r1", "job_title": "Backend Developer", "job": _JOB,
                     "pending": ["resume"], "answers": {}},
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "i'll send it later")            # not a URL / doc / skip
    assert "link" in out["response"].lower()                 # re-asked
    assert rt._test_applications == []                       # not finalized


async def test_apply_no_live_write_when_flag_off(monkeypatch):
    """With register_in_db off, the apply stays Redis-only — no
    private_job_applications write."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "register_in_db", False)
    rt = _runtime()
    _stub_memory(
        rt, registration=_REG, job_lookup=_JOB,
        application_ids={"jobSeekerId": "live-seeker-9", "profileId": "live-profile-9"},
        apply_state={"job_ref": "r1", "job_title": "Backend Developer", "job": _JOB,
                     "pending": ["resume"], "answers": {}},
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    await _handle(rt, "https://cv/me.pdf")
    assert rt._test_app_db == []                              # type: ignore[attr-defined]


async def test_apply_skips_live_write_when_no_live_seeker(monkeypatch):
    """If the phone isn't found in the live job board, the application write is
    skipped (can't satisfy the FK) — the Redis copy still holds."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "register_in_db", True)
    rt = _runtime()
    _stub_memory(
        rt, registration=_REG, job_lookup=_JOB, application_ids=None,   # not in live DB
        apply_state={"job_ref": "r1", "job_title": "Backend Developer", "job": _JOB,
                     "pending": ["resume"], "answers": {}},
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "https://cv/me.pdf")
    assert rt._test_app_db == []                              # type: ignore[attr-defined]
    assert "applied" in out["response"].lower()              # apply still confirmed


async def test_apply_asks_for_missing_details_first():
    """Tapping Apply with a staged profile that lacks resume/salary starts the
    collection — asks for the resume and stages apply-state (not finalized)."""
    rt = _runtime()
    _stub_memory(rt, registration=_REG, job_lookup=_JOB)
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Apply", interactive_id="apply:r1")
    assert "resume" in out["response"].lower()
    saved = rt._test_apply_saves[0]                                  # type: ignore[attr-defined]
    assert saved["pending"][0] == "resume"
    assert rt._test_applications == []                              # not finalized yet


async def test_apply_answer_finalizes_application():
    """Answering the resume (a pasted link) finalizes: a DB-ready application
    record is staged (jobId + jobSeekerId resolved) and the candidate is confirmed."""
    rt = _runtime()
    _stub_memory(
        rt, registration=_REG, job_lookup=_JOB,
        apply_state={"job_ref": "r1", "job_title": "Backend Developer",
                     "pending": ["resume"], "answers": {}},
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "http://cv/me")                        # pasted link, no button
    assert rt._test_applications and rt._test_applications[0]["ref"] == "r1"  # type: ignore[attr-defined]
    rec = rt._test_applications[0]["record"]                       # type: ignore[attr-defined]
    assert rec["jobId"] == "job-db-1" and rec["jobSeekerId"] == "seeker1"
    assert rec["resume"] == "http://cv/me" and rec["status"] == "PENDING"
    assert "applied" in out["response"].lower()


async def test_apply_always_asks_resume_even_with_one_on_file():
    """Apply ALWAYS asks for the resume (skippable) — even when the profile
    already has one, the candidate is offered the Upload Resume step."""
    rt = _runtime()
    reg = {"private_job_seekers": {"id": "seeker1"},
           "job_seeker_profiles": {"id": "profile1", "resume": "http://cv"}}
    _stub_memory(rt, registration=reg, job_lookup=_JOB)
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Apply", interactive_id="apply:r1")
    assert "resume" in out["response"].lower()                     # resume IS asked
    assert rt._test_applications == []                             # not finalized yet
    saved = rt._test_apply_saves[0]                                # type: ignore[attr-defined]
    assert saved["pending"] == ["resume"]


async def test_apply_skip_uses_existing_resume_on_file():
    """Skipping the resume on apply falls back to the resume already on the
    profile, so the application still carries the candidate's CV."""
    rt = _runtime()
    reg = {"private_job_seekers": {"id": "seeker1"},
           "job_seeker_profiles": {"id": "profile1", "resume": "http://cv/on-file"}}
    _stub_memory(
        rt, registration=reg, job_lookup=_JOB,
        apply_state={"job_ref": "r1", "job_title": "Backend Developer",
                     "job": _JOB, "pending": ["resume"], "answers": {}},
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Skip", interactive_id="apply_skip")
    rec = rt._test_applications[0]["record"]                        # type: ignore[attr-defined]
    assert rec["resume"] == "http://cv/on-file" and "applied" in out["response"].lower()


async def test_apply_resume_offers_upload_and_skip_buttons(monkeypatch):
    """Tapping Apply sends TWO messages: a tappable 'Upload Resume' web button
    (cta_url) AND a separate 'Skip' reply button — not the old clip prompt."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "public_base_url", "https://abc.ngrok-free.app")
    rt = _runtime()
    _stub_memory(rt, registration=_REG, job_lookup=_JOB)
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Apply", interactive_id="apply:r1")
    msgs = out["whatsapp_messages"]
    assert len(msgs) == 2
    upload = msgs[0]["interactive"]
    assert upload["type"] == "cta_url"
    assert "Upload Resume" in upload["action"]["parameters"]["display_text"]
    assert upload["action"]["parameters"]["url"] == "https://abc.ngrok-free.app/onboard/resume?token=atok123"
    skip = msgs[1]["interactive"]
    assert skip["type"] == "button"
    assert skip["action"]["buttons"][0]["reply"]["id"] == "apply_skip"
    assert skip["action"]["buttons"][0]["reply"]["title"] == "Skip"
    assert "resume" in out["response"].lower() and "clip" not in out["response"].lower()
    saved = rt._test_apply_saves[0]                                 # type: ignore[attr-defined]
    assert saved["pending"] == ["resume"] and saved.get("job")


async def test_apply_resume_via_document_upload_finalizes():
    """Sending a resume DOCUMENT in chat (the fallback) is accepted and, since
    resume is the only apply field, finalizes the application."""
    rt = _runtime()
    _stub_memory(
        rt, registration=_REG, job_lookup=_JOB,
        apply_state={"job_ref": "r1", "job_title": "Backend Developer",
                     "job": _JOB, "pending": ["resume"], "answers": {}},
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(
        rt, "myresume.pdf",
        attachment={"kind": "document", "media_id": "MID123", "filename": "myresume.pdf"},
    )
    rec = rt._test_applications[0]["record"]                        # type: ignore[attr-defined]
    assert rec["resume"].startswith("myresume.pdf") and "MID123" in rec["resume"]
    assert "applied" in out["response"].lower()


async def test_apply_skip_finalizes_without_resume():
    """Tapping Skip on the resume step finalizes the application with no resume."""
    rt = _runtime()
    _stub_memory(
        rt, registration=_REG, job_lookup=_JOB,
        apply_state={"job_ref": "r1", "job_title": "Backend Developer",
                     "job": _JOB, "pending": ["resume"], "answers": {}},
    )
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(rt, "Skip", interactive_id="apply_skip")
    rec = rt._test_applications[0]["record"]                        # type: ignore[attr-defined]
    assert rec["resume"] is None and "applied" in out["response"].lower()


async def test_document_outside_apply_is_nudged():
    """A file sent when not applying → a friendly nudge, not a job search."""
    rt = _runtime()
    _stub_memory(rt)
    rt.llm = _FakeLLM(plans=[], reply="(should not be called)")
    out = await _handle(
        rt, "resume.pdf",
        attachment={"kind": "document", "media_id": "M", "filename": "resume.pdf"},
    )
    assert "apply" in out["response"].lower()
    assert rt.llm.json_calls == [] and rt.llm.chat_calls == []
