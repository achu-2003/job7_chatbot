"""Responder: length guard + hallucination enforcement (no fake confirmations)."""
from app.agent.nodes.responder import _SAFE_FALLBACK, _shorten, respond
from app.chatbot.validator import HallucinationValidator


class _CannedLLM:
    def __init__(self, text):
        self._text = text

    async def chat(self, **kwargs):
        return self._text, {}


async def test_unsupported_price_is_replaced_not_sent():
    # fabricated confirmation with a price not in any tool result or pinned product
    state = {"inbound_text": "yes", "working": {}, "short_term": [], "cached_product": {}}
    out = await respond(
        state, llm=_CannedLLM("Your order is confirmed for ₹9999."),
        validator=HallucinationValidator(), memory_context=None,
    )
    assert out["draft_response"] == _SAFE_FALLBACK


async def test_price_grounded_in_pinned_product_passes():
    state = {
        "inbound_text": "how much", "working": {}, "short_term": [],
        "cached_product": {"product_id": "p1", "doc": "Saree — ₹999.99 — Sarees"},
    }
    out = await respond(
        state, llm=_CannedLLM("It's ₹999.99."),
        validator=HallucinationValidator(), memory_context=None,
    )
    assert "999" in out["draft_response"]      # cached price grounds it → not replaced


async def test_application_status_is_reported_not_asked_for_identity():
    # Regression: "My application" lookups returned data but the weak model
    # sometimes asked for name/email instead. The responder now reports status
    # deterministically (no LLM), so identity is never wrongly requested.
    state = {
        "inbound_text": "My application",
        "working": {"tool_results": [{
            "tool": "get_application_status",
            "result": {"found": True, "applications": [
                {"job_title": "Kt developer", "status": "REJECTED"},
            ]},
        }]},
        "short_term": [], "cached_product": {},
    }
    # LLM would have asked for email — but the deterministic guard wins.
    out = await respond(
        state, llm=_CannedLLM("Need a bit more — what's your email?"),
        validator=HallucinationValidator(), memory_context=None,
    )
    assert "Kt developer" in out["draft_response"]
    assert "Rejected" in out["draft_response"]
    assert "email" not in out["draft_response"].lower()


async def test_apply_confirmation_hands_over_jobs7_app_link():
    # When submit_application returns the Jobs7 app link, the responder formats
    # it deterministically (no LLM, never asks for email) and attaches a tappable
    # cta_url button — with the link inline as the fallback text.
    state = {
        "inbound_text": "yes",
        "working": {"tool_results": [{
            "tool": "submit_application",
            "result": {
                "apply_via_app": True, "job_title": "Tester",
                "app_url": "https://play.google.com/store/apps/details?id=com.jobs7",
            },
        }]},
        "short_term": [], "cached_product": {},
        "customer_facts": {"full_name": "Achuthan E"},
    }
    out = await respond(
        state, llm=_CannedLLM("Need a bit more — what's your email?"),
        validator=HallucinationValidator(), memory_context=None,
    )
    assert out["used_llm"] is False
    assert out["single_bubble"] is True
    assert "Tester" in out["draft_response"]
    assert "Achuthan" in out["draft_response"]            # addressed by first name
    assert "id=com.jobs7" in out["draft_response"]        # link inline (fallback)
    assert "email" not in out["draft_response"].lower()   # never asks for email
    cta = out["whatsapp_interactive"]["interactive"]
    assert cta["type"] == "cta_url"
    assert cta["action"]["parameters"]["display_text"] == "Open in Jobs7"
    assert cta["action"]["parameters"]["url"].endswith("id=com.jobs7")


async def test_apply_link_http_has_no_cta_button():
    # A non-https app URL can't be a Meta cta button — link stays inline only.
    state = {
        "inbound_text": "apply",
        "working": {"tool_results": [{
            "tool": "submit_application",
            "result": {"apply_via_app": True, "job_title": "QA Engineer",
                       "app_url": "http://localhost/app"},
        }]},
        "short_term": [], "cached_product": {}, "customer_facts": {},
    }
    out = await respond(
        state, llm=_CannedLLM("x"),
        validator=HallucinationValidator(), memory_context=None,
    )
    assert out.get("whatsapp_interactive") is None
    assert "http://localhost/app" in out["draft_response"]


async def test_no_applications_gives_friendly_line():
    state = {
        "inbound_text": "my applications",
        "working": {"tool_results": [{
            "tool": "get_application_status", "result": {"found": False},
        }]},
        "short_term": [], "cached_product": {},
    }
    out = await respond(
        state, llm=_CannedLLM("whatever"),
        validator=HallucinationValidator(), memory_context=None,
    )
    assert "don't have any applications" in out["draft_response"].lower()


async def test_job_refs_grounded_by_search_jobs_pass():
    # Regression: search_jobs results must feed the grounding check. Without the
    # search_jobs branch in _grounding_rows, every JOB-XXXX the model quotes is
    # flagged unsupported and the reply is wrongly replaced with the fallback.
    state = {
        "inbound_text": "Can u show some python developer job",
        "working": {"tool_results": [{
            "tool": "search_jobs",
            "result": [
                {"job_ref": "JOB-AB1001", "title": "Python Developer", "location": "Bengaluru"},
                {"job_ref": "JOB-AB1002", "title": "Python Developer", "location": "Hyderabad"},
                {"job_ref": "JOB-AB1003", "title": "Python Developer", "location": "Pune"},
            ],
        }]},
        "short_term": [], "cached_product": {},
    }
    draft = (
        "• Python Developer — Bengaluru (JOB-AB1001)\n"
        "• Python Developer — Hyderabad (JOB-AB1002)\n"
        "• Python Developer — Pune (JOB-AB1003)"
    )
    out = await respond(
        state, llm=_CannedLLM(draft),
        validator=HallucinationValidator(), memory_context=None,
    )
    assert out["draft_response"] != _SAFE_FALLBACK
    assert "JOB-AB1001" in out["draft_response"]


async def test_category_browse_lists_every_job_deterministically():
    # A sizable result set (a category browse) is listed in full, bypassing the
    # LLM, and flagged for single-bubble delivery so nothing gets truncated.
    jobs = [
        {"job_ref": f"role-{i}", "title": f"Sales Exec {i}",
         "location": "Chennai", "department": "Sales & Marketing"}
        for i in range(15)
    ]
    state = {
        "inbound_text": "Sales & Marketing",
        "working": {"tool_results": [{"tool": "search_jobs", "result": jobs}]},
        "short_term": [], "cached_product": {},
    }
    out = await respond(
        state, llm=_CannedLLM("(should not be called)"),
        validator=HallucinationValidator(), memory_context=None,
    )
    assert out["used_llm"] is False
    assert out["single_bubble"] is True
    assert "all 15 Sales & Marketing roles" in out["draft_response"]
    for i in range(15):                                   # every job present, not truncated
        assert f"Sales Exec {i}" in out["draft_response"]


async def test_small_search_keeps_llm_reply():
    # <= 5 hits is an ordinary search → the natural LLM reply, NOT the list format.
    jobs = [{"job_ref": "JOB-1", "title": "Python Dev", "location": "Pune"}]
    state = {
        "inbound_text": "python jobs",
        "working": {"tool_results": [{"tool": "search_jobs", "result": jobs}]},
        "short_term": [], "cached_product": {},
    }
    out = await respond(
        state, llm=_CannedLLM("Found a Python Dev role in Pune (JOB-1)!"),
        validator=HallucinationValidator(), memory_context=None,
    )
    assert out["used_llm"] is True
    assert not out.get("single_bubble")


def test_short_reply_unchanged():
    s = "Yes! Banarasi Silk Saree — ₹2,499, 7 in stock 🌹 Want colours?"
    assert _shorten(s) == s


def test_product_bullets_pass_through():
    s = ("Here are 3 🌹\n• Saree — ₹999 (5 left)\n"
         "• Kurti — ₹699 (2 left)\n• Top — ₹499 (8 left)")
    assert _shorten(s) == s.strip()        # well under the cap


def test_paragraph_is_trimmed_at_a_sentence_boundary():
    rambling = "Thank you so much for reaching out to us today. " * 12  # >300 chars
    out = _shorten(rambling, limit=120)
    assert len(out) <= 121
    assert out.endswith(".") or out.endswith("…")
