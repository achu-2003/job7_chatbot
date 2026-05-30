"""persist node remembers conversationally-supplied name/email so the next
turn has them in customer_facts (regression for the WhatsApp re-ask loop)."""
from app.agent.nodes.persist import persist


class _FakeSemantic:
    def __init__(self):
        self.facts: dict[str, str] = {}

    async def remember_fact(self, *, tenant_id, customer_id, key, value, confidence, source):
        self.facts[key] = value


class _FakeGateway:
    def __init__(self):
        self.semantic = _FakeSemantic()

    async def persist(self, **kw):
        return None

    async def set_focus_product(self, **kw):
        return None

    async def reset_session(self, **kw):
        return None


def _state(text, *, short_term=None, facts=None):
    return {
        "tenant_id": "t", "customer_id": "91", "conversation_id": "wa_91",
        "session_id": "s1", "inbound_text": text, "draft_response": "ok",
        "working": {}, "short_term": short_term or [], "customer_facts": facts or {},
    }


async def test_email_is_remembered():
    gw = _FakeGateway()
    await persist(_state("sandhanapandiyanmurugan@outlook.com"), gateway=gw)
    assert gw.semantic.facts["email"] == "sandhanapandiyanmurugan@outlook.com"


async def test_name_remembered_after_name_prompt():
    gw = _FakeGateway()
    state = _state(
        "Sandhanapandiyanmurugan",
        short_term=[{"role": "assistant", "content": "Need your full name to proceed."}],
    )
    await persist(state, gateway=gw)
    assert gw.semantic.facts["full_name"] == "Sandhanapandiyanmurugan"


async def test_non_identity_message_writes_nothing():
    # An intent phrase isn't a name and has no email → no fact churn, even with
    # a name already on file.
    gw = _FakeGateway()
    state = _state(
        "I need a python developer job",
        short_term=[{"role": "assistant", "content": "what's your full name?"}],
        facts={"full_name": "Already Known"},
    )
    await persist(state, gateway=gw)
    assert gw.semantic.facts == {}


async def test_new_email_overwrites_old():
    # Regression for the verify-loop: candidate gives a different email → it
    # replaces the stored one instead of being ignored.
    gw = _FakeGateway()
    state = _state(
        "eachudhan2003@gmail.com",
        facts={"email": "sandhanapandiyanmurugan@outlook.com"},
    )
    await persist(state, gateway=gw)
    assert gw.semantic.facts["email"] == "eachudhan2003@gmail.com"


async def test_same_email_not_rewritten():
    # No change → no write (avoids needless fact churn / updated_at bumps).
    gw = _FakeGateway()
    state = _state(
        "eachudhan2003@gmail.com",
        facts={"email": "eachudhan2003@gmail.com"},
    )
    await persist(state, gateway=gw)
    assert gw.semantic.facts == {}


async def test_new_name_overwrites_old_after_prompt():
    gw = _FakeGateway()
    state = _state(
        "Achuthan E",
        short_term=[{"role": "assistant", "content": "Need your full name again?"}],
        facts={"full_name": "Sandhanapandiyanmurugan"},
    )
    await persist(state, gateway=gw)
    assert gw.semantic.facts["full_name"] == "Achuthan E"


async def test_intent_phrase_never_stored_as_name():
    gw = _FakeGateway()
    state = _state(
        "I need a python developer job",
        short_term=[{"role": "assistant", "content": "what's your full name?"}],
    )
    await persist(state, gateway=gw)
    assert "full_name" not in gw.semantic.facts
