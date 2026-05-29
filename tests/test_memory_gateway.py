"""MemoryGateway — pure decisions + load/persist fan-out (mocked stores)."""
import datetime as dt

from app.memory import repositories as repo
from app.memory.gateway import MemoryGateway


class _FakeVector:
    async def query(self, *a, **k):
        return []

    async def upsert(self, *a, **k):
        pass


class _FakeShortTerm:
    def __init__(self, turns=None):
        self._turns = turns or []
        self.appended: list[tuple] = []

    async def history(self, conv, *, tenant_id=None):
        return self._turns

    async def append(self, conv, role, content, *, tenant_id=None):
        self.appended.append((role, content))

    async def get_last_product(self, conv, *, tenant_id=None):
        return None

    async def clear(self, conv, *, tenant_id=None):
        self.cleared = getattr(self, "cleared", [])
        self.cleared.append(conv)


def _gw(turns=None):
    return MemoryGateway(vector=_FakeVector(), short_term=_FakeShortTerm(turns))


# --- pure decisions ---------------------------------------------------------


def test_is_resumed_by_gap():
    now = dt.datetime(2026, 5, 22, 12, 0, tzinfo=dt.timezone.utc)
    assert MemoryGateway.is_resumed(None, now=now) is False
    assert MemoryGateway.is_resumed(now - dt.timedelta(hours=5), now=now) is True
    assert MemoryGateway.is_resumed(now - dt.timedelta(minutes=2), now=now) is False


def test_should_summarize_threshold():
    gw = _gw()
    assert gw.should_summarize([{"x": 1}] * 12) is True
    assert gw.should_summarize([{"x": 1}] * 3) is False
    assert gw.should_summarize([]) is False


# --- load -------------------------------------------------------------------


def _patch_repos(monkeypatch, *, get_ret, facts=None):
    async def fake_get(*, tenant_id, conversation_id):
        return get_ret

    async def fake_goc(*, tenant_id, customer_id, conversation_id):
        return {"id": "sess-new", "rolling_summary": "", "last_active_at": None}

    async def fake_touch(sid):
        return None

    async def fake_pending(sid):
        return []

    async def fake_goals(sid):
        return []

    async def fake_facts(tid, cid):
        return facts or {}

    monkeypatch.setattr(repo.AgentSessionRepository, "get", staticmethod(fake_get))
    monkeypatch.setattr(repo.AgentSessionRepository, "get_or_create", staticmethod(fake_goc))
    monkeypatch.setattr(repo.AgentSessionRepository, "touch", staticmethod(fake_touch))
    monkeypatch.setattr(repo.PendingActionRepository, "list_pending", staticmethod(fake_pending))
    monkeypatch.setattr(repo.GoalRepository, "list_active", staticmethod(fake_goals))
    monkeypatch.setattr(repo.CustomerFactRepository, "all_for", staticmethod(fake_facts))


async def test_load_new_session(monkeypatch):
    _patch_repos(monkeypatch, get_ret=None, facts={"name": "Asha"})
    gw = _gw(turns=[{"role": "user", "content": "hi"}])
    snap = await gw.load(
        tenant_id="t", customer_id="91", conversation_id="wa_91", query="red saree"
    )
    assert snap["session_id"] == "sess-new"
    assert snap["session_status"] == "active"
    assert snap["customer_facts"] == {"name": "Asha"}
    assert snap["short_term"] == [{"role": "user", "content": "hi"}]


async def test_load_resumed_after_gap(monkeypatch):
    old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=6)
    existing = {"id": "sess-old", "rolling_summary": "was browsing sarees", "last_active_at": old}
    _patch_repos(monkeypatch, get_ret=existing)
    snap = await _gw().load(
        tenant_id="t", customer_id="91", conversation_id="wa_91", query="hi"
    )
    assert snap["session_id"] == "sess-old"
    assert snap["session_status"] == "resumed"
    assert snap["rolling_summary"] == "was browsing sarees"


# --- persist ----------------------------------------------------------------


async def test_reset_session_clears_and_closes(monkeypatch):
    closed: list = []

    async def fake_close(*, tenant_id, conversation_id):
        closed.append((tenant_id, conversation_id))

    monkeypatch.setattr(repo.AgentSessionRepository, "close", staticmethod(fake_close))
    gw = _gw()
    await gw.reset_session(tenant_id="t", conversation_id="wa_91")
    assert gw.short_term.cleared == ["wa_91"]      # short-term + pin cleared
    assert closed == [("t", "wa_91")]              # session closed + summary wiped


async def test_persist_writes_all_tiers(monkeypatch):
    episodes: list[dict] = []

    async def fake_add(**kw):
        episodes.append(kw)

    monkeypatch.setattr(repo.EpisodeRepository, "add", staticmethod(fake_add))
    gw = _gw()
    await gw.persist(
        tenant_id="t", customer_id="91", conversation_id="wa_91", session_id="s",
        user_text="any red sarees?", assistant_text="Yes! Here are a few 😊",
    )
    assert gw.short_term.appended == [
        ("user", "any red sarees?"),
        ("assistant", "Yes! Here are a few 😊"),
    ]
    assert [e["role"] for e in episodes] == ["user", "assistant"]
