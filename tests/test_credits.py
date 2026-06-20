"""Job-posting credit model (app/credits.py) + the Activate Job screen render."""
import pytest

from app import credits as c
from app.api.routes.employer import _activate_job_html


@pytest.mark.parametrize("days,mult", [(15, 1), (30, 2), (45, 3), (99, 1)])
def test_validity_multiplier(days, mult):
    assert c.validity_multiplier(days) == mult


@pytest.mark.parametrize("districts,days,need", [
    (3, 15, 3), (3, 30, 6), (3, 45, 9),       # the 3-district example
    (1, 15, 1), (1, 30, 2), (1, 45, 3),       # single district
    (37, 15, 37), (37, 30, 74), (37, 45, 111),  # the 37-district example
])
def test_credits_required(districts, days, need):
    assert c.credits_required(districts, days) == need


def test_credits_required_floors_at_one_district():
    # zero/garbage districts still charge for at least one
    assert c.credits_required(0, 15) == 1
    assert c.credits_required(None, 30) == 2


def test_credit_quote_matches_screenshots():
    # Have 1, 3-district 15-day job → need 3, buy 2, pay ₹1298 (2 × 649)
    assert c.credit_quote(1, 3) == {"have": 1, "need": 3, "buy": 2, "pay": 1298, "sufficient": False}
    # Have 1, 1-district 15-day job → sufficient, nothing to buy
    assert c.credit_quote(1, 1) == {"have": 1, "need": 1, "buy": 0, "pay": 0, "sufficient": True}
    # Have 1, 37-district 15-day job → buy 36 × 649 = ₹23364
    q = c.credit_quote(1, 37)
    assert q["buy"] == 36 and q["pay"] == 23364 and not q["sufficient"]


def test_credit_quote_surplus_is_sufficient():
    assert c.credit_quote(40, 1)["sufficient"] is True
    assert c.credit_quote(40, 1)["buy"] == 0


class _FakeRedis:
    """Minimal async Redis stand-in (get/setex/set/delete) for memory tests."""
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.store[key] = value

    async def set(self, key, value, nx=False, ex=None):
        self.store[key] = value
        return True

    async def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)


async def _mem_with_employer():
    from app.chatbot.memory import ConversationMemory
    mem = ConversationMemory()
    mem._redis = _FakeRedis()   # type: ignore[assignment]
    await mem.save_employer("919042177457", {"private_employers": {"id": "emp1"}}, tenant_id="t1")
    return mem


async def test_wallet_seeds_once_then_debits():
    mem = await _mem_with_employer()
    # first read seeds from the live/welcome balance; later reads ignore the seed
    assert await mem.ensure_job_credits("919042177457", tenant_id="t1", seed=1) == 1
    assert await mem.ensure_job_credits("919042177457", tenant_id="t1", seed=99) == 1
    # debit 1 credit → 0; debit floors at 0 (never negative)
    assert await mem.adjust_job_credits("919042177457", -1, tenant_id="t1") == 0
    assert await mem.adjust_job_credits("919042177457", -5, tenant_id="t1") == 0
    # simulated purchase tops it up
    assert await mem.adjust_job_credits("919042177457", 3, tenant_id="t1") == 3


async def test_job_draft_stage_get_clear():
    mem = await _mem_with_employer()
    await mem.stage_job_draft("tokX", {"ref": "JOB-ABC", "private_jobs": {"title": "Dev"}})
    got = await mem.get_job_draft("tokX")
    assert got and got["ref"] == "JOB-ABC"
    await mem.clear_job_draft("tokX")
    assert await mem.get_job_draft("tokX") is None


def test_activate_page_renders_validity_and_credits():
    out = _activate_job_html("tok123", title="Fullstack Developer",
                             district_names=["Chennai", "Vellore", "Tirupattur"], have=1,
                             key_id="rzp_test_x", test_mode=True)
    assert 'action="/employer/post-job/activate"' in out
    assert 'name="token" value="tok123"' in out
    assert 'name="validity_days"' in out
    # validity pills 15/30/45 with multipliers
    for days in (15, 30, 45):
        assert f'data-days="{days}"' in out
    # district chips + count + starting balance baked in
    assert "Tirupattur" in out and "Districts (3)" in out
    assert "var HAVE = 1;" in out and f"var PRICE = {c.JOB_CREDIT_PRICE};" in out
    # the activate button + the recompute logic that flips it to "Pay … & Activate"
    assert ">Activate Now<" in out and "& Activate" in out


def test_activate_page_wires_razorpay_for_shortfall():
    out = _activate_job_html("tok", title="Job", district_names=["Chennai"], have=0,
                             key_id="rzp_test_x", test_mode=True)
    assert "checkout.razorpay.com/v1/checkout.js" in out
    assert "/employer/post-job/credits/order" in out
    assert "/employer/post-job/credits/verify" in out
    assert "new Razorpay(" in out
    # only a shortfall opens checkout; covered credits use the normal POST
    assert "if(need - HAVE <= 0) return;" in out


async def test_finalize_job_debits_and_posts(monkeypatch):
    import app.api.routes.employer as emp

    async def _noop(*a, **k):
        return None
    monkeypatch.setattr(emp.wa_delivery, "send_message", _noop)

    mem = await _mem_with_employer()
    await mem.ensure_job_credits("919042177457", tenant_id="t1", seed=2)
    await mem.stage_job_draft("tokF", {"ref": "JOB-XYZ",
        "private_jobs": {"title": "Dev", "preferredDistrictIds": ["d1", "d2"]}})
    # 2 districts × 15-day (1x) = 2 credits → wallet 2 → 0; job posted
    summary = await emp._finalize_job(mem, "919042177457", "t1", "tokF", "15")
    assert summary["need"] == 2 and summary["ref"] == "JOB-XYZ"
    rec = await mem.get_employer("919042177457", tenant_id="t1")
    assert rec["walletJobCredits"] == 0
    pj = rec["jobs"][-1]["private_jobs"]
    assert pj["status"] == "PENDING"
    # validity maps to expiresAt (a real private_jobs column), NOT validityDays
    assert "validityDays" not in pj and pj.get("expiresAt")
    assert rec["jobs"][-1]["validityDays"] == 15              # kept as Redis meta
    assert rec["lastActivated"]["ref"] == "JOB-XYZ"
    assert await mem.get_job_draft("tokF") is None        # draft consumed


async def test_finalize_job_writes_live_billing_when_flag_on(monkeypatch):
    """With JOB_POST_IN_DB on, activation calls the transactional live write
    (private_jobs + wallet debit + ledger) once, with the right amounts."""
    import app.api.routes.employer as emp
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "job_post_in_db", True)

    async def _noop(*a, **k):
        return None
    monkeypatch.setattr(emp.wa_delivery, "send_message", _noop)

    calls = []

    async def fake_billing(*, employer_id, job_payload, need, commit=True):
        calls.append({"employer_id": employer_id, "need": need})
        # the DB atomic debit returns the new authoritative balance
        return {"committed": True, "job_id": job_payload["id"], "credits": need,
                "balances": {"job": 0, "unlock": 0, "boost": 0}}
    monkeypatch.setattr(emp.CreditWalletRepository, "post_job_with_billing", staticmethod(fake_billing))

    mem = await _mem_with_employer()                       # employerId = "emp1"
    await mem.ensure_job_credits("919042177457", tenant_id="t1", seed=2)
    await mem.stage_job_draft("tokB", {"ref": "JOB-B1", "id": "j1",
        "private_jobs": {"id": "j1", "title": "Dev", "employerId": "emp1",
                         "preferredDistrictIds": ["d1", "d2"]}})
    await emp._finalize_job(mem, "919042177457", "t1", "tokB", "15")
    assert len(calls) == 1
    assert calls[0]["employer_id"] == "emp1" and calls[0]["need"] == 2   # atomic debit of 2


# --- subscription precedence: included free job posts ----------------------

async def test_plan_covers_post_logic(monkeypatch):
    """_plan_covers_post: covered only when an active sub has a free slot AND the
    job fits the plan's per-job location cap; off-flag / no-sub / over-cap → not."""
    import app.api.routes.employer as emp
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "subscriptions_in_db", True)

    async def mem_get_employer(phone, *, tenant_id=None):
        return {"private_employers": {"id": "emp1"}}
    mem = type("M", (), {"get_employer": staticmethod(mem_get_employer)})()

    ent = {"active": True, "max_active_jobs": 2, "max_locations_per_job": 2}

    async def fake_ent(eid):
        return ent
    monkeypatch.setattr(emp.SubscriptionRepository, "get_entitlement", staticmethod(fake_ent))

    async def fake_count(eid):
        return fake_count.n
    fake_count.n = 1
    monkeypatch.setattr(emp.JobPostRepository, "count_active_for_employer", staticmethod(fake_count))

    # active sub, 1/2 slots used, 2 districts ≤ 2 locations → covered
    assert await emp._plan_covers_post(mem, "9", "t", 2) is True
    # slots full (2/2) → not covered
    fake_count.n = 2
    assert await emp._plan_covers_post(mem, "9", "t", 1) is False
    fake_count.n = 1
    # job has more districts than the plan allows per job → not covered (pay credits)
    assert await emp._plan_covers_post(mem, "9", "t", 3) is False
    # inactive subscription → not covered
    ent2 = {"active": False, "max_active_jobs": 2, "max_locations_per_job": 2}
    monkeypatch.setattr(emp.SubscriptionRepository, "get_entitlement",
                        staticmethod(lambda eid: _aval(ent2)))
    assert await emp._plan_covers_post(mem, "9", "t", 1) is False
    # flag off → never covered (credit model)
    monkeypatch.setattr(get_settings(), "subscriptions_in_db", False)
    assert await emp._plan_covers_post(mem, "9", "t", 1) is False


async def _aval(v):
    return v


async def test_finalize_job_free_when_plan_covers(monkeypatch):
    """A plan-covered post charges 0 job credits (no wallet debit), still posts,
    and the summary/message flag it as covered."""
    import app.api.routes.employer as emp

    async def _noop(*a, **k):
        return None
    monkeypatch.setattr(emp.wa_delivery, "send_message", _noop)

    async def covered(*a, **k):
        return True
    monkeypatch.setattr(emp, "_plan_covers_post", covered)

    mem = await _mem_with_employer()
    await mem.ensure_job_credits("919042177457", tenant_id="t1", seed=5)   # has credits…
    await mem.stage_job_draft("tokP", {"ref": "JOB-PLN",
        "private_jobs": {"title": "Dev", "preferredDistrictIds": ["d1", "d2"]}})
    summary = await emp._finalize_job(mem, "919042177457", "t1", "tokP", "15")
    assert summary["need"] == 0 and summary["covered"] is True
    rec = await mem.get_employer("919042177457", tenant_id="t1")
    assert rec["walletJobCredits"] == 5                 # …but NONE were debited
    assert rec["jobs"][-1]["private_jobs"]["creditsUsed"] == 0
    assert rec["jobs"][-1]["coveredByPlan"] is True


async def test_job_quote_covered_needs_no_payment(monkeypatch):
    """When a plan covers the post, the quote is need=0 / sufficient / covered."""
    import app.api.routes.employer as emp

    async def covered(*a, **k):
        return True
    monkeypatch.setattr(emp, "_plan_covers_post", covered)
    mem = await _mem_with_employer()
    job = {"private_jobs": {"preferredDistrictIds": ["d1", "d2", "d3"]}}
    need, q = await emp._job_credit_quote(mem, "919042177457", "t1", job, "30")
    assert need == 0 and q["covered"] is True and q["sufficient"] is True


# --- lazy monthly renewal (every 30 days while active) ----------------------

class _RenewMem:
    def __init__(self):
        self.grants = []

    async def get_employer(self, phone, *, tenant_id=None):
        return {"private_employers": {"id": "emp1"}}

    async def grant_credits(self, phone, *, tenant_id, job=0, unlock=0, boost=0, description=""):
        self.grants.append({"unlock": unlock, "boost": boost, "desc": description})
        return {"job": 0, "unlock": unlock, "boost": boost}


async def test_apply_due_renewal_grants_when_due(monkeypatch):
    import app.api.routes.employer as emp
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "subscriptions_in_db", True)

    async def fake_claim(eid, *, cycle_days=30):
        return {"sub_id": "sub1", "monthly_credits": 200, "monthly_boosts": 5}
    monkeypatch.setattr(emp.SubscriptionRepository, "claim_due_renewal", staticmethod(fake_claim))
    recorded = []

    async def fake_record(**k):
        recorded.append(k); return {"recorded": True}
    monkeypatch.setattr(emp.SubscriptionRepository, "record_renewal_grant", staticmethod(fake_record))
    notes = []

    async def fake_notify(phone, body):
        notes.append(body)
    monkeypatch.setattr(emp, "_notify_subscription", fake_notify)

    mem = _RenewMem()
    await emp._apply_due_renewal(mem, "919", "t1")
    assert mem.grants == [{"unlock": 200, "boost": 5, "desc": "Plan monthly credits (renewal)"}]
    assert len(recorded) == 1 and recorded[0]["monthly_credits"] == 200
    # #5: the employer is notified of the monthly re-grant
    assert len(notes) == 1 and "renewed" in notes[0].lower() and "200 unlock" in notes[0]


async def test_apply_due_renewal_noop_when_not_due(monkeypatch):
    import app.api.routes.employer as emp
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "subscriptions_in_db", True)

    async def fake_claim(eid, *, cycle_days=30):
        return None
    monkeypatch.setattr(emp.SubscriptionRepository, "claim_due_renewal", staticmethod(fake_claim))
    mem = _RenewMem()
    await emp._apply_due_renewal(mem, "919", "t1")
    assert mem.grants == []          # nothing due → no grant


async def test_apply_due_renewal_gated_by_flag(monkeypatch):
    import app.api.routes.employer as emp
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "subscriptions_in_db", False)
    called = []

    async def boom(eid, *, cycle_days=30):
        called.append(eid); return None
    monkeypatch.setattr(emp.SubscriptionRepository, "claim_due_renewal", staticmethod(boom))
    await emp._apply_due_renewal(_RenewMem(), "919", "t1")
    assert called == []              # flag off → never touches the DB


# --- #4 plan status card + #5 expiry notify + #2 cancel ---------------------

from datetime import datetime, timedelta


async def test_plan_status_shapes_active_entitlement(monkeypatch):
    """_plan_status turns an active entitlement into the 'Your Plan' card data
    (name, days left, slots used/total, daily cap); inactive → None."""
    import app.api.routes.employer as emp

    async def fake_count(eid):
        return 1
    monkeypatch.setattr(emp.JobPostRepository, "count_active_for_employer", staticmethod(fake_count))
    ent = {"active": True, "plan_name": "Growth", "max_active_jobs": 2,
           "daily_apply_cap": None, "end_date": datetime.utcnow() + timedelta(days=12)}
    card = await emp._plan_status(ent, "emp1")
    assert card["name"] == "Growth" and card["slots_total"] == 2 and card["slots_used"] == 1
    assert 11 <= card["days_left"] <= 12
    # inactive / no employer → None
    assert await emp._plan_status({"active": False}, "emp1") is None
    assert await emp._plan_status({"active": True}, None) is None


async def test_resolve_entitlement_notifies_once_on_expiry(monkeypatch):
    """When get_entitlement reports a just-expired plan, the employer gets a
    one-time expiry notification."""
    import app.api.routes.employer as emp

    class _M:
        async def get_employer(self, phone, *, tenant_id=None):
            return {"private_employers": {"id": "emp1"}}

    async def expired(eid):
        return {"active": False, "plan_name": "Pro", "just_expired": True}
    monkeypatch.setattr(emp.SubscriptionRepository, "get_entitlement", staticmethod(expired))
    notes = []

    async def fake_notify(phone, body):
        notes.append(body)
    monkeypatch.setattr(emp, "_notify_subscription", fake_notify)
    ent, eid = await emp._resolve_entitlement(_M(), "919", "t1")
    assert eid == "emp1" and ent["active"] is False
    assert len(notes) == 1 and "expired" in notes[0].lower() and "Pro" in notes[0]


async def test_subscribe_cancel_confirm_endpoint(monkeypatch):
    """POST /subscribe/cancel/confirm cancels the active sub, clears the Redis
    cache, and notifies."""
    import app.api.routes.employer as emp

    cancelled = []

    async def fake_cancel(eid, *, commit=True):
        cancelled.append(eid); return {"cancelled": True, "plan_name": "Growth"}
    monkeypatch.setattr(emp.SubscriptionRepository, "cancel", staticmethod(fake_cancel))
    notes, cache = [], []

    async def fake_notify(phone, body):
        notes.append(body)
    monkeypatch.setattr(emp, "_notify_subscription", fake_notify)

    class _M:
        async def get_employer_identity(self, token):
            return {"customer_id": "919876543210", "tenant_id": "t1"}

        async def get_employer(self, phone, *, tenant_id=None):
            return {"private_employers": {"id": "emp1"}}

        async def update_employer(self, phone, fields, *, tenant_id=None):
            cache.append(fields); return None

    import types
    from urllib.parse import urlencode
    body = urlencode({"token": "tok"}).encode()
    app = types.SimpleNamespace(state=types.SimpleNamespace(memory=_M()))
    req = types.SimpleNamespace(app=app, body=lambda: _aret2(body))
    out = await emp.subscribe_cancel_do(req)
    assert cancelled == ["emp1"]
    assert cache == [{"subscription": None}]                       # Redis cache cleared
    assert len(notes) == 1 and "cancelled" in notes[0].lower()
    assert "cancelled" in out.body.decode().lower()


async def _aret2(v):
    return v
