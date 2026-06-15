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
    assert rec["jobs"][-1]["private_jobs"]["status"] == "PENDING"
    assert rec["jobs"][-1]["private_jobs"]["validityDays"] == 15
    assert rec["lastActivated"]["ref"] == "JOB-XYZ"
    assert await mem.get_job_draft("tokF") is None        # draft consumed
