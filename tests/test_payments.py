"""Razorpay subscription payment — signature verify, config, page, activation."""
import hashlib
import hmac

from app.config import get_settings
from app.payments import razorpay as rzp
from app.api.routes.employer import _subscribe_html


_PLANS = [
    {"id": "p_free", "type": "FREE", "name": "Free", "price": 0.0, "billingCycle": "DAYS_30",
     "maxActiveJobs": 1, "monthlyCredits": 0, "monthlyBoosts": 0},
    {"id": "p_growth", "type": "GROWTH", "name": "Growth", "price": 1999.0, "billingCycle": "DAYS_30",
     "maxActiveJobs": 2, "monthlyCredits": 60, "monthlyBoosts": 2},
]


def test_test_mode_detection():
    assert get_settings().razorpay_key_id.startswith("rzp_test_")  # the configured key
    assert get_settings().razorpay_test_mode is True
    assert get_settings().razorpay_enabled is True


def test_verify_signature_good_and_bad(monkeypatch):
    monkeypatch.setattr(get_settings(), "razorpay_key_id", "rzp_test_x")
    monkeypatch.setattr(get_settings(), "razorpay_key_secret", "shhh-secret")
    sig = hmac.new(b"shhh-secret", b"order_1|pay_1", hashlib.sha256).hexdigest()
    assert rzp.verify_payment_signature(order_id="order_1", payment_id="pay_1", signature=sig)
    assert not rzp.verify_payment_signature(order_id="order_1", payment_id="pay_1", signature="nope")
    # any blank input fails closed
    assert not rzp.verify_payment_signature(order_id="", payment_id="pay_1", signature=sig)
    assert not rzp.verify_payment_signature(order_id="order_1", payment_id="pay_1", signature="")


def test_subscribe_page_renders_plans_and_checkout():
    out = _subscribe_html("tok123", _PLANS, key_id="rzp_test_x", test_mode=True,
                          prefill_name="Asha", prefill_phone="919876543210", current_type="FREE")
    assert "checkout.razorpay.com/v1/checkout.js" in out
    assert "TEST MODE" in out
    assert "/employer/subscribe/order" in out and "/employer/subscribe/verify" in out
    assert "Growth" in out and 'data-price="1999.0"' in out
    assert "new Razorpay(" in out
    # the Free plan path (no payment) is handled
    assert "Activate Free Plan" in out or "o.free" in out


def test_individual_credit_packs_match_screenshots():
    from app import credits as c
    # Job credits: ₹649/credit; 10 for ₹5192
    assert c.find_pack("JOB", 1) == {"type": "JOB", "credits": 1, "price": 649}
    assert c.find_pack("JOB", 10)["price"] == 5192
    # Unlock: ₹49, 10→₹392, 100→₹3920
    assert c.find_pack("UNLOCK", 1)["price"] == 49
    assert c.find_pack("UNLOCK", 100)["price"] == 3920
    # Boost: ₹999, 10→₹792
    assert c.find_pack("BOOST", 1)["price"] == 999
    assert c.find_pack("BOOST", 10)["price"] == 792
    # unknown combos rejected (can't be tampered into a free grant)
    assert c.find_pack("JOB", 7) is None and c.find_pack("XXX", 1) is None


def test_buy_credits_page_renders_bundles_and_individual():
    from app.api.routes.employer import _buy_credits_html
    bundles = [{"id": "b1", "bundleType": "STARTER", "name": "Starter", "price": 999.0,
                "jobCredits": 1, "unlockCredits": 20, "boostCredits": 1}]
    out = _buy_credits_html("tok", bundles=bundles, balance={"job": 1, "unlock": 0, "boost": 0},
                            key_id="rzp_test_x", test_mode=True, prefill_name="A", prefill_phone="9000000000")
    assert "Current Balance" in out and "TEST MODE" in out
    assert ">Bundles<" in out and ">Individual Credits<" in out
    assert 'data-item="bundle:b1"' in out and "Starter" in out
    assert 'data-item="pack:JOB:1"' in out and 'data-item="pack:UNLOCK:100"' in out
    assert "/employer/buy-credits/order" in out and "/employer/buy-credits/verify" in out
    assert "new Razorpay(" in out


async def test_resolve_buy_item_and_grant():
    import app.api.routes.employer as emp
    # an individual pack resolves to its server price + single-bucket grant
    r = await emp._resolve_buy_item("pack:UNLOCK:10")
    assert r["price"] == 392.0 and r["grant"] == {"job": 0, "unlock": 10, "boost": 0}
    r2 = await emp._resolve_buy_item("pack:BOOST:1")
    assert r2["price"] == 999.0 and r2["grant"]["boost"] == 1
    # garbage item → None (rejected)
    assert await emp._resolve_buy_item("pack:JOB:3") is None
    assert await emp._resolve_buy_item("nonsense") is None


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.store[key] = value

    async def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)


async def _mem():
    from app.chatbot.memory import ConversationMemory
    m = ConversationMemory()
    m._redis = _FakeRedis()   # type: ignore[assignment]
    await m.save_employer("919876543210", {"private_employers": {"id": "e1"}}, tenant_id="t1")
    return m


async def test_payment_order_stage_get_clear():
    m = await _mem()
    await m.stage_payment_order("order_abc", {"token": "t", "plan_id": "p_growth", "phone": "919876543210"})
    got = await m.get_payment_order("order_abc")
    assert got and got["plan_id"] == "p_growth"
    await m.clear_payment_order("order_abc")
    assert await m.get_payment_order("order_abc") is None


class _StubMem:
    def __init__(self, *, employer_id, pending):
        self._pending, self.employer_id = pending, employer_id

    async def get_payment_order(self, oid):
        return self._pending

    async def grant_credits(self, phone, *, tenant_id, job=0, unlock=0, boost=0, description=""):
        return {"job": job, "unlock": unlock, "boost": boost}

    async def update_employer(self, phone, fields, *, tenant_id=None):
        return None

    async def clear_payment_order(self, oid):
        return None


def _ReqJSON(fields, mem):
    import types
    from urllib.parse import urlencode
    body = urlencode(fields).encode()
    app = types.SimpleNamespace(state=types.SimpleNamespace(memory=mem))
    return types.SimpleNamespace(app=app, body=lambda: _aret(body))


async def _aret(v):
    return v


async def test_buy_credits_records_live_purchase_when_flag_on(monkeypatch):
    """With CREDITS_PURCHASE_IN_DB on, a verified bundle purchase calls the live
    record_purchase with the bundle + post-purchase balances."""
    import app.api.routes.employer as emp
    from app.config import get_settings
    import types
    monkeypatch.setattr(get_settings(), "credits_purchase_in_db", True)
    monkeypatch.setattr(emp.rzp, "verify_payment_signature", lambda **k: True)

    async def _noop(*a, **k):
        return None
    monkeypatch.setattr(emp.wa_delivery, "send_message", _noop)

    async def fake_get_bundle(bid):
        return {"id": bid, "name": "Growth", "validityDays": 30}
    monkeypatch.setattr(emp.CreditBundleRepository, "get", staticmethod(fake_get_bundle))

    calls = []

    async def fake_record(*, employer_id, grants, balances_after, price, payment_id, bundle=None, commit=True):
        calls.append({"employer_id": employer_id, "grants": grants, "bundle": bundle, "price": price})
        return {"committed": True, "granted": grants}
    monkeypatch.setattr(emp.CreditWalletRepository, "record_purchase", staticmethod(fake_record))

    mem = _StubMem(employer_id="emp1", pending={
        "kind": "buy_credits", "token": "t", "phone": "919876543210", "tenant_id": "t1",
        "grant": {"job": 2, "unlock": 60, "boost": 2}, "label": "Growth bundle",
        "item": "bundle:b1", "price": 1999.0, "employer_id": "emp1"})
    req = _ReqJSON({"razorpay_order_id": "order_1", "razorpay_payment_id": "pay_1",
                    "razorpay_signature": "sig"}, mem)
    out = await emp.buy_credits_verify(req)
    import json as _j
    assert _j.loads(out.body)["ok"] is True
    assert len(calls) == 1
    assert calls[0]["employer_id"] == "emp1" and calls[0]["bundle"]["id"] == "b1"
    assert calls[0]["grants"] == {"job": 2, "unlock": 60, "boost": 2} and calls[0]["price"] == 1999.0


async def test_wallet_ledger_records_transactions():
    m = await _mem()
    # welcome seed → a 'Congratulations …FREE job credit' ledger row
    await m.ensure_wallet("919876543210", tenant_id="t1", seed={"job": 1, "unlock": 0, "boost": 0}, welcome=True)
    # a purchase + a debit, each with a description, get logged
    await m.grant_credits("919876543210", tenant_id="t1", unlock=20, description="Purchased Starter bundle")
    await m.adjust_job_credits("919876543210", -1, tenant_id="t1", description="Posted a job")
    led = await m.wallet_ledger("919876543210", tenant_id="t1")
    assert len(led) == 3
    assert led[0]["description"] == "Posted a job" and led[0]["action"] == "debit"   # newest first
    assert led[0]["creditType"] == "JOB" and led[0]["balance"] == 0
    assert led[1]["creditType"] == "UNLOCK" and led[1]["amount"] == 20
    assert "Congratulations" in led[2]["description"] and led[2]["balance"] == 1


def test_wallet_page_renders_balance_and_history():
    from app.api.routes.employer import _wallet_html
    ledger = [
        {"creditType": "JOB", "action": "credit", "amount": 1, "balance": 1,
         "description": "Congratulations! You have received 1 FREE job credit.", "createdAt": 1750000000},
        {"creditType": "UNLOCK", "action": "credit", "amount": 20, "balance": 20,
         "description": "Purchased Starter bundle", "createdAt": 1750000500},
    ]
    out = _wallet_html("tok", balance={"job": 1, "unlock": 20, "boost": 0}, ledger=list(reversed(ledger)))
    assert "Credit Wallet" in out and "Recharge" in out
    assert "All Transactions" in out and ">Unlocks<" in out and ">Boosts<" in out
    assert "Congratulations" in out and "Bal: 20" in out
    assert 'data-type="UNLOCK"' in out and 'data-type="JOB"' in out
    assert "/employer/buy-credits?token=tok" in out          # Recharge / Add Credits link


async def test_wallet_ensure_and_grant_all_buckets():
    m = await _mem()
    # seeds each bucket from the live balance only the first time
    bal = await m.ensure_wallet("919876543210", tenant_id="t1", seed={"job": 1, "unlock": 5, "boost": 2})
    assert bal == {"job": 1, "unlock": 5, "boost": 2}
    bal = await m.ensure_wallet("919876543210", tenant_id="t1", seed={"job": 99, "unlock": 99, "boost": 99})
    assert bal == {"job": 1, "unlock": 5, "boost": 2}          # seed ignored after first
    # a purchase grants into the right buckets
    bal = await m.grant_credits("919876543210", tenant_id="t1", unlock=10)
    assert bal == {"job": 1, "unlock": 15, "boost": 2}
    bal = await m.grant_credits("919876543210", tenant_id="t1", job=2, boost=1)
    assert bal == {"job": 3, "unlock": 15, "boost": 3}
    assert await m.wallet_balances("919876543210", tenant_id="t1") == {"job": 3, "unlock": 15, "boost": 3}


async def test_activate_subscription_grants_credits():
    m = await _mem()
    sub = {"planType": "GROWTH", "planName": "Growth", "validityDays": 30}
    rec = await m.activate_subscription("919876543210", sub, grant_unlock=60, grant_boost=2, tenant_id="t1")
    assert rec["subscription"]["planType"] == "GROWTH"
    assert rec["walletUnlockCredits"] == 60 and rec["walletBoostCredits"] == 2
    # a second grant stacks
    rec = await m.activate_subscription("919876543210", sub, grant_unlock=60, grant_boost=2, tenant_id="t1")
    assert rec["walletUnlockCredits"] == 120 and rec["walletBoostCredits"] == 4
