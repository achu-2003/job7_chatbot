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


async def test_activate_subscription_grants_credits():
    m = await _mem()
    sub = {"planType": "GROWTH", "planName": "Growth", "validityDays": 30}
    rec = await m.activate_subscription("919876543210", sub, grant_unlock=60, grant_boost=2, tenant_id="t1")
    assert rec["subscription"]["planType"] == "GROWTH"
    assert rec["walletUnlockCredits"] == 60 and rec["walletBoostCredits"] == 2
    # a second grant stacks
    rec = await m.activate_subscription("919876543210", sub, grant_unlock=60, grant_boost=2, tenant_id="t1")
    assert rec["walletUnlockCredits"] == 120 and rec["walletBoostCredits"] == 4
