"""Post-a-job token isolation.

Register / KYC reuse one form token per employer, but each *post-job* must mint a
FRESH token so every post is a distinct link with its own staged draft — a reused
token let a previous post's draft (or a cached page) bleed into the next post, so
the activate step showed the OLD job. See ensure_employer_token(fresh=...)."""
from app.chatbot.memory import ConversationMemory


class _FakeRedis:
    def __init__(self):
        self.store = {}

    async def get(self, k):
        return self.store.get(k)

    async def setex(self, k, ttl, v):
        self.store[k] = v

    async def expire(self, k, ttl):
        return True

    async def delete(self, *ks):
        for k in ks:
            self.store.pop(k, None)


def _mem():
    m = ConversationMemory()
    m._redis = _FakeRedis()
    return m


_KW = dict(tenant_id="default", conversation_id="c1", name="Asha")


async def test_post_job_mints_a_fresh_token_each_time():
    m = _mem()
    t1 = await m.ensure_employer_token("919000000001", fresh=True, **_KW)
    t2 = await m.ensure_employer_token("919000000001", fresh=True, **_KW)
    assert t1 != t2, "each post-job must get its own token"
    # both remain resolvable to the same employer identity
    assert (await m.get_employer_identity(t1))["customer_id"] == "919000000001"
    assert (await m.get_employer_identity(t2))["customer_id"] == "919000000001"


async def test_register_reuses_one_token():
    m = _mem()
    t1 = await m.ensure_employer_token("919000000002", **_KW)
    t2 = await m.ensure_employer_token("919000000002", **_KW)
    assert t1 == t2, "register/KYC keep reusing one token per employer"


async def test_fresh_repoints_forward_key_so_later_reuse_gets_latest():
    m = _mem()
    reused = await m.ensure_employer_token("919000000003", **_KW)
    fresh = await m.ensure_employer_token("919000000003", fresh=True, **_KW)
    assert fresh != reused
    # a subsequent non-fresh call reuses the most recent (fresh) token
    again = await m.ensure_employer_token("919000000003", **_KW)
    assert again == fresh
