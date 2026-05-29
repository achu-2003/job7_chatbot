"""Webhook idempotency — ConversationMemory.mark_seen dedupes by message id."""
from app.chatbot.memory import ConversationMemory


class _FakeRedis:
    def __init__(self):
        self.store: set[str] = set()

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None          # already seen → SET NX returns nil
        self.store.add(key)
        return True


async def test_mark_seen_first_then_duplicate():
    mem = ConversationMemory()
    mem._redis = _FakeRedis()   # type: ignore[assignment]
    assert await mem.mark_seen("wamid.AAA") is True    # first delivery → process
    assert await mem.mark_seen("wamid.AAA") is False   # Meta retry → skip
    assert await mem.mark_seen("wamid.BBB") is True     # a different message
