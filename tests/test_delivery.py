"""WhatsApp paced delivery — sends bubbles, handles image fallback (no network)."""
from types import SimpleNamespace

from app.whatsapp import delivery


def _settings():
    return SimpleNamespace(
        whatsapp_graph_version="v22.0", meta_phone_number_id="PID", meta_access_token="TOK",
    )


class _Resp:
    def __init__(self, status):
        self.status_code = status
        self.text = "body"


class _FakeClient:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.posts: list[dict] = []

    async def post(self, url, json=None, headers=None):
        self.posts.append(json)
        return _Resp(self.statuses.pop(0) if self.statuses else 200)


async def _nosleep(_seconds):
    return None


async def test_sends_each_bubble_to_sender():
    client = _FakeClient([200, 200])
    plan = [
        {"text": "Hey! 👋", "typing_ms": 700},
        {"text": "Here are a few red sarees 🌹", "typing_ms": 900},
    ]
    sent = await delivery.deliver(_settings(), "9198", plan, sleep=_nosleep, client=client)
    assert sent == 2
    assert [p["type"] for p in client.posts] == ["text", "text"]
    assert all(p["to"] == "9198" for p in client.posts)


async def test_first_bubble_can_carry_image():
    client = _FakeClient([200])
    plan = [{"text": "Found this for you 🌹", "typing_ms": 700,
             "image_url": "https://cdn.example.com/saree.jpg"}]
    await delivery.deliver(_settings(), "9198", plan, sleep=_nosleep, client=client)
    assert client.posts[0]["type"] == "image"
    assert client.posts[0]["image"]["link"] == "https://cdn.example.com/saree.jpg"


async def test_rejected_image_resends_as_text():
    client = _FakeClient([400, 200])   # image rejected, then text accepted
    plan = [{"text": "Found this for you", "typing_ms": 700,
             "image_url": "https://cdn.example.com/saree.webp"}]
    await delivery.deliver(_settings(), "9198", plan, sleep=_nosleep, client=client)
    assert [p["type"] for p in client.posts] == ["image", "text"]
    assert client.posts[1]["text"]["body"] == "Found this for you"
