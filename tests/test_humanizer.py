"""humanizer — pure chunking + typing-delay logic for paced WhatsApp bubbles."""
from app.agent.nodes.humanizer import build_delivery_plan


def _plan(text, **over):
    cfg = dict(max_chars=30, max_chunks=3, cps=25.0, min_ms=700, max_ms=4000)
    cfg.update(over)
    return build_delivery_plan(text, **cfg)


def test_empty_text_no_bubbles():
    assert _plan("") == []
    assert _plan("   ") == []


def test_short_reply_is_one_bubble_at_min_delay():
    plan = _plan("Hi there!", max_chars=100)
    assert len(plan) == 1
    assert plan[0]["text"] == "Hi there!"
    assert plan[0]["typing_ms"] == 700          # clamped up to the floor


def test_typing_delay_clamps_to_max():
    plan = _plan("a" * 200, max_chars=500)        # 200/25*1000 = 8000 → clamp 4000
    assert len(plan) == 1
    assert plan[0]["typing_ms"] == 4000


def test_long_reply_splits_and_caps_at_max_chunks():
    text = ("First sentence here. Second sentence here. Third one here. "
            "Fourth one here. Fifth one here too.")
    plan = _plan(text, max_chars=30, max_chunks=3)
    assert 2 <= len(plan) <= 3
    joined = " ".join(b["text"] for b in plan)
    assert "First sentence" in joined and "Fifth one" in joined   # nothing dropped


def test_hard_cap_folds_overflow_into_last_bubble():
    text = "\n\n".join(f"Para {i} body text." for i in range(8))
    plan = _plan(text, max_chars=15, max_chunks=3)
    assert len(plan) == 3
