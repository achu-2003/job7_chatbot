"""Per-turn Budget — bounds the reasoning loop (deterministic via injected now)."""
from app.agent.budget import Budget


def test_budget_starts_with_deadline_offset():
    b = Budget.start(max_loops=4, deadline_seconds=10, now=100.0)
    assert b.deadline == 110.0
    assert b.loops == 0
    assert b.remaining_loops() == 4


def test_exhausts_on_loop_cap():
    b = Budget.start(max_loops=2, deadline_seconds=999, now=0.0)
    assert b.exhausted(now=1.0) is False
    b.tick()
    b.tick()
    assert b.loops == 2
    assert b.exhausted(now=1.0) is True       # hit loop cap
    assert b.remaining_loops() == 0


def test_exhausts_on_deadline():
    b = Budget.start(max_loops=99, deadline_seconds=5, now=0.0)
    assert b.exhausted(now=4.9) is False
    assert b.exhausted(now=5.0) is True        # deadline reached
    assert b.time_left(now=4.0) == 1.0
    assert b.time_left(now=6.0) == 0.0          # clamped, never negative
