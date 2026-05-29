"""Agent state constructors/accessors (pure logic — no I/O)."""
from app.agent.state import (
    active_goals,
    current_step,
    new_goal,
    new_step,
    plan_complete,
)


def test_new_goal_defaults():
    g = new_goal("find a red saree under 2000")
    assert g["id"].startswith("goal_")
    assert g["status"] == "active"
    assert g["priority"] == 0
    assert g["description"] == "find a red saree under 2000"


def test_new_step_defaults():
    s = new_step("search_products", tool="search_products", args={"query": "saree"})
    assert s["id"].startswith("step_")
    assert s["status"] == "pending"
    assert s["attempts"] == 0
    assert s["tool"] == "search_products"
    assert s["args"] == {"query": "saree"}


def test_current_step_tracks_cursor():
    a, b = new_step("a"), new_step("b")
    state = {"plan": [a, b], "cursor": 0}
    assert current_step(state) is a
    state["cursor"] = 1
    assert current_step(state) is b
    state["cursor"] = 2  # past the end
    assert current_step(state) is None
    assert current_step({}) is None  # no plan


def test_active_goals_filters_by_status():
    g1 = new_goal("active one")
    g2 = new_goal("done one", status="done")
    assert active_goals({"goals": [g1, g2]}) == [g1]
    assert active_goals({}) == []


def test_plan_complete():
    s1, s2 = new_step("a"), new_step("b")
    assert plan_complete({"plan": [s1, s2]}) is False  # both pending
    s1["status"], s2["status"] = "done", "failed"
    assert plan_complete({"plan": [s1, s2]}) is True
    assert plan_complete({"plan": []}) is False  # empty plan isn't "complete"
