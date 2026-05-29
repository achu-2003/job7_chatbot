"""AgentScheduler.run_once — drains due follow-ups, sends them, marks done."""
from app.memory import repositories as repo
from app.workers import agent_scheduler as sched_mod
from app.workers.agent_scheduler import AgentScheduler


async def test_run_once_sends_due_marks_all_and_parses_payloads(monkeypatch):
    due_rows = [
        {"id": "a1", "session_id": "s1", "kind": "followup",
         "payload": {"message": "Back in stock! 🎉", "customer_id": "9198"}},
        # payload as a JSON string (asyncpg may return JSONB as text)
        {"id": "a2", "session_id": "s1", "kind": "followup",
         "payload": '{"message": "Just checking in!", "customer_id": "9199"}'},
        # no message → skip the send but still resolve the action
        {"id": "a3", "session_id": "s1", "kind": "followup",
         "payload": {"message": "", "customer_id": "9100"}},
    ]

    async def fake_due(tenant_id):
        return due_rows

    marked: list[tuple] = []

    async def fake_mark(action_id, status):
        marked.append((action_id, status))

    monkeypatch.setattr(repo.PendingActionRepository, "due", staticmethod(fake_due))
    monkeypatch.setattr(repo.PendingActionRepository, "mark", staticmethod(fake_mark))

    delivered: list[tuple] = []

    async def fake_deliver(settings, to, plan, **kw):
        delivered.append((to, plan[0]["text"]))

    monkeypatch.setattr(sched_mod.wa_delivery, "deliver", fake_deliver)

    sch = AgentScheduler(tenant_id="t1", poll_seconds=999)
    sent = await sch.run_once()

    assert sent == 2  # a1 + a2 sent; a3 had no message
    assert ("9198", "Back in stock! 🎉") in delivered
    assert ("9199", "Just checking in!") in delivered
    # every due action is resolved so it isn't re-sent
    assert {m[0] for m in marked} == {"a1", "a2", "a3"}
    assert all(status == "done" for _, status in marked)


async def test_run_once_no_due_actions(monkeypatch):
    async def fake_due(tenant_id):
        return []

    monkeypatch.setattr(repo.PendingActionRepository, "due", staticmethod(fake_due))
    assert await AgentScheduler(tenant_id="t1").run_once() == 0
