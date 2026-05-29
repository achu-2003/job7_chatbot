"""Streamlit LIVE MONITOR for the WhatsApp agent.

You chat from the real WhatsApp app — this page is read-only and shows, in
real time (polled):
  * the complete conversation in the page body (customer ↔ bot)
  * the received Meta payload + client info (name, phone, wa_id, message id)
  * the LangGraph flow graph for the latest turn (active nodes highlighted)
  * the MCP tool calls and the memory snapshot

Run with:
    ./.venv/bin/streamlit run ui/agent_flow_ui.py
    # backend must be up (make run) with the WhatsApp webhook pointed at it.
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("CHATBOT_API_URL", "http://127.0.0.1:8000")

_NODE_LABELS = {
    "load_context": "load_context", "greeting_response": "greeting",
    "summarize": "summarize", "planner": "planner", "execute": "execute (tools)",
    "reflect": "reflect", "responder": "responder", "humanize": "humanize",
    "persist": "persist",
}
_EDGES = [
    ("load_context", "greeting_response"), ("load_context", "summarize"),
    ("summarize", "planner"), ("planner", "execute"), ("planner", "responder"),
    ("execute", "reflect"), ("reflect", "planner"), ("reflect", "responder"),
    ("greeting_response", "humanize"), ("responder", "humanize"),
    ("humanize", "persist"),
]


def fetch_feed() -> list[dict]:
    with httpx.Client(base_url=st.session_state.api_url, timeout=10.0) as c:
        r = c.get("/api/v1/admin/live-feed", params={"limit": 60})
        r.raise_for_status()
        return r.json().get("turns", [])


def reset_customer(phone: str) -> dict:
    with httpx.Client(base_url=st.session_state.api_url, timeout=15.0) as c:
        r = c.post("/api/v1/admin/reset-customer", params={"phone": phone})
        r.raise_for_status()
        return r.json()


def active_nodes(trace: list) -> set[str]:
    labels = [t[0] for t in (trace or [])]
    active = {"load_context", "humanize", "persist"}
    if "plan" in labels:
        active |= {"summarize", "planner"}
        if "tool" in labels:
            active.add("execute")
        if "reflect" in labels:
            active.add("reflect")
        if "respond" in labels:
            active.add("responder")
    else:
        active.add("greeting_response")
    return active


def _tool_ok(tc: dict) -> bool:
    res = tc.get("result")
    return not (isinstance(res, dict) and res.get("error"))


def flow_dot(turn: dict) -> str:
    """Compact graph with the active path lit + the actual tool calls (name +
    args + ok/error) branching off the execute node."""
    trace, steps = turn.get("trace"), (turn.get("steps") or {})
    active = active_nodes(trace)

    # annotate planner + reflect nodes with what they decided
    labels = dict(_NODE_LABELS)
    plan_note = next((e[1] for e in (trace or []) if e[0] == "plan"), None)
    if plan_note:
        labels["planner"] = f"planner\\n[{plan_note[:20]}]"
    refl = steps.get("reflections") or []
    if refl:
        labels["reflect"] = f"reflect\\n[{refl[-1].get('next', '')}]"

    parts = [
        "digraph G {",
        'graph [rankdir=TB, bgcolor=transparent, nodesep=0.16, ranksep=0.22, size="3.4,4.2", margin=0];',
        'node [fontname="Helvetica", shape=box, style="filled,rounded", '
        'fontsize=9, height=0.26, margin="0.07,0.03"];',
        "edge [arrowsize=0.55];",
    ]
    for name, label in labels.items():
        on = name in active
        parts.append(
            f'{name} [label="{label}", fillcolor="{"#2563eb" if on else "#eef2f7"}", '
            f'fontcolor="{"white" if on else "#94a3b8"}", color="{"#1d4ed8" if on else "#e2e8f0"}"];'
        )
    # actual tool calls as note-shaped nodes off `execute`
    for i, tc in enumerate(steps.get("tools") or []):
        ok = _tool_ok(tc)
        args = tc.get("args") or {}
        astr = ", ".join(f"{k}={v}" for k, v in list(args.items())[:2])
        astr = (astr[:24] + "…") if len(astr) > 24 else astr
        lbl = (f"{tc.get('tool')}\\n{astr}" if astr else str(tc.get("tool"))) + f"\\n{'ok' if ok else 'ERROR'}"
        parts.append(
            f'tool{i} [label="{lbl}", shape=note, fontsize=8, '
            f'fillcolor="{"#dcfce7" if ok else "#fee2e2"}", '
            f'color="{"#16a34a" if ok else "#dc2626"}", fontcolor="#0f172a"];'
        )
        parts.append(f'execute -> tool{i} [style=dashed, color="{"#16a34a" if ok else "#dc2626"}", arrowsize=0.5];')
    for a, b in _EDGES:
        lit = a in active and b in active
        parts.append(f'{a} -> {b} [color="{"#2563eb" if lit else "#e2e8f0"}", penwidth={2 if lit else 1}];')
    parts.append("}")
    return " ".join(parts)


def render_turn_details(t: dict) -> None:
    steps = t.get("steps") or {}
    mem = t.get("memory") or {}
    tools = steps.get("tools") or []
    refl = steps.get("reflections") or []

    # ---- client ----
    st.markdown(
        f"**{t.get('name') or 'Customer'}**  ·  `{t.get('phone') or '—'}`  ·  "
        f"_{t.get('ts', '')}_"
    )

    # ---- metrics row ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latency", f"{steps.get('latency_ms', 0)} ms")
    c2.metric("Tools", len(tools))
    c3.metric("Loops", steps.get("loop_count", 0))
    c4.metric("Reflects", len(refl))

    # ---- flow graph ----
    st.graphviz_chart(flow_dot(t))

    # ---- tool calls (deep: args + result) ----
    st.markdown("**🔧 Tool calls**")
    if tools:
        for tc in tools:
            ok = _tool_ok(tc)
            with st.expander(f"{'🟢' if ok else '🔴'}  {tc.get('tool')}", expanded=True):
                st.caption("args")
                st.json(tc.get("args") or {}, expanded=False)
                st.caption("result")
                st.json(tc.get("result"), expanded=False)
    else:
        st.caption("none — answered directly from memory / greeting")

    # ---- reflections ----
    if refl:
        st.markdown("**🔁 Reflection**")
        for r in refl:
            st.markdown(f"- → **{r.get('next')}** · satisfied=`{r.get('goal_satisfied')}`")

    # ---- memory ----
    st.markdown("**🧠 Memory**")
    st.markdown(
        f"- focus product: `{(mem.get('cached_product') or '—')[:70]}`\n"
        f"- session `{mem.get('session_status', '—')}` · short-term `{mem.get('short_term_turns', 0)}`"
        f" · pending `{mem.get('pending', 0)}`"
    )
    if mem.get("customer_facts"):
        st.markdown("- facts: " + ", ".join(f"`{k}={v}`" for k, v in mem["customer_facts"].items()))

    with st.expander("📩 received payload"):
        st.json(t.get("message_payload") or {}, expanded=False)
    with st.expander("raw trace"):
        st.json(t.get("trace") or [])


# ----------------------------------------------------------------------

st.set_page_config(page_title="WhatsApp Agent — Live Monitor", layout="wide")
st.session_state.setdefault("api_url", DEFAULT_API)

with st.sidebar:
    st.subheader("Monitor")
    st.session_state.api_url = st.text_input("API base URL", st.session_state.api_url)
    auto = st.toggle("Auto-refresh", value=True)
    interval = st.slider("refresh interval (s)", 2, 30, 4)
    phone_filter = st.text_input("Filter by phone (optional)", "")
    st.divider()
    reset_phone = st.text_input("Reset memory for phone", "")
    if st.button("♻️ Reset this customer", use_container_width=True) and reset_phone.strip():
        try:
            st.toast(str(reset_customer(reset_phone.strip())))
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

st.title("WhatsApp Agent — Live Monitor")
st.caption("Chat from the WhatsApp app; this page shows the live conversation, payload, flow and memory.")


@st.fragment(run_every=interval if auto else None)
def monitor() -> None:
    try:
        turns = fetch_feed()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Can't reach the API at {st.session_state.api_url} — is the server up (make run)?  {exc}")
        return
    if phone_filter.strip():
        turns = [t for t in turns if t.get("phone") == phone_filter.strip()]

    convo_col, detail_col = st.columns([3, 2], gap="large")
    # fixed-height panels → the page fits the screen; each side scrolls on its own
    with convo_col:
        st.subheader("Conversation (live)")
        with st.container(height=620):
            if not turns:
                st.info("No messages yet — send a WhatsApp message to the connected number.")
            for t in turns:
                with st.chat_message("user"):
                    st.markdown(t.get("inbound") or "(no text)")
                    st.caption(f"{t.get('name') or '?'} · {t.get('phone') or '?'} · {t.get('ts', '')}")
                with st.chat_message("assistant"):
                    st.markdown(t.get("reply") or "(no reply)")
    with detail_col:
        st.subheader("Latest turn")
        with st.container(height=620):
            if turns:
                render_turn_details(turns[-1])
            else:
                st.caption("waiting…")


monitor()
