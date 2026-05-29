"""Streamlit testing UI for the e-commerce CRM chatbot.

Run with:
    make ui                    # uses the venv + default API at 127.0.0.1:8000
    streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("CHATBOT_API_URL", "http://127.0.0.1:8000")


# ----------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------

def _client() -> httpx.Client:
    return httpx.Client(base_url=st.session_state.api_url, timeout=60.0)


def post_chat(payload: dict[str, Any]) -> dict[str, Any]:
    with _client() as c:
        r = c.post("/api/v1/chat", json=payload)
        r.raise_for_status()
        return r.json()


def post_reindex() -> dict[str, Any]:
    with _client() as c:
        r = c.post("/api/v1/admin/reindex")
        r.raise_for_status()
        return r.json()


def get_queue_depth() -> int:
    with _client() as c:
        r = c.get("/api/v1/admin/reindex/status")
        r.raise_for_status()
        return int(r.json().get("queue_depth", 0))


def get_health() -> dict[str, Any]:
    with _client() as c:
        r = c.get("/health/ready")
        r.raise_for_status()
        return r.json()


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------

def _init_state() -> None:
    st.session_state.setdefault("api_url", DEFAULT_API)
    st.session_state.setdefault("conversation_id", f"conv_{uuid.uuid4().hex[:12]}")
    st.session_state.setdefault("messages", [])             # list of {role, content, meta}
    st.session_state.setdefault("channel", "web")
    st.session_state.setdefault("customer_ref", "")
    st.session_state.setdefault("last_reindex", None)


def _new_conversation() -> None:
    st.session_state.conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
    st.session_state.messages = []


# ----------------------------------------------------------------------
# UI rendering
# ----------------------------------------------------------------------

st.set_page_config(page_title="Chatbot Test UI", page_icon=None, layout="wide")
_init_state()

st.title("E-commerce CRM Chatbot — Testing UI")
st.caption(
    "Talks to the FastAPI backend. Each assistant turn shows intent, routing, "
    "retrieval counts and validation result so you can verify the pipeline."
)

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Connection")
    st.session_state.api_url = st.text_input("API base URL", st.session_state.api_url)

    st.subheader("Conversation")
    st.text_input("conversation_id", st.session_state.conversation_id, disabled=True)
    if st.button("New conversation", use_container_width=True):
        _new_conversation()
        st.rerun()

    st.subheader("Request fields")
    st.session_state.channel = st.selectbox(
        "channel", ["web", "whatsapp", "mobile", "other"],
        index=["web", "whatsapp", "mobile", "other"].index(st.session_state.channel),
    )
    st.session_state.customer_ref = st.text_input(
        "customer_external_id (phone/email, optional)",
        st.session_state.customer_ref,
        help="Used by order_tracking. Match against orders.customer_phone / customer_email.",
    )

    st.divider()
    st.subheader("Admin")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Health", use_container_width=True):
            try:
                st.session_state.last_health = get_health()
            except Exception as exc:  # noqa: BLE001
                st.session_state.last_health = {"error": str(exc)}
    with col2:
        if st.button("Queue", use_container_width=True):
            try:
                st.session_state.last_depth = get_queue_depth()
            except Exception as exc:  # noqa: BLE001
                st.session_state.last_depth = f"err: {exc}"

    if st.button("Reindex now", use_container_width=True, type="primary"):
        try:
            st.session_state.last_reindex = post_reindex()
            st.toast("Reindex enqueued.")
        except Exception as exc:  # noqa: BLE001
            st.session_state.last_reindex = {"error": str(exc)}

    if "last_health" in st.session_state:
        st.write("**health**")
        st.json(st.session_state.last_health, expanded=False)
    if "last_depth" in st.session_state:
        st.write(f"**queue_depth:** {st.session_state.last_depth}")
    if st.session_state.last_reindex is not None:
        st.write("**last reindex**")
        st.json(st.session_state.last_reindex, expanded=False)

# ---------- Main chat ----------

for turn in st.session_state.messages:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        meta = turn.get("meta")
        if meta:
            with st.expander("Details", expanded=False):
                cols = st.columns(4)
                cols[0].metric("intent", meta.get("intent") or "-")
                cols[1].metric("validation", meta.get("validation") or "-")
                cols[2].metric("escalated", "yes" if meta.get("escalated") else "no")
                cols[3].metric("latency_ms", meta.get("latency_ms") or 0)
                routing = meta.get("routing") or {}
                st.write(
                    f"**routing** — sql={routing.get('sql_used')}, "
                    f"vector={routing.get('vector_used')}, "
                    f"llm={routing.get('llm_used')}, "
                    f"escalate={routing.get('escalate')}  \n"
                    f"_reason:_ {routing.get('reason')}"
                )
                st.code(meta.get("request_id", ""), language="text")

prompt = st.chat_input("Ask about products, orders, returns…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "message": prompt,
        "conversation_id": st.session_state.conversation_id,
        "channel": st.session_state.channel,
    }
    if st.session_state.customer_ref.strip():
        payload["customer_external_id"] = st.session_state.customer_ref.strip()

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = post_chat(payload)
                reply = result.get("response", "(empty)")
                st.markdown(reply)
                meta = {
                    "intent": result.get("intent"),
                    "routing": result.get("routing"),
                    "validation": result.get("validation"),
                    "escalated": result.get("escalated"),
                    "latency_ms": result.get("latency_ms"),
                    "request_id": result.get("request_id"),
                }
                with st.expander("Details", expanded=False):
                    cols = st.columns(4)
                    cols[0].metric("intent", meta["intent"] or "-")
                    cols[1].metric("validation", meta["validation"] or "-")
                    cols[2].metric("escalated", "yes" if meta["escalated"] else "no")
                    cols[3].metric("latency_ms", meta["latency_ms"] or 0)
                    routing = meta["routing"] or {}
                    st.write(
                        f"**routing** — sql={routing.get('sql_used')}, "
                        f"vector={routing.get('vector_used')}, "
                        f"llm={routing.get('llm_used')}, "
                        f"escalate={routing.get('escalate')}  \n"
                        f"_reason:_ {routing.get('reason')}"
                    )
                    st.code(meta["request_id"] or "", language="text")
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply, "meta": meta}
                )
            except httpx.HTTPStatusError as exc:
                err = f"API error {exc.response.status_code}: {exc.response.text[:300]}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as exc:  # noqa: BLE001
                err = f"Request failed: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
