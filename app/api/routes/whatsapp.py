"""WhatsApp Cloud API webhook.

Meta sends two flavors of traffic here:

* GET  /api/whatsapp/webhook  — verification handshake (echo hub.challenge).
* POST /api/whatsapp/webhook  — incoming customer messages.

The POST handler:
    1. Immediately marks the inbound message as read + starts a typing
       indicator (fire-and-forget) so the customer sees blue ticks and
       "typing…" while the graph runs.
    2. Dispatches text/button/list messages through the LangGraph chat
       runner (no HTTP loopback).
    3. Sends the reply back via the Meta Graph API using async ``httpx``.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import ORJSONResponse, PlainTextResponse

from app.api.deps import bind_tenant, get_graph, get_memory, get_vector
from app.chatbot import wa_format as wa
from app.chatbot.graph import ChatGraphRunner
from app.core.metrics import WEBHOOK_DEDUPE
from app.whatsapp import delivery as wa_delivery
from app.config import get_settings
from app.core import conversation_log as conv
from app.core.logging import get_logger, request_id_var
from app.vector.store import VectorStore

router = APIRouter()
log = get_logger("whatsapp")


@router.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(
    hub_mode: str | None = Query(default=None, alias="hub.mode"),
    hub_verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(default=None, alias="hub.challenge"),
) -> PlainTextResponse:
    settings = get_settings()
    log.info("whatsapp_verify_request", mode=hub_mode)

    if hub_verify_token == settings.whatsapp_verify_token and hub_challenge:
        log.info("whatsapp_verify_ok")
        return PlainTextResponse(hub_challenge, status_code=200)

    log.warning("whatsapp_verify_failed")
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhook", response_class=ORJSONResponse)
async def receive_webhook(
    request: Request,
) -> ORJSONResponse:
    settings = get_settings()
    request_id = request_id_var.get() or "REQ_unknown"

    try:
        payload: dict[str, Any] = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    log.info("whatsapp_webhook_received", payload=payload)

    conv.start()
    webhook_type = _inbound_message_type(payload)
    if webhook_type:
        conv.note("webhook", webhook_type)

    # Fire-and-forget read receipt + typing indicator — applies to every
    # customer-originated message regardless of type.
    inbound_message_id = _inbound_message_id(payload)

    # Idempotency: Meta retries webhooks until it gets a 200, which can deliver
    # the same message twice. Dedupe by message id so we never double-process.
    if inbound_message_id:
        try:
            is_new = await get_memory(request).mark_seen(inbound_message_id)
        except Exception:  # noqa: BLE001 — fail open: a rare dup beats dropping a message
            is_new = True
        if not is_new:
            WEBHOOK_DEDUPE.labels(result="duplicate").inc()
            log.info("whatsapp_duplicate_skipped", message_id=inbound_message_id)
            return ORJSONResponse({"status": "duplicate"}, status_code=200)
        WEBHOOK_DEDUPE.labels(result="new").inc()

    if inbound_message_id:
        asyncio.create_task(
            _send_read_and_typing(settings, inbound_message_id),
            name="wa_read_typing",
        )

    # The recruiting agent is text-only — there's no visual job search. If a
    # candidate sends an image, reply with a gentle nudge to describe what they
    # want in words rather than silently ignoring it.
    if webhook_type == "image":
        image_msg = _extract_image_message(payload)
        if image_msg is not None:
            return await _handle_unsupported_image(
                settings=settings,
                customer_number=_normalize_phone(image_msg["from"]),
            )

    message = _extract_text_message(payload)
    if message is None:
        # status updates, delivery receipts, non-text — ack and ignore (no
        # conversation_turn for these, they aren't customer turns)
        return ORJSONResponse({"status": "ignored"}, status_code=200)

    customer_number: str = _normalize_phone(message["from"])
    incoming_text: str = message["text"]

    wa_tenant = settings.whatsapp_tenant_id or settings.default_tenant_id
    tenant = bind_tenant(wa_tenant)

    # Conversation memory keys are tenant-scoped (t:<tenant>:conv:wa_<digits>:*),
    # so two tenants running on different phone_number_ids never share state.
    conversation_id = f"wa_{customer_number}"

    conv.note("sender", customer_number)
    conv.note("prompt", incoming_text)

    log.info(
        "whatsapp_message_in",
        sender=customer_number,
        tenant_id=tenant.id,
        conversation_id=conversation_id,
        text=incoming_text,
    )

    runner: ChatGraphRunner = get_graph(request)
    result = await runner.handle(
        request_id=request_id,
        tenant_id=tenant.id,
        conversation_id=conversation_id,
        customer_query=incoming_text,
        customer_external_id=customer_number,
        channel="whatsapp",
    )
    bot_reply: str = (
        result.get("response")
        or "Sorry, I couldn't process your request."
    )
    log.info("whatsapp_ai_reply", reply=bot_reply)

    # Production AgentRuntime path: paced, human-like multi-bubble delivery.
    delivery_plan = result.get("delivery_plan")
    if delivery_plan:
        # An interactive payload (e.g. the onboarding cta_url "Open form" button)
        # takes priority over the text bubbles. If Meta rejects it (cta_url not
        # enabled / non-https URL), fall back to the bubbles — whose text carries
        # the same link inline, so the candidate never loses it.
        interactive = result.get("whatsapp_interactive")
        if interactive:
            status = await _post_whatsapp_message(settings, customer_number, interactive)
            if status in (200, 201):
                sent = 1
                conv.note("reply to", f"{customer_number} (cta button)")
            else:
                log.warning("wa_cta_rejected_fallback_text", status=status)
                sent = await wa_delivery.deliver(settings, customer_number, delivery_plan)
                conv.note("reply to", f"{customer_number} ({sent} bubble(s), cta fallback)")
        else:
            sent = await wa_delivery.deliver(settings, customer_number, delivery_plan)
            conv.note("reply to", f"{customer_number} ({sent} bubble(s))")
        await _push_live_feed(
            request, tenant_id=tenant.id, payload=payload, phone=customer_number,
            inbound=incoming_text, reply=bot_reply,
            trace=conv.events(), memory=result.get("memory"), steps=result.get("steps"),
        )
        conv.flush()
        return ORJSONResponse(
            {
                "status": "success",
                "customer": customer_number,
                "message": incoming_text,
                "reply": bot_reply,
                "bubbles": sent,
            },
            status_code=200,
        )

    # ------------------------------------------------------------------
    # Legacy ChatGraphRunner path — single message. The graph may have
    # attached a fully-formed interactive payload (buttons / list); use it
    # when present and fall back to plain text otherwise.
    # ------------------------------------------------------------------
    to_number = customer_number
    graph_url = (
        f"https://graph.facebook.com/{settings.whatsapp_graph_version}/"
        f"{settings.meta_phone_number_id}/messages"
    )
    rich = result.get("whatsapp_message") or {
        "type": "text",
        "text": {"body": bot_reply},
    }
    wa_payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        **rich,
    }
    headers = {
        "Authorization": f"Bearer {settings.meta_access_token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(graph_url, json=wa_payload, headers=headers)
        except httpx.HTTPError as exc:
            log.exception("whatsapp_send_error", error=str(exc))
            raise HTTPException(status_code=502, detail="WhatsApp send failed")

    log.info(
        "whatsapp_send_response",
        status=resp.status_code,
        body=resp.text[:500],
    )
    if resp.status_code not in (200, 201):
        # Rich payload rejected (e.g. an interactive message with an
        # unsupported header image, or cta_url not enabled on the number).
        # Retry once as plain text so the customer always gets the answer
        # rather than silence.
        log.warning("whatsapp_rich_send_failed", status=resp.status_code, fallback="text")
        fallback = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "text",
            "text": {"body": bot_reply},
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(graph_url, json=fallback, headers=headers)
            except httpx.HTTPError as exc:
                log.exception("whatsapp_send_error", error=str(exc))
                raise HTTPException(status_code=502, detail="WhatsApp send failed")
        if resp.status_code not in (200, 201):
            log.error("whatsapp_fallback_send_failed", status=resp.status_code, body=resp.text[:300])
            raise HTTPException(status_code=502, detail="WhatsApp API rejected message")

    conv.note("reply to", to_number)
    conv.flush()

    return ORJSONResponse(
        {
            "status": "success",
            "customer": to_number,
            "message": incoming_text,
            "reply": bot_reply,
        },
        status_code=200,
    )


def _normalize_phone(raw: str) -> str:
    """Canonicalise a phone identifier for memory keys + outbound delivery.

    Strips whitespace, the leading ``+``, and any non-digit characters
    (Meta occasionally injects formatting). Returns a digits-only string.
    Same input → same output → same Redis key, so two messages from the
    same WhatsApp sender always share a memory session even if Meta
    changes its representation between requests.
    """
    return "".join(ch for ch in (raw or "").strip().lstrip("+") if ch.isdigit())


async def _push_live_feed(
    request: Request,
    *,
    tenant_id: str,
    payload: dict[str, Any],
    phone: str,
    inbound: str,
    reply: str,
    trace: list | None = None,
    memory: dict | None = None,
    steps: dict | None = None,
) -> None:
    """Record one WhatsApp turn (client info + received message + reply + flow +
    memory) into the per-tenant live feed the monitor UI polls. Best-effort."""
    try:
        value = payload["entry"][0]["changes"][0]["value"]
        msg = (value.get("messages") or [{}])[0]
        contact = (value.get("contacts") or [{}])[0]
    except (KeyError, IndexError, TypeError):
        msg, contact = {}, {}
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phone": phone,
        "name": (contact.get("profile") or {}).get("name"),
        "wa_id": contact.get("wa_id"),
        "message_id": msg.get("id"),
        "type": msg.get("type"),
        "inbound": inbound,
        "reply": reply,
        "trace": trace or [],
        "memory": memory,
        "steps": steps,                  # plan + tool calls (args/results) + reflections
        "message_payload": msg,          # the received message object (Meta payload)
    }
    try:
        await get_memory(request).push_feed(record, tenant_id=tenant_id)
    except Exception as exc:  # noqa: BLE001 — monitoring is best-effort
        log.warning("livefeed_push_failed", error=str(exc)[:200])


def _inbound_message_id(payload: dict[str, Any]) -> str | None:
    """Extract the inbound message id (``wamid....``) for the read receipt."""
    try:
        msgs = payload["entry"][0]["changes"][0]["value"]["messages"]
        return msgs[0].get("id")
    except (KeyError, IndexError, TypeError):
        return None


async def _send_read_and_typing(settings, message_id: str) -> None:
    """POST a combined read+typing acknowledgment to Meta.

    Cloud API v22+ supports ``typing_indicator`` on the same call as the
    read receipt — one round trip lights up blue ticks AND the "typing…"
    bubble. Typing indicator auto-clears when our next message lands or
    after ~25s, whichever first. Errors are swallowed and logged; this is
    pure UX polish, not critical path.
    """
    url = (
        f"https://graph.facebook.com/{settings.whatsapp_graph_version}/"
        f"{settings.meta_phone_number_id}/messages"
    )
    body = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {"type": "text"},
    }
    headers = {
        "Authorization": f"Bearer {settings.meta_access_token}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json=body, headers=headers)
        if resp.status_code not in (200, 201):
            log.warning(
                "whatsapp_read_typing_rejected",
                status=resp.status_code,
                body=resp.text[:200],
            )
    except httpx.HTTPError as exc:
        log.warning("whatsapp_read_typing_failed", error=str(exc))


def _inbound_message_type(payload: dict[str, Any]) -> str | None:
    """Classify the inbound webhook for the conversation tracer.

    Returns one of ``text``, ``button_reply``, ``list_reply``, ``status``
    (delivery receipts / read receipts), or ``other`` (image/voice/etc.).
    """
    try:
        value = payload["entry"][0]["changes"][0]["value"]
    except (KeyError, IndexError, TypeError):
        return None
    if value.get("statuses"):
        return "status"
    messages = value.get("messages") or []
    if not messages:
        return None
    msg = messages[0]
    mtype = msg.get("type")
    if mtype == "text":
        return "text"
    if mtype == "interactive":
        return (msg.get("interactive") or {}).get("type") or "interactive"
    return mtype or "other"


def _extract_text_message(payload: dict[str, Any]) -> dict[str, str] | None:
    """Pull a textual customer message out of any inbound WhatsApp event.

    Handles three message shapes we care about:

        * ``type=text``                       — normal typed messages
        * ``type=interactive`` button_reply   — customer tapped a reply button
        * ``type=interactive`` list_reply     — customer picked a list row

    For taps we use the button/row *title* as the message body, so the graph
    classifier handles it just like a typed message (no separate dispatcher
    required).
    """
    try:
        value = payload["entry"][0]["changes"][0]["value"]
    except (KeyError, IndexError, TypeError):
        return None

    messages = value.get("messages")
    if not messages:
        return None

    msg = messages[0]
    sender = msg.get("from")
    mtype = msg.get("type")
    body: str | None = None

    if mtype == "text":
        body = (msg.get("text") or {}).get("body", "")
    elif mtype == "interactive":
        interactive = msg.get("interactive") or {}
        itype = interactive.get("type")
        if itype == "button_reply":
            body = (interactive.get("button_reply") or {}).get("title", "")
        elif itype == "list_reply":
            body = (interactive.get("list_reply") or {}).get("title", "")

    body = (body or "").strip()
    if not body or not sender:
        return None
    return {"from": str(sender), "text": body}


def _extract_image_message(payload: dict[str, Any]) -> dict[str, str] | None:
    """Pull the inbound image's ``media_id`` + sender for visual search."""
    try:
        messages = payload["entry"][0]["changes"][0]["value"]["messages"]
    except (KeyError, IndexError, TypeError):
        return None
    if not messages:
        return None
    msg = messages[0]
    if msg.get("type") != "image":
        return None
    media_id = (msg.get("image") or {}).get("id")
    sender = msg.get("from")
    if not media_id or not sender:
        return None
    return {"from": str(sender), "media_id": str(media_id)}


async def _download_whatsapp_media(settings, media_id: str) -> bytes | None:
    """Two-step Meta media fetch: media_id → URL → bytes. Both calls need
    the bearer token. Returns None on any failure — visual search is
    best-effort, we want to fall back to a polite reply rather than 500."""
    auth = {"Authorization": f"Bearer {settings.meta_access_token}"}
    meta_url = f"https://graph.facebook.com/{settings.whatsapp_graph_version}/{media_id}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            meta_resp = await client.get(meta_url, headers=auth)
            if meta_resp.status_code != 200:
                log.warning(
                    "wa_media_metadata_failed",
                    status=meta_resp.status_code, body=meta_resp.text[:200],
                )
                return None
            url = (meta_resp.json() or {}).get("url")
            if not url:
                log.warning("wa_media_no_url", body=meta_resp.text[:200])
                return None
            bytes_resp = await client.get(url, headers=auth)
            if bytes_resp.status_code != 200:
                log.warning(
                    "wa_media_bytes_failed", status=bytes_resp.status_code,
                )
                return None
            return bytes_resp.content
    except httpx.HTTPError as exc:
        log.warning("wa_media_download_error", error=str(exc))
        return None


async def _post_whatsapp_message(
    settings, to_number: str, message_payload: dict[str, Any]
) -> int:
    """POST a Cloud API message payload. Returns the HTTP status code."""
    url = (
        f"https://graph.facebook.com/{settings.whatsapp_graph_version}/"
        f"{settings.meta_phone_number_id}/messages"
    )
    body = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        **message_payload,
    }
    headers = {
        "Authorization": f"Bearer {settings.meta_access_token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            log.exception("whatsapp_send_error", error=str(exc))
            return 502
    log.info(
        "whatsapp_send_response", status=resp.status_code, body=resp.text[:500],
    )
    return resp.status_code


async def _handle_unsupported_image(
    *,
    settings,
    customer_number: str,
) -> ORJSONResponse:
    """The recruiting agent has no visual search (jobs have no images). When a
    candidate sends a photo, ask them to describe what they want in words
    instead of silently dropping the message."""
    conv.note("sender", customer_number)
    conv.note("prompt", "<image — unsupported>")
    body = (
        "I can't read images here, but I can help you find and apply for jobs. "
        "Tell me what kind of role you're after and I'll take it from there."
    )
    await _post_whatsapp_message(settings, customer_number, wa.text_message(body))
    conv.note("reply to", customer_number)
    conv.flush()
    return ORJSONResponse({"status": "image_unsupported"}, status_code=200)
