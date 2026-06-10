"""Paced, human-like outbound delivery to the WhatsApp Cloud API.

Sends a ``delivery_plan`` (from the humanizer) as separate bubbles with a short
"typing" pause before each, so messages arrive spaced like a person typing
rather than one wall of text. The first bubble may carry a product image; if
Meta rejects it, we resend the words as plain text so the customer never loses
the answer.

``sleep`` and ``client`` are injectable so tests run instantly without network.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import httpx

from app.chatbot import wa_format as wa
from app.core.logging import get_logger

log = get_logger("wa_delivery")


def _graph_url(settings: Any) -> str:
    return (
        f"https://graph.facebook.com/{settings.whatsapp_graph_version}/"
        f"{settings.meta_phone_number_id}/messages"
    )


def _headers(settings: Any) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.meta_access_token}",
        "Content-Type": "application/json",
    }


def _bubble_payload(bubble: dict[str, Any]) -> dict[str, Any]:
    if bubble.get("image_url"):
        return wa.image_message(bubble["text"], bubble["image_url"])
    return wa.text_message(bubble["text"])


async def _post(settings: Any, to_number: str, payload: dict[str, Any], client: httpx.AsyncClient) -> int:
    body = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        **payload,
    }
    try:
        resp = await client.post(_graph_url(settings), json=body, headers=_headers(settings))
    except httpx.HTTPError as exc:
        log.warning("wa_delivery_post_error", error=str(exc))
        return 502
    if resp.status_code not in (200, 201):
        log.warning("wa_delivery_rejected", status=resp.status_code, body=resp.text[:200])
    return resp.status_code


async def send_message(
    settings: Any,
    to_number: str,
    payload: dict[str, Any],
    *,
    client: httpx.AsyncClient | None = None,
) -> int:
    """Send a single Cloud API message payload (text or interactive) and return
    the HTTP status. Used for proactive pushes (e.g. the post-registration menu)."""
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30.0)
    try:
        return await _post(settings, to_number, payload, client)
    finally:
        if own_client:
            await client.aclose()


async def deliver(
    settings: Any,
    to_number: str,
    delivery_plan: list[dict[str, Any]],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    client: httpx.AsyncClient | None = None,
) -> int:
    """Send each bubble after its typing pause. Returns how many were sent."""
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30.0)
    sent = 0
    try:
        for bubble in delivery_plan:
            await sleep(bubble.get("typing_ms", 0) / 1000)
            status = await _post(settings, to_number, _bubble_payload(bubble), client)
            if status not in (200, 201) and bubble.get("image_url"):
                # image rejected (bad/unreachable URL) → resend the words as text
                await _post(settings, to_number, wa.text_message(bubble["text"]), client)
            sent += 1
    finally:
        if own_client:
            await client.aclose()
    return sent
