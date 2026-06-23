"""Localize OUT-OF-BAND WhatsApp pushes into the recipient's language.

Confirmations / notifications sent outside the agent turn — the resume-received
message, "job submitted", purchase/renewal/expiry notices — don't pass through the
agent's translation hook, so they'd arrive in English for a Tamil/Hindi user. This
helper resolves the recipient's stored language (the ``lang`` fact) and translates
the payload before delivery. Best-effort: any failure just sends the English text.
"""
from __future__ import annotations

from typing import Any, Callable

from app import i18n
from app.config import get_settings
from app.core.logging import get_logger
from app.memory.repositories import CustomerFactRepository
from app.whatsapp import delivery as wa_delivery

log = get_logger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from app.llm.client import LLMClient
        _llm = LLMClient()
    return _llm


async def user_lang(tenant_id: str | None, customer_id: str | None) -> str:
    """The recipient's stored UI language (``ta`` / ``hi``), or ``en``."""
    if not (tenant_id and customer_id):
        return "en"
    try:
        facts = await CustomerFactRepository.all_for(tenant_id, customer_id)
    except Exception as exc:  # noqa: BLE001 — a lookup miss must not block the push
        log.warning("user_lang_failed", error=str(exc)[:160])
        return "en"
    lang = (facts or {}).get("lang") or "en"
    return lang if i18n.is_supported(lang) else "en"


def _payload_slots(payload: dict) -> list[tuple[Callable[[str], None], str]]:
    """``(setter, text)`` for the user-visible text of a single WhatsApp payload —
    a text message body or an interactive's labels (ids/urls excluded)."""
    out: list[tuple[Callable[[str], None], str]] = []
    if payload.get("type") == "text" and isinstance(payload.get("text"), dict):
        body = payload["text"]
        if isinstance(body.get("body"), str) and body["body"].strip():
            out.append(((lambda v, o=body: o.__setitem__("body", v)), body["body"]))
    out += i18n._interactive_slots(payload)
    return out


async def send(settings: Any, phone: str, payload: dict, *, tenant_id: str | None) -> None:
    """Translate ``payload`` into the recipient's stored language (best-effort), then
    send it. No-op localization for English / when multilang is off."""
    if get_settings().multilang_enabled and phone:
        try:
            lang = await user_lang(tenant_id, phone)
            if lang != "en":
                slots = _payload_slots(payload)
                if slots:
                    setters = [s for s, _ in slots]
                    translated = await i18n.translate_many(
                        _get_llm(), [t for _, t in slots], to_lang=lang)
                    for setter, tr in zip(setters, translated):
                        setter(tr)
        except Exception as exc:  # noqa: BLE001 — never block the push on a translation error
            log.warning("push_localize_failed", error=str(exc)[:200])
    await wa_delivery.send_message(settings, phone, payload)
