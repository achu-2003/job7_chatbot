"""Lightweight multilingual support (translate-pivot).

The bot's intent routing + templated replies are English. To serve Tamil / Hindi
users we translate the INBOUND message to English (so the existing English routing
and search keep working) and translate the OUTBOUND text back to the user's
language. Script-based detection (no extra dependency); LLM-backed translation with
an in-memory cache so repeated templated strings (greetings, menus) translate once.

Phase 1 covers the conversational TEXT (``draft_response`` + the paced delivery
bubbles). Interactive button/list labels, job-card bodies, and the web forms stay
English for now (Phase 2). Everything here is best-effort — a translation error
never breaks the turn (the original English text is used).
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.llm.client import LLMClient

log = get_logger(__name__)

# Supported languages: English is the pivot; add more by extending these maps.
SUPPORTED_LANGS = {"en", "ta", "hi"}
_LANG_NAME = {"en": "English", "ta": "Tamil", "hi": "Hindi"}

# Unicode script ranges used for detection.
_TAMIL = re.compile(r"[஀-௿]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")        # Hindi (and other Devanagari)
# "has a letter in any supported script" — purely-numeric/punctuation/emoji strings
# (and bare ids) are never sent to the translator.
_HAS_LETTER = re.compile(r"[A-Za-zऀ-ॿ஀-௿]")


def detect_lang(text: str | None) -> str:
    """The script's language: ``ta`` (Tamil), ``hi`` (Devanagari/Hindi), else
    ``en``. Romanized text (Tanglish/Hinglish) is treated as ``en`` for now."""
    t = text or ""
    if _TAMIL.search(t):
        return "ta"
    if _DEVANAGARI.search(t):
        return "hi"
    return "en"


def is_supported(lang: str | None) -> bool:
    return lang in SUPPORTED_LANGS


def _translatable(s: str | None) -> bool:
    return bool(s and s.strip() and _HAS_LETTER.search(s))


# Curated, hand-verified translations for the FIXED job-category names. The LLM
# mistranslates some out of context (e.g. "Hospitality & Tourism" → "World
# Transport"), so these authoritative strings are used instead — guaranteeing
# correct, CONSISTENT labels (which also makes the reverse-match reliable). Keyed
# by the exact catalog name (case-insensitive). Extendable to other fixed UI terms.
_GLOSSARY: dict[str, dict[str, str]] = {
    "ta": {
        "accounting & financee": "கணக்கியல் & நிதி",
        "accounting & finance": "கணக்கியல் & நிதி",
        "administration": "நிர்வாகம்",
        "agriculture": "விவசாயம்",
        "banking & insurance": "வங்கி & காப்பீடு",
        "construction": "கட்டுமானம்",
        "customer service": "வாடிக்கையாளர் சேவை",
        "design & creative": "வடிவமைப்பு & படைப்பாக்கம்",
        "education & training": "கல்வி & பயிற்சி",
        "engineering": "பொறியியல்",
        "healthcare": "சுகாதாரம்",
        "hospitality & tourism": "விருந்தோம்பல் & சுற்றுலா",
        "human resources": "மனிதவளம்",
        "information technology": "தகவல் தொழில்நுட்பம்",
        "legal": "சட்டம்",
        "logistics & supply chain": "தளவாடம் & விநியோகச் சங்கிலி",
        "manufacturing": "உற்பத்தி",
        "media & communications": "ஊடகம் & தொடர்பு",
        "operations": "செயல்பாடுகள்",
        "retail": "சில்லறை விற்பனை",
        "sales & marketing": "விற்பனை & சந்தைப்படுத்தல்",
        "security services": "பாதுகாப்பு சேவைகள்",
        "technician / maintenance": "தொழில்நுட்பர் / பராமரிப்பு",
    },
    "hi": {
        "accounting & financee": "लेखांकन और वित्त",
        "accounting & finance": "लेखांकन और वित्त",
        "administration": "प्रशासन",
        "agriculture": "कृषि",
        "banking & insurance": "बैंकिंग और बीमा",
        "construction": "निर्माण",
        "customer service": "ग्राहक सेवा",
        "design & creative": "डिज़ाइन और रचनात्मक",
        "education & training": "शिक्षा और प्रशिक्षण",
        "engineering": "इंजीनियरिंग",
        "healthcare": "स्वास्थ्य सेवा",
        "hospitality & tourism": "आतिथ्य और पर्यटन",
        "human resources": "मानव संसाधन",
        "information technology": "सूचना प्रौद्योगिकी",
        "legal": "कानूनी",
        "logistics & supply chain": "रसद और आपूर्ति श्रृंखला",
        "manufacturing": "विनिर्माण",
        "media & communications": "मीडिया और संचार",
        "operations": "संचालन",
        "retail": "खुदरा",
        "sales & marketing": "बिक्री और विपणन",
        "security services": "सुरक्षा सेवाएं",
        "technician / maintenance": "तकनीशियन / रखरखाव",
    },
}


def _glossary(text: str, lang: str) -> str | None:
    """An authoritative translation for a fixed UI/category term, or None."""
    return _GLOSSARY.get(lang, {}).get((text or "").strip().lower())


# Module-level LRU cache: (text, to_lang) -> translation. Templated replies repeat
# across users, so this avoids re-translating the same string every turn.
_CACHE: "OrderedDict[tuple[str, str], str]" = OrderedDict()
_CACHE_MAX = 2000


def _cache_get(key: tuple[str, str]) -> str | None:
    v = _CACHE.get(key)
    if v is not None:
        _CACHE.move_to_end(key)
    return v


def _cache_put(key: tuple[str, str], val: str) -> None:
    _CACHE[key] = val
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


async def to_english(llm: "LLMClient", text: str, *, source_lang: str) -> str:
    """Translate a user's message to English (for the English routing/search).
    No-op for English / non-wordy input. Best-effort — returns the original on
    error so a translation hiccup never blocks the turn."""
    if source_lang == "en" or not _translatable(text):
        return text
    try:
        content, _ = await llm.chat(
            purpose="translate_in",
            messages=[
                {"role": "system", "content":
                    "Translate the user's message to English. Reply with ONLY the "
                    "English translation — no quotes, no notes, no extra words."},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        return content.strip() or text
    except Exception as exc:  # noqa: BLE001 — never break the turn on a translation error
        log.warning("translate_in_failed", error=str(exc)[:200])
        return text


async def _batch_translate(llm: "LLMClient", items: list[str], to_lang: str) -> dict[int, str]:
    """One batched JSON call → ``{index: translation}`` for the items it returned.
    Returns ``{}`` on any error (the caller then falls back per-string)."""
    try:
        payload = {str(j): s for j, s in enumerate(items)}
        system = (
            f"You are a translator. Translate each VALUE in the JSON object to "
            f"{_LANG_NAME[to_lang]}. Keep the SAME keys. Preserve emoji, *bold* markers, "
            "line breaks, numbers, prices, and any code/reference tokens (slugs, ids, "
            "URLs) exactly. Return ONLY a JSON object mapping each key to its translated "
            "string."
        )
        content, _ = await llm.chat(
            purpose="translate_out",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=1500,
        )
        data = json.loads(content)
        return {j: data[str(j)] for j in range(len(items))
                if isinstance(data.get(str(j)), str) and data[str(j)].strip()}
    except Exception as exc:  # noqa: BLE001 — caller falls back per-string
        log.warning("translate_batch_failed", error=str(exc)[:200])
        return {}


async def _one_translate(llm: "LLMClient", text: str, to_lang: str) -> str | None:
    """Translate a single string (robust fallback — no JSON to misparse). None on
    error so the caller keeps the English original."""
    try:
        content, _ = await llm.chat(
            purpose="translate_out_one",
            messages=[
                {"role": "system", "content":
                    f"Translate the user's text to {_LANG_NAME[to_lang]}. Preserve emoji, "
                    "*bold* markers, line breaks, numbers, prices, and code/reference tokens. "
                    "Reply with ONLY the translation — nothing else."},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=600,
        )
        return content.strip() or None
    except Exception as exc:  # noqa: BLE001
        log.warning("translate_one_failed", error=str(exc)[:200])
        return None


async def translate_many(llm: "LLMClient", texts: list[str], *, to_lang: str) -> list[str]:
    """Translate strings to ``to_lang``, returning a same-length list. No-op for
    English / unsupported lang / empty. Per-string cached. ROBUST: tries ONE batched
    JSON call for efficiency, then falls back to a per-string call (in parallel) for
    anything the batch missed or mangled — so a single malformed batch never English-
    dumps the whole turn. Best-effort: originals are kept where translation fails."""
    if to_lang == "en" or to_lang not in SUPPORTED_LANGS or not texts:
        return list(texts)
    out: list[str | None] = [None] * len(texts)
    todo: list[tuple[int, str]] = []
    for i, s in enumerate(texts):
        if not _translatable(s):
            out[i] = s
            continue
        glossed = _glossary(s, to_lang)            # authoritative fixed-term override
        if glossed is not None:
            out[i] = glossed
            continue
        cached = _cache_get((s, to_lang))
        if cached is not None:
            out[i] = cached
        else:
            todo.append((i, s))
    if todo:
        batched = await _batch_translate(llm, [s for _, s in todo], to_lang)
        missing: list[tuple[int, str]] = []
        for k, (i, s) in enumerate(todo):
            tr = batched.get(k)
            if tr:
                _cache_put((s, to_lang), tr)
                out[i] = tr
            else:
                missing.append((i, s))
        if missing:                                  # per-string fallback (parallel)
            results = await asyncio.gather(
                *[_one_translate(llm, s, to_lang) for _, s in missing])
            for (i, s), tr in zip(missing, results):
                if tr:
                    _cache_put((s, to_lang), tr)
                    out[i] = tr
                else:
                    out[i] = s
    return [o if o is not None else texts[k] for k, o in enumerate(out)]


# --- interactive-payload label localization ---------------------------------
# WhatsApp Cloud API length caps per field — a translated label can be longer
# than its English source, so each is re-truncated to its cap (else Meta rejects
# the whole interactive message).
_WA_CAPS = {"body": 1024, "header": 60, "list_btn": 20, "section": 24,
            "row_title": 24, "row_desc": 72, "reply": 20, "cta": 20}


def _trunc(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: max(0, n - 1)].rstrip() + "…"


def _interactive_slots(payload: dict | None) -> list[tuple[Callable[[str], None], str]]:
    """``(setter, text)`` for every USER-VISIBLE label in a WhatsApp interactive
    payload — body, text header, list button/section/row titles + descriptions,
    reply-button titles, and the cta display_text. Each setter re-truncates to the
    field's WhatsApp cap. IDs and URLs are NEVER included (so taps still route)."""
    out: list[tuple[Callable[[str], None], str]] = []
    inter = (payload or {}).get("interactive") or {}

    def slot(obj: Any, key: str, cap: int) -> None:
        if isinstance(obj, dict) and isinstance(obj.get(key), str) and obj[key].strip():
            out.append(((lambda v, o=obj, k=key, c=cap: o.__setitem__(k, _trunc(v, c))), obj[key]))

    slot(inter.get("body"), "text", _WA_CAPS["body"])
    header = inter.get("header")
    if isinstance(header, dict) and header.get("type") == "text":
        slot(header, "text", _WA_CAPS["header"])
    action = inter.get("action") or {}
    slot(action, "button", _WA_CAPS["list_btn"])             # list "View roles" button
    for sec in action.get("sections") or []:
        slot(sec, "title", _WA_CAPS["section"])
        for row in sec.get("rows") or []:
            slot(row, "title", _WA_CAPS["row_title"])
            slot(row, "description", _WA_CAPS["row_desc"])
    for btn in action.get("buttons") or []:
        slot(btn.get("reply"), "title", _WA_CAPS["reply"])   # NOT reply.id
    slot(action.get("parameters"), "display_text", _WA_CAPS["cta"])  # NOT parameters.url
    return out


def collect_localizable(result: dict) -> tuple[list[str], list[Callable[[str], None]]]:
    """``(texts, setters)`` for ALL user-visible text in an agent result: the reply,
    the paced delivery bubbles, and every interactive label (job cards, role/category
    lists, buttons, cta). IDs/URLs are never included. Translate ``texts`` and call
    each ``setter`` with the result to localize the whole turn in place."""
    texts: list[str] = []
    setters: list[Callable[[str], None]] = []

    def add(text: str | None, setter: Callable[[str], None]) -> None:
        if text and text.strip():
            texts.append(text)
            setters.append(setter)

    dr = result.get("draft_response")
    if dr:
        add(dr, lambda v: result.__setitem__("draft_response", v))
    for bubble in result.get("delivery_plan") or []:
        if isinstance(bubble, dict) and bubble.get("text"):
            add(bubble["text"], (lambda v, b=bubble: b.__setitem__("text", v)))
    for setter, text in _interactive_slots(result.get("whatsapp_interactive")):
        add(text, setter)
    for msg in result.get("whatsapp_messages") or []:
        for setter, text in _interactive_slots(msg):
            add(text, setter)
    return texts, setters


def _norm(s: str | None) -> str:
    """Normalize for comparison: lowercase + collapse whitespace + drop trailing
    punctuation (so 'தயாரிப்பு' matches 'தயாரிப்பு.' / 'தயாரிப்பு ')."""
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return s.strip(" .,:;!?-—·()[]")


async def category_from_translation(
    llm: "LLMClient", text: str, categories: list[str], lang: str,
) -> str | None:
    """Reverse-map a phrase typed in ``lang`` to its English category by matching it
    against the bot's OWN translations of the category names. This makes the round
    trip CONSISTENT: if the category list showed "Manufacturing" as "தயாரிப்பு",
    typing "தயாரிப்பு" maps straight back to "Manufacturing" — no fragile English
    round-trip. Returns the exact English category name or None."""
    if lang == "en" or not text or not categories:
        return None
    norm = _norm(text)
    if not norm:
        return None
    translated = await translate_many(llm, categories, to_lang=lang)  # cached
    pairs = list(zip(categories, translated))
    for cat, tr in pairs:                       # exact match first
        if _norm(tr) == norm:
            return cat
    for cat, tr in pairs:                       # then a contained match (multi-word)
        nt = _norm(tr)
        if nt and (nt in norm or norm in nt):
            return cat
    return None


# --- LLM category classification (synonym / translated-term fallback) --------
async def classify_category(llm: "LLMClient", text: str, categories: list[str]) -> str | None:
    """Map a free-text phrase to ONE known job category via the LLM — but only when
    it names a broad field/area/industry (so a synonym or translated term like
    "transport" → "Logistics & Supply Chain" resolves), NOT a specific role/skill.
    Returns the EXACT category name from ``categories`` or None. Best-effort."""
    text = (text or "").strip()
    if not text or not categories:
        return None
    try:
        cat_list = "\n".join(f"- {c}" for c in categories)
        content, _ = await llm.chat(
            purpose="classify_category",
            messages=[
                {"role": "system", "content":
                    "Map the user's phrase to ONE job category from the list. Reply with the "
                    "EXACT category name (copied from the list) if the phrase refers to a broad "
                    "job field, area, or industry. Reply EXACTLY 'NONE' if it's a specific job "
                    "title/role, a skill, or doesn't clearly fit a category. Output ONLY the "
                    "category name or NONE.\nCategories:\n" + cat_list},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=30,
        )
        ans = (content or "").strip().strip('"').strip()
        if not ans or ans.upper() == "NONE":
            return None
        for c in categories:                       # accept only an exact catalog name
            if c.lower() == ans.lower():
                return c
        return None
    except Exception as exc:  # noqa: BLE001 — a classify miss must never break the turn
        log.warning("classify_category_failed", error=str(exc)[:200])
        return None
