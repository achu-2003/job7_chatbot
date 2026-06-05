"""Identity extraction — pull the candidate's name & email out of the turn.

The application flow needs ``full_name`` + ``email`` before it can call
``submit_application``. Those details arrive conversationally ("my email is …",
or a bare name in reply to "what's your name?"), so we extract them here and the
``persist`` node writes them to ``customer_facts``. Once stored, every later
turn loads them back via ``customer_facts`` — so the bot stops re-asking.

Both functions are pure (text in, value out) so they're trivially unit-tested.
"""
from __future__ import annotations

import re

# A pragmatic email match — good enough for WhatsApp-typed addresses, not RFC
# 5322. We take the first address in the message.
_EMAIL_RX = re.compile(r"\b([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b")

# "my name is X", "I'm X", "this is X", "name: X" — capture the trailing words.
_NAME_LEAD_RX = re.compile(
    r"\b(?:my name is|i am|i'm|im|this is|name[:\-]?\s*|name is)\s+([A-Za-z][A-Za-z .'\-]{1,60})$",
    re.IGNORECASE,
)
# Did the assistant's previous line ask for a name? Then a name-shaped reply is
# almost certainly the name (handles the bare "Sandhanapandiyan" case).
_ASKED_FOR_NAME_RX = re.compile(r"\b(full name|your name|name to proceed)\b", re.IGNORECASE)
# A reply that is *just* a name: 1-4 capitalised-or-plain word tokens, letters
# only, no digits/@/punctuation that would signal a query, email or sentence.
_NAME_ONLY_RX = re.compile(r"^[A-Za-z][A-Za-z .'\-]{1,60}$")

# Words that look like a name reply but are really chit-chat / intent, so we
# must NOT store them as a name. Lowercased, whole-string match. (Common
# affirmations like "interested"/"fine" land here — otherwise a reply to
# "what's your name?" gets stored as the name, e.g. the "Hi instrested!" bug.)
_NOT_A_NAME = {
    "hi", "hello", "hey", "yes", "no", "ok", "okay", "thanks", "thank you",
    "yeah", "yep", "nope", "sure", "please", "help", "bye", "good morning",
    "good evening", "good afternoon", "namaste",
    "interested", "not interested", "fine", "good", "great", "cool", "nice",
    "hmm", "hello there", "test", "testing",
}
# If a "name-shaped" reply contains any of these tokens it's an intent/question,
# not a name ("I need a python developer job" is letters-only but not a name).
_INTENT_TOKENS = {
    "job", "jobs", "role", "roles", "developer", "engineer", "apply",
    "application", "status", "salary", "want", "need", "show", "list", "find",
    "search", "looking", "vacancy", "vacancies", "position", "opening",
}


def extract_email(text: str) -> str | None:
    m = _EMAIL_RX.search(text or "")
    return m.group(1).lower() if m else None


def extract_name(text: str, *, assistant_prompt: str | None = None) -> str | None:
    """Best-effort candidate name from the message.

    Two paths: an explicit lead-in ("my name is X") anywhere, or — when the
    assistant just asked for the name — a bare name-shaped reply. Returns the
    cleaned name, or ``None`` if it looks like chit-chat or an intent phrase.
    """
    text = (text or "").strip()
    if not text:
        return None

    lead = _NAME_LEAD_RX.search(text)
    if lead:
        return _clean_name(lead.group(1))

    asked = bool(assistant_prompt and _ASKED_FOR_NAME_RX.search(assistant_prompt))
    if asked and _NAME_ONLY_RX.match(text) and _looks_like_name(text):
        return _clean_name(text)
    return None


def is_plausible_name(value: str | None) -> bool:
    """True if a value actually looks like a person's name.

    Used to validate a *stored* ``full_name`` before trusting it — legacy junk
    like ``"Searching Python Developer Job"`` (captured by an earlier, buggier
    version) should not greet the candidate or make a number look onboarded.
    Self-heals that whole class on the next turn: an implausible cached name is
    dropped and the sender is re-asked.
    """
    if not value:
        return False
    value = value.strip()
    return bool(_NAME_ONLY_RX.match(value)) and _looks_like_name(value)


def _looks_like_name(text: str) -> bool:
    low = text.strip().lower()
    if low in _NOT_A_NAME:
        return False
    tokens = re.findall(r"[a-z']+", low)
    if not tokens or len(tokens) > 4:          # real names are 1-4 words
        return False
    if any(tok in _INTENT_TOKENS for tok in tokens):
        return False
    return True


def _clean_name(raw: str) -> str:
    name = " ".join(raw.split()).strip(" .'-")
    # Title-case only all-lower / all-upper input; leave mixed case (e.g. a
    # legitimately styled name the user typed) as-is.
    if name.islower() or name.isupper():
        name = name.title()
    return name
