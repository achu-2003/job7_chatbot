"""humanizer node — turn one reply into paced WhatsApp bubbles.

A wall of text reads like a bot. Humans send a couple of short messages with
little pauses. ``build_delivery_plan`` (pure, testable) splits the reply at
natural boundaries into ≤ ``max_chunks`` bubbles and assigns each a "typing"
delay proportional to its length (clamped), which the delivery layer turns into
real pacing.
"""
from __future__ import annotations

import re
from typing import Any

# Sentence boundary: keep the punctuation with the sentence.
_SENTENCE_RX = re.compile(r"(?<=[.!?])\s+")


def build_delivery_plan(
    text: str,
    *,
    max_chars: int,
    max_chunks: int,
    cps: float,
    min_ms: int,
    max_ms: int,
) -> list[dict[str, Any]]:
    """Return ``[{text, typing_ms}, ...]`` — the bubbles to send, in order."""
    chunks = _split(text, max_chars=max_chars, max_chunks=max_chunks)
    plan: list[dict[str, Any]] = []
    for chunk in chunks:
        typing_ms = int(min(max_ms, max(min_ms, len(chunk) / max(cps, 1.0) * 1000)))
        plan.append({"text": chunk, "typing_ms": typing_ms})
    return plan


def _split(text: str, *, max_chars: int, max_chunks: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    # 1) paragraphs, then over-long paragraphs broken on sentence boundaries
    pieces: list[str] = []
    for para in (p.strip() for p in text.split("\n\n") if p.strip()):
        if len(para) <= max_chars:
            pieces.append(para)
        else:
            pieces.extend(_pack_sentences(para, max_chars))

    # 2) greedily merge adjacent pieces that still fit (fewer, fuller bubbles)
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) + 1 + len(piece) <= max_chars:
            merged[-1] = f"{merged[-1]}\n{piece}"
        else:
            merged.append(piece)

    # 3) cap the bubble count — fold the overflow into the last bubble
    if len(merged) > max_chunks:
        head = merged[: max_chunks - 1]
        tail = "\n".join(merged[max_chunks - 1:])
        merged = head + [tail]
    return merged


def _pack_sentences(para: str, max_chars: int) -> list[str]:
    out: list[str] = []
    current = ""
    for sentence in _SENTENCE_RX.split(para):
        sentence = sentence.strip()
        if not sentence:
            continue
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            out.append(current)
            current = sentence
    if current:
        out.append(current)
    # a single sentence longer than max_chars: hard-wrap as a last resort
    wrapped: list[str] = []
    for chunk in out:
        while len(chunk) > max_chars:
            wrapped.append(chunk[:max_chars])
            chunk = chunk[max_chars:]
        wrapped.append(chunk)
    return wrapped
