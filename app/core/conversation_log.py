"""Per-turn conversation tracer.

Collects compact ``(label, value)`` events through a single chat turn and
emits them as one structured log record at the end. The pretty renderer
(``app.core.logging._pretty_console_renderer``) detects the special
``conversation_turn`` event name and prints a separator + block; JSON output
keeps the structured shape unchanged.

Lifecycle::

    start()                    # at top of webhook / chat handler
    note("webhook", "text")
    note("prompt",  customer_message)
    ...                        # branches call note(...) along the way
    note("reply to", to_number)
    flush()                    # emits the single log record

The collector is ContextVar-scoped so concurrent requests don't bleed into
each other.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from app.core.logging import get_logger

_log = get_logger("chat")

_events: ContextVar[list[tuple[str, str]] | None] = ContextVar(
    "_conv_events", default=None
)


def start() -> None:
    _events.set([])


def note(label: str, value: Any) -> None:
    events = _events.get()
    if events is None:
        return
    events.append((str(label), str(value)))


def events() -> list[tuple[str, str]]:
    """Snapshot of the events collected so far this turn (for the debug UI)."""
    return list(_events.get() or [])


def flush() -> None:
    events = _events.get()
    _events.set(None)
    if not events:
        return
    pad = max(len(lbl) for lbl, _ in events)
    block = "\n".join(f"{lbl:<{pad}}  {val}" for lbl, val in events)
    _log.info("conversation_turn", block=block)
