"""Structured logging via structlog.

JSON output (``JSON_LOGS=true``) is used in production for log aggregators.
The dev-friendly pretty renderer (``JSON_LOGS=false``) prints each log
record as a header line plus one indented ``key  value`` line per field,
with long values truncated and a blank line between records.

Example dev output::

    13:42:55  INFO   chat_request
      conversation_id  wa_919600648314
      intent           product_search
      latency_ms       1011
      response         "Found 5 orders on this number: …"
"""
from __future__ import annotations

import logging
import os
import re
import sys
from contextvars import ContextVar
from typing import Any

import structlog

from app.config import get_settings

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


# ---------------------------------------------------------------------------
# pretty renderer for dev output
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

_RESET = "\x1b[0m" if _USE_COLOR else ""
_DIM = "\x1b[2m" if _USE_COLOR else ""
_BOLD = "\x1b[1m" if _USE_COLOR else ""
_LEVEL_COLORS = {
    "debug": "\x1b[2m",
    "info": "\x1b[36m",       # cyan
    "warning": "\x1b[33m",    # yellow
    "error": "\x1b[31m",      # red
    "critical": "\x1b[1;31m", # bold red
} if _USE_COLOR else {}

# Single-line values longer than this get truncated. Multi-line values
# (containing newlines) bypass the cap and render as a sub-block under
# the key — useful for prompts and TOON-encoded context dumps.
_MAX_FIELD_LEN = 600

# Fields that aren't worth showing in dev (already represented elsewhere
# or just noise). We still emit them in JSON mode.
_SUPPRESS_KEYS_IN_DEV = {
    "stack_info",
}

# Width of the *** separator drawn around each conversation_turn block.
_CONV_SEP = "*" * 60


def _pretty_console_renderer(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> str:
    ts = event_dict.pop("timestamp", "")
    short_ts = ts[11:19] if "T" in ts else ts        # HH:MM:SS
    level = str(event_dict.pop("level", "info")).lower()
    event = str(event_dict.pop("event", ""))

    # Special case: a finished chat turn rendered as one separated block.
    # Timestamp appears once on the header line; the block body has its
    # own labels and never repeats the time.
    if event == "conversation_turn":
        block = str(event_dict.pop("block", "")).rstrip()
        lines = [_CONV_SEP, f"{_DIM}{short_ts}{_RESET}"]
        for sub in block.splitlines():
            lines.append(f"  {sub}")
        lines.append(_CONV_SEP)
        lines.append("")  # trailing blank line for breathing room
        return "\n".join(lines) + "\n"

    color_l = _LEVEL_COLORS.get(level, "")
    header = (
        f"{_DIM}{short_ts}{_RESET}  "
        f"{color_l}{level.upper():<7}{_RESET}  "
        f"{_BOLD}{event}{_RESET}"
    )

    # Strip noise + compute alignment width
    items = [
        (k, v) for k, v in event_dict.items()
        if k not in _SUPPRESS_KEYS_IN_DEV
    ]
    if not items:
        return header + "\n"

    max_k = min(max(len(k) for k, _ in items), 24)
    lines = [header]
    # Sort for stable output; request_id pinned first if present.
    items.sort(key=lambda kv: (kv[0] != "request_id", kv[0]))
    for k, v in items:
        s = v if isinstance(v, str) else repr(v)
        if "\n" in s:
            # Multi-line value (prompt, TOON context, formatted hits) →
            # print the key on its own line, then each value line indented
            # under it. No truncation: these blocks are the whole point of
            # detailed logs.
            lines.append(f"  {_DIM}{k}{_RESET}:")
            for sub in s.splitlines() or [""]:
                lines.append(f"      {sub}")
        else:
            if len(s) > _MAX_FIELD_LEN:
                s = s[: _MAX_FIELD_LEN - 1].rstrip() + "…"
            lines.append(f"  {_DIM}{k:<{max_k}}{_RESET}  {s}")
    # blank line after each record for breathing room
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------
# Every log record is scrubbed before rendering so phone numbers, names, tokens
# and secrets never land in the logs (or a log aggregator) in the clear. Keyed by
# FIELD NAME (not by scanning arbitrary text) to avoid masking unrelated numbers
# like amounts or timestamps. Recurses into nested dicts/lists (e.g. the raw Meta
# webhook payload) up to a bounded depth.

_SECRET_KEY_RX = re.compile(
    r"(secret|password|authorization|api[_-]?key|access[_-]?token|_token$|^token$"
    r"|signature|key_secret)",
    re.I,
)
# Keys whose value is a phone number → keep country code + last 4, mask the middle.
_PHONE_KEYS = frozenset({
    "phone", "phone_number", "customer_number", "customer_id", "customer",
    "customer_external_id", "from", "from_", "to", "to_number", "wa_id",
    "msisdn", "recipient", "display_phone_number",
})
# Keys whose value is a person/company name → keep the first char only.
_NAME_KEYS = frozenset({
    "full_name", "first_name", "last_name", "company_name", "companyname",
})
# Masked only inside a webhook-payload subtree (the contact's "name" is PII there,
# but a bare "name" elsewhere — a plan/category/logger name — is not).
_DEEP_NAME_KEYS = frozenset({"name"})
_PAYLOAD_ROOT_KEYS = frozenset({"payload"})
# Deep enough to reach phones/names inside a Meta webhook payload
# (payload → entry[] → changes[] → value → contacts[] → profile → name ≈ 10),
# but bounded so a pathological structure can't recurse without end.
_REDACT_MAX_DEPTH = 14


def _mask_phone(v: Any) -> str:
    digits = re.sub(r"\D", "", str(v))
    if len(digits) < 7:
        return "***"
    return digits[:2] + "*" * (len(digits) - 6) + digits[-4:]


def _mask_name(v: Any) -> str:
    s = str(v).strip()
    return (s[0] + "***") if s else "***"


def _scrub(obj: Any, depth: int = 0, in_payload: bool = False) -> Any:
    if depth >= _REDACT_MAX_DEPTH:
        return obj
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, val in obj.items():
            kl = str(k).lower()
            child_in_payload = in_payload or kl in _PAYLOAD_ROOT_KEYS
            if _SECRET_KEY_RX.search(kl):
                out[k] = "***"
            elif kl in _PHONE_KEYS and val not in (None, ""):
                out[k] = _mask_phone(val)
            elif (kl in _NAME_KEYS or (child_in_payload and kl in _DEEP_NAME_KEYS)) \
                    and isinstance(val, str) and val:
                out[k] = _mask_name(val)
            else:
                out[k] = _scrub(val, depth + 1, child_in_payload)
        return out
    if isinstance(obj, (list, tuple)):
        return [_scrub(x, depth + 1, in_payload) for x in obj]
    return obj


def _redact_pii(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    return _scrub(event_dict, 0)


# ---------------------------------------------------------------------------
# request-id middleware hook
# ---------------------------------------------------------------------------

def _add_request_id(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    rid = request_id_var.get()
    if rid is not None:
        event_dict.setdefault("request_id", rid)
    return event_dict


# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    settings = get_settings()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )
    # Silence noisy libs
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_request_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _redact_pii,                 # scrub PII/secrets before any rendering
    ]
    if settings.json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(_pretty_console_renderer)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
