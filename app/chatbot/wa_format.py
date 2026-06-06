"""WhatsApp Cloud API message builders.

The graph emits a plain ``response`` string that's perfect for the JSON chat
API but renders as a wall of paragraph text on WhatsApp. This module turns
each reply into the right Cloud API message shape:

    * Greeting / out-of-scope / single product → text + reply buttons
      (interactive type=button, 3 max).
    * Multi-product browse                     → interactive list with up
      to 10 tappable rows.
    * Order card / facet / FAQ / LLM           → markdown text message.

Each response node in the graph attaches the fully-formed payload dict to
``ChatState["whatsapp_message"]``; the WhatsApp route POSTs it verbatim. The
HTTP chat endpoint keeps using the plain ``response`` string, so web and
mobile clients are unaffected.

Cloud API limits (Nov 2024) — anything over is silently rejected by Meta,
so every builder truncates defensively::

    text body              4096
    interactive body       1024
    header                   60
    footer                   60
    list button text         20
    list section title       24
    list row title           24
    list row description     72
    list rows total          10
    reply button title       20
    reply buttons total       3
"""
from __future__ import annotations

from typing import Any, Sequence

# ---------------------------------------------------------------------------
# limits + helpers
# ---------------------------------------------------------------------------

_TEXT_BODY_MAX = 4096
_INTERACTIVE_BODY_MAX = 1024
_HEADER_MAX = 60
_FOOTER_MAX = 60
_LIST_BUTTON_MAX = 20
_LIST_ROW_TITLE_MAX = 24
_LIST_ROW_DESC_MAX = 72
_BUTTON_TITLE_MAX = 20


def _trunc(s: str | None, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)].rstrip() + "…"


def _money(amount: Any) -> str:
    if amount is None:
        return "—"
    try:
        return f"₹{float(amount):,.0f}"
    except (TypeError, ValueError):
        return f"₹{amount}"


# ---------------------------------------------------------------------------
# low-level payload builders
# ---------------------------------------------------------------------------


def text_message(body: str) -> dict[str, Any]:
    return {"type": "text", "text": {"body": _trunc(body, _TEXT_BODY_MAX)}}


def image_message(caption: str, image_url: str) -> dict[str, Any]:
    """Image message with a markdown caption — no buttons. The caption shares
    the interactive body limit (~1024). ``image_url`` must be a JPEG/PNG https
    link (Meta rejects http and webp)."""
    return {
        "type": "image",
        "image": {"link": image_url, "caption": _trunc(caption, _INTERACTIVE_BODY_MAX)},
    }


def buttons_message(
    body: str,
    buttons: Sequence[tuple[str, str]],
    *,
    image_url: str | None = None,
) -> dict[str, Any]:
    """Reply buttons with an optional image header.

    ``buttons`` is a sequence of ``(id, title)`` tuples. Max 3 — extras are
    dropped to stay within Meta's limit. When ``image_url`` is provided it
    is set as the header (Cloud API supports image/video/document/text
    headers on reply-button messages)."""
    items = [
        {
            "type": "reply",
            "reply": {"id": bid, "title": _trunc(title, _BUTTON_TITLE_MAX)},
        }
        for bid, title in list(buttons)[:3]
    ]
    interactive: dict[str, Any] = {
        "type": "button",
        "body": {"text": _trunc(body, _INTERACTIVE_BODY_MAX)},
        "action": {"buttons": items},
    }
    if image_url:
        interactive["header"] = {"type": "image", "image": {"link": image_url}}
    return {"type": "interactive", "interactive": interactive}


def cta_url_message(
    *,
    body: str,
    display_text: str,
    url: str,
    image_url: str | None = None,
) -> dict[str, Any]:
    """Interactive cta_url message — one tap opens the URL in the browser.

    Replaces reply buttons when a store product URL is configured; better
    for conversion since the customer lands on the PDP directly."""
    interactive: dict[str, Any] = {
        "type": "cta_url",
        "body": {"text": _trunc(body, _INTERACTIVE_BODY_MAX)},
        "action": {
            "name": "cta_url",
            "parameters": {
                "display_text": _trunc(display_text, _BUTTON_TITLE_MAX),
                "url": url,
            },
        },
    }
    if image_url:
        interactive["header"] = {"type": "image", "image": {"link": image_url}}
    return {"type": "interactive", "interactive": interactive}


def interactive_fallback_text(payload: dict[str, Any]) -> str:
    """Plain-text fallback for an interactive payload (used when Meta rejects it,
    e.g. cta_url/list not enabled). Returns the body text; for a cta_url button it
    appends the URL so the link isn't lost."""
    interactive = (payload or {}).get("interactive") or {}
    body = ((interactive.get("body") or {}).get("text") or "").strip()
    if interactive.get("type") == "cta_url":
        url = ((interactive.get("action") or {}).get("parameters") or {}).get("url")
        if url:
            body = f"{body}\n{url}".strip()
    return body


def first_image_url(row: dict[str, Any]) -> str | None:
    """Best-effort extract of a usable https product image URL.

    The Prisma schema stores images as JSONB; in practice we've seen three
    shapes:
        ["https://…", "https://…"]
        [{"url": "https://…"}, …]
        {"primary": "https://…", "thumbnail": "https://…"}
    Returns the first https URL found or None. Meta rejects http URLs.
    """
    raw = row.get("images")
    if not raw:
        return None
    candidates: list[Any] = []
    if isinstance(raw, list):
        candidates = list(raw)
    elif isinstance(raw, dict):
        candidates = list(raw.values())
    elif isinstance(raw, str):
        candidates = [raw]
    for c in candidates:
        url: str | None = None
        if isinstance(c, str):
            url = c
        elif isinstance(c, dict):
            url = c.get("url") or c.get("link") or c.get("src")
        if url and isinstance(url, str) and url.startswith("https://"):
            return url
    return None


_WA_HEADER_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def whatsapp_header_image_url(row: dict[str, Any]) -> str | None:
    """Return a product image URL only if WhatsApp can render it as a message
    header. Meta supports JPEG/PNG headers — webp/svg/gif get the *entire*
    interactive message rejected with a 400, so we drop the header instead of
    losing the whole reply. Many catalogue images are webp, hence this guard.
    """
    url = first_image_url(row)
    if not url:
        return None
    path = url.split("?")[0].lower()
    return url if path.endswith(_WA_HEADER_IMAGE_EXTS) else None


def render_product_url(template: str, row: dict[str, Any]) -> str | None:
    """Format the store URL template with the product's id/slug."""
    if not template:
        return None
    title = str(row.get("title") or "")
    slug = "".join(
        ch.lower() if ch.isalnum() else "-" for ch in title
    ).strip("-")
    try:
        return template.format(id=row.get("id"), slug=slug or row.get("id"))
    except (KeyError, IndexError):
        return None


def list_message(
    *,
    body: str,
    button_text: str,
    rows: Sequence[dict[str, str]],
    header: str | None = None,
    footer: str | None = None,
    section_title: str = "Matches",
) -> dict[str, Any]:
    """Build an interactive *list* message. Each row dict must have ``id``
    and ``title``; ``description`` is optional. Up to 10 rows."""
    out_rows = []
    for r in list(rows)[:10]:
        item: dict[str, str] = {
            "id": _trunc(r["id"], 200),
            "title": _trunc(r["title"], _LIST_ROW_TITLE_MAX),
        }
        if r.get("description"):
            item["description"] = _trunc(r["description"], _LIST_ROW_DESC_MAX)
        out_rows.append(item)

    interactive: dict[str, Any] = {
        "type": "list",
        "body": {"text": _trunc(body, _INTERACTIVE_BODY_MAX)},
        "action": {
            "button": _trunc(button_text, _LIST_BUTTON_MAX),
            "sections": [
                {
                    "title": _trunc(section_title, _LIST_ROW_TITLE_MAX),
                    "rows": out_rows,
                }
            ],
        },
    }
    if header:
        interactive["header"] = {"type": "text", "text": _trunc(header, _HEADER_MAX)}
    if footer:
        interactive["footer"] = {"text": _trunc(footer, _FOOTER_MAX)}
    return {"type": "interactive", "interactive": interactive}


# ---------------------------------------------------------------------------
# domain renderers — used by graph response nodes
# ---------------------------------------------------------------------------


def product_card_markdown(row: dict[str, Any]) -> str:
    """Single-product markdown card. Used as the body of a buttons message
    and for image-match replies, so it carries the full product detail set:
    title, price, category, fabric, colors, sizes, stock, and a short
    description when present."""
    title = row.get("title") or "this product"
    price = _money(row.get("base_price"))
    mrp = row.get("suggested_mrp")
    price_line = f"*{price}*"
    try:
        if mrp and float(mrp) != float(row.get("base_price") or 0):
            price_line += f"  ~{_money(mrp)}~"
    except (TypeError, ValueError):
        pass

    colors = ", ".join(row.get("available_colors") or []) or "—"
    sizes = ", ".join(row.get("available_sizes") or []) or "—"
    stock = int(row.get("total_stock", 0) or 0)
    stock_line = f"{stock} in stock" if stock > 0 else "Out of stock"

    lines = [
        f"*{title}*",
        price_line,
        f"_{row.get('category_name') or 'Uncategorised'}_",
    ]
    if row.get("fabric"):
        lines.append(f"*Fabric:* {row['fabric']}")
    lines.append(f"*Colors:* {colors}")
    lines.append(f"*Sizes:*  {sizes}")
    lines.append(stock_line)
    if row.get("description"):
        desc = " ".join(str(row["description"]).split())  # collapse whitespace
        lines.append("")
        lines.append(f"_{_trunc(desc, 280)}_")
    return "\n".join(lines)


def product_list_markdown(rows: Sequence[dict[str, Any]]) -> str:
    """Markdown product list — used as the ``response`` string (HTTP chat
    + WhatsApp text fallback when interactive lists aren't appropriate)."""
    lines = ["*Here are matches from our catalogue:*", ""]
    for i, r in enumerate(rows[:5], 1):
        price = _money(r.get("base_price"))
        mrp = r.get("suggested_mrp")
        price_str = f"*{price}*"
        try:
            if mrp and float(mrp) != float(r.get("base_price") or 0):
                price_str += f"  ~{_money(mrp)}~"
        except (TypeError, ValueError):
            pass
        title = r.get("title") or "(no title)"
        cat = r.get("category_name") or "—"
        stock = int(r.get("total_stock", 0) or 0)
        stock_line = f"{stock} in stock" if stock > 0 else "Out of stock"
        lines += [
            f"*{i}. {title}*",
            f"  {price_str}",
            f"  _{cat}_",
            f"  {stock_line}",
            "",
        ]
    if len(rows) > 5:
        lines.append(f"_…and {len(rows) - 5} more._")
    return "\n".join(lines).rstrip()


def product_list_payload(
    rows: Sequence[dict[str, Any]], *, query: str
) -> dict[str, Any]:
    """Interactive *list* payload — multi-product browse."""
    items = []
    for r in rows[:10]:
        price = _money(r.get("base_price"))
        stock = int(r.get("total_stock", 0) or 0)
        stock_note = f"{stock} in stock" if stock > 0 else "Out of stock"
        items.append(
            {
                "id": f"product:{r.get('id')}",
                "title": r.get("title") or "(no title)",
                "description": f"{price} · {stock_note}",
            }
        )
    n = min(len(rows), 10)
    body = f"Found {len(rows)} match{'es' if len(rows) != 1 else ''} for *{_trunc(query, 40)}* — tap one for details:"
    return list_message(
        body=body,
        button_text="View matches",
        rows=items,
        header="Catalogue matches",
        section_title=f"Top {n}",
    )


def order_card_markdown(row: dict[str, Any]) -> str:
    """Single-order tracking card."""
    status = str(row.get("order_status") or "—")
    pay = row.get("payment_status") or "—"
    total = _money(row.get("total_amount"))
    lines = [
        f"*Order {row.get('order_number')}*",
        f"*Status:*   {status}",
        f"*Payment:*  {pay}",
        f"*Total:*    {total}",
    ]
    if row.get("tracking_number"):
        lines.append(f"*Tracking:* `{row['tracking_number']}`")
    if row.get("courier_name"):
        lines.append(f"*Courier:*  {row['courier_name']}")
    if row.get("tracking_url"):
        lines.append(row["tracking_url"])
    return "\n".join(lines)


def orders_markdown(rows: Sequence[dict[str, Any]]) -> str:
    """Order tracking — single or list."""
    if not rows:
        return (
            "I couldn't find any orders linked to this WhatsApp number.\n"
            "Please share your order number (e.g. _SS-XX0000000000_) and I'll look it up."
        )
    if len(rows) == 1:
        return order_card_markdown(rows[0])
    lines = [f"*Found {len(rows)} orders on this number:*", ""]
    for r in rows:
        placed = r.get("created_at")
        placed_str = placed.strftime("%Y-%m-%d") if placed else "—"
        lines.append(
            f"• *{r.get('order_number')}*  "
            f"·  {r.get('order_status')}  "
            f"·  {_money(r.get('total_amount'))}  "
            f"·  _{placed_str}_"
        )
    return "\n".join(lines)
