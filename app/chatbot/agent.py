"""MCP tool-calling agent — the single responder for every non-greeting turn.

The graph (``app.chatbot.graph``) sends every message here except pure
greetings (which take a 0-LLM fast-path). The loop:

    1. seeds the model with conversation history + the last-discussed product
       (so facet follow-ups like "what colors?" don't always need a search),
    2. lets the model call MCP tools (search_products, get_order_status,
       request_human_handoff, …) as needed,
    3. feeds each tool result back and repeats, up to ``max_iterations``,
    4. validates the final reply against everything it saw, best-effort,
    5. surfaces any products it found so the graph can render a WhatsApp
       card/list.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.chatbot.toon import encode as toon_encode
from app.chatbot.validator import HallucinationValidator
from app.config import get_settings
from app.core import conversation_log as conv
from app.core.logging import get_logger
from app.llm.client import LLMClient
from app.mcp.tools import ToolContext, ToolRegistry

log = get_logger("agent")

AGENT_SYSTEM_PROMPT = """\
You are a warm, friendly shopping assistant for an eCommerce store, chatting with
a customer on WhatsApp. Talk like a real, helpful human sales assistant — natural
and conversational — NOT like a form, a database dump, or a list of fields.

You're given CONTEXT retrieved from the store's live catalogue (semantic search
over our product database). You can also call tools for more: search the
catalogue again with filters, look up the customer's OWN orders, read
policies/FAQs, or hand off to a human.

How to reply:
- Ground every answer in the CONTEXT and tool results. Lead with a friendly,
  natural sentence, then the key details (price, what's in stock), and end with a
  helpful nudge or question ("want me to show more?", "shall I check sizes?").
- Recommend like a good salesperson. No emojis. Keep it concise
  (WhatsApp-style markdown), a few short lines.
- For returns, refunds, cancellations, complaints, or anything you can't do
  yourself, call request_human_handoff with a short summary.
- If you genuinely can't find something, say so warmly and offer to help another way.

IMPORTANT — never make things up:
- ONLY use product info from the CONTEXT and tools. NEVER invent prices, stock,
  colours, sizes, variants, order numbers, tracking, delivery dates or discounts.
- Quote prices and stock exactly as given.
- Only ever discuss the current customer's own orders.\
"""

# Returned when the LLM is unavailable (rate-limited, timeout, provider down)
# so a transient failure degrades to a polite ask-again instead of a 500.
_BUSY_REPLY = (
    "I'm handling a lot of messages right now and couldn't get to yours — "
    "please send it again in a few seconds."
)


@dataclass
class AgentResult:
    response: str
    used_llm: bool = True
    tool_calls: int = 0
    # Compact products (with id) from the agent's last search_products call —
    # the graph re-hydrates these into a WhatsApp card/list.
    products: list[dict[str, Any]] = field(default_factory=list)


class MCPAgent:
    """Drives the LLM ⇄ MCP-tool loop for a single turn."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        registry: ToolRegistry,
        validator: HallucinationValidator | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.validator = validator or HallucinationValidator()

    async def run(
        self,
        *,
        customer_query: str,
        history: list[dict[str, str]],
        ctx: ToolContext,
        cached_product: dict[str, str] | None = None,
        memory_context: str | None = None,
        seed_sql_rows: list[dict[str, Any]] | None = None,
        seed_vector_hits: list[dict[str, Any]] | None = None,
    ) -> AgentResult:
        s = get_settings()
        max_iterations = max(1, s.mcp_agent_max_iterations)
        model = s.mcp_agent_model or s.llm_model_chat
        tools = self.registry.openai_schemas()

        messages = self._seed_messages(
            customer_query, history, cached_product, memory_context,
            seed_sql_rows, seed_vector_hits,
        )
        # Authoritative rows the model has been shown — grows as tools return,
        # then feeds the grounding validator at the end.
        grounding: list[dict[str, Any]] = list(seed_sql_rows or [])
        # Products from the most recent non-empty search — the graph renders
        # these into a WhatsApp card/list.
        surfaced_products: list[dict[str, Any]] = []
        total_tool_calls = 0

        for step in range(max_iterations):
            # On the final allowed step, drop tools so the model is forced to
            # answer in words instead of asking for yet another tool.
            last_step = step == max_iterations - 1
            try:
                message, _ = await self.llm.chat_with_tools(
                    purpose="agent_step",
                    messages=messages,
                    tools=tools,
                    tool_choice="none" if last_step else "auto",
                    model=model,
                    temperature=0.2,
                )
            except Exception as exc:  # noqa: BLE001 — never 500 the turn on an LLM error
                log.warning(
                    "agent_llm_unavailable",
                    request_id=ctx.request_id,
                    error=str(exc)[:200],
                )
                return self._finalize(
                    _BUSY_REPLY, customer_query, grounding, seed_vector_hits,
                    surfaced_products, total_tool_calls,
                )
            tool_calls = getattr(message, "tool_calls", None) or []

            if not tool_calls:
                text = (message.content or "").strip()
                return self._finalize(
                    text, customer_query, grounding, seed_vector_hits,
                    surfaced_products, total_tool_calls,
                )

            # Record the assistant's tool-call turn, then run each tool.
            messages.append(_assistant_tool_msg(message, tool_calls))
            for tc in tool_calls:
                total_tool_calls += 1
                name = tc.function.name
                args = _safe_json(tc.function.arguments)
                conv.note("tool", f"{name}({_compact_args(args)})")
                result = await self.registry.dispatch(name, args, ctx)
                _accumulate_grounding(grounding, name, result)
                if name == "search_products" and isinstance(result, list) and result:
                    surfaced_products = result
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    }
                )

        # Exhausted iterations without a textual answer — shouldn't happen
        # because the last step disables tools, but guard anyway.
        return self._finalize(
            "I'm sorry, I couldn't complete that just now. Could you rephrase?",
            customer_query, grounding, seed_vector_hits, surfaced_products, total_tool_calls,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _seed_messages(
        self,
        customer_query: str,
        history: list[dict[str, str]],
        cached_product: dict[str, str] | None,
        memory_context: str | None,
        seed_sql_rows: list[dict[str, Any]] | None,
        seed_vector_hits: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]
        for h in history[-_history_turns():]:
            role, content = h.get("role"), h.get("content")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        parts: list[str] = []
        # Durable memory about this customer (summary + facts + recalled notes),
        # so the agent stays continuous across sessions.
        if memory_context:
            parts.append(f"WHAT YOU REMEMBER ABOUT THIS CUSTOMER:\n{memory_context}")
        # The product discussed last turn — lets facet follow-ups ("what
        # colors?", "in blue") be answered without re-searching.
        if cached_product and cached_product.get("doc"):
            parts.append(f"RECENTLY DISCUSSED PRODUCT:\n{cached_product['doc']}")
        context_block = self._build_context(seed_sql_rows, seed_vector_hits)
        if context_block:
            parts.append(f"CONTEXT (TOON):\n{context_block}")
        parts.append(f"CUSTOMER QUERY: {customer_query}")

        messages.append({"role": "user", "content": "\n\n".join(parts)})
        return messages

    @staticmethod
    def _build_context(
        seed_sql_rows: list[dict[str, Any]] | None,
        seed_vector_hits: list[dict[str, Any]] | None,
    ) -> str:
        rows = (seed_sql_rows or [])[:3]
        hits = (seed_vector_hits or [])[:3]
        if not rows and not hits:
            return ""
        # Flatten colors/sizes to strings so TOON uses its compact *table* form
        # (list fields would force the verbose record-per-block fallback).
        products = [
            {
                "title": r.get("title"),
                "price": r.get("base_price"),
                "mrp": r.get("suggested_mrp"),
                "category": r.get("category_name"),
                "colors": ", ".join(r.get("available_colors") or []) or "-",
                "sizes": ", ".join(r.get("available_sizes") or []) or "-",
                "stock": r.get("total_stock", 0),
            }
            for r in rows
        ]
        documents = [
            {"title": (h.get("metadata") or {}).get("title"), "doc": (h.get("document") or "")[:300]}
            for h in hits
        ]
        return toon_encode({"products": products, "documents": documents})

    def _finalize(
        self,
        text: str,
        customer_query: str,
        grounding: list[dict[str, Any]],
        seed_vector_hits: list[dict[str, Any]] | None,
        products: list[dict[str, Any]],
        tool_calls: int,
    ) -> AgentResult:
        if not text:
            text = "I couldn't find an answer for that. Could you give me more detail?"
        v = self.validator.validate(
            text,
            sql_rows=grounding,
            vector_hits=seed_vector_hits or [],
            customer_query=customer_query,
        )
        if not v.valid:
            log.warning("agent_response_invalid", offending=v.offending)
        return AgentResult(
            response=text, used_llm=True, tool_calls=tool_calls, products=products,
        )


# ---------------------------------------------------------------------------
# module helpers
# ---------------------------------------------------------------------------


def _history_turns() -> int:
    # Last 4 turns (~2 exchanges). Kept small on purpose: every token in the
    # prompt counts against the provider's tokens-per-minute budget, and the
    # tool schemas + system prompt are already a fixed overhead each call.
    return 4


def _safe_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        log.warning("agent_bad_tool_args", raw=(raw or "")[:200])
        return {}


def _compact_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())[:120]


def _assistant_tool_msg(message: Any, tool_calls: list[Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ],
    }


def _accumulate_grounding(
    grounding: list[dict[str, Any]], tool_name: str, result: Any
) -> None:
    """Pull authoritative rows out of a tool result so the final reply can be
    grounding-checked. Maps the compact product shape onto the keys the
    HallucinationValidator understands (it reads ``price`` / ``suggested_mrp``).
    """
    if tool_name == "search_products" and isinstance(result, list):
        for p in result:
            if isinstance(p, dict):
                grounding.append(
                    {
                        "title": p.get("title"),
                        "price": p.get("price"),
                        "suggested_mrp": p.get("mrp"),
                    }
                )
    elif tool_name == "get_order_status" and isinstance(result, dict):
        order = result.get("order")
        if isinstance(order, dict):
            grounding.append(order)
    elif tool_name == "get_recent_orders" and isinstance(result, list):
        grounding.extend(o for o in result if isinstance(o, dict))
