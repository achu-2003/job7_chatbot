# MCP layer

Model Context Protocol support for the conversational-commerce assistant.

The store's capabilities are defined **once** in [`tools.py`](./tools.py) and
consumed two ways:

| Consumer | File | Who calls it | Transport |
|---|---|---|---|
| **In-process agent** | `app/chatbot/agent.py` | the WhatsApp bot itself | in-process (no network hop) |
| **Standalone server** | `app/mcp/server.py` | external agents (Claude Desktop, MCP Inspector, staff tools) | MCP protocol (stdio) |

Same tools, two trust boundaries (see *Security* below).

## Tools

| Tool | What it does | Backed by |
|---|---|---|
| `search_products` | Semantic catalogue search → authoritative price/stock from Postgres | `VectorStore` + `ProductRepository` |
| `get_order_status` | One order's status / courier / tracking by number | `OrderRepository.get_by_number` |
| `get_recent_orders` | The customer's recent orders | `OrderRepository.latest_for_customer` |
| `search_policies` | Returns / refund / shipping policy lookup | `VectorStore` (policies) |
| `search_faq` | FAQ lookup | `VectorStore` (faqs) |
| `request_human_handoff` | Escalate returns / refunds / complaints to a human | `EscalationService` |

> **Why no `create_return` / `cancel_order`?** `app/db/repositories.py` is
> read-only by design (with a SQL-safety guard). Rather than add write paths to
> the production orders table, those actions route to a human via
> `request_human_handoff`. Add real write tools later if/when you add audited
> write repositories.

## How the WhatsApp bot uses the agent

The agent is the **single brain** for every message (`app/chatbot/graph.py`):

```
msg → load_memory ─(greeting?)─► greeting_response   (regex, 0 LLM calls)
                  └─(else)──────► agent_response       (MCP tool loop)
                                       │
                                       └─ render products → WhatsApp card / list
```

- **Greeting fast-path:** a one-line regex catches "hi"/"hello" and replies
  from a fixed string — the cheapest, most common turn stays at **zero** LLM
  calls.
- **Everything else** goes to the agent, which picks tools itself (no intent
  regex, no templates). It's seeded with the last-discussed product so facet
  follow-ups ("what colors?") don't always need a fresh search.
- **Product rendering (no buttons):** the agent returns text; when it surfaced
  products the graph re-hydrates them from Postgres and replies with a product
  **card + image** (1 match) or a **markdown list** (many). No reply buttons,
  CTA links, or interactive lists.

```bash
# .env — the agent is always on; just tune it
MCP_AGENT_MAX_ITERATIONS=4    # LLM↔tool round-trips cap (last step forces an answer)
MCP_AGENT_MODEL=              # empty → LLM_MODEL_CHAT (llama-3.1-8b-instant on Groq)
```

Single model, single provider: **`llama-3.1-8b-instant` on Groq** for both chat
and the agent loop.

> **Two caveats with an 8B model:** (1) small models pick tools less reliably,
> so the agent may occasionally mis-call or skip a tool; (2) every non-greeting
> message costs ≥1 LLM call (more when it chains tools), so on Groq's free tier
> watch for 429s — the agent degrades to a polite "try again" rather than
> crashing. The greeting fast-path is the main built-in cost saver.

## Running the standalone MCP server

For external agents / staff tools (this surface takes an explicit
`customer_phone` for order tools — it's for **trusted** operators):

```bash
python -m app.mcp.server          # stdio (default)
```

Point Claude Desktop at it (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ecom-store": {
      "command": "/abs/path/.venv/bin/python",
      "args": ["-m", "app.mcp.server"],
      "cwd": "/abs/path/to/Chatbot Rag",
      "env": { "TENANT_ID": "default" }
    }
  }
}
```

The server reads the same `.env` (needs Postgres + Qdrant reachable to answer).

## Security

`tenant_id` and the customer's phone are **never** model-controlled in the
customer-facing path. The agent injects them via `ToolContext` at dispatch
time, and the order tools deliberately omit `customer_ref` from their JSON
schema — so a customer can only ever read **their own** orders, never someone
else's by spoofing a number. The standalone server, used by trusted staff,
exposes `customer_phone` explicitly instead.
```
