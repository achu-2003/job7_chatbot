# WhatsApp AI Agent — Production Architecture

A stateful, autonomous WhatsApp agent built on a **planner → executor → reflection**
loop with four-tier memory, dynamic MCP tool routing, and human-like conversational
pacing. This document is the build contract: folder layout, agent state, LangGraph
flow, memory schema, prompt architecture, tool/retry strategy, and a phased roadmap.

It is an **evolution of the existing codebase**, not a rewrite. We reuse: FastAPI +
lifespan, LangGraph, Qdrant `VectorStore` (hybrid search), Redis `ConversationMemory`,
the MCP `ToolRegistry`, `LLMClient` (Groq, graceful 429), `HallucinationValidator`,
structlog + Prometheus, multi-tenancy (ContextVar), the Redis worker/queue, and the
WhatsApp webhook (which already sends read receipts + typing indicators).

---

## 1. Design principles

1. **Stateful, not request-response.** Every turn loads and updates durable agent
   state (goals, plan, memory, session). The reply is a side effect of advancing a
   goal, not a one-shot answer.
2. **Reason in steps, reflect, retry.** The model plans, executes, then critiques
   its own result and decides whether to retry, replan, ask the user, or hand off.
3. **Ground everything.** Replies are grounded in retrieved memory + tool results;
   the `HallucinationValidator` gates output. No invented prices/orders/facts.
4. **Two DBs, two trust levels.** Business data (products/orders) stays **read-only**
   (Prisma-owned). The agent's memory/goals/sessions live in **agent-owned**
   read-write tables (`agent_*`). Never mix.
5. **Human pacing is a feature.** Chunked messages, typing delays, acknowledgements,
   and natural phrasing are first-class, not afterthoughts.
6. **Bounded autonomy.** Every loop has iteration, token, and wall-clock budgets, plus
   a kill-switch to graceful degradation.

---

## 2. High-level architecture

```
                WhatsApp Cloud API  ◄────── outbound: chunked + paced
                       │
                       ▼  webhook (idempotent, deduped by message id)
              ┌──────────────────┐
              │  FastAPI edge     │  bind tenant, ack fast, enqueue
              └──────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │      LangGraph Agent Runtime       │
        │  load_context → summarize → intent │
        │      → PLAN ⇄ ROUTE ⇄ EXECUTE      │
        │              ⇄ REFLECT             │
        │      → respond → humanize → persist│
        └──────────────────────────────────┘
            │            │             │
        ┌───┘        ┌───┘         ┌───┘
        ▼            ▼             ▼
   MCP tools     Memory         Delivery
  (products,   (Redis STM,    (chunk, typing,
   orders,      Postgres        pacing, acks)
   policies,    episodic/goals,
   handoff,     Qdrant semantic)
   external)
            │
            ▼
   Background worker (Redis queue): pending actions, resume-after-hours,
   proactive follow-ups, embedding/indexing.
```

---

## 3. Folder structure

```
app/
  agent/
    __init__.py
    runtime.py            # AgentRuntime: builds + runs the compiled graph (entry point)
    state.py              # AgentState (TypedDict) + Goal/Plan/Step/Reflection models
    graph.py              # LangGraph wiring of all nodes + conditional edges + loops
    budget.py             # iteration / token / time budgets + kill-switch
    nodes/
      load_context.py     # Memory node: hydrate STM, working, episodic, semantic, session
      summarizer.py       # Context summarizer node (rolling summary to cap tokens)
      intent.py           # Intent + continuity: shift detection, resume unfinished task
      planner.py          # Planner node: produce/[re]plan steps + update goal stack
      tool_router.py      # Dynamic tool routing node: choose + dispatch MCP tools
      executor.py         # Execute a plan step (tool calls / sub-answers)
      reflection.py       # Reflection node: critique, decide retry/replan/finish
      responder.py        # Response generation: natural, emotionally-aware draft
      humanizer.py        # Chunking + pacing + typing plan + acknowledgements
      persist.py          # Write episodic/semantic/session/goals/pending actions
    prompts/
      persona.py          # System persona + voice + safety rules (versioned)
      planner.py          # Planner prompt (JSON plan schema)
      reflection.py       # Reflection/critique prompt (JSON verdict)
      responder.py        # Response prompt (natural WhatsApp style)
      summarizer.py       # Summarizer prompt
      registry.py         # Prompt versioning + selection
  memory/
    __init__.py
    working.py            # In-state scratchpad helpers (this turn)
    short_term.py         # Redis recent turns (wraps existing ConversationMemory)
    episodic.py           # Postgres: turn/session episodes (what happened)
    semantic.py           # Qdrant customer-memory + Postgres customer_facts (durable)
    session.py            # Postgres: agent_sessions, goals, pending_actions
    repositories.py       # READ-WRITE repos for agent_* tables (separate from business)
    schema.sql            # DDL for agent_* tables (bootstrap / Alembic migration source)
  whatsapp/
    delivery.py           # send chunks, typing indicator, inter-chunk delay, retries
    inbound.py            # parse/normalize inbound (text/interactive/image), dedupe
  mcp/                    # EXISTING — tools.py, server.py (tool surface, unchanged-ish)
  workers/
    agent_scheduler.py    # NEW: process pending_actions + proactive follow-ups (Redis queue)
    embedding_worker.py   # EXISTING
  api/ core/ db/ llm/ schemas/ vector/   # EXISTING (reused)
docs/
  AGENT_ARCHITECTURE.md   # this file
```

The current `app/chatbot/graph.py` becomes the seed of `app/agent/` (its MCP agent
loop is reused as the `executor` + `responder` baseline) and is retired once the new
runtime is wired.

---

## 4. Agent state design

A single rich `AgentState` flows through the graph. It is **assembled from durable
stores** at `load_context` and **flushed back** at `persist`.

```python
class Goal(TypedDict):
    id: str
    description: str            # "track order SS-AB123", "find a red saree under 2k"
    status: Literal["active", "blocked", "done", "abandoned"]
    created_at: float

class Step(TypedDict):
    id: str
    intent: str                 # "search_products", "lookup_order", "answer", "ask_user"
    tool: str | None            # MCP tool name, if any
    args: dict[str, Any]
    status: Literal["pending", "running", "done", "failed"]
    result: Any
    attempts: int

class Reflection(TypedDict):
    goal_satisfied: bool
    grounded: bool
    issues: list[str]
    next: Literal["finish", "retry_step", "replan", "ask_user", "handoff"]

class AgentState(TypedDict, total=False):
    # identity / routing
    tenant_id: str
    customer_id: str            # WhatsApp phone (digits)
    conversation_id: str
    session_id: str
    request_id: str
    inbound_text: str
    inbound_kind: str           # text | button | list | image
    received_at: float

    # memory (hydrated by load_context)
    short_term: list[dict[str, str]]      # recent role-tagged turns (Redis)
    rolling_summary: str                  # compressed older history
    semantic_hits: list[dict[str, Any]]   # retrieved durable facts (vector)
    customer_facts: dict[str, Any]        # structured prefs (name, sizes, budget…)
    catalog_hits: list[dict[str, Any]]    # vector product retrieval for this turn
    working: dict[str, Any]               # scratchpad for this turn

    # session / continuity
    session_status: str                   # active | dormant | resumed
    last_active_at: float | None
    pending_actions: list[dict[str, Any]]
    intent: str
    intent_shifted: bool

    # reasoning
    goals: list[Goal]
    plan: list[Step]
    cursor: int                           # index of current step
    reflections: list[Reflection]
    loop_count: int

    # output
    draft_response: str
    message_chunks: list[str]
    delivery_plan: list[dict[str, Any]]   # [{text, typing_ms, delay_ms, image_url?}]
    used_llm: bool
    latency_ms: int
```

---

## 5. LangGraph flow

```
START
  │
  ▼
load_context ── hydrate STM, summary, semantic facts, session, pending actions
  │
  ▼
summarize ──── refresh rolling_summary when history exceeds the token budget
  │
  ▼
intent ─────── classify intent, detect intent_shift, detect resume-after-gap
  │
  ▼
planner ◄───────────────────────────────────┐  (re-entry on replan)
  │   produce/refresh plan + update goals     │
  ▼                                           │
route ── needs a tool? ──yes──► execute ──► reflect
  │                              (tool        │  verdict:
  │ no (direct answer)           router +     │   finish     → responder
  │                              retry)       │   retry_step → execute
  ▼                                           │   replan     → planner ─┘
responder ◄───────────────────────────────────┘   ask_user   → responder
  │   draft a natural, grounded reply             handoff     → responder (+ tool)
  ▼
humanize ───── chunk + typing plan + pacing + acknowledgements
  │
  ▼
persist ────── write episodic + semantic + session + goals + pending actions
  │
  ▼
END  ──► delivery layer streams chunks to WhatsApp with typing + delays
```

**Loop control** (`budget.py`): `loop_count` capped (e.g. 4), plus a token budget and a
wall-clock budget. When exhausted, the graph forces `responder` with whatever was
gathered (never spins forever; never 500s). LLM errors degrade to a warm "give me a
moment" reply (existing `_BUSY_REPLY` pattern).

Edges are conditional (`add_conditional_edges`) keyed on planner output (`route`) and
reflection verdict (`next`), mirroring the existing pattern in `chatbot/graph.py`.

---

## 6. Memory system

Four tiers, each with a clear store and lifetime:

| Tier | Holds | Store | Lifetime |
|---|---|---|---|
| **Working** | this turn's scratchpad, plan, tool results | `AgentState` (in-process) | one turn |
| **Short-term** | last N role-tagged turns | Redis (`ConversationMemory`, exists) | TTL (hours) |
| **Episodic** | "what happened" — per-turn/session event summaries | Postgres `agent_episodes` | durable |
| **Semantic** | durable facts & preferences about the customer | Qdrant `customer_memory` + Postgres `customer_facts` | durable |

Plus **rolling summary** (compressed older dialogue) kept on the session so token cost
stays flat across long conversations.

### Postgres schema (`agent_*`, read-write, separate from Prisma business tables)

```sql
-- A conversation thread with a customer (resumable across days).
CREATE TABLE agent_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT NOT NULL,
    customer_id     TEXT NOT NULL,                 -- WhatsApp phone (digits)
    conversation_id TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active', -- active | dormant | closed
    rolling_summary TEXT DEFAULT '',
    last_active_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, conversation_id)
);

-- Goal stack: what the customer is trying to achieve.
CREATE TABLE agent_goals (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
    tenant_id   TEXT NOT NULL,
    description TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',    -- active | blocked | done | abandoned
    priority    INT  NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Episodic memory: a compact record of each turn / notable event.
CREATE TABLE agent_episodes (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
    tenant_id   TEXT NOT NULL,
    role        TEXT NOT NULL,                     -- user | assistant | system
    summary     TEXT NOT NULL,                     -- 1-2 line "what happened"
    tool_calls  JSONB DEFAULT '[]',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON agent_episodes (session_id, created_at);

-- Deferred / unfinished work to resume or run proactively.
CREATE TABLE agent_pending_actions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
    goal_id     UUID REFERENCES agent_goals(id) ON DELETE SET NULL,
    tenant_id   TEXT NOT NULL,
    kind        TEXT NOT NULL,                     -- followup | reminder | retry_tool
    payload     JSONB NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending | done | cancelled
    run_after   TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON agent_pending_actions (status, run_after);

-- Structured semantic facts (the queryable half of semantic memory).
CREATE TABLE customer_facts (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    key         TEXT NOT NULL,                     -- name | size | budget | favourite_category
    value       TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.7,
    source      TEXT,                              -- which turn/tool asserted it
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, customer_id, key)
);
```

> These ship as `app/memory/schema.sql` and get applied via an **Alembic** migration
> (alembic is already a dependency; we `alembic init` once). The agent gets a
> **write-capable** repository layer (`app/memory/repositories.py`) — distinct from the
> read-only business `ProductRepository`/`OrderRepository`.

### Vector memory (Qdrant)

A new collection `customer_memory` (single dense vector, same 384-dim model), payload
`{tenant_id, customer_id, kind, text, ts}`, tenant-scoped exactly like the existing
collections. Used to recall the free-text half of semantic memory ("last time they
asked about gift options for a wedding"). The existing `VectorStore` is extended with
`remember()` / `recall()` helpers.

---

## 7. Prompt architecture

Layered, versioned, single-responsibility prompts (in `app/agent/prompts/`), composed
per node — never one mega-prompt:

```
persona (identity + voice + safety, stable)
  ├── planner prompt      → emits JSON: {goals[], steps[{intent,tool,args}]}
  ├── reflection prompt   → emits JSON: {goal_satisfied, grounded, issues[], next}
  ├── responder prompt    → natural WhatsApp reply, given plan results + memory
  └── summarizer prompt   → compress old turns into rolling_summary
```

- **Persona** carries the human voice rules (warm, concise, emoji-light, sales-savvy)
  and the hard anti-hallucination constraints.
- **Structured nodes** (planner, reflection) request JSON via the existing
  `LLMClient.json_chat`, so their output is machine-checkable.
- **Responder** is free-text and validated by `HallucinationValidator`.
- **Versioning**: `prompts/registry.py` maps `name → version → template`; we log the
  prompt version on every call for eval/rollback.

---

## 8. Tool execution & retry strategy

- **Surface**: the existing MCP `ToolRegistry` (search_products, get_order_status,
  get_recent_orders, search_policies, search_faq, request_human_handoff) + room for
  external MCP servers (courier, payments) added by config.
- **Dynamic routing**: `tool_router` takes the planner's chosen `tool`+`args`, validates
  args against the tool schema, injects identity via `ToolContext` (tenant + customer —
  never model-set), and dispatches.
- **Retry ladder**:
  1. *Transport* — tenacity backoff inside `LLMClient`/tool (exists).
  2. *Step* — on `failed`, `reflection` may return `retry_step` (bounded by
     `Step.attempts`).
  3. *Plan* — repeated failure → `replan` with the failure noted in context.
  4. *Human* — give up gracefully → `request_human_handoff` + warm message.
  5. *Defer* — if the action can wait, write a `agent_pending_actions` row and tell the
     customer we'll follow up (resume later via the scheduler).
- **Idempotency**: action tools carry an idempotency key (session+goal+kind) so a retry
  never double-acts.

---

## 9. Human-like interaction

`humanizer` turns one draft into a delivery plan; `whatsapp/delivery.py` executes it.

- **Chunking**: split at sentence/paragraph boundaries into 1–3 bubbles (long answers
  feel robotic as one wall of text). Hard cap per bubble.
- **Typing simulation**: before each chunk, send Meta's typing indicator (the webhook
  already does this on inbound); delay ≈ `len(chunk) / typing_speed`, clamped to
  ~1.2–4s, so it reads as if a person is typing.
- **Acknowledgements**: before a slow tool op, send a quick natural ack ("Let me check
  that for you… 👀") so the customer isn't left hanging.
- **Pacing**: small inter-chunk delay; never dump all bubbles instantly.
- **Voice**: phrasing rules in the persona prompt. Banned: "Please provide more
  details." Preferred: "Looks like I need a bit more info to keep going — what's the
  order number?"

---

## 10. Conversation lifecycle (async + resumable)

- **Async**: the webhook acks fast and the agent runs; outbound is paced separately.
- **Resume after hours/days**: `load_context` reads `agent_sessions.last_active_at`;
  if the gap is large, `intent` node marks `session_status="resumed"` and the responder
  opens with a brief re-orient ("Welcome back! Earlier you were looking at red sarees —
  want to pick that back up?") sourced from the rolling summary + active goals.
- **Intent shift**: if the new message doesn't match the active goal, `intent` sets
  `intent_shifted`; planner pushes a new goal and parks the old one (`blocked`) rather
  than losing it.
- **Pending actions**: unfinished work is persisted and surfaced next turn, or run by
  the `agent_scheduler` worker (e.g. "your size is back in stock" follow-up).
- **Session state**: every turn updates the session row + episodic log.

---

## 11. Production concerns

- **Idempotent webhook**: dedupe by Meta `message_id` (Redis SETNX) so retried webhooks
  don't double-process.
- **Observability**: structlog conversation tracer (exists) extended with plan/reflect
  steps; Prometheus counters for loop_count, tool retries, reflection verdicts, handoffs.
- **Budgets & safety**: per-turn iteration/token/time caps; graceful degradation on
  rate-limit/outage; `HallucinationValidator` on every outbound.
- **Multi-tenant**: tenant_id threaded everywhere + ContextVar (exists); all agent
  tables and the vector memory are tenant-scoped.
- **Scaling**: stateless API workers; durable state in Postgres/Redis/Qdrant; the
  scheduler worker is horizontally shardable by tenant/customer hash.
- **Eval harness**: a scripted suite of conversations asserting goal completion,
  grounding, and resumption — run in CI with a mocked LLM + recorded fixtures.

---

## 12. Phased roadmap

| Phase | Deliverable | Reuses |
|---|---|---|
| **0 — Foundation** | `app/agent/` skeleton, `AgentState`, `runtime.py`, port the current MCP agent loop in as `executor+responder`; `agent_*` tables + Alembic + write repos | MCP agent, LangGraph, session_scope |
| **1 — Memory** | `load_context` + `persist` + `summarizer`; episodic + session + semantic (vector `customer_memory` + `customer_facts`); short-term via existing Redis | VectorStore, ConversationMemory |
| **2 — Reasoning** | `planner` + `reflection` + `tool_router` loop, goal tracking, retry/replan ladder, budgets | MCP ToolRegistry, json_chat |
| **3 — Human-like** | `humanizer` + `whatsapp/delivery.py`: chunking, typing, pacing, acks; persona voice | existing typing-indicator code |
| **4 — Lifecycle** | intent-shift, resume-after-gap, `agent_pending_actions` + `agent_scheduler` worker, proactive follow-ups | Redis worker/queue |
| **5 — Hardening** | webhook idempotency, eval harness, dashboards, prompt versioning, load test | structlog, Prometheus |

Each phase is independently shippable and leaves the bot working.
```
