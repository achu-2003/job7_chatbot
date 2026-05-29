# Job Application Agent — WhatsApp Recruiting Assistant

> **Domain migration note (2026-05).** This codebase was repurposed from an
> e-commerce support agent into a **job-application / recruiting agent**. The
> architecture is unchanged — only the domain layer (tools, repositories,
> prompts, content) was swapped. Candidates search open roles, apply
> conversationally (name/email/experience captured over turns), and check their
> application status, all over WhatsApp.
>
> **The stack is also newer than older sections of this README describe.** The
> live stack is **Qdrant** (vectors, hybrid dense+sparse), **fastembed** (local
> embeddings/rerank), **LangGraph** (the `AgentRuntime` planner→execute→reflect
> loop), **MCP** tools, and an **OpenAI-compatible client pointed at Groq** —
> *not* ChromaDB/OpenAI. Trust `docs/AGENT_ARCHITECTURE.md` + the `app/` code
> over the legacy prose below.
>
> **Job tools:** `search_jobs`, `submit_application` (idempotent write),
> `get_application_status`, `search_policies`, `search_faq`,
> `request_human_handoff`, `schedule_followup`.
> **Three data tiers:** read-only `jobs`/`departments`; candidate-owned
> read+write `candidates`/`applications`; agent-owned `agent_*` memory.
> Bootstrap the job schema with `python scripts/init_jobs_db.py --seed`.

---

Production-grade, low-resource (2 vCPU / 4 GB RAM) agent for recruiting
over WhatsApp, web, and mobile. Hybrid architecture:
PostgreSQL is the **source of truth** for everything transactional (jobs +
applications), Qdrant is used for **semantic retrieval**, the LLM is used
**only** for language understanding and grounded response generation.

---

## 1. Architecture

```
                                    ┌──────────────────────────┐
   WhatsApp / Web / Mobile ──HTTP──▶│        nginx (80)        │
                                    └────────────┬─────────────┘
                                                 │
                                    ┌────────────▼─────────────┐
                                    │   FastAPI (async, 2 wkr) │
                                    │   • intent classifier    │
                                    │   • query router         │
                                    │   • structured extract   │
                                    │   • semantic rewriter    │
                                    │   • response generator   │
                                    │   • hallucination guard  │
                                    └─┬──────────┬────────┬────┘
                                      │          │        │
                                      │          │        │ memory
                                      ▼          ▼        ▼
                              ┌──────────┐  ┌───────┐ ┌──────────┐
                              │ Postgres │  │Chroma │ │  Redis   │
                              │  (truth) │  │ (RAG) │ │ (mem+q)  │
                              └────┬─────┘  └───▲───┘ └─────▲────┘
                                   │ LISTEN/    │           │
                                   │ NOTIFY     │           │
                                   ▼            │           │
                              ┌──────────────────────────────┐
                              │   Embedding Worker (async)   │
                              │   • subscribes to vector_sync│
                              │   • UPSERTs into Chroma      │
                              └──────────────────────────────┘

   Observability: Prometheus  ◀─── /metrics ───  FastAPI + Worker
                  Grafana     ◀─── dashboards
                  structlog JSON logs (stdout)
```

### Core principles

| Principle | Implementation |
|-----------|----------------|
| Postgres is the source of truth | All prices, stock, orders, payments come from parameterised SQL |
| Vector DB never stores transactional fields | `embedding_worker._product_document` builds the doc from `name`, `description`, `category`, `color`, `style`, `occasion`, `gender` — **never** price/stock |
| LLM cannot hallucinate | `HallucinationValidator` rejects responses with prices/SKUs/order numbers not present in retrieved context |
| Realtime sync | Postgres triggers fire `pg_notify('vector_sync', ...)` on INSERT/UPDATE/DELETE → worker UPSERTs Chroma |
| SQL safety | All chat-path SQL is hand-written and parameterised. The `assert_safe_select` gatekeeper rejects any non-SELECT or multi-statement query, used wherever AI-generated SQL might be considered |
| Low resource | gunicorn 2 workers, asyncpg, Chroma server (mmap), Redis 256 MB cap, no Celery |

---

## 2. Folder structure

```
.
├── app/
│   ├── api/
│   │   ├── deps.py
│   │   └── routes/{chat,health,admin}.py
│   ├── chatbot/
│   │   ├── orchestrator.py     # end-to-end pipeline
│   │   ├── router.py           # SQL vs vector vs escalate
│   │   ├── memory.py           # Redis conversation memory
│   │   ├── validator.py        # anti-hallucination layer
│   │   └── escalation.py       # human handoff
│   ├── core/{logging,middleware,metrics,exceptions}.py
│   ├── db/
│   │   ├── models.py           # SQLAlchemy 2.0 async ORM
│   │   ├── repositories.py     # parameterised SQL only
│   │   ├── session.py
│   │   └── notifications.py    # asyncpg LISTEN/NOTIFY
│   ├── llm/
│   │   ├── client.py           # OpenAI async wrapper
│   │   ├── intent.py           # 10-class classifier
│   │   ├── extraction.py       # structured filter extraction
│   │   ├── rewriter.py         # query rewrite for vector
│   │   └── prompts.py          # all system prompts
│   ├── schemas/{chat,domain}.py
│   ├── vector/
│   │   ├── embeddings.py       # OpenAI embeddings client
│   │   └── store.py            # ChromaDB facade
│   ├── workers/
│   │   ├── queue.py            # Redis FIFO queue
│   │   └── embedding_worker.py # LISTEN → enqueue → upsert
│   ├── config.py
│   └── main.py
├── db/
│   ├── schema.sql              # full DDL
│   ├── triggers.sql            # vector_sync NOTIFY
│   └── seed.sql                # sample data
├── docker/
│   ├── Dockerfile              # api image
│   ├── Dockerfile.worker
│   └── nginx.conf
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/dashboards/chatbot.json
├── tests/
│   ├── test_router.py
│   ├── test_validator.py
│   ├── test_extraction.py
│   ├── test_sql_safety.py
│   └── test_api_health.py
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

---

## 3. Deployment

### 3a. Native run (no Docker)

This mode runs the chatbot **read-only against a live Prisma-managed
ecommerce DB** (SheScale). The chatbot **never writes to that database** —
no schema is applied, no triggers are installed. All chatbot-side content
(FAQs, policies) lives as JSON in `content/`, the audit log is the
structlog stream, and conversation memory is in Redis.

ChromaDB runs embedded inside the FastAPI process (no vector server), and
the embedding worker runs as a background asyncio task in that same
process — so you only start one Python process.

**Prerequisites:**
- Network access to the remote Postgres (DSN in `.env`)
- Redis 6+ running locally (`sudo apt install redis-server`)
- Python 3.12+

**One-time setup:**

```bash
make install                 # venv + pip install
make env                     # copies .env.native -> .env (skip if .env already exists)
$EDITOR .env                 # set OPENAI_API_KEY and DATABASE_URL
make check                   # verify remote Postgres + local Redis reachable
```

> No `make setup-db` step. The chatbot uses whatever schema is already in
> the connected DB. The repositories in `app/db/repositories.py` are mapped
> to the SheScale Prisma schema (products, product_variants, categories,
> orders, shipments). If you point this at a different store, rewrite those
> queries.

**Run:**

```bash
make run                     # uvicorn on 127.0.0.1:8000
# in another terminal:
make reindex                 # builds embeddings: live products + content/*.json
curl http://127.0.0.1:8000/api/v1/admin/reindex/status   # watch queue depth -> 0
```

What `/admin/reindex` does:
- Pulls every `status='ACTIVE'` row from `public.products` (with category,
  variant colors/sizes) and enqueues an upsert job per product.
- Loads `content/faqs.json` and `content/policies.json` and enqueues one
  upsert per item.
- The bundled worker consumes the queue and UPSERTs into the embedded
  ChromaDB collections `products`, `faqs`, `policies`.

Then:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'content-type: application/json' \
  -d '{"message":"Show me blue sarees under 2000"}'
```

What changes between native and Docker mode:

| Setting | Native (`.env.native`) | Docker (`.env.example`) |
|---------|------------------------|-------------------------|
| `VECTOR_MODE` | `embedded` | `http` |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | (unused — server) |
| `RUN_WORKER_IN_API` | `true` | `false` (separate container) |
| `POSTGRES_HOST` | `127.0.0.1` | `postgres` |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | `redis://redis:6379/0` |

### 3b. Docker / single-host

```bash
cp .env.example .env          # set OPENAI_API_KEY + POSTGRES_PASSWORD
docker compose build
docker compose up -d
curl -X POST http://localhost/api/v1/admin/reindex
```

The Docker stack uses **server-mode** ChromaDB and a separate worker container.
The split is intentional for production scaling. For dev, prefer §3a.

Endpoints:
- `POST http://localhost/api/v1/chat`
- `GET  http://localhost/health/ready`
- `GET  http://localhost:9090`     (Prometheus)
- `GET  http://localhost:3000`     (Grafana — `admin / admin`)

### Resource budget (deploy.resources.limits in docker-compose.yml)

| Service   | RAM   | CPU |
|-----------|-------|-----|
| nginx     | 128 M | 0.25|
| api       | 1.0 G | 0.75|
| worker    | 768 M | 0.5 |
| postgres  | 768 M | 0.5 |
| redis     | 320 M | 0.25|
| chroma    | 768 M | 0.5 |
| prom/graf | 512 M | 0.5 |
| **Total** | ~4.0 G| ~3.0|

(The CPU limits sum >2; under 2 vCPU they're contention shares, not caps.)

---

## 4. End-to-end request flow

```
   user message
       │
       ▼
[RequestContextMiddleware]   ← assigns request_id, starts latency timer
       │
       ▼
[Orchestrator.handle]
   ├── IntentClassifier (gpt-4o-mini, json mode)
   │       → {intent, confidence, needs_sql, needs_vector, needs_escalation}
   │
   ├── QueryRouter
   │       → RoutingDecision {sql_used, vector_used, llm_used, escalate}
   │
   ├── If escalate ─────────────────────────────► EscalationService → end
   │
   ├── If sql_used:
   │       FilterExtractor → ProductFilters
   │       ProductRepository.search(...)     ← parameterised SQL only
   │
   ├── If vector_used:
   │       QueryRewriter → 1-2 neutral phrases
   │       VectorStore.query(collection, ...) (cosine top-k)
   │       merge + dedupe across rewrites
   │
   ├── ConversationMemory.history(conv_id)   ← last N turns from Redis
   │
   ├── LLM grounded generation
   │       system prompt enforces: no hallucinations, no fake prices,
   │       only facts present in CONTEXT block
   │
   ├── HallucinationValidator
   │       ├── prices in response ⊆ prices in SQL rows
   │       ├── order/tracking numbers ⊆ those returned by SQL
   │       └── quoted product names ⊆ retrieved names
   │       INVALID → 1 strict-regenerate; still INVALID → escalate
   │
   ├── ConversationMemory.append(user, assistant)
   ├── AuditRepository.save(full payload)
   └── structlog.info("chat_request", **payload)
```

---

## 5. Intent → routing matrix

| Intent             | SQL | Vector | LLM | Notes |
|--------------------|:---:|:------:|:---:|-------|
| product_search     |  ✅  |   ✅    |  ✅  | Hybrid: SQL rows authoritative for price/stock; vector for nuance |
| recommendation     |  ✅  |   ✅    |  ✅  | Vector picks ids → SQL hydrates rows |
| faq                |     |   ✅    |  ✅  | Pure RAG against FAQ collection |
| return_request     |     |   ✅    |  ✅  | RAG against policies collection |
| order_tracking     |  ✅  |        |  ✅  | Parameterised lookup by order_number/customer |
| payment_issue      |  ✅  |        |  ✅  | + escalate (severity=high) |
| customer_support   |     |        |  ✅  | Escalate |
| escalation         |     |        |  ✅  | Escalate |
| greeting           |     |        |     | Canned response (no token spend) |
| unknown            |     |        |  ✅  | Asks one clarifying question |

Plus override: any message containing keywords in `ESCALATION_KEYWORDS`
(`refund, fraud, legal, complaint, manager`) forces escalation.

---

## 6. Realtime PostgreSQL → Vector sync

`db/triggers.sql` installs `AFTER INSERT/UPDATE/DELETE` triggers on
`products`, `faqs`, `policies` that call `pg_notify('vector_sync', payload)`.

For `products`, the trigger first checks whether any **semantic** field
actually changed (`name, description, category, color, style, occasion,
gender, is_active`). Price/stock-only updates produce **no notification**
— preventing wasteful re-embedding.

`app/db/notifications.py` opens a dedicated asyncpg connection in the
worker, registers a listener, and pushes each event onto Redis.
`app/workers/embedding_worker.py` consumes the queue, fetches the
current row (for INSERT/UPDATE) or simply deletes the id (for DELETE),
builds a semantic-only document, and `UPSERT`s into Chroma. Upserts make
the worker idempotent.

---

## 7. Anti-hallucination strategy

Defence in depth:

1. **Retrieval** — orchestrator only allows the LLM to generate when SQL
   rows or vector hits exist (greetings/escalations short-circuit).
2. **System prompt** (`RESPONSE_SYSTEM` in `app/llm/prompts.py`) enforces
   absolute grounding rules.
3. **Response validator** (`HallucinationValidator`) verifies:
   - every `₹NNN` / `Rs NNN` / `INR NNN` price appears in SQL rows;
   - every `ORDxxxx` / `TRKxxxx` matches retrieved values;
   - quoted product names overlap with retrieved names.
4. **One strict regeneration** if validation fails; escalate if it fails
   again. Counter `chatbot_validation_total{result=...}` exposes
   `VALID / REGENERATED / INVALID / ESCALATED` for monitoring.

---

## 8. Logging & observability

### Structured JSON log (one line per chat request)

```json
{
  "request_id": "REQ_1001",
  "timestamp": "2026-05-16T12:00:00+00:00",
  "channel": "web",
  "customer_query": "Need blue saree under 2000",
  "conversation_id": "conv_5e2f...",
  "intent": "product_search",
  "intent_confidence": 0.93,
  "routing": {
    "sql_used": true,
    "vector_used": true,
    "llm_used": true,
    "escalate": false,
    "reason": "intent=product_search, conf=0.93"
  },
  "extracted_filters": {"category":"saree","color":"blue","max_price":2000},
  "sql_query": "SELECT ... FROM products WHERE is_active=TRUE AND LOWER(category)=LOWER(:category) AND LOWER(color)=LOWER(:color) AND price<=:max_price ...",
  "vector_query": ["blue saree", "lightweight blue saree casual"],
  "sql_rows_count": 2,
  "vector_hits_count": 4,
  "llm_response": "Yes — we have the Royal Blue Banarasi Saree at ₹1899 ...",
  "response_validation": "VALID",
  "validation_offending": null,
  "escalated": false,
  "latency_ms": 1180
}
```

### Prometheus metrics

| Metric | Type | Labels |
|--------|------|--------|
| `chatbot_http_requests_total` | counter | method, path, status |
| `chatbot_http_request_latency_seconds` | histogram | method, path |
| `chatbot_sql_latency_seconds` | histogram | op |
| `chatbot_vector_latency_seconds` | histogram | collection |
| `chatbot_openai_latency_seconds` | histogram | model, purpose |
| `chatbot_openai_tokens_total` | counter | model, kind |
| `chatbot_intent_total` | counter | intent |
| `chatbot_routing_total` | counter | sql_used, vector_used |
| `chatbot_validation_total` | counter | result |
| `chatbot_errors_total` | counter | layer |
| `chatbot_embed_queue_depth` | histogram | — |

Grafana dashboard: `monitoring/grafana/dashboards/chatbot.json`.

---

## 9. Testing

```bash
pip install -r requirements.txt
pytest -q
```

Unit tests cover:
- routing matrix (`test_router.py`)
- SQL safety gatekeeper (`test_sql_safety.py`)
- structured filter extraction model (`test_extraction.py`)
- hallucination validator (`test_validator.py`)
- FastAPI `/health/live` smoke test (`test_api_health.py`)

Integration tests should run inside docker-compose against real
Postgres/Redis/Chroma; mocking those defeats the purpose.

---

## 10. Production security checklist

- [ ] Set `POSTGRES_PASSWORD`, `GRAFANA_ADMIN_PASSWORD`, `OPENAI_API_KEY`
      via secret manager, not `.env` in repo.
- [ ] Run nginx behind TLS termination (Caddy / ALB / Cloudflare).
- [ ] Restrict `/api/v1/admin/*` (mTLS, IP allowlist, or auth middleware).
- [ ] Enable Postgres `ssl=on` and connect with `sslmode=require`.
- [ ] Add per-customer rate limit at nginx (`limit_req_zone`).
- [ ] Pin all images by SHA in production, not just tag.
- [ ] Scan images (`trivy`, `grype`) in CI.

---

## 11. Sample requests & responses

### A — product search (hybrid SQL + vector)

```bash
curl -s -X POST http://localhost/api/v1/chat \
  -H 'content-type: application/json' \
  -d '{
    "message": "I need a blue saree under 2000 for a festival",
    "customer_external_id": "CUST001",
    "channel": "web"
  }' | jq
```

```json
{
  "request_id": "REQ_3f1c8a...",
  "conversation_id": "conv_8aa1c2...",
  "intent": "product_search",
  "response": "Yes — we have the Royal Blue Banarasi Saree at ₹1899, a handwoven silk saree with a gold zari border, perfect for festivals. Would you like me to add it to your cart?",
  "routing": {"sql_used": true, "vector_used": true, "llm_used": true, "escalate": false, "reason": "intent=product_search, conf=0.93"},
  "validation": "VALID",
  "escalated": false,
  "latency_ms": 1180
}
```

### B — order tracking (SQL only)

```bash
curl -s -X POST http://localhost/api/v1/chat \
  -d '{"message":"Where is my order ORD1001?","customer_external_id":"CUST001"}'
```

```json
{
  "intent": "order_tracking",
  "response": "Your order ORD1001 has shipped via BlueDart (tracking TRK-555-9988). Payment is captured.",
  "routing": {"sql_used": true, "vector_used": false, "llm_used": true, "escalate": false},
  "validation": "VALID"
}
```

### C — FAQ (vector only)

```bash
curl -s -X POST http://localhost/api/v1/chat -d '{"message":"What is your return policy?"}'
```

```json
{
  "intent": "faq",
  "response": "You can return unworn items with original tags within 7 days of delivery. Refunds reach the original payment method within 5–7 business days after we inspect the return.",
  "routing": {"sql_used": false, "vector_used": true, "llm_used": true, "escalate": false},
  "validation": "VALID"
}
```

### D — escalation (payment dispute)

```bash
curl -s -X POST http://localhost/api/v1/chat \
  -d '{"message":"I was charged but the order is still pending. I want a refund."}'
```

```json
{
  "intent": "payment_issue",
  "response": "I'll connect you with one of our human agents who can help further. Your conversation has been forwarded and someone will reach out shortly.",
  "routing": {"sql_used": false, "vector_used": false, "llm_used": true, "escalate": true, "reason": "keyword_escalation"},
  "escalated": true,
  "validation": "ESCALATED"
}
```

---

## 12. Operational runbook

| Symptom | First check |
|---------|-------------|
| `validation` is `INVALID` or `ESCALATED` spiking | Grafana → hallucination rate panel; inspect `request_audit.payload` for offending fields |
| Vector results stale | Worker container logs; `XLEN embedding_jobs` in Redis; check Postgres `pg_listening_channels()` |
| API latency P95 climbing | OpenAI latency vs SQL latency panels; if OpenAI: switch `OPENAI_MODEL_CHAT` to a faster model |
| Chroma OOM | Lower `OPENAI_EMBED_DIMENSIONS` from 512 → 256; shrink top-k |
| Worker not consuming | Verify `pg_notify` arrives: `LISTEN vector_sync;` in psql; if silent, triggers missing |

Re-index everything from scratch:

```bash
curl -X POST http://localhost/api/v1/admin/reindex
```
