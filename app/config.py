from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- app ----
    app_env: Literal["development", "staging", "production"] = "production"
    # Opt-in production-readiness guard. When True, startup runs
    # ``validate_runtime_config`` and REFUSES to boot on a critical misconfig
    # (test keys / test DB / missing app secret / non-https base url). Default
    # False so local dev/test is never blocked — set ENFORCE_PROD_CONFIG=true
    # only on the real production server.
    enforce_prod_config: bool = False
    app_name: str = "ecom-crm-chatbot"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    json_logs: bool = True

    # ---- postgres ----
    database_url: str
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "ecom_crm"
    postgres_user: str = "ecom"
    postgres_password: str = ""

    # ---- redis ----
    redis_url: str = "redis://redis:6379/0"
    redis_queue_key: str = "embedding_jobs"
    redis_memory_ttl_seconds: int = 3600

    # ---- vector store (Qdrant) ----
    # Connection mode: "http" (REST) or "embedded" (local on-disk store).
    vector_mode: Literal["http", "embedded"] = "http"
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333          # REST
    qdrant_grpc_port: int = 6334
    qdrant_api_key: str = Field(default="")
    qdrant_use_grpc: bool = False
    qdrant_persist_dir: str = "./data/qdrant"
    # Job-domain collections. The setting key stays ``..._products`` for code
    # stability (it's the catalog collection everywhere), but the default value
    # is the job-postings collection now that this is a recruiting agent.
    vector_collection_products: str = "jobs"
    vector_collection_faq: str = "faqs"
    vector_collection_policies: str = "policies"
    # Optional visual collection — unused by the recruiting agent (no image
    # search for jobs). Kept so the vector layer's schema doesn't change.
    vector_collection_product_images: str = "product_images"
    # Durable semantic memory about each candidate (the free-text half of
    # semantic memory; the structured half is the customer_facts table).
    vector_collection_customer_memory: str = "customer_memory"

    # ---- visual search (CLIP) ----
    # Customer sends an image on WhatsApp → embed with CLIP → search the
    # product_images collection. Separate from text vectors because dims
    # differ (CLIP is 512, our text model is 384).
    image_embed_model: str = "Qdrant/clip-ViT-B-32-vision"
    image_embed_dimensions: int = 512
    # Cosine similarity below this → treat as "no match" and reply with
    # the "couldn't find this in our store" message rather than guessing.
    # CLIP "same product, different photo" typically scores 0.45–0.65, so
    # 0.7 is too strict; 0.55 balances recall vs false positives. Tune via
    # IMAGE_MATCH_THRESHOLD after watching real top_score values in the logs.
    image_match_threshold: float = 0.55
    # Reindex http client timeout when downloading product images.
    image_download_timeout_seconds: int = 15

    # ---- multi-tenant ----
    # When True, every request to /api/v1/* must carry X-API-Key. Tenant key
    # is mapped through `tenant_api_keys` (JSON: {"<api_key>": "<tenant_id>"}).
    # When False, every request runs as `default_tenant_id` (legacy mode).
    multi_tenant_required: bool = False
    default_tenant_id: str = "default"
    # JSON map of api_key -> tenant_id. Kept as raw string here; parsed in
    # app.core.tenancy.
    tenant_api_keys: str = Field(default="")
    # JSON map of tenant_id -> display name. Optional; used for logging only.
    tenant_names: str = Field(default="")
    # When True, repositories add `tenant_id = :tenant_id` to product/order
    # queries. Requires the column to exist in the Prisma schema. Default
    # False so single-tenant deployments keep working unchanged.
    enforce_db_tenant_isolation: bool = False

    # ---- native dev ----
    run_worker_in_api: bool = False  # bundle worker as asyncio task in api

    # ---- LLM (any OpenAI-compatible provider — Groq, Together, Ollama, …) ----
    # Each field accepts both the new LLM_* env var and the legacy OPENAI_*
    # alias so existing deployments keep working without an env change.
    llm_enabled: bool = True
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    # Single chat model used everywhere (chat + the MCP agent), on Groq.
    llm_model_chat: str = Field(
        default="llama-3.1-8b-instant",
        validation_alias=AliasChoices("LLM_MODEL_CHAT", "OPENAI_MODEL_CHAT"),
    )
    # Kept for config compatibility; not used as a separate model today
    # (filter extraction reuses llm_model_chat).
    llm_model_router: str = Field(
        default="llama-3.1-8b-instant",
        validation_alias=AliasChoices("LLM_MODEL_ROUTER", "OPENAI_MODEL_ROUTER"),
    )
    llm_model_embed: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("LLM_MODEL_EMBED", "OPENAI_MODEL_EMBED"),
    )
    llm_embed_dimensions: int = Field(
        default=512,
        validation_alias=AliasChoices("LLM_EMBED_DIMENSIONS", "OPENAI_EMBED_DIMENSIONS"),
    )
    llm_timeout_seconds: int = Field(
        default=20,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS", "OPENAI_TIMEOUT_SECONDS"),
    )
    llm_max_retries: int = Field(
        default=2,
        validation_alias=AliasChoices("LLM_MAX_RETRIES", "OPENAI_MAX_RETRIES"),
    )

    # ---- local LLM endpoint (llama.cpp, Ollama, vLLM — any OpenAI-shape) ----
    llm_base_url: str = Field(default="")
    # Multilingual by default (~50 languages incl. Hindi/Tamil). Same 384-dim
    # as bge-small so the Qdrant collection schema doesn't change, but the
    # embedding space is different — reindex when switching models.
    local_embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    local_embed_dimensions: int = 384

    # ---- MCP agent (tool-calling automation) ----
    # The MCP tool-calling agent is the single responder for every non-greeting
    # turn: it searches products, looks up the customer's own orders, reads
    # policies, and hands off to a human via MCP tools. See app/mcp/README.md.
    # Hard cap on LLM<->tool round-trips per turn. The final step disables tools
    # to force a textual answer, so a confused model can't loop forever.
    mcp_agent_max_iterations: int = 4
    # Model for the agent loop. Empty -> falls back to llm_model_chat.
    mcp_agent_model: str = Field(default="")

    # ---- production agent runtime (see docs/AGENT_ARCHITECTURE.md) ----
    # Bounds the planner<->executor<->reflection loop so it can never spin or
    # blow the webhook latency window. On exhaustion the graph forces a reply.
    agent_max_loops: int = 3
    agent_deadline_seconds: float = 25.0
    # Serve WhatsApp via the production AgentRuntime (planner/memory/reflection +
    # human-like delivery). Set false to fall back to the legacy ChatGraphRunner.
    agent_runtime_enabled: bool = True
    # Human-like delivery: split a reply into bubbles and pace them. Per-bubble
    # "typing" delay ≈ len(chunk)/cps, clamped to [min, max] ms.
    agent_typing_cps: float = 25.0
    agent_typing_min_ms: int = 700
    agent_typing_max_ms: int = 4000
    # Minimalist replies are short, so 1-2 small bubbles is plenty.
    agent_chunk_max_chars: int = 320
    agent_max_chunks: int = 2
    # Proactive follow-up scheduler (drains due agent_pending_actions and sends
    # them on WhatsApp). OFF by default — business-initiated messages are
    # sensitive and, beyond Meta's 24h window, require approved templates.
    agent_scheduler_enabled: bool = False
    agent_scheduler_poll_seconds: int = 60

    # ---- retrieval ----
    vector_top_k: int = 5
    sql_max_rows: int = 10
    intent_confidence_threshold: float = 0.55
    escalation_keywords: str = "discrimination,complaint,withdraw,legal,recruiter,urgent"

    # ---- hybrid search + reranker ----
    # Hybrid = RRF fusion of dense embeddings + BM25-like sparse vectors.
    # When True (default), collections are created with named "dense"/"sparse"
    # vector slots and every upsert/query uses both. Switching this requires
    # re-indexing — collection schema is fixed at creation time.
    hybrid_search_enabled: bool = True
    # Sparse model. Qdrant/bm25 is purely statistical (language-agnostic, no
    # ONNX inference) — fastest and works for any language.
    sparse_embed_model: str = "Qdrant/bm25"
    # Cross-encoder reranker. Adds ~50–150ms per turn for a big precision lift.
    # bge-reranker-v2-m3 is ~570MB; bge-reranker-base ~280MB (English-leaning).
    # Default off — flip on after the first reindex when you're ready for the
    # one-time model download.
    rerank_enabled: bool = False
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 5
    # Pre-fetch this many hits per branch before fusion+rerank; the reranker
    # then narrows to vector_top_k. Bigger = better recall, more compute.
    hybrid_prefetch_limit: int = 25

    # ---- observability ----
    prometheus_enabled: bool = True

    # ---- onboarding form (new-candidate profile) ----
    # New WhatsApp numbers finish onboarding via a short self-hosted web form
    # (email, experience, preferred role/location). Submissions are kept in
    # Redis only — never the business DB. This is the base URL the form link
    # points at, so it MUST be publicly reachable for the WhatsApp link to work
    # (e.g. your domain or an ngrok tunnel); localhost is fine for web testing.
    public_base_url: str = "http://localhost:8000"
    # How long a generated form link / stored submission lives in Redis.
    onboarding_ttl_seconds: int = 7 * 24 * 3600
    # When True, a submitted onboarding form is ALSO written to the live job board
    # (private_job_seekers + job_seeker_profiles + child rows). Default False keeps
    # the safe Redis-only staging; flip to true (REGISTER_IN_DB=true) once verified.
    register_in_db: bool = False
    # When True, a submitted company registration is ALSO written to the live
    # ``private_employers`` table (in addition to the Redis staging). Default False
    # keeps the safe Redis-only flow; flip to true (EMPLOYER_REGISTER_IN_DB=true)
    # once verified. The write is idempotent on the phone (no duplicate rows).
    employer_register_in_db: bool = False
    # When True, an activated job post is ALSO written to the live job board in one
    # transaction: INSERT private_jobs + debit credit_wallets.jobCredits + a
    # DEBIT_JOB_POST credit_ledger row (full billing fidelity). Idempotent on the
    # job. Default False keeps the Redis-only flow; flip with JOB_POST_IN_DB=true.
    job_post_in_db: bool = False
    # When True, a completed credit PURCHASE (Razorpay) is ALSO recorded live:
    # credit_wallets top-up + credit_ledger CREDIT_* rows + a bundle_purchases row
    # for bundles. Idempotent on the payment id. Flip with CREDITS_PURCHASE_IN_DB=true.
    credits_purchase_in_db: bool = False
    # When True, every Razorpay ORDER is persisted to private_payments at creation
    # (status PAYMENT_PENDING, full context in metadata) and flipped to SUCCESS on
    # verify — so a Redis loss can't strand a paid order (verify reconciles the
    # order context from the DB). Idempotent on razorpayOrderId. PAYMENTS_IN_DB=true.
    payments_in_db: bool = False
    # When True, the bot serves Tamil / Hindi users via translate-pivot: the inbound
    # message is translated to English for routing/search and the conversational
    # text reply is translated back to the user's language (LLM-backed, best-effort).
    # Default False keeps the English-only flow. MULTILANG_ENABLED=true.
    multilang_enabled: bool = False
    # When True, a seeker's "Save" tap is ALSO written to the live ``private_saved_jobs``
    # table (so the bookmark survives a Redis loss and syncs with the main app).
    # Idempotent on (jobSeekerId, jobId); needs the seeker in the DB (REGISTER_IN_DB).
    # SAVED_JOBS_IN_DB=true.
    saved_jobs_in_db: bool = False
    # When True, an activated subscription is ALSO written to the live ``subscriptions``
    # table (status ACTIVE, endDate = now + plan validity, paymentId → private_payments)
    # and the plan's monthly credit grant is mirrored to credit_wallets + a CREDIT_PLAN
    # credit_ledger row. Supersedes the employer's prior ACTIVE sub; idempotent on the
    # payment (paid) or one ACTIVE per employer+plan (free). SUBSCRIPTIONS_IN_DB=true.
    subscriptions_in_db: bool = False
    # Employer (job-poster) flow — Stages 1-5 are staged in Redis ONLY for testing
    # (company profile shaped like private_employers, KYC, posted jobs, the
    # candidate-unlock entitlement). When ``employer_kyc_auto_verify`` is True a
    # submitted KYC is approved immediately so the gate can be demoed end-to-end;
    # set EMPLOYER_KYC_AUTO_VERIFY=false to leave it PENDING ("under review") for a
    # manual/admin approval design.
    employer_kyc_auto_verify: bool = True
    # The business's DIALABLE WhatsApp number (digits, international format, e.g.
    # "919876543210" — NOT the meta_phone_number_id). Used to build the "Back to
    # chat" wa.me deep link on the form's success page so the candidate returns to
    # the chat after submitting. Empty → the success page shows a plain Close button.
    whatsapp_business_number: str = Field(default="")
    # Applying to a role is completed in the Jobs7 mobile app, not from chat.
    # Once an identified candidate confirms a role we hand them this Play Store
    # link (as a tappable "Open in Jobs7" WhatsApp button); an https URL is
    # required for the button (Meta rejects http/localhost), else it's inline text.
    jobs7_app_url: str = "https://play.google.com/store/apps/details?id=com.jobs7"
    # Jobs7 Employer app — where an employer goes to unlock full candidate
    # details (payment/subscription handled there, not in chat).
    employer_portal_url: str = "https://play.google.com/store/apps/details?id=in.jobs7.employer"

    # ---- razorpay (employer subscription / credit payments) ----
    # TEST keys start with "rzp_test_" (no real money — pay with Razorpay test
    # cards); LIVE keys start with "rzp_live_". The secret signs/creates orders
    # and verifies the payment callback signature; never expose it to the browser.
    razorpay_key_id: str = Field(default="")
    razorpay_key_secret: str = Field(default="")

    @property
    def razorpay_enabled(self) -> bool:
        return bool(self.razorpay_key_id and self.razorpay_key_secret)

    @property
    def razorpay_test_mode(self) -> bool:
        return self.razorpay_key_id.startswith("rzp_test_")

    # ---- whatsapp / meta cloud api ----
    meta_access_token: str = Field(default="")
    meta_phone_number_id: str = Field(default="")
    whatsapp_verify_token: str = Field(default="")
    # Meta App Secret — used to verify the X-Hub-Signature-256 HMAC on every
    # inbound webhook POST so spoofed payloads are rejected. When empty,
    # signature verification is SKIPPED (dev/test); set it in production.
    meta_app_secret: str = Field(default="")
    whatsapp_graph_version: str = "v22.0"
    # When True, a single-product reply is sent as an image message with the
    # product card as its caption (no buttons). Requires p.images to hold a
    # usable JPEG/PNG https URL (Meta rejects http and webp).
    whatsapp_send_images: bool = True
    # Tenant that owns the connected WhatsApp number. Webhook requests come
    # from Meta and cannot carry X-API-Key, so we map the phone_number_id to
    # a tenant_id at config time. Falls back to default_tenant_id.
    whatsapp_tenant_id: str = Field(default="")

    @property
    def escalation_keyword_list(self) -> list[str]:
        return [k.strip().lower() for k in self.escalation_keywords.split(",") if k.strip()]

    @field_validator("vector_collection_products", "vector_collection_faq", "vector_collection_policies")
    @classmethod
    def _no_empty_collection(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("collection name must be non-empty")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def validate_runtime_config(settings: Settings) -> tuple[list[str], list[str]]:
    """Inspect ``settings`` for production-readiness. Returns ``(criticals,
    warnings)``: *criticals* are unsafe-to-go-live misconfigs (insecure or
    pointed at test infra) that should block boot; *warnings* are likely-wrong
    settings worth surfacing but not fatal. Pure + side-effect-free so it's easy
    to unit-test; the startup hook decides whether to raise."""
    criticals: list[str] = []
    warnings: list[str] = []

    # --- security / wrong-environment (block) ---
    if not settings.meta_app_secret:
        criticals.append(
            "META_APP_SECRET is empty — inbound webhooks are NOT signature-verified.")
    if not settings.whatsapp_verify_token:
        criticals.append("WHATSAPP_VERIFY_TOKEN is empty.")
    if not settings.meta_access_token:
        criticals.append("META_ACCESS_TOKEN is empty — cannot send WhatsApp replies.")
    if not settings.public_base_url.lower().startswith("https://"):
        criticals.append(
            f"PUBLIC_BASE_URL must be a public https:// URL (got '{settings.public_base_url}').")
    if settings.razorpay_enabled and settings.razorpay_test_mode:
        criticals.append(
            "Razorpay is in TEST mode (rzp_test_…) — switch to live keys before go-live.")
    if "jobs7uat" in settings.database_url.lower():
        criticals.append("DATABASE_URL points at the jobs7uat TEST database.")
    if settings.llm_enabled and not settings.llm_api_key:
        criticals.append("LLM_ENABLED is true but LLM_API_KEY is empty.")

    # --- likely-wrong (warn) ---
    flag_map = {
        "REGISTER_IN_DB": settings.register_in_db,
        "EMPLOYER_REGISTER_IN_DB": settings.employer_register_in_db,
        "JOB_POST_IN_DB": settings.job_post_in_db,
        "CREDITS_PURCHASE_IN_DB": settings.credits_purchase_in_db,
        "PAYMENTS_IN_DB": settings.payments_in_db,
        "SUBSCRIPTIONS_IN_DB": settings.subscriptions_in_db,
        "SAVED_JOBS_IN_DB": settings.saved_jobs_in_db,
    }
    off = [name for name, on in flag_map.items() if not on]
    if off:
        warnings.append(
            "Persistence flags OFF (data won't be written to the DB): " + ", ".join(off))
    if not settings.multilang_enabled:
        warnings.append("MULTILANG_ENABLED is off — Tamil/Hindi replies disabled.")
    if settings.employer_kyc_auto_verify:
        warnings.append(
            "EMPLOYER_KYC_AUTO_VERIFY is on — every employer is auto-approved with no vetting.")
    if not settings.razorpay_enabled:
        warnings.append("Razorpay is not configured — payments/credits will fail.")
    if not settings.qdrant_api_key:
        warnings.append("QDRANT_API_KEY is empty — the vector store is unauthenticated.")

    return criticals, warnings
