"""MemoryGateway — one facade over all four memory tiers.

The agent's ``load_context`` / ``summarizer`` / ``persist`` nodes call this; it
fans out to Redis (short-term), Postgres (`agent_*` episodic/session/goals), and
Qdrant + `customer_facts` (semantic). Keeping the fan-out here means the nodes
stay declarative and this class is the single unit to unit-test.

Pure decisions (resume-after-gap, summary trigger) take an injectable ``now`` so
they're deterministic in tests.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from app.chatbot.memory import ConversationMemory
from app.core.logging import get_logger
from app.memory.repositories import (
    AgentSessionRepository,
    EpisodeRepository,
    PendingActionRepository,
)
from app.memory.semantic import SemanticMemory
from app.vector.store import VectorStore

log = get_logger("memory_gateway")

# A gap larger than this since the customer last spoke → treat the turn as a
# resumption (the responder re-orients: "Welcome back! Earlier you were…").
_RESUME_GAP_SECONDS = 3 * 3600
# Summarise once short-term history reaches this many turns (matches the Redis
# cap), folding older turns into the rolling summary to keep tokens flat.
_SUMMARY_AFTER_TURNS = 12

_SUMMARY_PROMPT = (
    "You maintain a running summary of a WhatsApp shopping conversation. Merge "
    "the previous summary with the recent turns into a concise, factual summary "
    "(<= 120 words): what the customer wants, key preferences, decisions made, "
    "and any unfinished task. No fluff, no greetings."
)


class MemoryGateway:
    def __init__(self, *, vector: VectorStore, short_term: ConversationMemory) -> None:
        self.short_term = short_term
        self.semantic = SemanticMemory(vector)

    # ---- pure decisions ---------------------------------------------

    @staticmethod
    def resume_gap_seconds(
        last_active_at: datetime | None, now: datetime | None = None
    ) -> float:
        if last_active_at is None:
            return 0.0
        if last_active_at.tzinfo is None:
            last_active_at = last_active_at.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return (current - last_active_at).total_seconds()

    @classmethod
    def is_resumed(
        cls,
        last_active_at: datetime | None,
        now: datetime | None = None,
        threshold: float = _RESUME_GAP_SECONDS,
    ) -> bool:
        return cls.resume_gap_seconds(last_active_at, now) > threshold

    @staticmethod
    def should_summarize(
        short_term: list[dict[str, str]], threshold: int = _SUMMARY_AFTER_TURNS
    ) -> bool:
        return len(short_term or []) >= threshold

    # ---- load / persist ---------------------------------------------

    async def load(
        self, *, tenant_id: str, customer_id: str, conversation_id: str, query: str
    ) -> dict[str, Any]:
        """Hydrate the full memory snapshot for a turn."""
        existing = await AgentSessionRepository.get(
            tenant_id=tenant_id, conversation_id=conversation_id
        )
        if existing:
            session = existing
            resumed = self.is_resumed(existing.get("last_active_at"))
            await AgentSessionRepository.touch(existing["id"])
        else:
            session = await AgentSessionRepository.get_or_create(
                tenant_id=tenant_id,
                customer_id=customer_id,
                conversation_id=conversation_id,
            )
            resumed = False

        sid = str(session["id"])
        short_term, facts, semantic_hits, pending, cached_product = await asyncio.gather(
            self.short_term.history(conversation_id, tenant_id=tenant_id),
            self.semantic.facts(tenant_id=tenant_id, customer_id=customer_id),
            self.semantic.recall(query, tenant_id=tenant_id, customer_id=customer_id),
            PendingActionRepository.list_pending(sid),
            self.short_term.get_last_product(conversation_id, tenant_id=tenant_id),
        )
        return {
            "session_id": sid,
            "rolling_summary": session.get("rolling_summary") or "",
            "last_active_at": session.get("last_active_at"),
            "session_status": "resumed" if resumed else "active",
            "short_term": short_term,
            "customer_facts": facts,
            "semantic_hits": semantic_hits,
            "pending_actions": pending,
            # the product currently in focus ("this"/"it" refer to it)
            "cached_product": cached_product,
            # goals are per-turn working memory (created by the planner), not
            # loaded from the DB — see app/agent/nodes/planner.py.
        }

    # ---- onboarding form (Redis-only new-candidate profile) ---------

    async def onboarding(
        self, *, tenant_id: str, conversation_id: str
    ) -> dict[str, Any] | None:
        """The submitted onboarding form for this conversation (the gate), or
        None if the new candidate hasn't completed it yet."""
        return await self.short_term.get_onboarding(
            conversation_id, tenant_id=tenant_id
        )

    async def onboarding_token(
        self, *, tenant_id: str, customer_id: str, conversation_id: str, name: str | None
    ) -> str:
        """Get/create this conversation's form token, used to build the link."""
        return await self.short_term.ensure_onboarding_token(
            conversation_id, tenant_id=tenant_id, customer_id=customer_id, name=name,
        )

    async def mark_onboarding_welcomed(
        self, *, tenant_id: str, conversation_id: str
    ) -> None:
        """Mark the submitted form as acknowledged in chat (so the success
        message is sent once)."""
        await self.short_term.mark_onboarding_welcomed(
            conversation_id, tenant_id=tenant_id
        )

    # ---- employer flow (Redis-only job-poster profile) --------------

    async def employer(
        self, *, tenant_id: str, phone: str
    ) -> dict[str, Any] | None:
        """The staged employer record for this phone, or None if not registered."""
        return await self.short_term.get_employer(phone, tenant_id=tenant_id)

    async def update_employer(
        self, *, tenant_id: str, phone: str, fields: dict[str, Any]
    ) -> dict[str, Any] | None:
        return await self.short_term.update_employer(phone, fields, tenant_id=tenant_id)

    async def employer_token(
        self, *, tenant_id: str, phone: str, conversation_id: str, name: str | None,
        fresh: bool = False,
    ) -> str:
        """Get/create this employer's form token (register / KYC / post-job).
        ``fresh=True`` always mints a new token (used for post-job so each post is
        an isolated link — see ``ensure_employer_token``)."""
        return await self.short_term.ensure_employer_token(
            phone, tenant_id=tenant_id, conversation_id=conversation_id, name=name,
            fresh=fresh,
        )

    async def set_focus_product(
        self, *, conversation_id: str, tenant_id: str, product_id: str, doc: str
    ) -> None:
        """Pin the product the customer is now discussing, so the next turn's
        'this'/'it'/'details' resolves to it."""
        await self.short_term.set_last_product(
            conversation_id, product_id, doc, tenant_id=tenant_id,
        )

    # ---- job-browse flow (category → role → location) ---------------

    async def browse_state(
        self, *, tenant_id: str, conversation_id: str
    ) -> dict[str, Any] | None:
        """Where the candidate is in the tappable job-browse flow, or None."""
        return await self.short_term.get_browse_state(
            conversation_id, tenant_id=tenant_id
        )

    async def set_browse_state(
        self, *, tenant_id: str, conversation_id: str, state: dict[str, Any]
    ) -> None:
        await self.short_term.set_browse_state(
            conversation_id, state, tenant_id=tenant_id
        )

    async def save_job(
        self, *, tenant_id: str, conversation_id: str, ref: str, job: dict[str, Any]
    ) -> None:
        """Add a job to the candidate's saved list (Redis only)."""
        await self.short_term.save_job(
            conversation_id, ref, job, tenant_id=tenant_id
        )

    async def record_interest(
        self, *, tenant_id: str, conversation_id: str, ref: str, job: dict[str, Any]
    ) -> None:
        """Record an Apply tap as interest (Redis only — a recruiter follows up)."""
        await self.short_term.record_interest(
            conversation_id, ref, job, tenant_id=tenant_id
        )

    # ---- apply-time top-up (progressive profiling) ------------------

    async def registration(
        self, *, tenant_id: str, conversation_id: str
    ) -> dict[str, Any] | None:
        """The staged DB-ready registration payload (seeker + profile + child rows)."""
        return await self.short_term.get_registration(
            conversation_id, tenant_id=tenant_id
        )

    async def update_registration_profile(
        self, *, tenant_id: str, conversation_id: str, fields: dict[str, Any]
    ) -> None:
        await self.short_term.update_registration_profile(
            conversation_id, fields, tenant_id=tenant_id
        )

    async def apply_state(
        self, *, tenant_id: str, conversation_id: str
    ) -> dict[str, Any] | None:
        return await self.short_term.get_apply_state(
            conversation_id, tenant_id=tenant_id
        )

    async def set_apply_state(
        self, *, tenant_id: str, conversation_id: str, state: dict[str, Any]
    ) -> None:
        await self.short_term.set_apply_state(
            conversation_id, state, tenant_id=tenant_id
        )

    async def clear_apply_state(self, *, tenant_id: str, conversation_id: str) -> None:
        await self.short_term.clear_apply_state(conversation_id, tenant_id=tenant_id)

    async def apply_token(self, *, tenant_id: str, conversation_id: str) -> str:
        """Mint the token the apply-resume upload page resolves to this chat."""
        return await self.short_term.ensure_apply_token(conversation_id, tenant_id=tenant_id)

    async def save_application(
        self, *, tenant_id: str, conversation_id: str, ref: str, record: dict[str, Any]
    ) -> None:
        """Stage a DB-ready application record in Redis (no business-DB write)."""
        await self.short_term.save_application(
            conversation_id, ref, record, tenant_id=tenant_id
        )

    async def reset_session(
        self, *, tenant_id: str, conversation_id: str
    ) -> None:
        """Refresh ONE customer's conversation when their session ends: clear
        short-term history + pinned product (Redis) and close the session +
        wipe its rolling summary (Postgres). Durable customer_facts are KEPT
        (name/size/prefs carry over). Strictly scoped to this (tenant,
        conversation) — one customer's reset never affects another's memory.
        """
        await self.short_term.clear(conversation_id, tenant_id=tenant_id)
        await AgentSessionRepository.close(
            tenant_id=tenant_id, conversation_id=conversation_id,
        )
        log.info("session_reset", tenant_id=tenant_id, conversation_id=conversation_id)

    async def persist(
        self,
        *,
        tenant_id: str,
        customer_id: str,
        conversation_id: str,
        session_id: str,
        user_text: str,
        assistant_text: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Write the turn across short-term, episodic and semantic memory."""
        await self.short_term.append(
            conversation_id, "user", user_text, tenant_id=tenant_id
        )
        await EpisodeRepository.add(
            session_id=session_id, tenant_id=tenant_id, role="user",
            summary=user_text[:300],
        )
        if assistant_text:
            await self.short_term.append(
                conversation_id, "assistant", assistant_text, tenant_id=tenant_id
            )
            await EpisodeRepository.add(
                session_id=session_id, tenant_id=tenant_id, role="assistant",
                summary=assistant_text[:300], tool_calls=tool_calls or [],
            )
            await self.semantic.remember(
                f"Customer: {user_text}\nAssistant: {assistant_text}",
                tenant_id=tenant_id, customer_id=customer_id, kind="exchange",
            )

    async def summarize(
        self, *, llm: Any, short_term: list[dict[str, str]], prior_summary: str
    ) -> str:
        """Fold the conversation into an updated rolling summary (one LLM call)."""
        convo = "\n".join(
            f"{t.get('role')}: {t.get('content')}"
            for t in short_term if t.get("content")
        )
        messages = [
            {"role": "system", "content": _SUMMARY_PROMPT},
            {
                "role": "user",
                "content": (
                    f"PREVIOUS SUMMARY:\n{prior_summary or '(none)'}\n\n"
                    f"RECENT CONVERSATION:\n{convo}\n\nWrite the updated summary."
                ),
            },
        ]
        text, _ = await llm.chat(
            purpose="memory_summary", messages=messages, temperature=0.2, max_tokens=220,
        )
        return text.strip()
