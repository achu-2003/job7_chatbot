"""Cross-encoder reranker for the chat retrieval path.

After hybrid search produces a candidate set, this module re-scores each
candidate against the query with a cross-encoder model and keeps the top-K.
Cross-encoders beat bi-encoders on ranking accuracy because they attend
jointly to query and document tokens — but they cost ~10–30 ms per pair, so
we only run them on the already-narrowed top-N from hybrid search.

The model is loaded lazily on first use (~280–570 MB depending on choice).
Failures are swallowed and the original order is preserved — reranking is
pure precision polish, never on the critical path.

Disabled by default via ``settings.rerank_enabled``. Flip it on after the
first reindex so the model download isn't a surprise.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Sequence

from app.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import LLM_LATENCY

log = get_logger("reranker")

_model: Any | None = None
_lock = asyncio.Lock()


async def _get_model() -> Any | None:
    """Lazy-load fastembed's Rerank model. Returns None if the dependency
    or chosen model isn't available — caller treats that as no-op."""
    global _model
    if _model is not None:
        return _model
    async with _lock:
        if _model is not None:
            return _model
        s = get_settings()
        loop = asyncio.get_running_loop()
        try:
            # Newer fastembed (≥0.4) exposes a Rerank class. Older versions
            # might not — degrade gracefully rather than crash startup.
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:
            log.warning("fastembed_rerank_unavailable")
            return None
        try:
            _model = await loop.run_in_executor(
                None, lambda: TextCrossEncoder(model_name=s.rerank_model)
            )
            log.info("reranker_loaded", model=s.rerank_model)
        except Exception as exc:  # noqa: BLE001
            log.warning("reranker_load_failed", model=s.rerank_model, error=str(exc))
            return None
    return _model


async def rerank(
    query: str, hits: Sequence[dict[str, Any]], *, top_k: int | None = None
) -> list[dict[str, Any]]:
    """Re-score ``hits`` against ``query`` with the cross-encoder.

    Each hit dict must carry ``document`` (the text the cross-encoder scores
    against). Returns a new list ordered by descending cross-encoder score,
    truncated to ``top_k`` (defaults to ``rerank_top_k`` from settings).
    Original ``score``/``distance`` fields are preserved; a new
    ``rerank_score`` field is added.
    """
    s = get_settings()
    if not hits:
        return []
    k = top_k or s.rerank_top_k
    model = await _get_model()
    if model is None:
        return list(hits)[:k]

    docs = [h.get("document", "") for h in hits]
    start = time.perf_counter()
    loop = asyncio.get_running_loop()
    try:
        scores = await loop.run_in_executor(
            None, lambda: list(model.rerank(query, docs))
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("rerank_call_failed", error=str(exc))
        return list(hits)[:k]
    LLM_LATENCY.labels(model=s.rerank_model, purpose="rerank").observe(
        time.perf_counter() - start
    )

    scored: list[tuple[float, dict[str, Any]]] = []
    for hit, score in zip(hits, scores):
        new_hit = dict(hit)
        new_hit["rerank_score"] = float(score)
        scored.append((float(score), new_hit))

    scored.sort(key=lambda kv: kv[0], reverse=True)
    return [h for _, h in scored[:k]]
