"""Local embedding clients backed by fastembed (ONNX, CPU)."""
from __future__ import annotations

import asyncio
import io
import time
from typing import Any, Literal, Sequence

from fastembed import ImageEmbedding, SparseTextEmbedding, TextEmbedding

from app.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import LLM_LATENCY

log = get_logger("embeddings")

_model: TextEmbedding | None = None
_model_lock = asyncio.Lock()

_sparse_model: SparseTextEmbedding | None = None
_sparse_lock = asyncio.Lock()

_image_model: ImageEmbedding | None = None
_image_lock = asyncio.Lock()


async def _get_model() -> TextEmbedding:
    """Lazy-load the embedding model once per process."""
    global _model
    if _model is not None:
        return _model
    async with _model_lock:
        if _model is None:
            s = get_settings()
            loop = asyncio.get_running_loop()
            _model = await loop.run_in_executor(
                None, lambda: TextEmbedding(model_name=s.local_embed_model)
            )
            log.info("local_embed_model_loaded", model=s.local_embed_model)
    return _model


def _needs_e5_prefix(model_name: str) -> bool:
    """E5 family models expect a ``query: ``/``passage: `` prefix on the
    input text. Skipping it costs ~5–10% accuracy on retrieval tasks. Other
    fastembed models (bge, gte, jina, …) don't use prefixes."""
    return "e5" in model_name.lower()


def _apply_prefix(
    texts: Sequence[str], model_name: str, mode: Literal["query", "passage"] | None
) -> list[str]:
    if mode is None or not _needs_e5_prefix(model_name):
        return list(texts)
    prefix = "query: " if mode == "query" else "passage: "
    return [prefix + t for t in texts]


class EmbeddingClient:
    """Async-friendly wrapper around fastembed's synchronous embed()."""

    def __init__(self, _client: object | None = None) -> None:
        # _client kept for signature parity with the previous OpenAI version.
        self._settings = get_settings()

    async def embed(
        self,
        texts: Sequence[str],
        *,
        mode: Literal["query", "passage"] | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts.

        ``mode`` is honoured for E5-family models — pass ``"query"`` for
        search queries and ``"passage"`` for documents being indexed. Other
        models ignore it. Leaving it None disables the prefix entirely.
        """
        if not texts:
            return []
        model = await _get_model()
        prepared = _apply_prefix(texts, self._settings.local_embed_model, mode)
        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(
            None, lambda: [v.tolist() for v in model.embed(prepared)]
        )
        LLM_LATENCY.labels(
            model=self._settings.local_embed_model, purpose=f"embed:{mode or 'plain'}"
        ).observe(time.perf_counter() - start)
        return vectors


# ---------------------------------------------------------------------------
# sparse embeddings (BM25-like for hybrid search)
# ---------------------------------------------------------------------------


async def _get_sparse_model() -> SparseTextEmbedding:
    """Lazy-load the sparse model. Qdrant/bm25 is pure-Python and tiny;
    other choices (bm42, splade) require ONNX inference."""
    global _sparse_model
    if _sparse_model is not None:
        return _sparse_model
    async with _sparse_lock:
        if _sparse_model is None:
            s = get_settings()
            loop = asyncio.get_running_loop()
            _sparse_model = await loop.run_in_executor(
                None, lambda: SparseTextEmbedding(model_name=s.sparse_embed_model)
            )
            log.info("sparse_embed_model_loaded", model=s.sparse_embed_model)
    return _sparse_model


class SparseEmbeddingClient:
    """Async wrapper over fastembed's sparse text embedding. Returns
    ``(indices, values)`` tuples ready for ``qdrant_client.SparseVector``."""

    async def embed(
        self, texts: Sequence[str], *, mode: Literal["query", "passage"] = "passage"
    ) -> list[tuple[list[int], list[float]]]:
        if not texts:
            return []
        model = await _get_sparse_model()
        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        # BM25-style sparse models expose two methods: query_embed / embed.
        # query_embed weights tokens differently for queries (IDF only) vs
        # passages (TF·IDF). Falls back to .embed if not available.
        def _run() -> list[Any]:
            fn = getattr(model, "query_embed", None) if mode == "query" else None
            if fn is None:
                fn = model.embed
            return list(fn(list(texts)))

        raw = await loop.run_in_executor(None, _run)
        LLM_LATENCY.labels(
            model=get_settings().sparse_embed_model, purpose=f"sparse:{mode}"
        ).observe(time.perf_counter() - start)
        out: list[tuple[list[int], list[float]]] = []
        for sv in raw:
            out.append(([int(i) for i in sv.indices], [float(v) for v in sv.values]))
        return out


# ---------------------------------------------------------------------------
# image embeddings (CLIP — visual product search)
# ---------------------------------------------------------------------------


async def _get_image_model() -> ImageEmbedding:
    """Lazy-load the CLIP vision model (~200 MB on first use)."""
    global _image_model
    if _image_model is not None:
        return _image_model
    async with _image_lock:
        if _image_model is None:
            s = get_settings()
            loop = asyncio.get_running_loop()
            _image_model = await loop.run_in_executor(
                None, lambda: ImageEmbedding(model_name=s.image_embed_model)
            )
            log.info("image_embed_model_loaded", model=s.image_embed_model)
    return _image_model


class ImageEmbeddingClient:
    """Async wrapper around fastembed's CLIP vision embedder.

    Accepts raw bytes from any source (WhatsApp media download, scraped
    product image, local file). Converts to PIL.Image internally — fastembed
    requires a PIL Image or filesystem path, not raw bytes.
    """

    async def embed_bytes(
        self, images: Sequence[bytes]
    ) -> list[list[float]]:
        if not images:
            return []
        # Imported lazily so the module import path doesn't require PIL when
        # visual search is unused. (PIL ships with fastembed anyway.)
        from PIL import Image

        model = await _get_image_model()
        loop = asyncio.get_running_loop()
        start = time.perf_counter()

        def _run() -> list[list[float]]:
            pil_images = []
            for raw in images:
                try:
                    img = Image.open(io.BytesIO(raw))
                    # CMYK / palette images → CLIP wants RGB.
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                except Exception as exc:  # noqa: BLE001
                    # Undecodable bytes (SVG/HEIC/corrupt). Raise a clean,
                    # specific error so callers can skip rather than logging
                    # an opaque stack trace.
                    raise ValueError(f"undecodable image: {exc}") from exc
                pil_images.append(img)
            return [v.tolist() for v in model.embed(pil_images)]

        vectors = await loop.run_in_executor(None, _run)
        LLM_LATENCY.labels(
            model=get_settings().image_embed_model, purpose="embed:image"
        ).observe(time.perf_counter() - start)
        return vectors
