"""Cross-encoder reranking utilities for refining retrieved code chunks."""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from config import settings

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Return a lazily initialized cross-encoder model used for reranking."""

    global _model
    if _model is None:
        try:
            _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Failed to load reranker model: {exc}") from exc
    return _model


def rerank(query: str, docs: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
    """Return top-k reranked documents scored by cross-encoder relevance."""

    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not docs:
        return []

    k = top_k or settings.rerank_top_k
    if k <= 0:
        raise ValueError("top_k must be greater than zero")

    model = _get_model()
    pairs = [(query, str(doc.get("text", ""))[:1000]) for doc in docs]

    try:
        scores = model.predict(pairs)
    except (RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(f"Reranker inference failed: {exc}") from exc

    ranked = sorted(zip(scores, docs), key=lambda item: float(item[0]), reverse=True)
    return [doc for _, doc in ranked[:k]]
