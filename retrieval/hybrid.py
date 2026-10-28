"""Hybrid retrieval using Reciprocal Rank Fusion over vector and BM25 results."""

from __future__ import annotations

from typing import Any

from config import settings
from retrieval.bm25_index import bm25_search
from retrieval.query_rewriter import rewrite_query
from retrieval.retriever import vector_search

RRF_K = 60


def hybrid_search(query: str, repo_url: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Return fused top-k documents by combining vector and BM25 rankings."""

    k = top_k or settings.retrieval_top_k
    if k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    original_query = query.strip()
    rewritten_query = rewrite_query(original_query)
    print(f"[hybrid_search] original query: {original_query}")
    print(f"[hybrid_search] rewritten query: {rewritten_query}")

    vector_results = vector_search(rewritten_query, repo_url, k)
    bm25_results = bm25_search(rewritten_query, repo_url, k)

    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}

    for rank, doc in enumerate(vector_results):
        chunk_id = str(doc.get("chunk_id", ""))
        if not chunk_id:
            continue
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        docs[chunk_id] = doc

    for rank, doc in enumerate(bm25_results):
        chunk_id = str(doc.get("chunk_id", ""))
        if not chunk_id:
            continue
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        docs[chunk_id] = doc

    ranked_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [docs[cid] for cid in ranked_chunk_ids[:k]]
