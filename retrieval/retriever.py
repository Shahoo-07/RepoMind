"""Vector retrieval wrapper for querying LanceDB-backed semantic search."""

from __future__ import annotations

from config import settings
from vectorstore.lance_store import search_similar


def vector_search(query: str, repo_url: str, top_k: int | None = None) -> list[dict]:
    """Return vector-search results for a query and repository URL."""

    k = top_k or settings.retrieval_top_k
    if k <= 0:
        raise ValueError("top_k must be greater than zero")
    return search_similar(query, repo_url, k)
