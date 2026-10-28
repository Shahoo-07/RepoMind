"""BM25 retrieval index cached per repository URL for lexical search."""

from __future__ import annotations

from typing import Any

import lancedb
from rank_bm25 import BM25Okapi

from config import settings

_cache: dict[str, tuple[BM25Okapi, list[dict[str, Any]]]] = {}


def _escape_sql_literal(value: str) -> str:
    """Return a safely quoted literal value for LanceDB filter predicates."""

    return value.replace("'", "''")


def _load_repo_docs(repo_url: str) -> list[dict[str, Any]]:
    """Return all stored chunk documents for a repository URL from LanceDB."""

    if not repo_url or not repo_url.strip():
        raise ValueError("repo_url must be a non-empty string")

    try:
        db = lancedb.connect(settings.lancedb_uri)
        table = db.open_table(settings.lancedb_table)
        predicate = f"repo_url = '{_escape_sql_literal(repo_url.strip())}'"
        return table.search().where(predicate, prefilter=True).to_list()
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Failed loading BM25 documents for repo '{repo_url}': {exc}") from exc


def _get_index(repo_url: str) -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """Return cached BM25 index and source docs for the specified repository URL."""

    if repo_url not in _cache:
        docs = _load_repo_docs(repo_url)
        if not docs:
            raise FileNotFoundError(f"No indexed documents found for repo '{repo_url}'")

        corpus = [str(doc.get("text", "")).lower().split() for doc in docs]
        _cache[repo_url] = (BM25Okapi(corpus), docs)
    return _cache[repo_url]


def bm25_search(query: str, repo_url: str, top_k: int) -> list[dict[str, Any]]:
    """Return top-k BM25-ranked documents for a query and repository URL."""

    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    index, docs = _get_index(repo_url)
    scores = index.get_scores(query.lower().split())
    ranked_indexes = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return [docs[idx] for idx in ranked_indexes[:top_k]]


def invalidate_cache(repo_url: str) -> None:
    """Invalidate the BM25 cache entry for a repository URL if it exists."""

    _cache.pop(repo_url, None)
