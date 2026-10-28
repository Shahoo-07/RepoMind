"""Query rewriting utility to improve retrieval precision without changing user intent."""

from __future__ import annotations

from typing import Any

from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import settings

_REWRITE_PROMPT = (
    "Rewrite the user query to be more specific for retrieving relevant code from a codebase. "
    "Do not change intent. Do not add facts that are not in the query. "
    "Return only the rewritten query."
)


def _build_llm() -> Any:
    """Return a configured chat model client for deterministic query rewriting."""

    if settings.openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI model requested but langchain_openai is unavailable or incompatible"
            ) from exc
        return ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.0,
        )

    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_host,
        temperature=0.0,
    )


def rewrite_query(query: str) -> str:
    """Return a retrieval-friendly rewrite for a user query while preserving intent."""

    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    stripped = query.strip()
    llm = _build_llm()
    messages = [
        SystemMessage(content=_REWRITE_PROMPT),
        HumanMessage(content=stripped),
    ]

    try:
        response = llm.invoke(messages)
    except (RuntimeError, ValueError, OSError) as exc:
        # Fall back to the original query if rewrite fails to avoid breaking retrieval.
        print(f"[query_rewriter] rewrite failed, using original query: {exc}")
        return stripped

    content = getattr(response, "content", "")
    rewritten = content.strip() if isinstance(content, str) else ""
    if not rewritten:
        return stripped

    return rewritten
