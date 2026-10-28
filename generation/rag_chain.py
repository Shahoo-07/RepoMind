"""RAG answer generation utilities with citation-oriented prompting."""

from __future__ import annotations

from typing import Any

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import settings

SYSTEM_PROMPT = (
    "You are a codebase analysis assistant. "
    "Answer the user's question using only the provided code chunks. "
    "Always cite source file paths and line ranges in your answer. "
    "If the answer is not present in the chunks, say so explicitly."
)


def _build_llm() -> Any:
    """Return a configured chat model client for answer generation."""

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
            temperature=0.2,
        )

    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_host,
        temperature=0.2,
    )


def build_context(docs: list[dict[str, Any]]) -> str:
    """Return formatted retrieval context string with file and line metadata."""

    parts: list[str] = []
    for doc in docs:
        file_path = str(doc.get("file_path", ""))
        start_line = int(doc.get("start_line", 0))
        end_line = int(doc.get("end_line", 0))
        text = str(doc.get("text", ""))
        header = f"[{file_path} lines {start_line}-{end_line}]"
        parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(parts)


def _history_messages(history: list[dict[str, Any]] | None) -> list[Any]:
    """Return converted LangChain message objects from a history payload."""

    messages: list[Any] = []
    if not history:
        return messages

    for turn in history[-6:]:
        role = str(turn.get("role", ""))
        content = str(turn.get("content", ""))
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def answer_query(query: str, docs: list[dict[str, Any]], history: list[dict[str, Any]] | None = None) -> str:
    """Return a grounded answer for a query using retrieved documents and optional history."""

    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    context = build_context(docs)
    messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(_history_messages(history))
    messages.append(HumanMessage(content=f"Code context:\n\n{context}\n\nQuestion: {query.strip()}"))

    llm = _build_llm()
    try:
        response = llm.invoke(messages)
    except (RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(f"LLM invocation failed: {exc}") from exc

    content = getattr(response, "content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM returned an empty or invalid response")
    return content
