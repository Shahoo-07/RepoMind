"""LangGraph ReAct agent wiring for repository-aware codebase Q&A."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from langchain.schema import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from ollama import ResponseError

from agent.tools import TOOLS
from config import settings


def _build_agent_llm() -> Any:
    """Return a configured chat model compatible with LangGraph tool calling."""

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
            temperature=0,
        )

    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_host,
        temperature=0,
    )


agent_executor = create_react_agent(_build_agent_llm(), TOOLS)


def _is_runner_terminated_error(exc: BaseException) -> bool:
    """Return True when an exception indicates Ollama runner termination."""

    return "runner process has terminated" in str(exc).lower()


def _reset_ollama_chat_model() -> None:
    """Reset the configured Ollama chat model process using the local CLI."""

    ollama_cli = shutil.which("ollama")
    if ollama_cli is None:
        raise FileNotFoundError("'ollama' CLI was not found on PATH for runner recovery")

    try:
        subprocess.run(
            [ollama_cli, "stop", settings.llm_model],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Failed resetting Ollama model '{settings.llm_model}': {exc}") from exc


def _invoke_agent(messages: list[Any]) -> dict[str, Any]:
    """Invoke the LangGraph agent and return the raw execution result dictionary."""

    return agent_executor.invoke({"messages": messages})


def run_agent(query: str, repo_url: str, history: list[dict] | None = None) -> str:
    """Return the final agent response for a query scoped to a repository URL."""

    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not repo_url or not repo_url.strip():
        raise ValueError("repo_url must be a non-empty string")

    messages: list[Any] = []
    if history:
        for turn in history[-4:]:
            role = str(turn.get("role", ""))
            content = str(turn.get("content", ""))
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    full_query = (
        "Use tool inputs as JSON strings. "
        f"Repository: {repo_url.strip()}. "
        f"User question: {query.strip()}"
    )
    messages.append(HumanMessage(content=full_query))

    try:
        result = _invoke_agent(messages)
    except (ResponseError, RuntimeError, ValueError, OSError) as exc:
        if _is_runner_terminated_error(exc):
            _reset_ollama_chat_model()
            try:
                result = _invoke_agent(messages)
            except (ResponseError, RuntimeError, ValueError, OSError) as retry_exc:
                raise RuntimeError(f"Agent execution failed after Ollama recovery: {retry_exc}") from retry_exc
        else:
            raise RuntimeError(f"Agent execution failed: {exc}") from exc

    output_messages = result.get("messages", []) if isinstance(result, dict) else []
    if not output_messages:
        raise RuntimeError("Agent returned no messages")

    last = output_messages[-1]
    content = getattr(last, "content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Agent returned an empty response")
    return content
