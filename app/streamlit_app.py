"""Streamlit frontend that communicates with the FastAPI backend over HTTP only."""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st

API = "http://localhost:8000"


def _safe_json(response: requests.Response) -> dict[str, Any]:
    """Return parsed JSON payload from a response or raise ValueError."""

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(f"Response was not valid JSON: {response.text[:200]}") from exc
    if not isinstance(payload, dict):
        raise ValueError("API response payload must be a JSON object")
    return payload


def _fetch_repos() -> list[str]:
    """Return indexed repository URLs from the backend API."""

    try:
        response = requests.get(f"{API}/repos", timeout=20)
        response.raise_for_status()
        payload = _safe_json(response)
        repos = payload.get("repos", [])
        if not isinstance(repos, list):
            raise ValueError("'repos' field is not a list")
        return [str(repo) for repo in repos if str(repo)]
    except (requests.RequestException, ValueError):
        return []


def _index_repo(repo_url: str) -> str:
    """Request background indexing for a repo URL and return status text."""

    if not repo_url or not repo_url.strip():
        raise ValueError("repo_url must be a non-empty string")

    try:
        response = requests.post(
            f"{API}/repos/index",
            json={"repo_url": repo_url.strip()},
            timeout=30,
        )
        response.raise_for_status()
        payload = _safe_json(response)
        return str(payload.get("status", "indexing_started"))
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to start indexing: {exc}") from exc


def _index_status(repo_url: str) -> dict[str, Any]:
    """Return latest backend indexing status for a repository URL."""

    if not repo_url or not repo_url.strip():
        raise ValueError("repo_url must be a non-empty string")

    try:
        response = requests.get(
            f"{API}/repos/index/status",
            params={"repo_url": repo_url.strip()},
            timeout=20,
        )
        response.raise_for_status()
        return _safe_json(response)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch indexing status: {exc}") from exc


def _chat(query: str, repo_url: str, history: list[dict[str, str]]) -> str:
    """Send chat payload to API and return assistant answer text."""

    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not repo_url or not repo_url.strip():
        raise ValueError("repo_url must be a non-empty string")

    payload = {
        "query": query.strip(),
        "repo_url": repo_url.strip(),
        "history": history,
    }

    try:
        response = requests.post(f"{API}/chat", json=payload, timeout=180)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to call chat API: {exc}") from exc

    if not response.ok:
        raise RuntimeError(f"Chat API returned {response.status_code}: {response.text[:300]}")

    data = _safe_json(response)
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Chat API returned an invalid or empty 'answer'")
    return answer


def _build_graph(repo_url: str) -> dict[str, Any]:
    """Request graph build for repository URL and return backend response payload."""

    if not repo_url or not repo_url.strip():
        raise ValueError("repo_url must be a non-empty string")
    try:
        response = requests.post(
            f"{API}/graph/build",
            json={"repo_url": repo_url.strip()},
            timeout=180,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to call graph build API: {exc}") from exc

    if not response.ok:
        raise RuntimeError(f"Graph build API returned {response.status_code}: {response.text[:300]}")
    return _safe_json(response)


def _graph_dependencies(repo_url: str, symbol: str, max_depth: int) -> dict[str, Any]:
    """Request dependency analysis from backend graph endpoint and return payload."""

    if not repo_url or not repo_url.strip():
        raise ValueError("repo_url must be a non-empty string")
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    try:
        response = requests.get(
            f"{API}/graph/dependencies",
            params={"repo_url": repo_url.strip(), "symbol": symbol.strip(), "max_depth": max_depth},
            timeout=60,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to call graph dependencies API: {exc}") from exc

    if not response.ok:
        raise RuntimeError(f"Graph dependencies API returned {response.status_code}: {response.text[:300]}")
    return _safe_json(response)


def _graph_impact(repo_url: str, symbol: str, max_depth: int) -> dict[str, Any]:
    """Request impact analysis from backend graph endpoint and return payload."""

    if not repo_url or not repo_url.strip():
        raise ValueError("repo_url must be a non-empty string")
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    try:
        response = requests.get(
            f"{API}/graph/impact",
            params={"repo_url": repo_url.strip(), "symbol": symbol.strip(), "max_depth": max_depth},
            timeout=60,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to call graph impact API: {exc}") from exc

    if not response.ok:
        raise RuntimeError(f"Graph impact API returned {response.status_code}: {response.text[:300]}")
    return _safe_json(response)


def _graph_symbols(repo_url: str, node_type: str | None = None, limit: int = 500) -> list[dict[str, str]]:
    """Return available symbols from repository graph for UI selection."""

    if not repo_url or not repo_url.strip():
        raise ValueError("repo_url must be a non-empty string")

    params: dict[str, Any] = {"repo_url": repo_url.strip(), "limit": limit}
    if node_type:
        params["node_type"] = node_type

    try:
        response = requests.get(f"{API}/graph/symbols", params=params, timeout=60)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to call graph symbols API: {exc}") from exc

    if not response.ok:
        raise RuntimeError(f"Graph symbols API returned {response.status_code}: {response.text[:300]}")

    payload = _safe_json(response)
    symbols = payload.get("symbols", [])
    if not isinstance(symbols, list):
        raise RuntimeError("Graph symbols API returned invalid 'symbols' payload")

    cleaned: list[dict[str, str]] = []
    for item in symbols:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip()
        if not symbol:
            continue
        cleaned.append(
            {
                "symbol": symbol,
                "node_id": str(item.get("node_id", "")),
                "node_type": str(item.get("node_type", "unknown")),
            }
        )

    return cleaned


def main() -> None:
    """Render the Streamlit UI and handle repository indexing and chat requests."""

    st.set_page_config(page_title="Codebase RAG", layout="wide")
    st.title("Codebase RAG")

    with st.sidebar:
        st.header("Repository")
        repo_input = st.text_input("GitHub URL", placeholder="https://github.com/...")
        if st.button("Index Repo"):
            if not repo_input.strip():
                st.error("Enter a repository URL before indexing.")
            else:
                try:
                    status = _index_repo(repo_input)
                    st.session_state["last_index_repo"] = repo_input.strip()
                    st.success(status)
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))

        repos_for_status = _fetch_repos()
        default_repo = st.session_state.get("last_index_repo", "")
        status_target = st.text_input(
            "Check Index Status For",
            value=str(default_repo),
            placeholder="https://github.com/...",
        )
        if repos_for_status:
            st.caption("Or choose from indexed repos")
            selected_status_repo = st.selectbox("Indexed repos", repos_for_status)
            if selected_status_repo:
                status_target = selected_status_repo

        if st.button("Refresh Index Status"):
            st.rerun()

        if status_target.strip():
            try:
                job = _index_status(status_target.strip())
                job_status = str(job.get("status", "unknown"))
                job_message = str(job.get("message", ""))
                if job_status == "completed":
                    st.success(f"Index status: {job_status} - {job_message}")
                elif job_status == "failed":
                    st.error(f"Index status: {job_status} - {job_message}")
                else:
                    st.info(f"Index status: {job_status} - {job_message}")
            except (ValueError, RuntimeError) as exc:
                st.warning(str(exc))

        st.divider()
        active_repo_options: list[str] = []
        candidates = [*repos_for_status, str(default_repo), str(status_target)]
        for candidate in candidates:
            repo = candidate.strip()
            if repo and repo not in active_repo_options:
                active_repo_options.append(repo)

        selected_repo = st.selectbox("Active repo", active_repo_options) if active_repo_options else None

        st.divider()
        st.header("Knowledge Graph")
        kg_symbol = st.text_input("Symbol", placeholder="module.function_name")
        kg_depth = st.slider("Graph depth", min_value=1, max_value=5, value=2, step=1)

        symbol_choices: list[dict[str, str]] = []
        if selected_repo:
            if st.button("Load Symbols"):
                st.session_state["kg_symbols_repo"] = selected_repo
                st.rerun()

            symbols_repo = str(st.session_state.get("kg_symbols_repo", "")).strip()
            if symbols_repo == selected_repo:
                try:
                    symbol_choices = _graph_symbols(selected_repo, limit=500)
                except (ValueError, RuntimeError) as exc:
                    st.warning(str(exc))

        if symbol_choices:
            display_options = [f"{item['symbol']} ({item['node_type']})" for item in symbol_choices]
            selected_display = st.selectbox("Pick Symbol", display_options)
            selected_index = display_options.index(selected_display)
            kg_symbol = symbol_choices[selected_index]["symbol"]

        if st.button("Build Graph"):
            if not selected_repo:
                st.error("Select an active repo to build graph.")
            else:
                try:
                    result = _build_graph(selected_repo)
                    st.session_state["kg_symbols_repo"] = selected_repo
                    st.success(
                        f"Graph built: nodes={result.get('node_count', 0)}, edges={result.get('edge_count', 0)}"
                    )
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))

        if st.button("Analyze Dependencies"):
            if not selected_repo:
                st.error("Select an active repo first.")
            elif not kg_symbol.strip():
                st.error("Enter a symbol first.")
            else:
                try:
                    result = _graph_dependencies(selected_repo, kg_symbol, kg_depth)
                    st.json(result)
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))

        if st.button("Analyze Impact"):
            if not selected_repo:
                st.error("Select an active repo first.")
            elif not kg_symbol.strip():
                st.error("Enter a symbol first.")
            else:
                try:
                    result = _graph_impact(selected_repo, kg_symbol, kg_depth)
                    st.json(result)
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about the codebase...")
    if prompt:
        if not selected_repo:
            st.error("Select or index a repo first.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = _chat(prompt, selected_repo, st.session_state.messages[:-1])
                except (ValueError, RuntimeError) as exc:
                    answer = str(exc)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
