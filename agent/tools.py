"""LangChain Tool definitions for code search and file-level repository access."""

from __future__ import annotations

import json
from typing import Any

import lancedb
from langchain.tools import Tool

from config import settings
from generation.rag_chain import answer_query
from knowledge_graph.service import build_repo_graph, get_dependencies, get_impact
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank


def _escape_sql_literal(value: str) -> str:
    """Return a safely quoted literal value for LanceDB filter predicates."""

    return value.replace("'", "''")


def _parse_json_input(raw: str) -> dict[str, Any]:
    """Return parsed JSON object for a tool input string."""

    if not raw or not raw.strip():
        raise ValueError("Tool input must be a non-empty JSON string")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON tool input: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Tool input JSON must be an object")
    return parsed


def _search_and_answer(query_and_repo: str) -> str:
    """Return a grounded answer for tool input JSON containing query and repo_url."""

    data = _parse_json_input(query_and_repo)
    query = str(data.get("query", "")).strip()
    repo_url = str(data.get("repo_url", "")).strip()
    history = data.get("history")

    if not query:
        raise ValueError("search_code requires 'query' in input JSON")
    if not repo_url:
        raise ValueError("search_code requires 'repo_url' in input JSON")
    if history is not None and not isinstance(history, list):
        raise ValueError("search_code optional 'history' must be a list")

    docs = hybrid_search(query, repo_url)
    reranked_docs = rerank(query, docs)
    return answer_query(query, reranked_docs, history)


def _list_files(args: str) -> str:
    """Return newline-separated indexed file paths for tool input JSON with repo_url."""

    data = _parse_json_input(args)
    repo_url = str(data.get("repo_url", "")).strip()
    if not repo_url:
        raise ValueError("list_files requires 'repo_url' in input JSON")

    try:
        db = lancedb.connect(settings.lancedb_uri)
        table = db.open_table(settings.lancedb_table)
        predicate = f"repo_url = '{_escape_sql_literal(repo_url)}'"
        rows = table.search().where(predicate, prefilter=True).to_pandas()[["file_path"]].drop_duplicates()
    except (RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(f"Failed listing files for repo '{repo_url}': {exc}") from exc

    file_paths = [str(path) for path in rows["file_path"].tolist() if str(path)]
    if not file_paths:
        raise FileNotFoundError(f"No files found for repo '{repo_url}'")
    return "\n".join(file_paths)


def _get_file(args: str) -> str:
    """Return concatenated file chunk content for tool input JSON with repo and file path."""

    data = _parse_json_input(args)
    repo_url = str(data.get("repo_url", "")).strip()
    file_path = str(data.get("file_path", "")).strip()
    if not repo_url:
        raise ValueError("get_file requires 'repo_url' in input JSON")
    if not file_path:
        raise ValueError("get_file requires 'file_path' in input JSON")

    predicate = (
        f"repo_url = '{_escape_sql_literal(repo_url)}' "
        f"AND file_path = '{_escape_sql_literal(file_path)}'"
    )

    try:
        db = lancedb.connect(settings.lancedb_uri)
        table = db.open_table(settings.lancedb_table)
        rows = table.search().where(predicate, prefilter=True).to_list()
    except (RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(
            f"Failed reading file '{file_path}' for repo '{repo_url}': {exc}"
        ) from exc

    if not rows:
        raise FileNotFoundError(f"No indexed chunks found for '{file_path}' in repo '{repo_url}'")

    ordered = sorted(rows, key=lambda row: int(row.get("start_line", 0)))
    return "\n".join(str(row.get("text", "")) for row in ordered if str(row.get("text", "")))


def _build_graph(args: str) -> str:
    """Build repository knowledge graph for tool input JSON with repo_url."""

    data = _parse_json_input(args)
    repo_url = str(data.get("repo_url", "")).strip()
    if not repo_url:
        raise ValueError("build_graph requires 'repo_url' in input JSON")

    result = build_repo_graph(repo_url)
    return json.dumps(result)


def _graph_dependencies(args: str) -> str:
    """Return dependency analysis JSON for a symbol in a repository graph."""

    data = _parse_json_input(args)
    repo_url = str(data.get("repo_url", "")).strip()
    symbol = str(data.get("symbol", "")).strip()
    max_depth_raw = int(data.get("max_depth", 2))
    if not repo_url:
        raise ValueError("graph_dependencies requires 'repo_url' in input JSON")
    if not symbol:
        raise ValueError("graph_dependencies requires 'symbol' in input JSON")

    result = get_dependencies(repo_url, symbol, max_depth=max_depth_raw)
    return json.dumps(result)


def _graph_impact(args: str) -> str:
    """Return impact analysis JSON for a symbol in a repository graph."""

    data = _parse_json_input(args)
    repo_url = str(data.get("repo_url", "")).strip()
    symbol = str(data.get("symbol", "")).strip()
    max_depth_raw = int(data.get("max_depth", 2))
    if not repo_url:
        raise ValueError("graph_impact requires 'repo_url' in input JSON")
    if not symbol:
        raise ValueError("graph_impact requires 'symbol' in input JSON")

    result = get_impact(repo_url, symbol, max_depth=max_depth_raw)
    return json.dumps(result)


TOOLS = [
    Tool(
        name="search_code",
        func=_search_and_answer,
        description="Search code and answer a question. Input JSON: {\"query\":\"...\",\"repo_url\":\"...\",\"history\":[]}",
    ),
    Tool(
        name="list_files",
        func=_list_files,
        description="List indexed files for a repo. Input JSON: {\"repo_url\":\"...\"}",
    ),
    Tool(
        name="get_file",
        func=_get_file,
        description="Get full indexed content of a file. Input JSON: {\"repo_url\":\"...\",\"file_path\":\"...\"}",
    ),
    Tool(
        name="build_graph",
        func=_build_graph,
        description="Build knowledge graph for a repo. Input JSON: {\"repo_url\":\"...\"}",
    ),
    Tool(
        name="graph_dependencies",
        func=_graph_dependencies,
        description="Get what a symbol depends on. Input JSON: {\"repo_url\":\"...\",\"symbol\":\"...\",\"max_depth\":2}",
    ),
    Tool(
        name="graph_impact",
        func=_graph_impact,
        description="Get what is impacted by a symbol. Input JSON: {\"repo_url\":\"...\",\"symbol\":\"...\",\"max_depth\":2}",
    ),
]
