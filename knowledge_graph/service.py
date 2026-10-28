"""Service helpers for loading and querying persisted knowledge graphs."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from config import settings
from knowledge_graph.builder import build_knowledge_graph


def _repo_hash(repo_url: str) -> str:
    """Return stable repository hash used for knowledge graph file names."""

    return hashlib.sha256(repo_url.encode("utf-8")).hexdigest()[:16]


def _graph_file(repo_url: str) -> Path:
    """Return expected graph file path for repository URL."""

    root = Path(settings.knowledge_graph_dir).resolve()
    return root / f"kg_{_repo_hash(repo_url)}.json"


def _load_graph(repo_url: str) -> nx.DiGraph:
    """Load persisted graph for repository URL and return directed graph object."""

    if not repo_url or not repo_url.strip():
        raise ValueError("repo_url must be a non-empty string")

    graph_path = _graph_file(repo_url.strip())
    if not graph_path.exists():
        raise FileNotFoundError(f"Knowledge graph not found for repo '{repo_url}'. Build it first.")

    try:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed reading knowledge graph file '{graph_path}': {exc}") from exc

    graph_payload = payload.get("graph")
    if not isinstance(graph_payload, dict):
        raise ValueError(f"Graph payload is invalid in '{graph_path}'")

    return json_graph.node_link_graph(graph_payload, directed=True, edges="edges")


def _resolve_node(graph: nx.DiGraph, symbol: str) -> str:
    """Resolve symbol string to best-matching graph node id."""

    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    target = symbol.strip()
    if target in graph.nodes:
        return target

    for node_id, attrs in graph.nodes(data=True):
        node_symbol = str(attrs.get("symbol", ""))
        if node_symbol == target or node_symbol.endswith(f".{target}"):
            return str(node_id)

    raise FileNotFoundError(f"Symbol '{symbol}' was not found in the knowledge graph")


def _traverse(graph: nx.DiGraph, start: str, reverse: bool, max_depth: int) -> list[dict[str, Any]]:
    """Traverse graph neighbors up to max_depth and return unique relation records."""

    results: list[dict[str, Any]] = []
    seen_nodes: set[str] = {start}
    queue: deque[tuple[str, int]] = deque([(start, 0)])

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue

        edges = graph.in_edges(current, data=True) if reverse else graph.out_edges(current, data=True)
        for source, target, edge_data in edges:
            neighbor = source if reverse else target
            neighbor_str = str(neighbor)
            attrs = graph.nodes.get(neighbor_str, {})
            results.append(
                {
                    "node_id": neighbor_str,
                    "symbol": str(attrs.get("symbol", neighbor_str)),
                    "node_type": str(attrs.get("node_type", "unknown")),
                    "edge_type": str(edge_data.get("edge_type", "related")),
                    "depth": depth + 1,
                }
            )
            if neighbor_str not in seen_nodes:
                seen_nodes.add(neighbor_str)
                queue.append((neighbor_str, depth + 1))

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for item in results:
        key = (item["node_id"], item["edge_type"])
        if key not in unique or item["depth"] < unique[key]["depth"]:
            unique[key] = item

    return sorted(unique.values(), key=lambda row: (row["depth"], row["symbol"]))


def build_repo_graph(repo_url: str) -> dict[str, Any]:
    """Build and persist graph for repository URL and return snapshot dictionary."""

    snapshot = build_knowledge_graph(repo_url)
    return {
        "repo_url": snapshot.repo_url,
        "graph_path": snapshot.graph_path,
        "node_count": snapshot.node_count,
        "edge_count": snapshot.edge_count,
        "created_at": snapshot.created_at,
    }


def get_dependencies(repo_url: str, symbol: str, max_depth: int = 2) -> dict[str, Any]:
    """Return forward dependency relationships for symbol from repository graph."""

    if max_depth <= 0:
        raise ValueError("max_depth must be greater than zero")

    graph = _load_graph(repo_url)
    start = _resolve_node(graph, symbol)
    deps = _traverse(graph, start, reverse=False, max_depth=max_depth)
    return {
        "repo_url": repo_url,
        "symbol": symbol,
        "node_id": start,
        "max_depth": max_depth,
        "dependencies": deps,
        "count": len(deps),
    }


def get_impact(repo_url: str, symbol: str, max_depth: int = 2) -> dict[str, Any]:
    """Return reverse impact relationships for symbol from repository graph."""

    if max_depth <= 0:
        raise ValueError("max_depth must be greater than zero")

    graph = _load_graph(repo_url)
    start = _resolve_node(graph, symbol)
    impact = _traverse(graph, start, reverse=True, max_depth=max_depth)
    return {
        "repo_url": repo_url,
        "symbol": symbol,
        "node_id": start,
        "max_depth": max_depth,
        "impact": impact,
        "count": len(impact),
    }


def list_symbols(repo_url: str, node_type: str | None = None, limit: int = 500) -> dict[str, Any]:
    """Return symbols from repository graph, optionally filtered by node type."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    graph = _load_graph(repo_url)
    wanted_type = (node_type or "").strip().lower()

    rows: list[dict[str, str]] = []
    for node_id, attrs in graph.nodes(data=True):
        current_type = str(attrs.get("node_type", "unknown")).strip().lower()
        if wanted_type and current_type != wanted_type:
            continue

        symbol = str(attrs.get("symbol", node_id)).strip()
        if not symbol:
            continue

        rows.append(
            {
                "node_id": str(node_id),
                "symbol": symbol,
                "node_type": current_type or "unknown",
            }
        )

    rows.sort(key=lambda item: (item["symbol"], item["node_id"]))
    sliced = rows[:limit]

    return {
        "repo_url": repo_url,
        "node_type": wanted_type or None,
        "limit": limit,
        "count": len(sliced),
        "symbols": sliced,
    }
