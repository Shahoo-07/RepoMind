"""Dataclasses for knowledge graph nodes, edges, and persisted snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    """Represent one knowledge-graph node with identifying metadata."""

    node_id: str
    node_type: str
    repo_url: str
    file_path: str = ""
    symbol: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    """Represent one directed graph edge between two graph nodes."""

    source: str
    target: str
    edge_type: str
    repo_url: str
    file_path: str = ""


@dataclass(frozen=True)
class KnowledgeGraphSnapshot:
    """Represent persisted graph metadata for one indexed repository."""

    repo_url: str
    graph_path: str
    node_count: int
    edge_count: int
    created_at: str
