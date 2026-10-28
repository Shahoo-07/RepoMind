"""Build and persist repository-level code dependency knowledge graphs."""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from config import settings
from ingestion.repo_loader import load_repo
from shared.kg_models import GraphEdge, GraphNode, KnowledgeGraphSnapshot


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _repo_hash(repo_url: str) -> str:
    """Return stable repository hash used for persisted graph file names."""

    return hashlib.sha256(repo_url.encode("utf-8")).hexdigest()[:16]


def _module_name(file_path: str) -> str:
    """Return Python module name derived from repository-relative file path."""

    module = file_path.replace("/", ".")
    if module.endswith(".py"):
        module = module[:-3]
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return module or file_path


def _node_id(node_type: str, symbol: str) -> str:
    """Return canonical node id from node type and symbol string."""

    return f"{node_type}:{symbol}"


def _collect_definitions(tree: ast.AST, module_name: str) -> dict[str, str]:
    """Return mapping of defined function symbols to graph node identifiers."""

    definitions: dict[str, str] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if self.class_stack:
                symbol = f"{module_name}.{'.'.join(self.class_stack)}.{node.name}"
            else:
                symbol = f"{module_name}.{node.name}"
            definitions[symbol] = _node_id("function", symbol)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.visit_FunctionDef(node)

    Visitor().visit(tree)
    return definitions


def _call_name(call_node: ast.Call) -> str:
    """Return normalized textual name for a function call AST node."""

    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = [func.attr]
        value = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        parts.reverse()
        return ".".join(parts)
    return "<dynamic_call>"


def _add_node(graph: nx.DiGraph, node: GraphNode) -> None:
    """Add one graph node and associated metadata to the directed graph."""

    graph.add_node(
        node.node_id,
        node_type=node.node_type,
        repo_url=node.repo_url,
        file_path=node.file_path,
        symbol=node.symbol,
        metadata=node.metadata,
    )


def _add_edge(graph: nx.DiGraph, edge: GraphEdge) -> None:
    """Add one typed directed edge and associated metadata to the graph."""

    graph.add_edge(
        edge.source,
        edge.target,
        edge_type=edge.edge_type,
        repo_url=edge.repo_url,
        file_path=edge.file_path,
    )


def _persist_graph(repo_url: str, graph: nx.DiGraph) -> str:
    """Persist one repository graph to JSON and return saved absolute file path."""

    root = Path(settings.knowledge_graph_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    graph_file = root / f"kg_{_repo_hash(repo_url)}.json"

    payload = {
        "repo_url": repo_url,
        "created_at": _utc_now_iso(),
        "graph": json_graph.node_link_data(graph, edges="edges"),
    }
    try:
        graph_file.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Failed writing knowledge graph file '{graph_file}': {exc}") from exc

    return graph_file.as_posix()


def build_knowledge_graph(source: str) -> KnowledgeGraphSnapshot:
    """Build and persist a knowledge graph for a repository source URL or local path."""

    repo_url, files = load_repo(source)
    graph = nx.DiGraph()

    for file_item in files:
        file_path = str(file_item.get("path", ""))
        language = str(file_item.get("language", ""))
        content = str(file_item.get("content", ""))

        if not file_path or language != "python":
            continue

        module = _module_name(file_path)
        module_id = _node_id("module", module)
        _add_node(
            graph,
            GraphNode(
                node_id=module_id,
                node_type="module",
                repo_url=repo_url,
                file_path=file_path,
                symbol=module,
            ),
        )

        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError:
            continue

        definitions = _collect_definitions(tree, module)
        local_lookup = {symbol.rsplit(".", 1)[-1]: node_id for symbol, node_id in definitions.items()}

        for symbol, node_id_value in definitions.items():
            _add_node(
                graph,
                GraphNode(
                    node_id=node_id_value,
                    node_type="function",
                    repo_url=repo_url,
                    file_path=file_path,
                    symbol=symbol,
                ),
            )

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name
                    import_id = _node_id("import", target)
                    _add_node(
                        graph,
                        GraphNode(
                            node_id=import_id,
                            node_type="import",
                            repo_url=repo_url,
                            file_path=file_path,
                            symbol=target,
                        ),
                    )
                    _add_edge(
                        graph,
                        GraphEdge(
                            source=module_id,
                            target=import_id,
                            edge_type="imports",
                            repo_url=repo_url,
                            file_path=file_path,
                        ),
                    )
            elif isinstance(node, ast.ImportFrom):
                module_base = node.module or ""
                for alias in node.names:
                    target = f"{module_base}.{alias.name}" if module_base else alias.name
                    import_id = _node_id("import", target)
                    _add_node(
                        graph,
                        GraphNode(
                            node_id=import_id,
                            node_type="import",
                            repo_url=repo_url,
                            file_path=file_path,
                            symbol=target,
                        ),
                    )
                    _add_edge(
                        graph,
                        GraphEdge(
                            source=module_id,
                            target=import_id,
                            edge_type="imports",
                            repo_url=repo_url,
                            file_path=file_path,
                        ),
                    )

        class CallVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.class_stack: list[str] = []
                self.current_function: str = ""

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.class_stack.append(node.name)
                self.generic_visit(node)
                self.class_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                if self.class_stack:
                    symbol = f"{module}.{'.'.join(self.class_stack)}.{node.name}"
                else:
                    symbol = f"{module}.{node.name}"
                self.current_function = symbol
                self.generic_visit(node)
                self.current_function = ""

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.visit_FunctionDef(node)

            def visit_Call(self, node: ast.Call) -> None:
                if not self.current_function:
                    self.generic_visit(node)
                    return

                source_id = definitions.get(self.current_function)
                if source_id is None:
                    self.generic_visit(node)
                    return

                called = _call_name(node)
                target_id = local_lookup.get(called)
                if target_id is None:
                    target_symbol = f"{module}.{called}" if called not in {"<dynamic_call>"} else called
                    target_id = _node_id("external_call", target_symbol)
                    _add_node(
                        graph,
                        GraphNode(
                            node_id=target_id,
                            node_type="external_call",
                            repo_url=repo_url,
                            file_path=file_path,
                            symbol=target_symbol,
                        ),
                    )

                _add_edge(
                    graph,
                    GraphEdge(
                        source=source_id,
                        target=target_id,
                        edge_type="calls",
                        repo_url=repo_url,
                        file_path=file_path,
                    ),
                )
                self.generic_visit(node)

        CallVisitor().visit(tree)

    graph_path = _persist_graph(repo_url, graph)
    return KnowledgeGraphSnapshot(
        repo_url=repo_url,
        graph_path=graph_path,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        created_at=_utc_now_iso(),
    )
