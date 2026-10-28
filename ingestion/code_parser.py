"""Tree-sitter powered parser that extracts semantic code blocks per file."""

from __future__ import annotations

from typing import Any

from tree_sitter import Language, Parser
from tree_sitter_javascript import language as javascript_language
from tree_sitter_python import language as python_language


def _build_parser(language: str) -> Parser | None:
    """Return a configured tree-sitter parser for supported languages."""

    parser = Parser()
    if language == "python":
        parser.language = Language(python_language())
        return parser
    if language == "javascript":
        parser.language = Language(javascript_language())
        return parser
    return None


def _line_text(lines: list[str], start_line: int, end_line: int) -> str:
    """Return source text segment between 1-based line boundaries."""

    # tree-sitter rows are 0-based; our contracts are 1-based inclusive lines.
    return "\n".join(lines[start_line - 1 : end_line])


def _node_name(node: Any, content: bytes) -> str:
    """Return a normalized semantic name for a syntax node."""

    name_node = node.child_by_field_name("name")
    if name_node is None:
        return "module"
    raw = content[name_node.start_byte : name_node.end_byte]
    return raw.decode("utf-8", errors="ignore") or "module"


def parse_file(file_path: str, language: str, content: str) -> list[dict[str, Any]]:
    """Return list of parsed semantic nodes for one file."""

    parser = _build_parser(language)
    if parser is None:
        end_line = content.count("\n")
        return [
            {
                "name": file_path,
                "type": "module",
                "start_line": 0,
                "end_line": end_line,
                "text": content,
            }
        ]

    source = content.encode("utf-8", errors="ignore")
    tree = parser.parse(source)
    root = tree.root_node
    lines = content.splitlines()

    if language == "python":
        target_types = {"function_definition", "class_definition"}
    else:
        target_types = {
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "method_definition",
            "arrow_function",
        }

    stack = [root]
    chunks: list[dict[str, Any]] = []

    while stack:
        node = stack.pop()
        stack.extend(reversed(node.children))

        if node.type not in target_types:
            continue

        start_line = node.start_point[0]
        end_line = node.end_point[0]
        if end_line < start_line:
            continue

        snippet = _line_text(lines, start_line + 1, end_line + 1)
        if not snippet.strip():
            continue

        kind = "class" if "class" in node.type else "function"
        chunks.append(
            {
                "name": _node_name(node, source),
                "type": kind,
                "start_line": start_line,
                "end_line": end_line,
                "text": snippet,
            }
        )

    if not chunks:
        end_line = content.count("\n")
        chunks.append(
            {
                "name": file_path,
                "type": "module",
                "start_line": 0,
                "end_line": end_line,
                "text": content,
            }
        )

    return chunks


def parse_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return parsed semantic nodes with file metadata for each input file."""

    parsed: list[dict[str, Any]] = []

    for file_item in files:
        if "path" not in file_item or "language" not in file_item or "content" not in file_item:
            raise ValueError("Each file item must contain 'path', 'language', and 'content'")

        file_path = str(file_item["path"])
        language = str(file_item["language"])
        content = str(file_item["content"])

        for node in parse_file(file_path, language, content):
            parsed.append(
                {
                    "file_path": file_path,
                    "language": language,
                    "name": node["name"],
                    "type": node["type"],
                    "start_line": node["start_line"],
                    "end_line": node["end_line"],
                    "text": node["text"],
                }
            )

    return parsed
