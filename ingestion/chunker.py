"""Chunk construction utilities for deterministic code chunk generation."""

from __future__ import annotations

import hashlib
from typing import Any

from ingestion.code_parser import parse_file
from ingestion.repo_loader import load_repo
from shared.models import CodeChunk


def _chunk_id(repo_url: str, file_path: str, start_line: int) -> str:
    """Return deterministic chunk id from repo/file/start line."""

    payload = f"{repo_url}|{file_path}|{start_line}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def ingest_repo(source: str) -> tuple[str, list[CodeChunk]]:
    """Main Phase 1 entrypoint that returns repo_url and CodeChunk list."""

    repo_url, files = load_repo(source)

    chunks: list[CodeChunk] = []
    for file_item in files:
        file_path = str(file_item["path"])
        language = str(file_item["language"])
        content = str(file_item["content"])
        parsed_nodes = parse_file(file_path, language, content)

        for node in parsed_nodes:
            start_line = int(node["start_line"])
            end_line = int(node["end_line"])
            chunk_id = _chunk_id(repo_url, file_path, start_line)
            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    repo_url=repo_url,
                    file_path=file_path,
                    language=language,
                    chunk_type=str(node["type"]),
                    name=str(node.get("name", "")),
                    start_line=start_line,
                    end_line=end_line,
                    text=str(node["text"]),
                )
            )

    return repo_url, chunks


def build_chunks(source: str) -> list[CodeChunk]:
    """Compatibility wrapper that returns only chunks."""

    _, chunks = ingest_repo(source)

    return chunks


def chunks_to_dicts(chunks: list[CodeChunk]) -> list[dict[str, Any]]:
    """Return serializable dictionaries preserving CodeChunk field names."""

    return [
        {
            "chunk_id": chunk.chunk_id,
            "repo_url": chunk.repo_url,
            "file_path": chunk.file_path,
            "language": chunk.language,
            "chunk_type": chunk.chunk_type,
            "name": chunk.name,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "text": chunk.text,
            "summary": chunk.summary,
            "embedding": chunk.embedding,
        }
        for chunk in chunks
    ]
