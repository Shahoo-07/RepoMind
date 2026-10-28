"""Shared dataclasses used across all layers of the Codebase RAG pipeline."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CodeChunk:
    """Represent one parsed code chunk with metadata and optional embedding."""

    # Identity
    chunk_id: str
    repo_url: str
    file_path: str
    language: str
    chunk_type: str
    name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0

    # Content
    text: str = ""
    summary: str = ""

    # Embedding (filled in later phases)
    embedding: list[float] = field(default_factory=list)
