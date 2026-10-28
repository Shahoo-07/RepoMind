"""LanceDB persistence and retrieval utilities for embedded code chunks."""

from __future__ import annotations

from typing import Any

import lancedb
import pyarrow as pa
from lancedb.table import Table

from config import settings
from embeddings.embedder import embed_texts
from shared.models import CodeChunk

SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string()),
        pa.field("repo_url", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("language", pa.string()),
        pa.field("chunk_type", pa.string()),
        pa.field("name", pa.string()),
        pa.field("start_line", pa.int32()),
        pa.field("end_line", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("summary", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), settings.embedding_dim)),
    ]
)


def _escape_sql_literal(value: str) -> str:
    """Return a safely quoted literal value for LanceDB filter predicates."""

    return value.replace("'", "''")


def _repo_filter(repo_url: str) -> str:
    """Return a validated repo_url equality predicate string."""

    if not repo_url or not repo_url.strip():
        raise ValueError("repo_url must be a non-empty string")
    escaped = _escape_sql_literal(repo_url.strip())
    return f"repo_url = '{escaped}'"


def _connect_db() -> Any:
    """Return a LanceDB database connection for the configured URI."""

    try:
        return lancedb.connect(settings.lancedb_uri)
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to connect LanceDB at '{settings.lancedb_uri}': {exc}") from exc


def _schema_matches(table: Table) -> bool:
    """Return True when an existing table schema matches the expected Phase 2 schema."""

    try:
        existing = table.schema
    except (RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(f"Failed reading existing table schema: {exc}") from exc

    expected_names = [field.name for field in SCHEMA]
    existing_names = [field.name for field in existing]
    if expected_names != existing_names:
        return False

    expected_embedding = SCHEMA.field("embedding").type
    existing_embedding = existing.field("embedding").type
    return expected_embedding == existing_embedding


def _get_table() -> Table:
    """Return an opened LanceDB table, creating it with schema when missing."""

    db = _connect_db()
    try:
        if settings.lancedb_table in db.table_names():
            table = db.open_table(settings.lancedb_table)
            if _schema_matches(table):
                return table
            db.drop_table(settings.lancedb_table)
        return db.create_table(settings.lancedb_table, schema=SCHEMA)
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to open/create table '{settings.lancedb_table}': {exc}") from exc


def _validate_chunks(chunks: list[CodeChunk]) -> str:
    """Return validated repo_url after ensuring chunk list consistency and non-emptiness."""

    if not chunks:
        raise ValueError("chunks must not be empty")

    repo_url = chunks[0].repo_url
    if not repo_url:
        raise ValueError("chunks[0].repo_url must not be empty")

    for index, chunk in enumerate(chunks):
        if chunk.repo_url != repo_url:
            raise ValueError(
                f"All chunks must share the same repo_url; mismatch at index {index} "
                f"('{chunk.repo_url}' != '{repo_url}')"
            )
        if not chunk.chunk_id:
            raise ValueError(f"chunk_id is missing at index {index}")
        if not chunk.file_path:
            raise ValueError(f"file_path is missing at index {index}")
        if not chunk.text:
            raise ValueError(f"text is missing at index {index}")

    return repo_url


def _rows_from_chunks(chunks: list[CodeChunk], vectors: list[list[float]]) -> list[dict[str, Any]]:
    """Return LanceDB row dictionaries mapped 1-for-1 from CodeChunk fields and vectors."""

    if len(chunks) != len(vectors):
        raise ValueError(
            "Embedding result count mismatch: "
            f"expected {len(chunks)}, got {len(vectors)}"
        )

    rows: list[dict[str, Any]] = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        if len(vector) != settings.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch at index {index}: "
                f"expected {settings.embedding_dim}, got {len(vector)}"
            )

        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "repo_url": chunk.repo_url,
                "file_path": chunk.file_path,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
                "name": chunk.name or "",
                "start_line": int(chunk.start_line),
                "end_line": int(chunk.end_line),
                "text": chunk.text,
                "summary": chunk.summary,
                "embedding": [float(value) for value in vector],
            }
        )

    return rows


def upsert_chunks(chunks: list[CodeChunk]) -> int:
    """Embed and upsert chunks idempotently for one repository and return stored count."""

    if not chunks:
        return 0

    repo_url = _validate_chunks(chunks)
    table = _get_table()
    predicate = _repo_filter(repo_url)

    try:
        table.delete(predicate)
    except (RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(f"Failed deleting existing rows for repo '{repo_url}': {exc}") from exc

    vectors = embed_texts([chunk.text for chunk in chunks])
    rows = _rows_from_chunks(chunks, vectors)

    try:
        table.add(rows)
    except (RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(f"Failed inserting rows for repo '{repo_url}': {exc}") from exc

    return len(rows)


def search_similar(query: str, repo_url: str, top_k: int) -> list[dict[str, Any]]:
    """Return top-k vector-similar rows filtered to a single repository URL."""

    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    table = _get_table()
    query_vector = embed_texts([query.strip()])[0]

    try:
        return (
            table.search(query_vector)
            .where(_repo_filter(repo_url), prefilter=True)
            .limit(top_k)
            .to_list()
        )
    except (RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(f"Vector search failed for repo '{repo_url}': {exc}") from exc


def list_repos() -> list[str]:
    """Return distinct repository URLs currently stored in LanceDB."""

    table = _get_table()
    try:
        dataframe = table.to_pandas()[["repo_url"]].drop_duplicates()
    except (RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(f"Failed reading repository list from LanceDB: {exc}") from exc
    except ImportError as exc:
        raise RuntimeError(f"Failed reading repository list from LanceDB: {exc}") from exc

    return [str(value) for value in dataframe["repo_url"].tolist() if str(value)]


def delete_repo(repo_url: str) -> None:
    """Delete all rows associated with a repository URL from LanceDB."""

    table = _get_table()
    try:
        table.delete(_repo_filter(repo_url))
    except (RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(f"Failed deleting repo '{repo_url}' from LanceDB: {exc}") from exc
