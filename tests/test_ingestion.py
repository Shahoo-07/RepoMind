"""Phase 1 ingestion smoke test for repository loading, parsing, and chunking."""

from __future__ import annotations

from ingestion.chunker import ingest_repo


def main() -> None:
    """Run a local ingestion smoke test and print deterministic status output."""

    url = "https://github.com/tiangolo/fastapi"
    repo_url, chunks = ingest_repo(url)

    assert len(chunks) > 0, "No chunks produced"
    sample = chunks[0]
    assert sample.chunk_id and sample.file_path and sample.text

    print(f"OK  {len(chunks)} chunks from {repo_url}")
    print(f"    First: [{sample.chunk_type}] {sample.name} @ {sample.file_path}:{sample.start_line}")


if __name__ == "__main__":
    main()
