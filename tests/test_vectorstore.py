"""Phase 2 vectorstore smoke test for embedding, upsert, and retrieval."""

from __future__ import annotations

from ingestion.chunker import ingest_repo
from vectorstore.lance_store import list_repos, search_similar, upsert_chunks


def main() -> None:
    """Run Phase 2 smoke test and print deterministic status output."""

    url = "https://github.com/tiangolo/fastapi"
    _, chunks = ingest_repo(url)
    small = chunks[:50]

    stored = upsert_chunks(small)
    assert stored == 50, f"Expected 50 stored, got {stored}"

    results = search_similar("authentication login user", url, top_k=5)
    assert len(results) > 0, "No vector search results returned"
    assert "text" in results[0] and "file_path" in results[0], "Missing expected result fields"

    repos = list_repos()
    assert url in repos, "Repository URL not found in LanceDB"

    print(f"OK  stored={stored}, top result: {results[0]['file_path']}")


if __name__ == "__main__":
    main()
