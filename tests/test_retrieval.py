"""Phase 3 smoke test for agentic retrieval and grounded answer generation."""

from __future__ import annotations

from agent.graph import run_agent
from vectorstore.lance_store import list_repos


def main() -> None:
    """Run the Phase 3 smoke test and print a short answer preview."""

    repos = list_repos()
    assert repos, "No repos indexed. Run test_vectorstore.py first."

    repo_url = repos[0]
    answer = run_agent("What does this project do?", repo_url)
    assert isinstance(answer, str) and len(answer) > 20

    print(f"OK  agent answered ({len(answer)} chars):")
    print(f"    {answer[:200]}...")


if __name__ == "__main__":
    main()
