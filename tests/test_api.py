"""Phase 4 smoke test for FastAPI endpoints and agent-backed chat."""

from __future__ import annotations

import requests

API = "http://localhost:8000"


def main() -> None:
    """Run API health, repository listing, and chat checks against local server."""

    health_response = requests.get(f"{API}/health", timeout=20)
    health_response.raise_for_status()
    assert health_response.json().get("status") == "ok"

    repos_response = requests.get(f"{API}/repos", timeout=20)
    repos_response.raise_for_status()
    repos = repos_response.json().get("repos", [])
    assert repos, "No repos in store"
    repo_url = repos[0]

    chat_response = requests.post(
        f"{API}/chat",
        json={"query": "What does this project do?", "repo_url": repo_url},
        timeout=180,
    )
    assert chat_response.ok, chat_response.text
    answer = chat_response.json().get("answer", "")
    assert isinstance(answer, str) and len(answer) > 20
    print(f"OK  API answered: {answer[:200]}")


if __name__ == "__main__":
    main()
