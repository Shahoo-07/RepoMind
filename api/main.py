"""FastAPI application exposing repository indexing and codebase chat endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from agent.graph import run_agent
from config import settings
from ingestion.chunker import ingest_repo
from knowledge_graph.service import build_repo_graph, get_dependencies, get_impact, list_symbols
from vectorstore.lance_store import delete_repo, list_repos, upsert_chunks

app = FastAPI(title="Codebase RAG API", version="2.0")

_INDEX_STATUS: dict[str, dict[str, Any]] = {}
_INDEX_STATUS_LOCK = Lock()


class IndexRequest(BaseModel):
    """Request payload containing the repository URL to index."""

    repo_url: str


class ChatRequest(BaseModel):
    """Request payload containing query, repo URL, and optional chat history."""

    query: str
    repo_url: str
    history: list[dict[str, Any]] | None = None


class GraphBuildRequest(BaseModel):
    """Request payload containing repository URL for graph building."""

    repo_url: str


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _set_index_status(repo_url: str, status: str, message: str = "", stored: int | None = None) -> None:
    """Store index job status details for a repository URL."""

    with _INDEX_STATUS_LOCK:
        record = {
            "repo_url": repo_url,
            "status": status,
            "message": message,
            "updated_at": _utc_now_iso(),
        }
        if stored is not None:
            record["stored"] = stored
        _INDEX_STATUS[repo_url] = record


def _get_index_status(repo_url: str) -> dict[str, Any] | None:
    """Return latest index job status for a repository URL if available."""

    with _INDEX_STATUS_LOCK:
        return _INDEX_STATUS.get(repo_url)


def _index_repo_job(repo_url: str) -> None:
    """Ingest and upsert all chunks for a repository URL in the background."""

    if not repo_url or not repo_url.strip():
        raise ValueError("repo_url must be a non-empty string")

    repo = repo_url.strip()
    _set_index_status(repo, "running", "Indexing started")

    try:
        resolved_repo_url, chunks = ingest_repo(repo)
        stored = upsert_chunks(chunks)
        if stored <= 0:
            raise RuntimeError(f"No chunks were stored for repo '{resolved_repo_url}'")
        _set_index_status(repo, "completed", f"Indexed {stored} chunks", stored=stored)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        _set_index_status(repo, "failed", str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    """Return service liveness status."""

    return {"status": "ok"}


@app.get("/repos")
def get_repos() -> dict[str, list[str]]:
    """Return all indexed repository URLs from the vector store."""

    return {"repos": list_repos()}


@app.post("/repos/index")
async def index_repo(req: IndexRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Queue repository indexing as a background task and return immediately."""

    if not req.repo_url or not req.repo_url.strip():
        raise HTTPException(status_code=422, detail="repo_url must be a non-empty string")

    repo_url = req.repo_url.strip()
    current = _get_index_status(repo_url)
    if current and current.get("status") == "running":
        return {"status": "indexing_running", "repo_url": repo_url}

    _set_index_status(repo_url, "queued", "Indexing queued")
    background_tasks.add_task(_index_repo_job, repo_url)
    return {"status": "indexing_started", "repo_url": repo_url}


@app.get("/repos/index/status")
def index_status(repo_url: str) -> dict[str, Any]:
    """Return latest indexing status details for a repository URL."""

    if not repo_url or not repo_url.strip():
        raise HTTPException(status_code=422, detail="repo_url query parameter must be provided")

    repo = repo_url.strip()
    status = _get_index_status(repo)
    if status is not None:
        return status

    if repo in list_repos():
        return {
            "repo_url": repo,
            "status": "completed",
            "message": "Repository already indexed",
            "updated_at": _utc_now_iso(),
        }

    return {
        "repo_url": repo,
        "status": "not_found",
        "message": "No indexing job found for this repository",
        "updated_at": _utc_now_iso(),
    }


@app.delete("/repos")
def remove_repo(repo_url: str) -> dict[str, str]:
    """Delete all indexed data for a repository URL and return operation status."""

    if not repo_url or not repo_url.strip():
        raise HTTPException(status_code=422, detail="repo_url query parameter must be provided")

    try:
        delete_repo(repo_url.strip())
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed deleting repository: {exc}") from exc

    return {"status": "deleted", "repo_url": repo_url.strip()}


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, str]:
    """Return a grounded agent answer for a query scoped to one repository URL."""

    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="query must be a non-empty string")
    if not req.repo_url or not req.repo_url.strip():
        raise HTTPException(status_code=422, detail="repo_url must be a non-empty string")

    repo_url = req.repo_url.strip()
    repos = list_repos()
    if repo_url not in repos:
        raise HTTPException(status_code=404, detail=f"{repo_url} not indexed")

    try:
        answer = run_agent(req.query.strip(), repo_url, req.history)
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed generating answer: {exc}") from exc

    return {"answer": answer, "repo_url": repo_url}


@app.post("/graph/build")
def graph_build(req: GraphBuildRequest) -> dict[str, Any]:
    """Build and persist knowledge graph for a repository URL."""

    if not req.repo_url or not req.repo_url.strip():
        raise HTTPException(status_code=422, detail="repo_url must be a non-empty string")

    try:
        return build_repo_graph(req.repo_url.strip())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/graph/dependencies")
def graph_dependencies(repo_url: str, symbol: str, max_depth: int = 2) -> dict[str, Any]:
    """Return dependency relationships for a symbol in the repository graph."""

    try:
        return get_dependencies(repo_url=repo_url, symbol=symbol, max_depth=max_depth)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/graph/impact")
def graph_impact(repo_url: str, symbol: str, max_depth: int = 2) -> dict[str, Any]:
    """Return reverse impact relationships for a symbol in the repository graph."""

    try:
        return get_impact(repo_url=repo_url, symbol=symbol, max_depth=max_depth)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/graph/symbols")
def graph_symbols(repo_url: str, node_type: str | None = None, limit: int = 500) -> dict[str, Any]:
    """Return symbols available in the repository graph for UI/tooling discovery."""

    try:
        return list_symbols(repo_url=repo_url, node_type=node_type, limit=limit)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=False)
