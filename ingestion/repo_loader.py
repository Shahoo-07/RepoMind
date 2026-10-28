"""Repository loader for cloning/walking source repos and reading supported files."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from git import GitCommandError, Repo

from config import settings


SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "javascript",
    ".java": "java",
    ".go": "go",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".md": "markdown",
    ".txt": "text",
    ".toml": "config",
    ".yaml": "config",
    ".yml": "config",
    ".json": "config",
}


def _is_repo_url(source: str) -> bool:
    """Return True when the source string looks like an HTTP(S) git URL."""

    return source.startswith("http://") or source.startswith("https://")


def _resolve_language(path: Path) -> str | None:
    """Return normalized language name for a file path or None if unsupported."""

    if path.name.lower() == ".env.example":
        return "config"

    ext = path.suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]

    if ext in settings.include_exts:
        return "other"

    return None


def _read_file_text(path: Path) -> str:
    """Return UTF-8 text from disk, ignoring invalid bytes."""

    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        raise OSError(f"Failed to read file '{path}': {exc}") from exc


def _walk_repo_files(root: Path) -> list[dict[str, Any]]:
    """Return file dictionaries with path/language/content for supported repo files."""

    files: list[dict[str, Any]] = []
    excluded = set(settings.exclude_dirs)

    for path in root.rglob("*"):
        if any(part in excluded for part in path.parts):
            continue
        if not path.is_file():
            continue

        try:
            if path.stat().st_size > settings.max_file_bytes:
                continue
        except OSError as exc:
            raise OSError(f"Failed to stat file '{path}': {exc}") from exc

        language = _resolve_language(path)
        if language is None:
            continue

        content = _read_file_text(path)
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "language": language,
                "content": content,
            }
        )

    return files


def _clone_repo(source: str) -> Path:
    """Clone a repository URL and return the cloned root directory path."""

    target_dir = Path(tempfile.mkdtemp(prefix="codebase_rag_repo_"))
    try:
        Repo.clone_from(source, target_dir.as_posix(), depth=1)
    except GitCommandError as exc:
        raise ValueError(f"Failed to clone repository '{source}': {exc}") from exc
    return target_dir


def load_repo(source: str) -> tuple[str, list[dict[str, Any]]]:
    """Return repository URL/path and raw file dictionaries from URL or local path."""

    if not source or not source.strip():
        raise ValueError("source must be a non-empty repository URL or local path")

    source = source.strip()

    if _is_repo_url(source):
        root = _clone_repo(source)
        repo_url = source
    else:
        root = Path(source).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Local repository path does not exist: '{root}'")
        if not root.is_dir():
            raise ValueError(f"Local repository path is not a directory: '{root}'")
        repo_url = root.as_posix()

    files = _walk_repo_files(root)
    if not files:
        raise ValueError(f"No supported source files were found under '{root}'")

    return repo_url, files
