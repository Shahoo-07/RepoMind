"""Ollama embedding utilities for Phase 2 vector indexing."""

from __future__ import annotations

import shutil
import subprocess
import ollama
from ollama import ResponseError

from config import settings


def _client() -> ollama.Client:
    """Return an initialized Ollama client for embedding requests."""

    if not settings.ollama_host:
        raise ValueError("ollama_host is not configured in settings")
    return ollama.Client(host=settings.ollama_host)


def _normalize_texts(texts: list[str]) -> list[str]:
    """Return sanitized and 8000-character-truncated text inputs for embedding."""

    normalized: list[str] = []
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("All embedding inputs must be strings")
        cleaned = text.replace("\x00", " ").strip()
        normalized.append((cleaned or " ")[:8000])
    return normalized


def _running_models(client: ollama.Client) -> list[str]:
    """Return currently loaded Ollama model names from the local runtime."""

    try:
        process_response = client.ps()
    except (ResponseError, RuntimeError, ValueError, OSError) as exc:
        raise RuntimeError(f"Failed querying Ollama runtime state: {exc}") from exc

    models = getattr(process_response, "models", [])
    names: list[str] = []
    for model in models:
        name = getattr(model, "name", "")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _stop_model(name: str) -> None:
    """Attempt to unload a loaded Ollama model by name using the CLI."""

    ollama_cli = shutil.which("ollama")
    if ollama_cli is None:
        raise FileNotFoundError("Cannot unload models because 'ollama' CLI is not available on PATH")

    try:
        subprocess.run(
            [ollama_cli, "stop", name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to unload running model '{name}' via ollama stop") from exc


def _preflight_embedding_runtime(client: ollama.Client) -> None:
    """Unload non-embedding models to reduce Ollama memory contention before embedding."""

    loaded = _running_models(client)
    embedding_model = settings.embedding_model

    to_stop = [name for name in loaded if name != embedding_model]
    for name in to_stop:
        _stop_model(name)


def _is_model_load_failure(exc: Exception) -> bool:
    """Return True when an exception indicates Ollama model load failure."""

    return "model failed to load" in str(exc).lower()


def _embed_batch(client: ollama.Client, batch: list[str]) -> list[list[float]]:
    """Return embeddings for a batch using the best available Ollama endpoint."""

    if hasattr(client, "embed"):
        response = client.embed(model=settings.embedding_model, input=batch)
        batch_vectors = response.get("embeddings", [])
        if not isinstance(batch_vectors, list):
            raise TypeError("Ollama embed response missing 'embeddings' list")
        return batch_vectors

    vectors: list[list[float]] = []
    for item in batch:
        response = client.embeddings(model=settings.embedding_model, prompt=item)
        vector = response.get("embedding", [])
        if not isinstance(vector, list):
            raise TypeError("Ollama embeddings response missing 'embedding' list")
        vectors.append(vector)
    return vectors


def _embed_with_fallback(client: ollama.Client, batch: list[str]) -> list[list[float]]:
    """Return embeddings for a batch and fall back to per-item calls on batch failure."""

    try:
        batch_vectors = _embed_batch(client, batch)
        if len(batch_vectors) == len(batch):
            return batch_vectors
    except (ResponseError, RuntimeError, ValueError, OSError, TypeError):
        pass

    vectors: list[list[float]] = []
    for index, item in enumerate(batch):
        try:
            single_vectors = _embed_batch(client, [item])
        except (ResponseError, RuntimeError, ValueError, OSError, TypeError) as exc:
            raise RuntimeError(
                f"Ollama embedding failed for item {index} in batch of size {len(batch)}: {exc}"
            ) from exc

        if len(single_vectors) != 1:
            raise RuntimeError(
                "Ollama returned an invalid single-item embedding result "
                f"for item {index}: expected 1 vector, got {len(single_vectors)}"
            )
        vectors.append(single_vectors[0])

    return vectors


def embed_texts(texts: list[str], batch_size: int = 128) -> list[list[float]]:
    """Return embedding vectors for input texts using batched Ollama API calls."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not texts:
        return []

    client = _client()
    prepared = _normalize_texts(texts)
    vectors: list[list[float]] = []
    preflight_done = False

    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        try:
            batch_vectors = _embed_with_fallback(client, batch)
        except (ResponseError, RuntimeError, ValueError, OSError, TypeError) as exc:
            if not preflight_done and _is_model_load_failure(exc):
                _preflight_embedding_runtime(client)
                preflight_done = True
                try:
                    batch_vectors = _embed_with_fallback(client, batch)
                except (ResponseError, RuntimeError, ValueError, OSError, TypeError) as retry_exc:
                    raise RuntimeError(f"Ollama embedding request failed: {retry_exc}") from retry_exc
            else:
                raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc

        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                "Ollama returned a mismatched embedding count "
                f"(expected {len(batch)}, got {len(batch_vectors)})"
            )

        for vector in batch_vectors:
            if not isinstance(vector, list):
                raise TypeError("Ollama embedding payload must contain lists of floats")
            vectors.append([float(value) for value in vector])

    return vectors
