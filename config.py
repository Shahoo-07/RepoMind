"""Centralized application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load typed configuration values used by all project modules."""

    openai_api_key: str = ""
    llm_model: str = "llama3.2:latest"
    embedding_model: str = "embeddinggemma:latest"
    embedding_dim: int = 768
    ollama_host: str = "http://127.0.0.1:11434"
    lancedb_uri: str = "./data/lancedb"
    lancedb_table: str = "code_chunks"
    knowledge_graph_dir: str = "./data/knowledge_graph"
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    repo_local_dir: str = "./data/repos"
    include_exts: tuple[str, ...] = (
        ".py",
        ".js",
        ".ts",
        ".java",
        ".go",
        ".cpp",
        ".c",
        ".h",
        ".md",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
    )
    exclude_dirs: tuple[str, ...] = (
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "dist",
        "build",
        ".pytest_cache",
    )
    max_file_bytes: int = 200_000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
