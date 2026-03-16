"""Application configuration via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_model: str = "gemini/gemini-2.0-flash"
    llm_api_key: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""

    # Storage paths
    database_path: Path = Path("data/kenquest.duckdb")
    graph_path: Path = Path("data/graph")
    chroma_path: Path = Path("data/chroma")

    # Embedding model (fixed for vector index stability)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # FalkorDB (graph traversal + visualization)
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379

    # Max concepts per chunk sent to LLM (keeps prompts focused)
    max_concepts_per_chunk: int = 10

    # Parallel LLM workers for graph building (intra-tier batches run concurrently).
    # Raise for faster builds; lower to avoid 429 rate-limit errors.
    graph_builder_workers: int = 4

    # Mastery threshold
    mastery_threshold: float = 0.7

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


settings = Settings()
