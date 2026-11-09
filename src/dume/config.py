"""Application configuration for DUM-E.

Manages application settings, hardware constraints, paths, model specifications,
and cache configuration using python-dotenv.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv

# Project root directory: parent of src/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Load default environment variables from .env if present
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable application settings for the DUM-E RAG pipeline.

    Attributes:
        PROJECT_ROOT: Root directory of the repository.
        DATA_DIR: Base directory for stored data.
        RAW_DIR: Directory for incoming raw documents.
        CHROMA_DIR: Persistent ChromaDB storage path.
        CACHE_DB: SQLite path for response and embedding caches.
        LOG_DIR: Directory for rotating application logs.
        EMBED_MODEL: Sentence transformer model identifier for embeddings.
        RERANK_MODEL: CrossEncoder model identifier for reranking.
        BATCH_SIZE: Batch size for CPU inference (must be <= 32).
        CHUNK_SIZE: Character count per document chunk.
        CHUNK_OVERLAP: Character overlap between consecutive chunks.
        TOP_K_RETRIEVAL: Initial candidate count from vector search.
        TOP_K_RERANK: Final chunk count after reranking.
        SCORE_THRESHOLD: Legacy reranker score threshold.
        MAX_ACTIVE_TURNS: Maximum conversation turns kept in active memory.
        SUMMARY_SENTENCES: Number of sentences in Sumy-generated summaries.
        CACHE_TTL_SECONDS: Time-to-live in seconds for exact-match cache entries.
        LLM_PROVIDER: Provider identifier (e.g., 'google' or 'openai').
        LLM_BASE_URL: Base URL for OpenAI-compatible LLM endpoint.
        LLM_MODEL: Model identifier passed to the LLM endpoint.
        LLM_API_KEY: Authentication key for the LLM endpoint.
        LLM_BACKUP_API_KEY: Optional failover key for the LLM endpoint.
        LLM_TEMPERATURE: Sampling temperature for generation.
        LLM_MAX_TOKENS: Maximum token generation limit.
        LLM_TIMEOUT_SECONDS: Network timeout for LLM inference requests.
    """

    # Paths (all pathlib relative to PROJECT_ROOT)
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    CHROMA_DIR: Path = PROJECT_ROOT / "data" / "chroma"
    CACHE_DB: Path = PROJECT_ROOT / "data" / "cache.sqlite"
    LOG_DIR: Path = PROJECT_ROOT / "logs"

    # Embedding and Reranking models (CPU friendly)
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    BATCH_SIZE: int = 32

    # Chunking parameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Retrieval and reranking parameters
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 3
    SCORE_THRESHOLD: float = 0.0
    MIN_RERANK_SCORE: float = float("-inf")

    # Memory caps
    MAX_ACTIVE_TURNS: int = 6
    SUMMARY_SENTENCES: int = 2

    # Response cache TTL (in seconds, default 24h)
    CACHE_TTL_SECONDS: int = 86400

    # LLM Settings (OpenAI-compatible endpoint, e.g. Google AI Studio)
    LLM_PROVIDER: str = "google"
    LLM_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    LLM_MODEL: str = "gemini-2.5-flash-lite"
    LLM_API_KEY: str = ""
    LLM_BACKUP_API_KEY: str = ""
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024
    LLM_TIMEOUT_SECONDS: float = 60.0

    def __post_init__(self) -> None:
        """Validate settings constraints for Windows CPU execution."""
        if self.BATCH_SIZE > 32:
            raise ValueError(
                f"BATCH_SIZE must be <= 32 for CPU-only execution, got {self.BATCH_SIZE}"
            )
        if self.BATCH_SIZE <= 0:
            raise ValueError(f"BATCH_SIZE must be positive, got {self.BATCH_SIZE}")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be strictly less than "
                f"CHUNK_SIZE ({self.CHUNK_SIZE})"
            )
        if self.TOP_K_RERANK > self.TOP_K_RETRIEVAL:
            raise ValueError(
                f"TOP_K_RERANK ({self.TOP_K_RERANK}) cannot exceed "
                f"TOP_K_RETRIEVAL ({self.TOP_K_RETRIEVAL})"
            )

    def config_hash(self) -> str:
        """Compute a deterministic SHA-256 hash representing answer-affecting configuration.

        Changes to prompts, models, chunking, retrieval counts, or temperatures
        alter the returned hash. Non-answer-affecting parameters (logging, cache DB paths,
        API keys, network timeouts) do not change the hash.

        Returns:
            str: 64-character hexadecimal SHA-256 hash.
        """
        answer_affecting: dict[str, Any] = {
            "embed_model": self.EMBED_MODEL,
            "rerank_model": self.RERANK_MODEL,
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "top_k_retrieval": self.TOP_K_RETRIEVAL,
            "top_k_rerank": self.TOP_K_RERANK,
            "score_threshold": self.SCORE_THRESHOLD,
            "min_rerank_score": self.MIN_RERANK_SCORE,
            "max_active_turns": self.MAX_ACTIVE_TURNS,
            "summary_sentences": self.SUMMARY_SENTENCES,
            "llm_provider": self.LLM_PROVIDER,
            "llm_model": self.LLM_MODEL,
            "llm_temperature": self.LLM_TEMPERATURE,
            "llm_max_tokens": self.LLM_MAX_TOKENS,
        }
        serialized = json.dumps(answer_affecting, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def ensure_directories(self) -> None:
        """Ensure all required filesystem directories exist."""
        for directory in (self.DATA_DIR, self.RAW_DIR, self.CHROMA_DIR, self.LOG_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        if self.CACHE_DB.parent:
            self.CACHE_DB.parent.mkdir(parents=True, exist_ok=True)


def load_settings(
    env_file: Path | str | None = None,
    **overrides: Any,
) -> Settings:
    """Load settings combining environment variables, .env file, and explicit overrides.

    Args:
        env_file: Path to an optional .env file. Defaults to PROJECT_ROOT / '.env'.
        **overrides: Key-value pairs that take precedence over environment variables.

    Returns:
        Settings: Configured frozen Settings instance.
    """
    root = PROJECT_ROOT
    resolved_env = Path(env_file) if env_file else root / ".env"
    file_vars: dict[str, str | None] = {}
    if resolved_env.exists():
        file_vars = dotenv_values(resolved_env)

    def _resolve(key: str, default: Any, caster: type = str) -> Any:
        if key in overrides:
            return overrides[key]
        raw = file_vars.get(key)
        if raw is None or raw == "":
            raw = os.getenv(key)
        if raw is None or raw == "":
            return default
        try:
            return caster(raw)
        except (ValueError, TypeError):
            return default

    def _resolve_path(key: str, default_subpath: str) -> Path:
        if key in overrides:
            return Path(overrides[key])
        raw = file_vars.get(key) or os.getenv(key)
        if raw:
            p = Path(raw)
            return p if p.is_absolute() else root / p
        return root / default_subpath

    kwargs: dict[str, Any] = {
        "PROJECT_ROOT": root,
        "DATA_DIR": _resolve_path("DATA_DIR", "data"),
        "RAW_DIR": _resolve_path("RAW_DIR", "data/raw"),
        "CHROMA_DIR": _resolve_path("CHROMA_DIR", "data/chroma"),
        "CACHE_DB": _resolve_path("CACHE_DB", "data/cache.sqlite"),
        "LOG_DIR": _resolve_path("LOG_DIR", "logs"),
        "EMBED_MODEL": _resolve("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2", str),
        "RERANK_MODEL": _resolve("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2", str),
        "BATCH_SIZE": _resolve("BATCH_SIZE", 32, int),
        "CHUNK_SIZE": _resolve("CHUNK_SIZE", 500, int),
        "CHUNK_OVERLAP": _resolve("CHUNK_OVERLAP", 50, int),
        "TOP_K_RETRIEVAL": _resolve("TOP_K_RETRIEVAL", 10, int),
        "TOP_K_RERANK": _resolve("TOP_K_RERANK", 3, int),
        "SCORE_THRESHOLD": _resolve("SCORE_THRESHOLD", 0.0, float),
        "MIN_RERANK_SCORE": _resolve("MIN_RERANK_SCORE", float("-inf"), float),
        "MAX_ACTIVE_TURNS": _resolve("MAX_ACTIVE_TURNS", 6, int),
        "SUMMARY_SENTENCES": _resolve("SUMMARY_SENTENCES", 2, int),
        "CACHE_TTL_SECONDS": _resolve("CACHE_TTL_SECONDS", 86400, int),
        "LLM_PROVIDER": _resolve("LLM_PROVIDER", "google", str),
        "LLM_BASE_URL": _resolve(
            "LLM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            str,
        ),
        "LLM_MODEL": _resolve("LLM_MODEL", "gemini-2.5-flash-lite", str),
        "LLM_API_KEY": _resolve("LLM_API_KEY", "", str),
        "LLM_BACKUP_API_KEY": _resolve("LLM_BACKUP_API_KEY", "", str),
        "LLM_TEMPERATURE": _resolve("LLM_TEMPERATURE", 0.2, float),
        "LLM_MAX_TOKENS": _resolve("LLM_MAX_TOKENS", 1024, int),
        "LLM_TIMEOUT_SECONDS": _resolve("LLM_TIMEOUT_SECONDS", 60.0, float),
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


_global_settings: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """Retrieve or initialize the singleton Settings instance.

    Args:
        reload: Force reload from disk and environment.

    Returns:
        Settings: Cached or freshly initialized settings.
    """
    global _global_settings
    if _global_settings is None or reload:
        _global_settings = load_settings()
    return _global_settings
