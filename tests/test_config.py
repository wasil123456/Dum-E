"""Unit tests for DUM-E application configuration and config_hash generation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from dume.config import Settings, get_settings, load_settings


def test_default_settings_loading() -> None:
    """Verify that default settings load with proper types, paths, and values."""
    settings = load_settings()

    # Paths are pathlib.Path instances
    assert isinstance(settings.PROJECT_ROOT, Path)
    assert isinstance(settings.DATA_DIR, Path)
    assert isinstance(settings.RAW_DIR, Path)
    assert isinstance(settings.CHROMA_DIR, Path)
    assert isinstance(settings.CACHE_DB, Path)
    assert isinstance(settings.LOG_DIR, Path)

    # Models & hardware bounds
    assert settings.EMBED_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.BATCH_SIZE <= 32
    assert settings.MAX_ACTIVE_TURNS == 6

    # Chunking bounds
    assert settings.CHUNK_SIZE > settings.CHUNK_OVERLAP

    # Retrieval bounds
    assert settings.TOP_K_RERANK <= settings.TOP_K_RETRIEVAL

    # Provider and model settings from .env
    assert settings.LLM_PROVIDER == "google"
    assert "googleapis.com" in settings.LLM_BASE_URL
    assert "gemini" in settings.LLM_MODEL.lower()


def test_settings_is_immutable() -> None:
    """Verify that Settings is a frozen dataclass and prevents mutation."""
    settings = Settings()
    with pytest.raises(FrozenInstanceError):
        settings.CHUNK_SIZE = 1000  # type: ignore[misc]


def test_settings_validation_errors() -> None:
    """Verify that invalid parameters raise clear ValueError exceptions."""
    # Batch size exceeds CPU limit
    with pytest.raises(ValueError, match="BATCH_SIZE must be <= 32"):
        Settings(BATCH_SIZE=64)

    # Negative/zero batch size
    with pytest.raises(ValueError, match="BATCH_SIZE must be positive"):
        Settings(BATCH_SIZE=0)

    # Overlap equals or exceeds chunk size
    with pytest.raises(ValueError, match="CHUNK_OVERLAP.*must be strictly less than"):
        Settings(CHUNK_SIZE=200, CHUNK_OVERLAP=200)

    # Rerank count exceeds retrieval count
    with pytest.raises(ValueError, match="TOP_K_RERANK.*cannot exceed TOP_K_RETRIEVAL"):
        Settings(TOP_K_RETRIEVAL=5, TOP_K_RERANK=10)


def test_config_hash_deterministic() -> None:
    """Verify that identical configurations yield the same hash."""
    s1 = Settings()
    s2 = Settings()
    assert s1.config_hash() == s2.config_hash()
    assert len(s1.config_hash()) == 64


def test_config_hash_changes_on_answer_affecting_fields() -> None:
    """Verify that changing any answer-affecting field alters the config hash."""
    base = Settings()
    base_hash = base.config_hash()

    answer_affecting_mods = [
        {"EMBED_MODEL": "sentence-transformers/all-mpnet-base-v2"},
        {"RERANK_MODEL": "cross-encoder/ms-marco-TinyBERT-L-2"},
        {"CHUNK_SIZE": 600},
        {"CHUNK_OVERLAP": 100},
        {"TOP_K_RETRIEVAL": 20},
        {"TOP_K_RERANK": 5},
        {"SCORE_THRESHOLD": 0.5},
        {"MAX_ACTIVE_TURNS": 4},
        {"SUMMARY_SENTENCES": 3},
        {"LLM_MODEL": "gemini-1.5-pro"},
        {"LLM_TEMPERATURE": 0.7},
        {"LLM_MAX_TOKENS": 2048},
        {"LLM_PROVIDER": "openai"},
    ]

    for mod in answer_affecting_mods:
        modified = replace(base, **mod)
        assert (
            modified.config_hash() != base_hash
        ), f"Modification {mod} did not alter config_hash"


def test_config_hash_unchanged_on_non_answer_affecting_fields() -> None:
    """Verify that changing non-answer-affecting fields leaves config hash unchanged."""
    base = Settings()
    base_hash = base.config_hash()

    non_affecting_mods = [
        {"LOG_DIR": Path("custom/logs")},
        {"CACHE_DB": Path("custom/cache.sqlite")},
        {"DATA_DIR": Path("custom/data")},
        {"RAW_DIR": Path("custom/raw")},
        {"CHROMA_DIR": Path("custom/chroma")},
        {"CACHE_TTL_SECONDS": 3600},
        {"LLM_API_KEY": "secret-different-key"},
        {"LLM_TIMEOUT_SECONDS": 120.0},
        {"BATCH_SIZE": 16},
    ]

    for mod in non_affecting_mods:
        modified = replace(base, **mod)
        assert (
            modified.config_hash() == base_hash
        ), f"Modification {mod} unexpectedly altered config_hash"


def test_load_settings_with_custom_env_file(tmp_path: Path) -> None:
    """Verify loading from a specific custom .env file."""
    custom_env = tmp_path / "custom.env"
    custom_env.write_text(
        "LLM_MODEL=custom-gemini\n"
        "CHUNK_SIZE=400\n"
        "CHUNK_OVERLAP=40\n"
        "LLM_PROVIDER=google-custom\n",
        encoding="utf-8",
    )

    settings = load_settings(env_file=custom_env)
    assert settings.LLM_MODEL == "custom-gemini"
    assert settings.CHUNK_SIZE == 400
    assert settings.CHUNK_OVERLAP == 40
    assert settings.LLM_PROVIDER == "google-custom"


def test_ensure_directories(tmp_path: Path) -> None:
    """Verify that ensure_directories creates paths accurately."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    chroma_dir = data_dir / "chroma"
    log_dir = tmp_path / "logs"
    cache_db = data_dir / "cache.sqlite"

    settings = Settings(
        DATA_DIR=data_dir,
        RAW_DIR=raw_dir,
        CHROMA_DIR=chroma_dir,
        LOG_DIR=log_dir,
        CACHE_DB=cache_db,
    )

    assert not data_dir.exists()
    settings.ensure_directories()
    assert data_dir.exists()
    assert raw_dir.exists()
    assert chroma_dir.exists()
    assert log_dir.exists()


def test_get_settings_singleton() -> None:
    """Verify that get_settings provides singleton behavior and respects reload."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2

    s3 = get_settings(reload=True)
    assert isinstance(s3, Settings)
