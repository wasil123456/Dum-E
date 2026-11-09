"""Unit tests for DUM-E response caching."""

from __future__ import annotations

from dataclasses import dataclass

from dume.cache import (
    ResponseCache,
    is_cacheable,
    make_cache_key,
    normalize_question,
)
from dume.models import Turn


@dataclass
class FakeClock:
    value: float = 100.0

    def __call__(self) -> float:
        return self.value


def test_normalization_and_context_sensitive_keys() -> None:
    assert normalize_question("  What, IS this?!  ") == "what is this"
    assert make_cache_key("Question!", 1, "a") == make_cache_key(" question ", 1, "a")
    assert make_cache_key("question", 1, "a") != make_cache_key("question", 2, "a")
    assert make_cache_key("question", 1, "a") != make_cache_key("question", 1, "b")


def test_cache_get_put_and_ttl(tmp_path) -> None:
    clock = FakeClock()
    cache = ResponseCache(tmp_path / "cache.sqlite", ttl_seconds=10, clock=clock)

    assert cache.get("What is DUM-E?", 1, "config") is None
    cache.put("What is DUM-E?", "A RAG backend.", 1, "config")
    assert cache.get(" what is dume ", 1, "config") == "A RAG backend."

    clock.value = 110
    assert cache.get("What is DUM-E?", 1, "config") is None
    assert cache.purge_stale() == 0


def test_purge_stale_removes_expired_entries(tmp_path) -> None:
    clock = FakeClock()
    cache = ResponseCache(tmp_path / "cache.sqlite", ttl_seconds=5, clock=clock)
    cache.put("one", "answer one", 1, "config")
    cache.put("two", "answer two", 1, "config")
    clock.value = 105

    assert cache.purge_stale() == 2
    assert cache.get("one", 1, "config") is None


def test_cacheability_skips_ambiguous_followups() -> None:
    prior = [Turn(query="What is retrieval?", answer="Vector search.")]

    assert is_cacheable("What is retrieval?", prior)
    assert is_cacheable("What is retrieval?", []) 
    assert not is_cacheable("What about it?", prior)
    assert not is_cacheable("Tell me more", prior)
    assert not is_cacheable("And why?", prior)
    assert not is_cacheable("", [])
