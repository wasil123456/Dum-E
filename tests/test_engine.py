"""Unit tests for DUM-E engine orchestration."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from dume.config import Settings
from dume.engine import DumeEngine, UNKNOWN_ANSWER
from dume.llm.base import LLMError
from dume.models import Chunk, RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0

    def retrieve(self, question):
        self.calls += 1
        return self.chunks


class FakeLLM:
    def __init__(self, response="generated answer", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.response


class FakeCache:
    def __init__(self):
        self.values = {}
        self.get_calls = 0
        self.put_calls = []

    def get(self, question, kb_version, config_hash):
        self.get_calls += 1
        return self.values.get((question.casefold().strip(), kb_version, config_hash))

    def put(self, question, response, kb_version, config_hash):
        self.put_calls.append((question, response, kb_version, config_hash))
        self.values[(question.casefold().strip(), kb_version, config_hash)] = response


def _settings(tmp_path):
    return Settings(DATA_DIR=tmp_path, CACHE_DB=tmp_path / "cache.sqlite")


def _chunk():
    return [
        RetrievedChunk(
            chunk=Chunk(chunk_id="1", text="grounded context", source="doc.txt"),
            score=0.9,
            rank=1,
        )
    ]


def test_successful_rag_flow_updates_memory_and_cache(tmp_path):
    settings = _settings(tmp_path)
    (tmp_path / "kb_version.json").write_text(json.dumps({"version": 4}))
    retriever = FakeRetriever(_chunk())
    llm = FakeLLM()
    cache = FakeCache()
    engine = DumeEngine(retriever=retriever, llm=llm, cache=cache, settings=settings)

    answer = engine.ask("s1", "What is this?")

    assert answer.text == "generated answer"
    assert answer.sources == _chunk()
    assert answer.cached is False
    assert retriever.calls == 1
    assert len(llm.calls) == 1
    assert len(cache.put_calls) == 1
    assert len(engine.get_memory_view("s1")["active_turns"]) == 1


def test_cache_hit_skips_retrieval_and_llm(tmp_path):
    settings = _settings(tmp_path)
    retriever = FakeRetriever(_chunk())
    llm = FakeLLM()
    cache = FakeCache()
    cache.values[("what is this?", 0, settings.config_hash())] = "cached"
    engine = DumeEngine(retriever=retriever, llm=llm, cache=cache, settings=settings)

    answer = engine.ask("s1", "What is this?")

    assert answer.cached is True
    assert answer.text == "cached"
    assert retriever.calls == 0
    assert llm.calls == []


def test_empty_retrieval_does_not_call_llm_or_cache(tmp_path):
    retriever = FakeRetriever([])
    llm = FakeLLM()
    cache = FakeCache()
    engine = DumeEngine(
        retriever=retriever, llm=llm, cache=cache, settings=_settings(tmp_path)
    )

    answer = engine.ask("s1", "Unknown question")

    assert answer.text == UNKNOWN_ANSWER
    assert llm.calls == []
    assert cache.put_calls == []
    assert engine.get_memory_view("s1")["active_turns"] == []


def test_ambiguous_followup_is_not_cached(tmp_path):
    cache = FakeCache()
    engine = DumeEngine(
        retriever=FakeRetriever(_chunk()),
        llm=FakeLLM(),
        cache=cache,
        settings=_settings(tmp_path),
    )
    engine.ask("s1", "What is this?")
    cache.put_calls.clear()

    answer = engine.ask("s1", "What about it?")

    assert answer.cached is False
    assert cache.put_calls == []


def test_llm_error_does_not_record_memory_or_cache(tmp_path):
    error = LLMError("failed")
    engine = DumeEngine(
        retriever=FakeRetriever(_chunk()),
        llm=FakeLLM(error=error),
        cache=FakeCache(),
        settings=_settings(tmp_path),
    )

    with pytest.raises(LLMError):
        engine.ask("s1", "What is this?")

    assert engine.get_memory_view("s1")["active_turns"] == []
    assert engine.cache.put_calls == []
    assert engine.stats()["llm_errors"] == 1


def test_reset_session_and_stats(tmp_path):
    engine = DumeEngine(
        retriever=FakeRetriever(_chunk()),
        llm=FakeLLM(),
        cache=FakeCache(),
        settings=_settings(tmp_path),
    )
    engine.ask("s1", "What is this?")
    engine.reset_session("s1")

    assert engine.get_memory_view("s1")["active_turns"] == []
    assert engine.stats()["asks"] == 1
