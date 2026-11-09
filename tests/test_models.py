"""Unit tests for DUM-E shared data models."""

from __future__ import annotations

import time

from dume.models import Answer, Chunk, RetrievedChunk, Turn


def test_chunk_attributes() -> None:
    """Verify Chunk dataclass creation and field access."""
    chunk = Chunk(
        chunk_id="chunk-1",
        text="Sample document text.",
        source="sample.pdf",
        metadata={"page": 1},
    )
    assert chunk.chunk_id == "chunk-1"
    assert chunk.text == "Sample document text."
    assert chunk.source == "sample.pdf"
    assert chunk.metadata == {"page": 1}


def test_retrieved_chunk_properties() -> None:
    """Verify RetrievedChunk wrapping and convenience properties."""
    chunk = Chunk(
        chunk_id="chunk-2",
        text="Retrieved context segment.",
        source="guide.txt",
        metadata={"section": "intro"},
    )
    retrieved = RetrievedChunk(chunk=chunk, score=0.92, rank=1)

    assert retrieved.chunk is chunk
    assert retrieved.score == 0.92
    assert retrieved.rank == 1
    assert retrieved.text == "Retrieved context segment."
    assert retrieved.source == "guide.txt"
    assert retrieved.chunk_id == "chunk-2"
    assert retrieved.metadata == {"section": "intro"}


def test_turn_attributes_and_aliases() -> None:
    """Verify Turn dataclass attributes and conversational property aliases."""
    now = time.time()
    turn = Turn(
        query="What is DUM-E?",
        answer="DUM-E is a CPU-friendly RAG backend.",
        timestamp=now,
        summary="DUM-E overview",
    )

    assert turn.query == "What is DUM-E?"
    assert turn.answer == "DUM-E is a CPU-friendly RAG backend."
    assert turn.question == "What is DUM-E?"
    assert turn.response == "DUM-E is a CPU-friendly RAG backend."
    assert turn.timestamp == now
    assert turn.summary == "DUM-E overview"


def test_answer_attributes_and_defaults() -> None:
    """Verify Answer dataclass attributes, defaults, and aliases."""
    chunk = Chunk(chunk_id="c1", text="Grounding text.")
    rc = RetrievedChunk(chunk=chunk, score=0.88, rank=1)

    answer = Answer(
        text="This is the answer.",
        sources=[rc],
        cached=False,
        latency_ms=124.5,
        config_hash="abc123hash",
    )

    assert answer.text == "This is the answer."
    assert answer.response == "This is the answer."
    assert len(answer.sources) == 1
    assert answer.cached is False
    assert answer.latency_ms == 124.5
    assert answer.config_hash == "abc123hash"

