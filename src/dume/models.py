"""Shared data models for DUM-E.

Defines pure dataclasses representing chunks, retrieval candidates,
conversational turns, and synthesized answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any


@dataclass
class Chunk:
    """A single segment of text extracted from an ingested document.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        text: Segment text content.
        source: Source document name or file path.
        metadata: Key-value metadata describing document attributes or chunk positioning.
    """

    chunk_id: str
    text: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A candidate chunk retrieved or reranked for a user query.

    Attributes:
        chunk: The underlying extracted document Chunk.
        score: Relevance or similarity score (higher indicates greater relevance).
        rank: Order position in the retrieval/rerank result list (1-based index).
    """

    chunk: Chunk
    score: float
    rank: int = 0

    @property
    def text(self) -> str:
        """Access the underlying chunk's text."""
        return self.chunk.text

    @property
    def source(self) -> str:
        """Access the underlying chunk's source."""
        return self.chunk.source

    @property
    def metadata(self) -> dict[str, Any]:
        """Access the underlying chunk's metadata."""
        return self.chunk.metadata

    @property
    def chunk_id(self) -> str:
        """Access the underlying chunk's identifier."""
        return self.chunk.chunk_id


@dataclass
class Turn:
    """A single exchange in a conversational session.

    Attributes:
        query: User input prompt or question.
        answer: Assistant response text.
        timestamp: Epoch timestamp representing when the turn concluded.
        summary: Extractive or abstractive summary for long-term buffer retention.
        metadata: Optional metadata for turn tracking or evaluation metrics.
    """

    query: str
    answer: str
    timestamp: float = field(default_factory=time.time)
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def question(self) -> str:
        """Alias for query."""
        return self.query

    @property
    def response(self) -> str:
        """Alias for answer."""
        return self.answer


@dataclass
class Answer:
    """The generated or cached output returned to the user.

    Attributes:
        text: Synthesized response message.
        sources: Ranked list of retrieved chunks used as grounding context.
        cached: Flag indicating whether this response was retrieved from the exact-match cache.
        latency_ms: Processing duration in milliseconds.
        config_hash: Hash of the active answer-affecting configuration.
        metadata: Auxiliary diagnostic or token usage information.
    """

    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    cached: bool = False
    latency_ms: float = 0.0
    config_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def response(self) -> str:
        """Alias for text."""
        return self.text
