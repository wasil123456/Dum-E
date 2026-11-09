"""Unit tests for DUM-E retrieval components: VectorStore, Reranker, and Retriever."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from dume.config import Settings
from dume.models import Chunk, RetrievedChunk
from dume.retrieval.reranker import Reranker
from dume.retrieval.retriever import Retriever
from dume.retrieval.vector_store import VectorStore


class FakeEmbedder:
    """Mock embedder returning fixed-dimension unit vectors."""

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for t in texts:
            val = float(len(t) % 10 + 1) / 10.0
            vectors.append([val] * self.dimension)
        return vectors


class FakeCrossEncoder:
    """Mock CrossEncoder returning predetermined or text-derived scores."""

    def __init__(self, score_map: dict[str, float] | None = None) -> None:
        self.score_map = score_map or {}
        self.recorded_pairs: list[list[str]] = []
        self.recorded_batch_sizes: list[int] = []

    def predict(
        self,
        pairs: list[list[str]],
        batch_size: int = 16,
        show_progress_bar: bool = False,
    ) -> list[float]:
        self.recorded_pairs.extend(pairs)
        self.recorded_batch_sizes.append(batch_size)
        scores: list[float] = []
        for query, text in pairs:
            if text in self.score_map:
                scores.append(self.score_map[text])
            else:
                # Default mock score based on text length
                scores.append(float(len(text)) / 100.0)
        return scores


# =====================================================================
# VectorStore Tests
# =====================================================================


def test_vector_store_empty_inputs(tmp_path: Path) -> None:
    """Verify VectorStore returns empty list for empty queries or empty collections."""
    import chromadb

    client = chromadb.EphemeralClient()
    fake_embedder = FakeEmbedder()
    store = VectorStore(
        settings=Settings(CHROMA_DIR=tmp_path),
        client=client,
        embedder=fake_embedder,
        collection_name="empty_test",
    )

    # Empty query string
    assert store.query("") == []
    assert store.query("   \t\n ") == []

    # Query against empty collection
    assert store.query("any query") == []


def test_vector_store_cosine_similarity_conversion(tmp_path: Path) -> None:
    """Verify that ChromaDB distance is converted to similarity score = 1.0 - distance."""
    import chromadb

    client = chromadb.EphemeralClient()
    fake_embedder = FakeEmbedder(dimension=8)
    store = VectorStore(
        settings=Settings(CHROMA_DIR=tmp_path, TOP_K_RETRIEVAL=5),
        client=client,
        embedder=fake_embedder,
        collection_name="test_sim",
    )

    # Pre-populate collection with known vectors
    col = store.collection
    col.upsert(
        ids=["c1", "c2"],
        documents=["Document one text", "Document two text"],
        embeddings=[[0.5] * 8, [0.1] * 8],
        metadatas=[{"source": "f1.txt", "page": 1}, {"source": "f2.txt", "page": 2}],
    )

    results = store.query("test search", top_k=2)

    assert len(results) == 2
    for r in results:
        assert isinstance(r, RetrievedChunk)
        assert isinstance(r.score, float)
        assert r.rank in (1, 2)
        assert r.chunk_id in ("c1", "c2")
        assert r.source in ("f1.txt", "f2.txt")


# =====================================================================
# Reranker Tests
# =====================================================================


def test_reranker_empty_inputs() -> None:
    """Verify Reranker returns empty list on empty query or empty candidates."""
    reranker = Reranker(cross_encoder=FakeCrossEncoder())
    assert reranker.rerank("", []) == []
    assert reranker.rerank("query", []) == []

    c1 = RetrievedChunk(chunk=Chunk(chunk_id="c1", text="Text"), score=0.5, rank=1)
    assert reranker.rerank("", [c1]) == []


def test_reranker_scoring_and_ordering() -> None:
    """Verify Reranker scores in batches of 16 and sorts candidates descending by score."""
    score_map = {
        "Chunk low relevance": 0.12,
        "Chunk high relevance": 0.95,
        "Chunk medium relevance": 0.65,
    }
    fake_ce = FakeCrossEncoder(score_map=score_map)
    reranker = Reranker(cross_encoder=fake_ce, batch_size=16)

    candidates = [
        RetrievedChunk(chunk=Chunk(chunk_id="1", text="Chunk low relevance"), score=0.8, rank=1),
        RetrievedChunk(chunk=Chunk(chunk_id="2", text="Chunk high relevance"), score=0.4, rank=2),
        RetrievedChunk(chunk=Chunk(chunk_id="3", text="Chunk medium relevance"), score=0.6, rank=3),
    ]

    reranked = reranker.rerank("sample query", candidates)

    assert len(reranked) == 3
    # Check descending order: high (0.95), medium (0.65), low (0.12)
    assert reranked[0].chunk.chunk_id == "2"
    assert reranked[0].score == 0.95
    assert reranked[0].rank == 1

    assert reranked[1].chunk.chunk_id == "3"
    assert reranked[1].score == 0.65
    assert reranked[1].rank == 2

    assert reranked[2].chunk.chunk_id == "1"
    assert reranked[2].score == 0.12
    assert reranked[2].rank == 3

    assert 16 in fake_ce.recorded_batch_sizes


# =====================================================================
# Retriever Pipeline Tests (Dependency Injected)
# =====================================================================


def test_retriever_pipeline_full_workflow() -> None:
    """Verify complete retrieval flow: vector search -> rerank -> filter cutoff -> top N."""
    # Setup mock vector store
    mock_vs = MagicMock()
    c1 = RetrievedChunk(chunk=Chunk(chunk_id="c1", text="Candidate one"), score=0.7, rank=1)
    c2 = RetrievedChunk(chunk=Chunk(chunk_id="c2", text="Candidate two"), score=0.6, rank=2)
    c3 = RetrievedChunk(chunk=Chunk(chunk_id="c3", text="Candidate three"), score=0.5, rank=3)
    mock_vs.query.return_value = [c1, c2, c3]

    # Setup mock reranker
    mock_reranker = MagicMock()
    r1 = RetrievedChunk(chunk=c2.chunk, score=0.85, rank=1)  # High score
    r2 = RetrievedChunk(chunk=c1.chunk, score=0.45, rank=2)  # Medium score
    r3 = RetrievedChunk(chunk=c3.chunk, score=0.10, rank=3)  # Low score (below threshold)
    mock_reranker.rerank.return_value = [r1, r2, r3]

    settings = Settings(
        TOP_K_RETRIEVAL=10,
        TOP_K_RERANK=2,
        MIN_RERANK_SCORE=0.30,
    )

    retriever = Retriever(
        vector_store=mock_vs,
        reranker=mock_reranker,
        settings=settings,
    )

    results = retriever.retrieve("What is DUM-E?", top_k=2, min_score=0.30)

    # c3 had score 0.10 < 0.30, so filtered out
    # Remaining: r1 (0.85) and r2 (0.45)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c2"
    assert results[0].score == 0.85
    assert results[0].rank == 1

    assert results[1].chunk.chunk_id == "c1"
    assert results[1].score == 0.45
    assert results[1].rank == 2


def test_retriever_filters_all_below_min_score() -> None:
    """Verify Retriever returns empty list when all candidate chunks fail MIN_RERANK_SCORE."""
    mock_vs = MagicMock()
    c1 = RetrievedChunk(chunk=Chunk(chunk_id="c1", text="Irrelevant context"), score=0.2, rank=1)
    mock_vs.query.return_value = [c1]

    mock_reranker = MagicMock()
    r1 = RetrievedChunk(chunk=c1.chunk, score=0.15, rank=1)
    mock_reranker.rerank.return_value = [r1]

    settings = Settings(MIN_RERANK_SCORE=0.50)
    retriever = Retriever(
        vector_store=mock_vs,
        reranker=mock_reranker,
        settings=settings,
    )

    results = retriever.retrieve("query with high cutoff", min_score=0.50)
    assert results == []


def test_retriever_empty_query() -> None:
    """Verify Retriever returns empty list on empty query without making downstream calls."""
    mock_vs = MagicMock()
    mock_reranker = MagicMock()

    retriever = Retriever(vector_store=mock_vs, reranker=mock_reranker)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   \t ") == []

    mock_vs.query.assert_not_called()
    mock_reranker.rerank.assert_not_called()
