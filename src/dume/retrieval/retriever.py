"""End-to-end context retrieval pipeline for DUM-E.

Combines ChromaDB vector search with CrossEncoder reranking, applies score
threshold filtering, and retains the top N candidate chunks for prompt grounding.
"""

from __future__ import annotations

import logging

from dume.config import Settings, get_settings
from dume.models import RetrievedChunk
from dume.retrieval.reranker import Reranker
from dume.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Orchestrates vector candidate retrieval, reranking, and threshold filtering.

    Attributes:
        settings: Application settings.
        vector_store: Vector store search client.
        reranker: CrossEncoder reranker.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize Retriever with optional dependency injection for isolated testing.

        Args:
            vector_store: VectorStore instance for vector candidate retrieval.
            reranker: Reranker instance for cross-encoder reranking.
            settings: Application settings. If None, loaded via get_settings().
        """
        self.settings = settings or get_settings()
        self.vector_store = vector_store or VectorStore(settings=self.settings)
        self.reranker = reranker or Reranker(settings=self.settings)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        subject: str | None = None,
        user_id: int | str | None = None,
    ) -> list[RetrievedChunk]:
        """Execute complete retrieval pipeline for a given user query.

        Steps:
            1. Vector search retrieving top initial candidates.
            2. Cross-encoder reranking scoring query-chunk pairs.
            3. Filtering out any candidates scoring below min_score (MIN_RERANK_SCORE).
            4. Retaining the top N qualified candidates (TOP_K_RERANK).

        Args:
            query: User search query or question.
            top_k: Number of final chunks to return. Defaults to TOP_K_RERANK.
            min_score: Minimum cross-encoder score cutoff. Defaults to MIN_RERANK_SCORE.

        Returns:
            list[RetrievedChunk]: Final list of ranked, filtered chunks (empty if none qualify).
        """
        stripped_query = query.strip()
        if not stripped_query:
            logger.debug("Empty query passed to retrieve(); returning empty list")
            return []

        # Step 1: Initial candidate retrieval from VectorStore
        initial_candidates = self.vector_store.query(
            stripped_query,
            top_k=self.settings.TOP_K_RETRIEVAL,
            subject=subject,
            user_id=user_id,
        )
        if not initial_candidates:
            logger.debug("No candidates found in vector store for query: '%s'", stripped_query)
            return []

        # Step 2: Rerank candidates with CrossEncoder
        reranked_candidates = self.reranker.rerank(stripped_query, initial_candidates)
        if not reranked_candidates:
            logger.debug("Reranking produced 0 results")
            return []

        # Step 3: Filter candidates below minimum score threshold
        threshold = (
            min_score
            if min_score is not None
            else getattr(
                self.settings, "MIN_RERANK_SCORE", float("-inf")
            )
        )

        qualified = [c for c in reranked_candidates if c.score >= threshold]
        if not qualified:
            logger.info(
                "All %d reranked candidates fell below score threshold %.3f; returning empty list",
                len(reranked_candidates),
                threshold,
            )
            return []

        # Step 4: Keep top N qualified candidates
        final_top_k = top_k if top_k is not None else self.settings.TOP_K_RERANK
        if final_top_k <= 0:
            return []

        selected = qualified[:final_top_k]

        # Re-assign 1-based ranks
        return [
            RetrievedChunk(chunk=c.chunk, score=c.score, rank=idx)
            for idx, c in enumerate(selected, start=1)
        ]
