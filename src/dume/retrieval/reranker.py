"""CrossEncoder reranking for DUM-E.

Wraps sentence_transformers.CrossEncoder to score query-chunk pairs in batches
of 16, updating relevance scores and ranks for retrieved candidates.
"""

from __future__ import annotations

import logging
from typing import Any

from dume.config import Settings, get_settings
from dume.models import RetrievedChunk

logger = logging.getLogger(__name__)

# Global cached CrossEncoder instance for lazy loading
_CROSS_ENCODER_INSTANCE: Any = None


def get_cross_encoder(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Any:
    """Lazy-load and return the CrossEncoder reranker model on CPU.

    Args:
        model_name: Cross-encoder model identifier.

    Returns:
        CrossEncoder: Initialized cross encoder model instance.
    """
    global _CROSS_ENCODER_INSTANCE
    if _CROSS_ENCODER_INSTANCE is None:
        logger.info("Lazy-loading CrossEncoder model: %s on CPU", model_name)
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER_INSTANCE = CrossEncoder(model_name, device="cpu")
    return _CROSS_ENCODER_INSTANCE


def reset_cross_encoder() -> None:
    """Reset the cached CrossEncoder instance (primarily for testing)."""
    global _CROSS_ENCODER_INSTANCE
    _CROSS_ENCODER_INSTANCE = None


class Reranker:
    """Reranks candidate chunks against a query using a CrossEncoder model.

    Attributes:
        settings: Application settings.
        batch_size: Processing batch size for inference (defaults to 16, capped at 32).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        cross_encoder: Any = None,
        batch_size: int = 16,
    ) -> None:
        """Initialize Reranker with optional dependency injection.

        Args:
            settings: Application settings. If None, loaded via get_settings().
            cross_encoder: Optional injected CrossEncoder model for tests.
            batch_size: Batch size for model inference (default 16, max 32).
        """
        self.settings = settings or get_settings()
        self._cross_encoder = cross_encoder
        self.batch_size = min(max(1, batch_size), 32)

    @property
    def cross_encoder(self) -> Any:
        """Retrieve or lazily initialize the shared CrossEncoder instance."""
        if self._cross_encoder is None:
            self._cross_encoder = get_cross_encoder(self.settings.RERANK_MODEL)
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Compute cross-encoder relevance scores for query and candidates.

        Pairs are scored in batches of 16 and sorted in descending order of score.

        Args:
            query: User search query.
            candidates: Retrieved candidates from vector search.

        Returns:
            list[RetrievedChunk]: Reranked candidates with new scores and updated ranks.
        """
        stripped_query = query.strip()
        if not stripped_query or not candidates:
            return []

        logger.debug(
            "Reranking %d candidate(s) in batches of %d",
            len(candidates),
            self.batch_size,
        )

        pairs = [[stripped_query, cand.text] for cand in candidates]

        raw_scores = self.cross_encoder.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        if hasattr(raw_scores, "tolist"):
            scores = raw_scores.tolist()
        elif isinstance(raw_scores, (int, float)):
            scores = [float(raw_scores)]
        else:
            scores = list(raw_scores)

        # Sort candidates descending by CrossEncoder score
        scored_pairs = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        reranked_chunks: list[RetrievedChunk] = []
        for rank_idx, (cand, score) in enumerate(scored_pairs, start=1):
            reranked_chunks.append(
                RetrievedChunk(
                    chunk=cand.chunk,
                    score=float(score),
                    rank=rank_idx,
                )
            )

        return reranked_chunks
