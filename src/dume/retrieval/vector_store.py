"""ChromaDB vector store querying and similarity scoring for DUM-E.

Translates semantic queries into vector representations using lazy-loaded
SentenceTransformer models and retrieves candidates with cosine similarity.
"""

from __future__ import annotations

import logging
from typing import Any

from dume.config import Settings, get_settings
from dume.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Interface for querying ChromaDB and scoring results with cosine similarity.

    Attributes:
        settings: Application settings configuration.
        collection_name: Name of ChromaDB collection.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any = None,
        embedder: Any = None,
        collection_name: str = "dume_knowledge",
    ) -> None:
        """Initialize VectorStore with optional dependency injection for testing.

        Args:
            settings: Application settings. If None, loaded via get_settings().
            client: Optional injected ChromaDB client.
            embedder: Optional injected embedding model.
            collection_name: Target Chroma collection name.
        """
        self.settings = settings or get_settings()
        self._client = client
        self._embedder = embedder
        self.collection_name = collection_name
        self._collection: Any = None

    @property
    def client(self) -> Any:
        """Retrieve or lazily initialize the ChromaDB client."""
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.settings.CHROMA_DIR))
        return self._client

    @property
    def collection(self) -> Any:
        """Retrieve or initialize the target Chroma collection with cosine distance."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def embedder(self) -> Any:
        """Retrieve or lazily initialize the shared SentenceTransformer embedder."""
        if self._embedder is None:
            from dume.ingestion.indexer import get_embedder

            self._embedder = get_embedder(self.settings.EMBED_MODEL)
        return self._embedder

    def count(self) -> int:
        """Return total number of chunks stored in the collection."""
        return self.collection.count()

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        subject: str | None = None,
        user_id: int | str | None = None,
    ) -> list[RetrievedChunk]:
        """Search the vector database for candidates relevant to query_text.

        Cosine distance d is mapped to similarity score: similarity = 1.0 - d.

        Args:
            query_text: Query string to match against indexed documents.
            top_k: Maximum candidate count to return. Defaults to TOP_K_RETRIEVAL.

        Returns:
            list[RetrievedChunk]: Candidate chunks ordered by similarity.
        """
        stripped = query_text.strip()
        if not stripped:
            return []

        k = top_k if top_k is not None else self.settings.TOP_K_RETRIEVAL
        if k <= 0:
            return []

        col = self.collection
        total_items = col.count()
        if total_items == 0:
            logger.debug("Collection '%s' contains 0 items; returning empty list", self.collection_name)
            return []

        n_results = min(k, total_items)

        # Generate query vector
        raw_embedding = self.embedder.encode(
            [stripped],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if hasattr(raw_embedding, "tolist"):
            query_embeddings = raw_embedding.tolist()
        else:
            query_embeddings = [list(e) for e in raw_embedding]

        filters: list[dict[str, object]] = []
        if user_id is not None:
            filters.append({"user_id": str(user_id)})
        if subject:
            filters.append({"subject": subject})
        where: dict[str, object] | None = None
        if len(filters) == 1:
            where = filters[0]
        elif filters:
            where = {"$and": filters}

        results = col.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for rank_idx, (chunk_id, doc, meta, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            # Convert cosine distance to cosine similarity
            similarity = 1.0 - float(distance)
            chunk = Chunk(
                chunk_id=chunk_id,
                text=doc,
                source=meta.get("source", "") if meta else "",
                metadata=meta or {},
            )
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=similarity,
                    rank=rank_idx,
                )
            )

        return retrieved
