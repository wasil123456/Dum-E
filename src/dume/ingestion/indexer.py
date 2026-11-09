"""Vector store indexing and incremental sync for DUM-E.

Manages ChromaDB storage with cosine distance, lazy-loaded sentence embeddings,
batch upserts (<= 32 items), and versioned state in data/kb_version.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

from dume.config import Settings, get_settings
from dume.ingestion.chunker import chunk_pages
from dume.ingestion.loader import LoadedPage
from dume.models import Chunk

logger = logging.getLogger(__name__)

# Global cached embedding model instance for lazy loading
_EMBEDDER_INSTANCE: Any = None


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Any:
    """Lazy-load and return the SentenceTransformer model on CPU.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        SentenceTransformer: Initialized sentence transformer model instance.
    """
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        logger.info("Lazy-loading SentenceTransformer model: %s on CPU", model_name)
        from sentence_transformers import SentenceTransformer

        _EMBEDDER_INSTANCE = SentenceTransformer(model_name, device="cpu")
    return _EMBEDDER_INSTANCE


def reset_embedder() -> None:
    """Reset the cached embedder instance (primarily for testing)."""
    global _EMBEDDER_INSTANCE
    _EMBEDDER_INSTANCE = None


@dataclass
class IndexingResult:
    """Summary of indexing operation results.

    Attributes:
        indexed_files: Number of newly indexed or modified files.
        skipped_files: Number of unchanged files skipped via incremental check.
        deleted_files: Number of deleted files removed from the index.
        chunks_indexed: Total number of chunks embedded and stored.
        kb_version: Updated knowledge base version counter.
    """

    indexed_files: int = 0
    skipped_files: int = 0
    deleted_files: int = 0
    chunks_indexed: int = 0
    kb_version: int = 0


def load_kb_version(path: Path) -> dict[str, Any]:
    """Load knowledge base version and file manifests from disk.

    Args:
        path: Path to kb_version.json.

    Returns:
        dict[str, Any]: Loaded version dictionary or default structure.
    """
    if not path.exists():
        return {
            "version": 0,
            "last_updated": None,
            "config_hash": "",
            "files": {},
        }
    try:
        content = path.read_text(encoding="utf-8")
        return json.loads(content)
    except Exception as exc:
        logger.warning(
            "Could not parse kb_version at %s (%s). Initializing fresh manifest.",
            path,
            exc,
        )
        return {
            "version": 0,
            "last_updated": None,
            "config_hash": "",
            "files": {},
        }


def save_kb_version(path: Path, data: dict[str, Any]) -> None:
    """Persist knowledge base version and file manifests to disk.

    Args:
        path: Path to kb_version.json.
        data: Version and files dictionary to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp_path.replace(path)


def index_documents(
    raw_dir: Path | str | None = None,
    chroma_dir: Path | str | None = None,
    kb_version_path: Path | str | None = None,
    reset: bool = False,
    settings: Settings | None = None,
    embedder: Any = None,
    chroma_client: Any = None,
    collection_name: str = "dume_knowledge",
    user_id: int | str | None = None,
) -> IndexingResult:
    """Perform incremental indexing of documents into ChromaDB.

    Args:
        raw_dir: Path to directory containing raw documents.
        chroma_dir: Directory where persistent ChromaDB files are stored.
        kb_version_path: Path to kb_version.json metadata file.
        reset: If True, clear vector store and index all documents from scratch.
        settings: Application settings. If None, loaded via get_settings().
        embedder: Optional injected embedder model for testing.
        chroma_client: Optional injected ChromaDB client for testing.
        collection_name: Target Chroma collection name.
        user_id: Optional owner ID added to chunk metadata.

    Returns:
        IndexingResult: Execution metrics.
    """
    active_settings = settings or get_settings()

    resolved_raw_dir = Path(raw_dir or active_settings.RAW_DIR).resolve()
    resolved_chroma_dir = Path(chroma_dir or active_settings.CHROMA_DIR).resolve()
    resolved_kb_path = Path(
        kb_version_path or (active_settings.DATA_DIR / "kb_version.json")
    ).resolve()

    resolved_chroma_dir.mkdir(parents=True, exist_ok=True)
    resolved_kb_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize Chroma client
    client = chroma_client
    if client is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(resolved_chroma_dir))

    # Handle reset
    if reset:
        logger.info("Reset requested: purging collection '%s'", collection_name)
        try:
            client.delete_collection(name=collection_name)
        except Exception as exc:
            logger.debug(
                "Collection '%s' did not exist during reset: %s",
                collection_name,
                exc,
            )
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        kb_data: dict[str, Any] = {
            "version": 0,
            "last_updated": None,
            "config_hash": active_settings.config_hash(),
            "files": {},
        }
    else:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        kb_data = load_kb_version(resolved_kb_path)

    # Scan raw documents on disk
    if not resolved_raw_dir.exists() or not resolved_raw_dir.is_dir():
        raise FileNotFoundError(f"Raw documents directory not found: {resolved_raw_dir}")

    matching_files = [
        p
        for p in resolved_raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}
    ]
    matching_files.sort()

    existing_files: dict[str, Any] = kb_data.get("files", {})

    # Detect deleted files
    current_source_names = {
        p.relative_to(resolved_raw_dir).as_posix() for p in matching_files
    }
    deleted_files = 0
    for stale_source in list(existing_files.keys()):
        if stale_source not in current_source_names:
            logger.info("Source file removed: purging chunks for %s", stale_source)
            stale_ids = existing_files[stale_source].get("chunk_ids", [])
            if stale_ids:
                try:
                    collection.delete(ids=stale_ids)
                except Exception as exc:
                    logger.warning("Error deleting stale chunk IDs: %s", exc)
            del existing_files[stale_source]
            deleted_files += 1

    # Determine files needing loading and indexing (fast hash check avoids opening unchanged files)
    from dume.ingestion.loader import compute_file_hash, load_single_file

    files_to_process: dict[str, tuple[str, list[Chunk]]] = {}
    skipped_files = 0

    for file_path in matching_files:
        source_name = file_path.relative_to(resolved_raw_dir).as_posix()
        current_hash = compute_file_hash(file_path)
        prev_entry = existing_files.get(source_name)

        if prev_entry and prev_entry.get("doc_hash") == current_hash and not reset:
            logger.debug("Skipping unchanged document: %s", source_name)
            skipped_files += 1
            continue

        if prev_entry:
            old_ids = prev_entry.get("chunk_ids", [])
            if old_ids:
                logger.info("Purging old chunks for modified document: %s", source_name)
                try:
                    collection.delete(ids=old_ids)
                except Exception as exc:
                    logger.warning("Error deleting modified chunk IDs: %s", exc)

        source_pages = load_single_file(file_path, resolved_raw_dir)
        chunks = chunk_pages(
            source_pages,
            chunk_size=active_settings.CHUNK_SIZE,
            chunk_overlap=active_settings.CHUNK_OVERLAP,
        )
        if user_id is not None:
            for chunk in chunks:
                chunk.metadata["user_id"] = str(user_id)
        files_to_process[source_name] = (current_hash, chunks)

    # Embed and upsert batches
    total_chunks_indexed = 0
    indexed_files = 0

    if files_to_process:
        active_embedder = embedder or get_embedder(active_settings.EMBED_MODEL)
        batch_size = min(active_settings.BATCH_SIZE, 32)

        for source_name, (doc_hash, chunks) in files_to_process.items():
            if not chunks:
                existing_files[source_name] = {
                    "doc_hash": doc_hash,
                    "chunk_count": 0,
                    "chunk_ids": [],
                }
                indexed_files += 1
                continue

            logger.info(
                "Upserting %d chunk(s) for '%s' in batches of <= %d",
                len(chunks),
                source_name,
                batch_size,
            )
            indexed_chunk_ids: list[str] = []

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_ids = [c.chunk_id for c in batch_chunks]
                batch_texts = [c.text for c in batch_chunks]
                batch_metadatas = [c.metadata for c in batch_chunks]

                raw_embeddings = active_embedder.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                if hasattr(raw_embeddings, "tolist"):
                    embeddings_list = raw_embeddings.tolist()
                else:
                    embeddings_list = [list(e) for e in raw_embeddings]

                collection.upsert(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=embeddings_list,
                    metadatas=batch_metadatas,
                )
                indexed_chunk_ids.extend(batch_ids)

                batch_idx = (i // batch_size) + 1
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                if batch_idx % 20 == 0 or batch_idx == total_batches:
                    logger.info(
                        "Progress for '%s': batch %d/%d (%d/%d chunks indexed)",
                        source_name,
                        batch_idx,
                        total_batches,
                        len(indexed_chunk_ids),
                        len(chunks),
                    )

            existing_files[source_name] = {
                "doc_hash": doc_hash,
                "chunk_count": len(chunks),
                "chunk_ids": indexed_chunk_ids,
            }
            total_chunks_indexed += len(chunks)
            indexed_files += 1

    # Update knowledge base manifest
    new_version = kb_data.get("version", 0) + 1
    kb_data["version"] = new_version
    kb_data["last_updated"] = datetime.now(timezone.utc).isoformat()
    kb_data["config_hash"] = active_settings.config_hash()
    kb_data["files"] = existing_files
    save_kb_version(resolved_kb_path, kb_data)

    logger.info(
        "Ingestion completed: %d indexed, %d skipped, %d deleted, %d total chunks (KB v%d)",
        indexed_files,
        skipped_files,
        deleted_files,
        total_chunks_indexed,
        new_version,
    )

    return IndexingResult(
        indexed_files=indexed_files,
        skipped_files=skipped_files,
        deleted_files=deleted_files,
        chunks_indexed=total_chunks_indexed,
        kb_version=new_version,
    )
