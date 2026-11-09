"""Document chunking for DUM-E.

Splits document text into chunks without breaking mid-word, adhering to
configured chunk sizes and overlaps, and generates deterministic SHA-1 IDs.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Iterable

from dume.ingestion.loader import LoadedPage
from dume.models import Chunk

logger = logging.getLogger(__name__)


def generate_chunk_id(
    source: str,
    page: int | str,
    chunk_index: int | str,
    text: str,
) -> str:
    """Generate a deterministic SHA-1 identifier for a chunk.

    ID is derived from sha1(source + page + chunk_index + text).

    Args:
        source: Relative path or identifier of the source document.
        page: Page number or index.
        chunk_index: Sequential index of the chunk within the page.
        text: Text content of the chunk.

    Returns:
        str: 40-character hexadecimal SHA-1 digest.
    """
    raw_key = f"{source}{page}{chunk_index}{text}"
    return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into segments of at most chunk_size characters without splitting mid-word.

    Args:
        text: Input string to segment.
        chunk_size: Maximum character length per chunk (positive int).
        chunk_overlap: Overlapping character count between consecutive chunks.

    Returns:
        list[str]: Extracted chunk text segments without mid-word splits.

    Raises:
        ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than "
            f"chunk_size ({chunk_size})"
        )

    stripped = text.strip()
    if not stripped:
        return []

    if len(stripped) <= chunk_size:
        return [stripped]

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # Skip leading whitespace
        while start < text_len and text[start].isspace():
            start += 1
        if start >= text_len:
            break

        end = start + chunk_size
        if end >= text_len:
            segment = text[start:].strip()
            if segment:
                chunks.append(segment)
            break

        # Check if boundary is clean
        if text[end].isspace():
            chunk_end = end
        else:
            # Look backward for nearest whitespace
            boundary = -1
            for idx in range(end - 1, start, -1):
                if text[idx].isspace():
                    boundary = idx
                    break

            if boundary != -1:
                chunk_end = boundary
            else:
                # No whitespace backward (single word longer than chunk_size)
                # Look forward for the next whitespace
                forward_boundary = -1
                for idx in range(end, text_len):
                    if text[idx].isspace():
                        forward_boundary = idx
                        break
                chunk_end = forward_boundary if forward_boundary != -1 else text_len

        segment = text[start:chunk_end].strip()
        if segment:
            chunks.append(segment)

        # Calculate next start with overlap
        next_start = chunk_end - chunk_overlap
        if next_start <= start:
            next_start = chunk_end

        # Avoid landing mid-word on next_start
        if next_start < text_len and not text[next_start].isspace():
            # Attempt to move backward to the start of the word
            word_start = next_start
            while word_start > start and not text[word_start - 1].isspace():
                word_start -= 1

            if word_start > start:
                next_start = word_start
            else:
                # If backing up reaches start, advance forward past current word
                while next_start < chunk_end and not text[next_start].isspace():
                    next_start += 1
                while next_start < chunk_end and text[next_start].isspace():
                    next_start += 1
                if next_start <= start:
                    next_start = chunk_end

        if next_start <= start:
            next_start = chunk_end

        start = next_start

    return chunks


def chunk_document_page(
    source_name: str,
    page: int,
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    doc_hash: str = "",
) -> list[Chunk]:
    """Convert an individual page's text into Chunk instances with deterministic IDs.

    Args:
        source_name: Relative path or identifier of the document.
        page: Page number of the document.
        text: Raw text content of the page.
        chunk_size: Maximum character length for each chunk.
        chunk_overlap: Overlap between consecutive chunks.
        doc_hash: SHA-256 hash of the parent file.

    Returns:
        list[Chunk]: List of Chunk dataclass objects.
    """
    segments = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []

    for idx, seg in enumerate(segments):
        chunk_id = generate_chunk_id(
            source=source_name,
            page=page,
            chunk_index=idx,
            text=seg,
        )
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=seg,
                source=source_name,
                metadata={
                    "source": source_name,
                    "subject": source_name.split("/", 1)[0]
                    if "/" in source_name
                    else "",
                    "page": page,
                    "chunk_index": idx,
                    "doc_hash": doc_hash,
                },
            )
        )

    return chunks


def chunk_pages(
    pages: Iterable[LoadedPage],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Chunk an iterable of LoadedPage objects into a flat list of Chunk models.

    Args:
        pages: Sequence of LoadedPage namedtuples.
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Overlapping character count.

    Returns:
        list[Chunk]: All generated Chunk models across the pages.
    """
    all_chunks: list[Chunk] = []
    for page_item in pages:
        page_chunks = chunk_document_page(
            source_name=page_item.source_name,
            page=page_item.page,
            text=page_item.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            doc_hash=page_item.doc_hash,
        )
        all_chunks.extend(page_chunks)
    return all_chunks
