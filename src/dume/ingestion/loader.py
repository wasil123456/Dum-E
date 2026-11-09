"""Document loading for DUM-E.

Recursively reads .txt, .md, and .pdf files from a directory, extracts text
per page, and computes file-level content hashes for incremental indexing.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import NamedTuple

import pypdf

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".pdf"})


class LoadedPage(NamedTuple):
    """Represents an extracted document page or text file segment.

    Attributes:
        source_name: Relative path or filename of the document.
        page: Page number (1-based index).
        text: Extracted text content of the page.
        doc_hash: Deterministic SHA-256 hash of the source file.
    """

    source_name: str
    page: int
    text: str
    doc_hash: str


def compute_file_hash(file_path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        file_path: Path to the target file.

    Returns:
        str: 64-character hexadecimal SHA-256 hash.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        OSError: If reading the file fails.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_text_file(file_path: Path, source_name: str, doc_hash: str) -> list[LoadedPage]:
    """Read a plain text or markdown file as a single page."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning(
            "UTF-8 decode failed for %s, falling back to latin-1 encoding",
            source_name,
        )
        content = file_path.read_text(encoding="latin-1")

    cleaned = content.strip()
    if not cleaned:
        logger.warning("Skipping empty text document: %s", source_name)
        return []

    return [LoadedPage(source_name=source_name, page=1, text=cleaned, doc_hash=doc_hash)]


def _load_pdf_file(file_path: Path, source_name: str, doc_hash: str) -> list[LoadedPage]:
    """Extract text from each page of a PDF document using pypdf."""
    pages: list[LoadedPage] = []
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*fontTools is required.*")
            reader = pypdf.PdfReader(str(file_path))
            for page_idx, page in enumerate(reader.pages, start=1):
                extracted = page.extract_text() or ""
                cleaned = extracted.strip()
                if cleaned:
                    pages.append(
                        LoadedPage(
                            source_name=source_name,
                            page=page_idx,
                            text=cleaned,
                            doc_hash=doc_hash,
                        )
                    )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to extract text from PDF '{source_name}': {exc}"
        ) from exc

    if not pages:
        logger.warning("No extractable text found in PDF: %s", source_name)

    return pages


def load_single_file(file_path: Path, raw_dir: Path) -> list[LoadedPage]:
    """Load and extract pages from a single supported file.

    Args:
        file_path: Absolute or resolved path to the file.
        raw_dir: Base directory to compute relative source names from.

    Returns:
        list[LoadedPage]: Extracted pages from this file.
    """
    rel_path = file_path.relative_to(raw_dir).as_posix()
    doc_hash = compute_file_hash(file_path)
    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return _load_text_file(file_path, rel_path, doc_hash)
    elif suffix == ".pdf":
        return _load_pdf_file(file_path, rel_path, doc_hash)
    return []


def load_documents(raw_dir: Path | str) -> list[LoadedPage]:
    """Recursively scan raw_dir for supported files and extract page-level text.

    Args:
        raw_dir: Directory containing documents (.txt, .md, .pdf).

    Returns:
        list[LoadedPage]: Extracted pages with source_name, page, text, and doc_hash.

    Raises:
        FileNotFoundError: If raw_dir does not exist or is not a directory.
    """
    directory = Path(raw_dir).resolve()
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Raw documents directory not found: {directory}")

    results: list[LoadedPage] = []
    matching_files = [
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    matching_files.sort()

    logger.info("Found %d supported document(s) in %s", len(matching_files), directory)

    for file_path in matching_files:
        try:
            pages = load_single_file(file_path, directory)
            results.extend(pages)
        except Exception as exc:
            logger.error("Error processing file %s: %s", file_path, exc)
            raise

    logger.info("Loaded %d total page(s) across %d file(s)", len(results), len(matching_files))
    return results
