"""Unit tests for DUM-E document ingestion pipeline: loader, chunker, indexer, and CLI."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dume.config import Settings
from dume.ingestion.chunker import chunk_document_page, chunk_pages, chunk_text, generate_chunk_id
from dume.ingestion.indexer import IndexingResult, index_documents, load_kb_version
from dume.ingestion.loader import LoadedPage, compute_file_hash, load_documents
from dume.models import Chunk
from scripts.ingest import parse_args


class FakeEmbedder:
    """Mock embedder returning fixed-dimension vectors for fast isolated tests."""

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension
        self.call_count = 0
        self.last_batch_size = 0

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> list[list[float]]:
        self.call_count += 1
        self.last_batch_size = batch_size
        vectors: list[list[float]] = []
        for t in texts:
            val = float(len(t) % 10 + 1) / 10.0
            vectors.append([val] * self.dimension)
        return vectors


# =====================================================================
# Loader Tests
# =====================================================================


def test_compute_file_hash(tmp_path: Path) -> None:
    """Verify SHA-256 calculation for file contents."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("DUM-E test content", encoding="utf-8")

    expected_hash = hashlib.sha256("DUM-E test content".encode("utf-8")).hexdigest()
    assert compute_file_hash(sample_file) == expected_hash


def test_load_documents_recursive(tmp_path: Path) -> None:
    """Verify loader finds .txt and .md files recursively and produces LoadedPage tuples."""
    raw_dir = tmp_path / "raw"
    sub_dir = raw_dir / "nested"
    sub_dir.mkdir(parents=True)

    file_a = raw_dir / "intro.txt"
    file_a.write_text("Hello from root text file.", encoding="utf-8")

    file_b = sub_dir / "details.md"
    file_b.write_text("# Markdown Title\nNested file body content.", encoding="utf-8")

    # Unsupported file that must be ignored
    ignored = raw_dir / "image.png"
    ignored.write_bytes(b"\x89PNG\r\n\x1a\n")

    pages = load_documents(raw_dir)

    assert len(pages) == 2
    sources = {p.source_name for p in pages}
    assert "intro.txt" in sources
    assert "nested/details.md" in sources or "nested\\details.md" in sources

    for p in pages:
        assert isinstance(p, LoadedPage)
        assert p.page == 1
        assert len(p.text) > 0
        assert len(p.doc_hash) == 64


def test_load_documents_missing_dir(tmp_path: Path) -> None:
    """Verify loading from non-existent directory raises FileNotFoundError."""
    missing = tmp_path / "non_existent_folder"
    with pytest.raises(FileNotFoundError, match="Raw documents directory not found"):
        load_documents(missing)


def test_load_pdf_file_mocked(tmp_path: Path) -> None:
    """Verify PDF extraction handling via pypdf with page numbering."""
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy")

    mock_page_1 = MagicMock()
    mock_page_1.extract_text.return_value = "Page one text content."
    mock_page_2 = MagicMock()
    mock_page_2.extract_text.return_value = "Page two text content."

    mock_reader = MagicMock()
    mock_reader.pages = [mock_page_1, mock_page_2]

    with patch("pypdf.PdfReader", return_value=mock_reader):
        pages = load_documents(tmp_path)

    assert len(pages) == 2
    assert pages[0].page == 1
    assert pages[0].text == "Page one text content."
    assert pages[1].page == 2
    assert pages[1].text == "Page two text content."


# =====================================================================
# Chunker Tests
# =====================================================================


def test_chunk_text_no_mid_word_split() -> None:
    """Verify chunking splits cleanly on word boundaries without breaking words."""
    text = "The quick brown fox jumps over the lazy dog and runs across the wide open fields."
    # With chunk_size=25 and overlap=5
    chunks = chunk_text(text, chunk_size=25, chunk_overlap=5)

    assert len(chunks) > 1
    # Check that no chunk starts or ends with partial broken words
    valid_words = set(text.replace(".", "").split())
    for chunk in chunks:
        chunk_words = chunk.replace(".", "").split()
        for w in chunk_words:
            assert w in valid_words, f"Word '{w}' was split mid-word!"


def test_chunk_text_edge_cases() -> None:
    """Verify boundary conditions and invalid parameters."""
    # Empty string
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []

    # Fits entirely in one chunk
    short = "Short sentence."
    assert chunk_text(short, chunk_size=50, chunk_overlap=10) == [short]

    # Single word longer than chunk_size
    long_word = "Supercalifragilisticexpialidocious"
    result = chunk_text(long_word, chunk_size=10, chunk_overlap=2)
    assert result == [long_word]

    # Invalid chunk_overlap >= chunk_size
    with pytest.raises(ValueError, match="chunk_overlap.*must be strictly less than"):
        chunk_text("test text", chunk_size=50, chunk_overlap=50)

    # Invalid non-positive chunk_size
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("test text", chunk_size=0, chunk_overlap=0)


def test_deterministic_chunk_id() -> None:
    """Verify SHA-1 ID is completely deterministic and sensitive to variations."""
    id1 = generate_chunk_id("file.txt", 1, 0, "Chunk text")
    id2 = generate_chunk_id("file.txt", 1, 0, "Chunk text")
    assert id1 == id2
    assert len(id1) == 40

    # Altering any component produces a different ID
    assert id1 != generate_chunk_id("file.txt", 2, 0, "Chunk text")
    assert id1 != generate_chunk_id("other.txt", 1, 0, "Chunk text")
    assert id1 != generate_chunk_id("file.txt", 1, 1, "Chunk text")
    assert id1 != generate_chunk_id("file.txt", 1, 0, "Different text")


def test_chunk_document_page_and_pages() -> None:
    """Verify chunk_document_page and chunk_pages produce valid Chunk models."""
    loaded_pages = [
        LoadedPage(
            source_name="doc1.txt",
            page=1,
            text="First sentence here. Second sentence follows.",
            doc_hash="hash123",
        ),
        LoadedPage(
            source_name="doc2.txt",
            page=1,
            text="Third sentence in another document.",
            doc_hash="hash456",
        ),
    ]

    chunks = chunk_pages(loaded_pages, chunk_size=30, chunk_overlap=5)
    assert len(chunks) >= 2
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.chunk_id
        assert c.text
        assert "doc_hash" in c.metadata


# =====================================================================
# Indexer & Incremental Sync Tests
# =====================================================================


def test_incremental_indexing_workflow(tmp_path: Path) -> None:
    """Verify full incremental indexing lifecycle: initial, skip unchanged, update modified, delete stale."""
    import chromadb

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    chroma_dir = tmp_path / "chroma"
    kb_path = tmp_path / "kb_version.json"

    file_1 = raw_dir / "guide.txt"
    file_1.write_text(
        "DUM-E is a CPU-friendly backend designed for Windows systems.",
        encoding="utf-8",
    )

    file_2 = raw_dir / "notes.md"
    file_2.write_text(
        "Memory management uses active window turns and Sumy summarization.",
        encoding="utf-8",
    )

    fake_embedder = FakeEmbedder(dimension=16)
    settings = Settings(
        RAW_DIR=raw_dir,
        CHROMA_DIR=chroma_dir,
        DATA_DIR=tmp_path,
        BATCH_SIZE=16,
        CHUNK_SIZE=50,
        CHUNK_OVERLAP=10,
    )

    # 1. Initial Ingestion
    res1 = index_documents(
        raw_dir=raw_dir,
        chroma_dir=chroma_dir,
        kb_version_path=kb_path,
        settings=settings,
        embedder=fake_embedder,
        collection_name="test_col",
    )

    assert res1.indexed_files == 2
    assert res1.skipped_files == 0
    assert res1.deleted_files == 0
    assert res1.chunks_indexed > 0
    assert res1.kb_version == 1

    kb_manifest = load_kb_version(kb_path)
    assert kb_manifest["version"] == 1
    assert "guide.txt" in kb_manifest["files"]
    assert "notes.md" in kb_manifest["files"]

    client = chromadb.PersistentClient(path=str(chroma_dir))
    col = client.get_collection("test_col")
    initial_count = col.count()
    assert initial_count == res1.chunks_indexed

    # 2. Second Ingestion without changes -> All files skipped
    res2 = index_documents(
        raw_dir=raw_dir,
        chroma_dir=chroma_dir,
        kb_version_path=kb_path,
        settings=settings,
        embedder=fake_embedder,
        collection_name="test_col",
    )

    assert res2.indexed_files == 0
    assert res2.skipped_files == 2
    assert res2.deleted_files == 0
    assert res2.chunks_indexed == 0
    assert col.count() == initial_count

    # 3. Modify file_1 -> Only file_1 re-indexed, file_2 skipped
    file_1.write_text(
        "DUM-E updated guide with brand new content requiring re-indexing.",
        encoding="utf-8",
    )

    res3 = index_documents(
        raw_dir=raw_dir,
        chroma_dir=chroma_dir,
        kb_version_path=kb_path,
        settings=settings,
        embedder=fake_embedder,
        collection_name="test_col",
    )

    assert res3.indexed_files == 1
    assert res3.skipped_files == 1
    assert res3.deleted_files == 0
    assert res3.kb_version == 3

    # 4. Delete file_2 -> file_2 purged from Chroma and manifest
    file_2.unlink()

    res4 = index_documents(
        raw_dir=raw_dir,
        chroma_dir=chroma_dir,
        kb_version_path=kb_path,
        settings=settings,
        embedder=fake_embedder,
        collection_name="test_col",
    )

    assert res4.deleted_files == 1
    assert res4.skipped_files == 1
    assert res4.indexed_files == 0
    kb_manifest4 = load_kb_version(kb_path)
    assert "notes.md" not in kb_manifest4["files"]
    assert "guide.txt" in kb_manifest4["files"]

    # 5. Reset flag -> Purges collection and re-indexes all
    res5 = index_documents(
        raw_dir=raw_dir,
        chroma_dir=chroma_dir,
        kb_version_path=kb_path,
        reset=True,
        settings=settings,
        embedder=fake_embedder,
        collection_name="test_col",
    )

    assert res5.indexed_files == 1
    assert res5.skipped_files == 0
    assert res5.deleted_files == 0


# =====================================================================
# CLI Script Argument Parsing Tests
# =====================================================================


def test_scripts_ingest_argument_parsing() -> None:
    """Verify CLI argument parsing defaults and flags."""
    args_default = parse_args([])
    assert args_default.reset is False
    assert args_default.raw_dir is None
    assert args_default.log_level == "INFO"

    args_custom = parse_args(["--reset", "--raw-dir", "custom/docs", "--log-level", "DEBUG"])
    assert args_custom.reset is True
    assert args_custom.raw_dir == Path("custom/docs")
    assert args_custom.log_level == "DEBUG"
