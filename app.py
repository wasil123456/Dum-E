"""Streamlit frontend for the DUM-E document-grounded assistant."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import time
from typing import Protocol, TypedDict

import streamlit as st

from dume.config import Settings, get_settings
from dume.engine import DumeEngine
from dume.ingestion.indexer import IndexingResult, index_documents
from dume.logging_setup import setup_logging

logger = logging.getLogger(__name__)
SESSION_ID = "streamlit_user"


class UploadedPdf(Protocol):
    """Minimal interface required from Streamlit's uploaded file object."""

    name: str

    def getvalue(self) -> bytes:
        """Return the uploaded file contents."""


class ResponseMetadata(TypedDict):
    """Metrics rendered below an assistant response."""

    source: str
    latency_ms: float
    source_count: int


@st.cache_resource
def get_engine() -> DumeEngine:
    """Create one reusable engine for the Streamlit process."""
    settings = get_settings()
    settings.ensure_directories()
    return DumeEngine(settings=settings)


def _save_uploaded_pdf(uploaded_file: UploadedPdf, settings: Settings) -> Path:
    """Persist an uploaded PDF below the configured raw-document directory."""
    filename = Path(str(uploaded_file.name)).name
    if not filename or Path(filename).suffix.casefold() != ".pdf":
        raise ValueError("Only PDF files can be uploaded.")

    content = uploaded_file.getvalue()
    if not content:
        raise ValueError("The uploaded PDF is empty.")

    settings.RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = settings.RAW_DIR / filename
    digest = hashlib.sha256(content).hexdigest()
    digest_path = destination.with_suffix(f"{destination.suffix}.sha256")
    if not destination.exists() or not digest_path.exists() or digest_path.read_text(
        encoding="ascii"
    ) != digest:
        destination.write_bytes(content)
        digest_path.write_text(digest, encoding="ascii")
    return destination


def _render_memory(engine: DumeEngine, settings: Settings) -> None:
    """Render the current conversation memory and engine counters."""
    st.sidebar.subheader("Memory")
    memory = engine.get_memory_view(SESSION_ID)
    active_turns = memory.get("active_turns", [])
    st.sidebar.caption(
        f"Active turns: {len(active_turns)} / {settings.MAX_ACTIVE_TURNS}"
    )
    summary = str(memory.get("long_term_summary", "")).strip()
    if summary:
        st.sidebar.write(summary)
    else:
        st.sidebar.caption("No long-term summary yet.")

    stats = engine.stats()
    if stats:
        st.sidebar.subheader("Engine stats")
        st.sidebar.json(stats)


def _render_answer_metadata(metadata: ResponseMetadata) -> None:
    """Display response diagnostics below an assistant message."""
    st.caption(
        " · ".join(
            [
                f"Source: {metadata['source']}",
                f"Latency: {metadata['latency_ms']:.0f} ms",
                f"Retrieved sources: {metadata['source_count']}",
            ]
        )
    )


def _render_sidebar(engine: DumeEngine, settings: Settings) -> None:
    """Render upload, ingestion, reset, and memory controls."""
    st.sidebar.header("Documents")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key="pdf_uploader",
    )
    if uploaded_file is not None:
        try:
            saved_path = _save_uploaded_pdf(uploaded_file, settings)
            st.session_state["uploaded_path"] = saved_path
            st.sidebar.success(f"Saved `{saved_path.name}`")
        except (OSError, ValueError) as exc:
            logger.exception("Could not save uploaded document")
            st.sidebar.error(f"Could not save document: {exc}")

    uploaded_path = st.session_state.get("uploaded_path")
    if isinstance(uploaded_path, Path) and uploaded_path.exists():
        if st.sidebar.button("Ingest Document", type="primary"):
            try:
                with st.sidebar.status("Indexing document..."):
                    result: IndexingResult = index_documents(settings=settings)
                st.sidebar.success(
                    f"Indexed {result.chunks_indexed} chunks (KB v{result.kb_version})."
                )
            except (OSError, RuntimeError, ValueError) as exc:
                logger.exception("Document ingestion failed")
                st.sidebar.error(f"Ingestion failed: {exc}")

    st.sidebar.divider()
    if st.sidebar.button("Clear / Reset Session"):
        engine.reset_session(SESSION_ID)
        st.session_state["messages"] = []
        st.session_state.pop("uploaded_path", None)
        st.rerun()

    _render_memory(engine, settings)


def main() -> None:
    """Run the DUM-E Streamlit application."""
    st.set_page_config(page_title="DUM-E", page_icon="🤖", layout="wide")
    settings = get_settings()
    settings.ensure_directories()
    setup_logging(level="INFO", log_file=settings.LOG_DIR / "dume.log")
    engine = get_engine()

    st.title("DUM-E")
    st.caption("Document-grounded assistant")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    _render_sidebar(engine, settings)

    for message in st.session_state["messages"]:
        with st.chat_message(str(message["role"])):
            st.markdown(str(message["content"]))
            if message["role"] == "assistant" and "metadata" in message:
                _render_answer_metadata(message["metadata"])

    question = st.chat_input("Ask a question about your indexed documents")
    if question is None or not question.strip():
        return

    clean_question = question.strip()
    st.session_state["messages"].append(
        {"role": "user", "content": clean_question}
    )
    with st.chat_message("user"):
        st.markdown(clean_question)

    with st.chat_message("assistant"):
        started = time.perf_counter()
        try:
            answer = engine.ask(SESSION_ID, clean_question)
        except Exception as exc:
            logger.exception("Question handling failed for Streamlit session")
            st.error(f"Unable to answer this question: {exc}")
            return

        elapsed_ms = (time.perf_counter() - started) * 1000
        metadata: ResponseMetadata = {
            "source": "cache" if answer.cached else "generated",
            "latency_ms": elapsed_ms,
            "source_count": len(answer.sources),
        }
        st.markdown(answer.text)
        _render_answer_metadata(metadata)
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer.text,
                "metadata": metadata,
            }
        )


if __name__ == "__main__":
    main()
