"""Document ingestion CLI script for DUM-E.

Scans the raw data folder, chunks documents, embeds text using CPU-friendly
sentence transformers, and upserts them into ChromaDB with incremental sync.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from dume.config import get_settings
from dume.ingestion.indexer import index_documents
from dume.logging_setup import setup_logging

logger = logging.getLogger("dume.ingest")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into DUM-E vector knowledge base.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Purge existing vector database and re-index all documents from scratch.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Optional custom directory path containing raw documents (.txt, .md, .pdf).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    return parser.parse_args(args)


def main(cli_args: list[str] | None = None) -> int:
    """Execute document ingestion workflow."""
    args = parse_args(cli_args)
    settings = get_settings()

    setup_logging(level=args.log_level)
    logger.info("Starting DUM-E document ingestion pipeline...")

    raw_dir = args.raw_dir or settings.RAW_DIR
    if not raw_dir.exists():
        logger.error("Raw documents directory does not exist: %s", raw_dir)
        return 1

    try:
        result = index_documents(
            raw_dir=raw_dir,
            reset=args.reset,
            settings=settings,
        )

        logger.info("==========================================")
        logger.info("Ingestion completed successfully!")
        logger.info("  Files indexed : %d", result.indexed_files)
        logger.info("  Files skipped : %d (unchanged)", result.skipped_files)
        logger.info("  Files deleted : %d", result.deleted_files)
        logger.info("  Chunks stored : %d", result.chunks_indexed)
        logger.info("  KB Version    : v%d", result.kb_version)
        logger.info("==========================================")
        return 0
    except Exception as exc:
        logger.exception("Ingestion failed with unhandled exception: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
