"""Thread-safe SQLite response caching for DUM-E."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
import hashlib
import re
import sqlite3
import string
import threading
import time
from pathlib import Path

from dume.config import Settings, get_settings
from dume.models import Turn

_WHITESPACE = re.compile(r"\s+")
_AMBIGUOUS_WORDS = frozenset(
    {
        "it",
        "that",
        "this",
        "they",
        "them",
        "those",
        "he",
        "she",
        "we",
        "there",
        "here",
        "again",
        "also",
        "more",
        "why",
        "how",
        "what about",
        "and",
        "continue",
        "tell me more",
    }
)


def normalize_question(question: str) -> str:
    """Normalize a question for exact-match cache lookup."""
    lowered = question.casefold().strip()
    without_punctuation = lowered.translate(
        str.maketrans("", "", string.punctuation)
    )
    return _WHITESPACE.sub(" ", without_punctuation).strip()


def make_cache_key(
    question: str,
    kb_version: int | str,
    config_hash: str,
    user_id: int | str | None = None,
) -> str:
    """Create a stable SHA-256 key from question and answer context versions."""
    payload = (
        f"{user_id if user_id is not None else ''}\x1f"
        f"{normalize_question(question)}\x1f{kb_version}\x1f{config_hash}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_cacheable(
    question: str,
    active_turns: Sequence[Turn] | None = None,
) -> bool:
    """Return whether a question is safe to reuse as an exact-match response.

    Questions with no active history are cacheable. With history, ambiguous
    pronoun/deictic or fragment-style follow-ups are excluded.
    """
    normalized = normalize_question(question)
    if not normalized:
        return False
    if not active_turns:
        return True

    words = normalized.split()
    if normalized in _AMBIGUOUS_WORDS:
        return False
    if len(words) <= 3 and words[0] in _AMBIGUOUS_WORDS:
        return False
    if words[0] in {"and", "also", "then", "but", "so"}:
        return False
    return not any(word in {"it", "this", "that", "they", "them", "those"} for word in words)


class ResponseCache:
    """SQLite-backed response cache with process-local thread synchronization."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        ttl_seconds: int | float | None = None,
        settings: Settings | None = None,
        *,
        clock: callable = time.time,
    ) -> None:
        """Create or open the cache database."""
        config = settings or get_settings()
        self.db_path = Path(db_path or config.CACHE_DB)
        self.ttl_seconds = (
            config.CACHE_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        )
        if self.ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        self._clock = clock
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30,
            isolation_level=None,
        )
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    response TEXT NOT NULL,
                    kb_version TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )

    def get(
        self,
        question: str,
        kb_version: int | str,
        config_hash: str,
        user_id: int | str | None = None,
    ) -> str | None:
        """Return a live cached response, or ``None`` on a miss."""
        key = make_cache_key(question, kb_version, config_hash, user_id)
        now = self._clock()
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT response, expires_at FROM response_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            if float(row[1]) <= now:
                connection.execute(
                    "DELETE FROM response_cache WHERE cache_key = ?", (key,)
                )
                return None
            return str(row[0])

    def put(
        self,
        question: str,
        response: str,
        kb_version: int | str,
        config_hash: str,
        user_id: int | str | None = None,
    ) -> None:
        """Store or replace a response with the configured TTL."""
        normalized = normalize_question(question)
        if not normalized:
            raise ValueError("question must contain non-whitespace text")
        now = self._clock()
        key = make_cache_key(question, kb_version, config_hash, user_id)
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO response_cache
                    (cache_key, question, response, kb_version, config_hash,
                     created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    question=excluded.question,
                    response=excluded.response,
                    kb_version=excluded.kb_version,
                    config_hash=excluded.config_hash,
                    created_at=excluded.created_at,
                    expires_at=excluded.expires_at
                """,
                (
                    key,
                    normalized,
                    response,
                    str(kb_version),
                    config_hash,
                    now,
                    now + self.ttl_seconds,
                ),
            )

    def purge_stale(self) -> int:
        """Delete expired entries and return the number removed."""
        with self._lock, closing(self._connect()) as connection:
            cursor = connection.execute(
                "DELETE FROM response_cache WHERE expires_at <= ?", (self._clock(),)
            )
            return cursor.rowcount
