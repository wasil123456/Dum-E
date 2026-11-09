"""Session-aware orchestration for the DUM-E RAG pipeline."""

from __future__ import annotations

from collections import Counter
import json
import logging
import threading
from typing import Any

from dume.cache.response_cache import ResponseCache, is_cacheable
from dume.config import Settings, get_settings
from dume.llm.base import LLMClient
from dume.memory.manager import MemoryManager
from dume.models import Answer
from dume.prompt import build_messages
from dume.retrieval.retriever import Retriever
from dume.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

UNKNOWN_ANSWER = "I don't know based on the provided documents."


class DumeEngine:
    """Coordinate retrieval, generation, memory, and response caching."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
        cache: ResponseCache | None = None,
        settings: Settings | None = None,
        memory_factory: Any = MemoryManager,
    ) -> None:
        """Create an engine with injectable collaborators for testing."""
        self.settings = settings or get_settings()
        self.retriever = retriever
        self.llm = llm
        self.cache = cache or ResponseCache(settings=self.settings)
        self._memory_factory = memory_factory
        self._retrievers: dict[str, Retriever] = {}
        self._memories: dict[str, MemoryManager] = {}
        self._lock = threading.RLock()
        self._stats: Counter[str] = Counter()

    def _get_retriever(self, user_id: int | str | None = None) -> Retriever:
        if self.retriever is not None:
            return self.retriever
        key = str(user_id) if user_id is not None else "default"
        with self._lock:
            retriever = self._retrievers.get(key)
            if retriever is None:
                collection_name = (
                    f"dume_user_{user_id}" if user_id is not None else "dume_knowledge"
                )
                retriever = Retriever(
                    settings=self.settings,
                    vector_store=VectorStore(
                        settings=self.settings,
                        collection_name=collection_name,
                    ),
                )
                self._retrievers[key] = retriever
            return retriever

    def _get_llm(self) -> LLMClient:
        if self.llm is None:
            from dume.llm.factory import build_llm

            self.llm = build_llm(settings=self.settings)
        return self.llm

    def _memory(self, session_id: str, user_id: int | str | None = None) -> MemoryManager:
        if not session_id.strip():
            raise ValueError("session_id must contain non-whitespace text")
        memory_key = (
            f"{user_id}:{session_id}" if user_id is not None else session_id
        )
        with self._lock:
            memory = self._memories.get(memory_key)
            if memory is None:
                memory = self._memory_factory(settings=self.settings)
                self._memories[memory_key] = memory
            return memory

    def _kb_version(self, user_id: int | str | None = None) -> int:
        path = (
            self.settings.DATA_DIR / "users" / str(user_id) / "kb_version.json"
            if user_id is not None
            else self.settings.DATA_DIR / "kb_version.json"
        )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return int(data.get("version", 0))
        except FileNotFoundError:
            return 0
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to read knowledge-base version: {path}") from exc

    def ask(
        self,
        session_id: str,
        question: str,
        subject: str | None = None,
        user_id: int | str | None = None,
    ) -> Answer:
        """Answer a question using the session's memory and RAG pipeline."""
        memory = self._memory(session_id, user_id)
        self._stats["asks"] += 1
        cacheable = is_cacheable(question, memory.active_turns())
        kb_version = self._kb_version(user_id)
        config_hash = self.settings.config_hash()
        cache_question = f"[subject:{subject}] {question}" if subject else question

        if cacheable:
            cached_text = (
                self.cache.get(cache_question, kb_version, config_hash, user_id)
                if user_id is not None
                else self.cache.get(cache_question, kb_version, config_hash)
            )
            if cached_text is not None:
                self._stats["cache_hits"] += 1
                memory.add(question, cached_text)
                return Answer(
                    text=cached_text,
                    cached=True,
                    config_hash=config_hash,
                )
            self._stats["cache_misses"] += 1
        else:
            self._stats["cache_skips"] += 1

        retriever = self._get_retriever(user_id)
        if user_id is not None:
            chunks = retriever.retrieve(
                question,
                subject=subject,
                user_id=user_id,
            )
        elif subject:
            chunks = retriever.retrieve(question, subject=subject)
        else:
            chunks = retriever.retrieve(question)
        if not chunks:
            self._stats["empty_retrievals"] += 1
            return Answer(text=UNKNOWN_ANSWER, config_hash=config_hash)

        messages = build_messages(
            query=question,
            context_chunks=chunks,
            active_turns=memory.active_turns(),
            long_term_summary=memory.long_term_summary() or None,
        )
        try:
            response = self._get_llm().generate(messages)
        except Exception:
            logger.exception("LLM generation failed for session %s", session_id)
            self._stats["llm_errors"] += 1
            raise

        if not response.strip():
            raise RuntimeError("LLM returned an empty response")

        memory.add(question, response)
        if cacheable:
            if user_id is not None:
                self.cache.put(
                    cache_question, response, kb_version, config_hash, user_id
                )
            else:
                self.cache.put(cache_question, response, kb_version, config_hash)
            self._stats["cache_writes"] += 1
        self._stats["generated"] += 1
        return Answer(
            text=response,
            sources=chunks,
            config_hash=config_hash,
        )

    def reset_session(self, session_id: str, user_id: int | str | None = None) -> None:
        """Discard all memory associated with a session."""
        with self._lock:
            memory_key = (
                f"{user_id}:{session_id}" if user_id is not None else session_id
            )
            self._memories.pop(memory_key, None)

    def get_memory_view(
        self, session_id: str, user_id: int | str | None = None
    ) -> dict[str, Any]:
        """Return a read-only snapshot of a session's memory."""
        memory = self._memory(session_id, user_id)
        return {
            "active_turns": memory.active_turns(),
            "long_term_summary": memory.long_term_summary(),
        }

    def stats(self) -> dict[str, int]:
        """Return engine counters as a snapshot."""
        with self._lock:
            return dict(self._stats)
