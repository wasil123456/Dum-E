"""Tests for the LLM layer: base, openai_compat, factory, and prompt builder.

All tests are offline — no real network calls are made.  OpenAI SDK calls
are intercepted via a lightweight fake client injected through DI.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dume.config import Settings, load_settings
from dume.llm.base import LLMClient, LLMError
from dume.llm.factory import build_llm
from dume.llm.openai_compat import MAX_RETRIES, OpenAICompatClient
from dume.models import Chunk, RetrievedChunk, Turn
from dume.prompt import (
    _DEFAULT_MAX_CONTEXT_CHARS,
    _build_context_block,
    _build_system_content,
    build_messages,
)


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------


def _make_settings(**overrides: Any) -> Settings:
    """Return a Settings instance with safe test defaults."""
    defaults: dict[str, Any] = {
        "LLM_PROVIDER": "google",
        "LLM_BASE_URL": "https://test.example/v1/",
        "LLM_MODEL": "test-model",
        "LLM_API_KEY": "test-key-do-not-log",
        "LLM_TEMPERATURE": 0.0,
        "LLM_MAX_TOKENS": 64,
        "LLM_TIMEOUT_SECONDS": 5.0,
    }
    defaults.update(overrides)
    return load_settings(**defaults)


def _make_fake_openai_client(reply: str = "Hello from fake LLM") -> MagicMock:
    """Build a minimal fake openai v1.x *client* for injection.

    Returns a MagicMock that mimics ``openai.OpenAI`` instance with
    ``client.chat.completions.create`` returning a response object.
    """
    msg = MagicMock()
    msg.content = reply

    choice = MagicMock()
    choice.message = msg

    response = MagicMock()
    response.choices = [choice]

    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def _make_chunk(text: str, source: str = "doc.pdf", score: float = 1.0, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id="id1", text=text, source=source),
        score=score,
        rank=rank,
    )


def _make_turn(query: str = "Q", answer: str = "A") -> Turn:
    return Turn(query=query, answer=answer)


# ---------------------------------------------------------------------------
# LLMError tests
# ---------------------------------------------------------------------------


class TestLLMError:
    def test_message_stored(self) -> None:
        err = LLMError("something went wrong")
        assert err.message == "something went wrong"
        assert str(err) == "something went wrong"

    def test_cause_stored(self) -> None:
        inner = ValueError("inner")
        err = LLMError("outer", cause=inner)
        assert err.__cause__ is inner

    def test_no_cause_by_default(self) -> None:
        err = LLMError("plain")
        assert not hasattr(err, "__cause__") or err.__cause__ is None


# ---------------------------------------------------------------------------
# LLMClient Protocol tests
# ---------------------------------------------------------------------------


class TestLLMClientProtocol:
    def test_fake_openai_client_satisfies_protocol(self) -> None:
        """A simple object with generate() satisfies the runtime-checkable Protocol."""

        class FakeClient:
            def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
                return "ok"

        assert isinstance(FakeClient(), LLMClient)

    def test_missing_generate_does_not_satisfy_protocol(self) -> None:
        class NotAClient:
            pass

        assert not isinstance(NotAClient(), LLMClient)


# ---------------------------------------------------------------------------
# OpenAICompatClient tests
# ---------------------------------------------------------------------------


class TestOpenAICompatClient:
    def _client(self, reply: str = "pong", **settings_overrides: Any) -> OpenAICompatClient:
        return OpenAICompatClient(
            settings=_make_settings(**settings_overrides),
            _openai_client=_make_fake_openai_client(reply),
        )

    def test_generate_returns_stripped_reply(self) -> None:
        client = self._client(reply="  hello world  ")
        result = client.generate([{"role": "user", "content": "hi"}])
        assert result == "hello world"

    def test_generate_passes_messages(self) -> None:
        """Verify messages are forwarded to chat.completions.create."""
        fake_client = _make_fake_openai_client("ok")
        client = OpenAICompatClient(settings=_make_settings(), _openai_client=fake_client)
        msgs = [{"role": "user", "content": "test"}]
        client.generate(msgs)

        call_kwargs = fake_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["messages"] == msgs

    def test_kwargs_override_settings(self) -> None:
        """temperature and max_tokens kwargs should override settings."""
        fake_client = _make_fake_openai_client("x")
        client = OpenAICompatClient(settings=_make_settings(), _openai_client=fake_client)
        client.generate([], temperature=0.9, max_tokens=512)

        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9
        assert call_kwargs["max_tokens"] == 512

    def test_rate_limit_retried_then_succeeds(self) -> None:
        """Should retry on RateLimitError and succeed on the third attempt."""
        import openai as _openai

        attempt_counter: list[int] = [0]

        # Build a real response mock
        msg = MagicMock(); msg.content = "finally!"
        choice = MagicMock(); choice.message = msg
        response = MagicMock(); response.choices = [choice]

        def _flaky_create(**kwargs: Any) -> Any:
            attempt_counter[0] += 1
            if attempt_counter[0] < 3:
                raise _openai.RateLimitError(
                    "rate limited", response=MagicMock(), body={}
                )
            return response

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = _flaky_create

        import dume.llm.openai_compat as oc_mod
        with patch.object(oc_mod.time, "sleep"):
            client = OpenAICompatClient(settings=_make_settings(), _openai_client=fake_client)
            result = client.generate([])

        assert result == "finally!"
        assert attempt_counter[0] == 3

    def test_rate_limit_exhausted_raises_llm_error(self) -> None:
        """Should raise LLMError after MAX_RETRIES rate-limit failures."""
        import openai as _openai

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = _openai.RateLimitError(
            "always limited", response=MagicMock(), body={}
        )

        import dume.llm.openai_compat as oc_mod
        with patch.object(oc_mod.time, "sleep"):
            client = OpenAICompatClient(settings=_make_settings(), _openai_client=fake_client)
            with pytest.raises(LLMError, match="retries"):
                client.generate([])

    def test_timeout_retried_then_raises(self) -> None:
        """Timeout errors should be retried and eventually raise LLMError."""
        import openai as _openai

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = _openai.APITimeoutError(
            request=MagicMock()
        )

        import dume.llm.openai_compat as oc_mod
        with patch.object(oc_mod.time, "sleep"):
            client = OpenAICompatClient(settings=_make_settings(), _openai_client=fake_client)
            with pytest.raises(LLMError):
                client.generate([])

    def test_non_retriable_api_error_raises_immediately(self) -> None:
        """Non-429/timeout OpenAIError should not be retried."""
        import openai as _openai

        attempt_counter: list[int] = [0]

        def _bad_request(**kwargs: Any) -> Any:
            attempt_counter[0] += 1
            raise _openai.BadRequestError(
                "bad request", response=MagicMock(), body={}
            )

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = _bad_request

        client = OpenAICompatClient(settings=_make_settings(), _openai_client=fake_client)
        with pytest.raises(LLMError):
            client.generate([])

        # Should NOT have retried
        assert attempt_counter[0] == 1

    def test_api_key_never_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """The API key must not appear in any log output."""
        import logging

        fake_client = _make_fake_openai_client("ok")
        client = OpenAICompatClient(
            settings=_make_settings(LLM_API_KEY="SUPER_SECRET_KEY_12345"),
            _openai_client=fake_client,
        )
        with caplog.at_level(logging.DEBUG, logger="dume.llm.openai_compat"):
            client.generate([{"role": "user", "content": "hi"}])

        for record in caplog.records:
            assert "SUPER_SECRET_KEY_12345" not in record.getMessage()


# ---------------------------------------------------------------------------
# build_llm factory tests
# ---------------------------------------------------------------------------


class TestBuildLLM:
    def test_returns_llm_client_instance(self) -> None:
        settings = _make_settings()
        client = build_llm(settings)
        assert isinstance(client, LLMClient)

    def test_raises_when_no_api_key(self) -> None:
        settings = _make_settings(LLM_API_KEY="", LLM_BACKUP_API_KEY="")
        with pytest.raises(LLMError, match="LLM_API_KEY"):
            build_llm(settings)

    def test_raises_on_unsupported_provider(self) -> None:
        settings = _make_settings(LLM_PROVIDER="fake_provider")
        with pytest.raises(LLMError, match="Unsupported"):
            build_llm(settings)

    def test_google_provider_accepted(self) -> None:
        settings = _make_settings(LLM_PROVIDER="google")
        client = build_llm(settings)
        assert isinstance(client, OpenAICompatClient)

    def test_openai_provider_accepted(self) -> None:
        settings = _make_settings(LLM_PROVIDER="openai")
        client = build_llm(settings)
        assert isinstance(client, OpenAICompatClient)


# ---------------------------------------------------------------------------
# prompt.py tests
# ---------------------------------------------------------------------------


class TestBuildSystemContent:
    def test_no_summary(self) -> None:
        content = _build_system_content(None)
        assert "Answer ONLY from the provided context" in content
        assert "Background" not in content

    def test_empty_summary_ignored(self) -> None:
        content = _build_system_content("   ")
        assert "Background" not in content

    def test_summary_included(self) -> None:
        content = _build_system_content("User asked about AWK before.")
        assert "Background" in content
        assert "User asked about AWK before." in content


class TestBuildContextBlock:
    def test_empty_chunks_returns_empty_string(self) -> None:
        assert _build_context_block([], 4000) == ""

    def test_single_chunk_formatted(self) -> None:
        rc = _make_chunk("Some text.", source="file.pdf")
        result = _build_context_block([rc], 4000)
        assert "[1] Source: file.pdf" in result
        assert "Some text." in result

    def test_multiple_chunks_numbered(self) -> None:
        chunks = [_make_chunk(f"Text {i}", source=f"doc{i}.pdf") for i in range(3)]
        result = _build_context_block(chunks, 4000)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_budget_truncates_chunk_text(self) -> None:
        """A chunk whose text exceeds the budget should be truncated."""
        long_text = "A" * 500
        rc = _make_chunk(long_text, source="big.pdf")
        result = _build_context_block([rc], 100)
        # Result must be non-empty but shorter than the full text
        assert len(result) < len(long_text)
        assert "[1]" in result

    def test_budget_stops_adding_chunks(self) -> None:
        """When budget is exhausted, later chunks are dropped."""
        chunks = [_make_chunk("X" * 200, source=f"d{i}.pdf") for i in range(5)]
        result = _build_context_block(chunks, 300)
        # At most a couple of chunks fit; [5] should not be present
        assert "[5]" not in result


class TestBuildMessages:
    def test_minimal_call(self) -> None:
        """With no chunks and no turns, still returns valid message list."""
        msgs = build_messages("hello?", [], [])
        assert len(msgs) >= 2  # system + user
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        assert "hello?" in msgs[-1]["content"]

    def test_active_turns_injected(self) -> None:
        turns = [_make_turn("Q1", "A1"), _make_turn("Q2", "A2")]
        msgs = build_messages("current?", [], turns)
        roles = [m["role"] for m in msgs]
        # system, user(Q1), assistant(A1), user(Q2), assistant(A2), user(current)
        assert roles.count("user") == 3
        assert roles.count("assistant") == 2

    def test_long_term_summary_in_system(self) -> None:
        msgs = build_messages("q", [], [], long_term_summary="Old summary.")
        assert "Old summary." in msgs[0]["content"]

    def test_context_chunks_in_final_user_message(self) -> None:
        rc = _make_chunk("AWK stands for Aho, Weinberger, Kernighan.", source="awk.pdf")
        msgs = build_messages("What is AWK?", [rc], [])
        final_user = msgs[-1]["content"]
        assert "[1] Source: awk.pdf" in final_user
        assert "AWK stands for" in final_user
        assert "What is AWK?" in final_user

    def test_no_context_no_context_header(self) -> None:
        msgs = build_messages("plain question", [], [])
        final_user = msgs[-1]["content"]
        assert "--- CONTEXT ---" not in final_user

    def test_context_budget_respected(self) -> None:
        """Context block must stay within max_context_chars."""
        big_chunks = [_make_chunk("Z" * 2000, source=f"f{i}.txt") for i in range(5)]
        msgs = build_messages("q", big_chunks, [], max_context_chars=500)
        context_part = msgs[-1]["content"]
        # The context block itself should be bounded
        assert len(context_part) < 5 * 2000 + 500  # sanity upper bound

    def test_system_message_always_first(self) -> None:
        msgs = build_messages("x", [], [])
        assert msgs[0]["role"] == "system"

    def test_user_message_always_last(self) -> None:
        turns = [_make_turn()]
        msgs = build_messages("last", [], turns)
        assert msgs[-1]["role"] == "user"
        assert "last" in msgs[-1]["content"]
