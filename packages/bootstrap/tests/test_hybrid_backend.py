"""Tests for the HybridBackend cognitive router."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from animus_bootstrap.gateway.cognitive import HybridBackend
from animus_bootstrap.gateway.cognitive_types import CognitiveResponse


def _run(coro):
    return asyncio.run(coro)


def _make_messages(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


class TestHybridClassification:
    """Test query classification logic."""

    def test_routes_self_referential_to_anthropic(self) -> None:
        anthropic = MagicMock()
        ollama = MagicMock()
        hybrid = HybridBackend(anthropic_backend=anthropic, ollama_backend=ollama)

        msgs = _make_messages("animus self improvement suggestions?")
        backend, reason = hybrid._classify_query(msgs)
        assert backend is anthropic
        assert "animus" in reason

    def test_routes_complex_analysis_to_anthropic(self) -> None:
        anthropic = MagicMock()
        ollama = MagicMock()
        hybrid = HybridBackend(anthropic_backend=anthropic, ollama_backend=ollama)

        msgs = _make_messages("analyze the architecture of this system")
        backend, reason = hybrid._classify_query(msgs)
        assert backend is anthropic
        assert "analyze" in reason or "architecture" in reason

    def test_routes_casual_to_ollama(self) -> None:
        anthropic = MagicMock()
        ollama = MagicMock()
        hybrid = HybridBackend(anthropic_backend=anthropic, ollama_backend=ollama)

        backend, reason = hybrid._classify_query(_make_messages("hello how are you"))
        assert backend is ollama
        assert "no complexity" in reason

    def test_routes_greeting_to_ollama(self) -> None:
        anthropic = MagicMock()
        ollama = MagicMock()
        hybrid = HybridBackend(anthropic_backend=anthropic, ollama_backend=ollama)

        backend, reason = hybrid._classify_query(_make_messages("good morning"))
        assert backend is ollama

    def test_routes_long_question_to_anthropic(self) -> None:
        anthropic = MagicMock()
        ollama = MagicMock()
        hybrid = HybridBackend(anthropic_backend=anthropic, ollama_backend=ollama)

        long_q = "can you help me with " + " ".join(["word"] * 45) + "?"
        backend, reason = hybrid._classify_query(_make_messages(long_q))
        assert backend is anthropic
        assert "long question" in reason

    def test_no_anthropic_always_ollama(self) -> None:
        ollama = MagicMock()
        hybrid = HybridBackend(anthropic_backend=None, ollama_backend=ollama)

        backend, reason = hybrid._classify_query(_make_messages("analyze the architecture"))
        assert backend is ollama
        assert "unavailable" in reason

    def test_uses_last_user_message(self) -> None:
        anthropic = MagicMock()
        ollama = MagicMock()
        hybrid = HybridBackend(anthropic_backend=anthropic, ollama_backend=ollama)

        messages = [
            {"role": "user", "content": "analyze my portfolio"},
            {"role": "assistant", "content": "sure thing"},
            {"role": "user", "content": "thanks"},
        ]
        backend, reason = hybrid._classify_query(messages)
        assert backend is ollama  # "thanks" has no complex keywords

    def test_keyword_tools_routes_to_anthropic(self) -> None:
        anthropic = MagicMock()
        ollama = MagicMock()
        hybrid = HybridBackend(anthropic_backend=anthropic, ollama_backend=ollama)

        backend, _ = hybrid._classify_query(_make_messages("what tools do I have?"))
        assert backend is anthropic


class TestHybridGenerateResponse:
    """Test generate_response routing and fallback."""

    def test_routes_complex_to_anthropic(self) -> None:
        anthropic = MagicMock()
        anthropic.generate_response = AsyncMock(return_value="anthropic response")
        ollama = MagicMock()
        ollama.generate_response = AsyncMock(return_value="ollama response")

        hybrid = HybridBackend(anthropic_backend=anthropic, ollama_backend=ollama)
        result = _run(hybrid.generate_response(_make_messages("analyze yourself")))

        assert result == "anthropic response"
        anthropic.generate_response.assert_awaited_once()
        ollama.generate_response.assert_not_awaited()

    def test_routes_casual_to_ollama(self) -> None:
        anthropic = MagicMock()
        anthropic.generate_response = AsyncMock(return_value="anthropic response")
        ollama = MagicMock()
        ollama.generate_response = AsyncMock(return_value="ollama response")

        hybrid = HybridBackend(anthropic_backend=anthropic, ollama_backend=ollama)
        result = _run(hybrid.generate_response(_make_messages("hello")))

        assert result == "ollama response"
        ollama.generate_response.assert_awaited_once()
        anthropic.generate_response.assert_not_awaited()

    def test_falls_back_on_anthropic_failure(self) -> None:
        anthropic = MagicMock()
        anthropic.generate_response = AsyncMock(side_effect=RuntimeError("API down"))
        ollama = MagicMock()
        ollama.generate_response = AsyncMock(return_value="ollama fallback")

        hybrid = HybridBackend(anthropic_backend=anthropic, ollama_backend=ollama)
        result = _run(hybrid.generate_response(_make_messages("analyze yourself")))

        assert result == "ollama fallback"
        anthropic.generate_response.assert_awaited_once()
        ollama.generate_response.assert_awaited_once()

    def test_no_anthropic_uses_ollama(self) -> None:
        ollama = MagicMock()
        ollama.generate_response = AsyncMock(return_value="ollama only")

        hybrid = HybridBackend(anthropic_backend=None, ollama_backend=ollama)
        result = _run(hybrid.generate_response(_make_messages("analyze the architecture")))

        assert result == "ollama only"
        ollama.generate_response.assert_awaited_once()


class TestHybridGenerateStructured:
    """Test generate_structured routing and fallback."""

    def test_routes_complex_to_anthropic(self) -> None:
        resp = CognitiveResponse(text="structured anthropic")
        anthropic = MagicMock()
        anthropic.generate_structured = AsyncMock(return_value=resp)
        ollama = MagicMock()
        ollama.generate_structured = AsyncMock()

        hybrid = HybridBackend(anthropic_backend=anthropic, ollama_backend=ollama)
        result = _run(hybrid.generate_structured(_make_messages("evaluate my tools")))

        assert result.text == "structured anthropic"
        anthropic.generate_structured.assert_awaited_once()

    def test_falls_back_on_anthropic_failure(self) -> None:
        fallback = CognitiveResponse(text="ollama fallback")
        anthropic = MagicMock()
        anthropic.generate_structured = AsyncMock(side_effect=RuntimeError("fail"))
        ollama = MagicMock()
        ollama.generate_structured = AsyncMock(return_value=fallback)

        hybrid = HybridBackend(anthropic_backend=anthropic, ollama_backend=ollama)
        result = _run(hybrid.generate_structured(_make_messages("evaluate my tools")))

        assert result.text == "ollama fallback"
        ollama.generate_structured.assert_awaited_once()


class TestRuntimeHybridWiring:
    """Test that runtime creates HybridBackend correctly."""

    def test_runtime_creates_hybrid_backend(self) -> None:
        from animus_bootstrap.runtime import AnimusRuntime

        rt = AnimusRuntime.__new__(AnimusRuntime)
        rt._started = False

        mock_config = MagicMock()
        mock_config.gateway.default_backend = "hybrid"
        mock_config.api.anthropic_key = "sk-ant-" + "x" * 80
        mock_config.ollama.host = "localhost"
        mock_config.ollama.port = 11434
        mock_config.ollama.model = "qwen2.5:14b"
        mock_config.ollama.code_model = "deepseek-coder-v2"
        rt._config = mock_config

        backend = rt._create_cognitive_backend()
        assert isinstance(backend, HybridBackend)
        assert backend._anthropic is not None
        assert backend._ollama is not None

    def test_runtime_hybrid_no_key_still_works(self) -> None:
        from animus_bootstrap.runtime import AnimusRuntime

        rt = AnimusRuntime.__new__(AnimusRuntime)
        rt._started = False

        mock_config = MagicMock()
        mock_config.gateway.default_backend = "hybrid"
        mock_config.api.anthropic_key = ""
        mock_config.ollama.host = "localhost"
        mock_config.ollama.port = 11434
        mock_config.ollama.model = "qwen2.5:14b"
        mock_config.ollama.code_model = ""
        rt._config = mock_config

        backend = rt._create_cognitive_backend()
        assert isinstance(backend, HybridBackend)
        assert backend._anthropic is None  # No key = no Anthropic
