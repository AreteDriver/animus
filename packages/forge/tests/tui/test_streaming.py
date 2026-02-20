"""Tests for the unified streaming adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_forge.providers.base import (
    CompletionResponse,
    ProviderError,
    ProviderType,
    StreamChunk,
)
from animus_forge.tui.streaming import (
    StreamResult,
    _stream_fallback,
    _stream_ollama,
    stream_completion,
)


def _make_provider(provider_type: ProviderType, initialized: bool = True):
    """Create a mock provider with the given type."""
    provider = MagicMock()
    provider.provider_type = provider_type
    provider._initialized = initialized
    provider.default_model = "test-model"
    provider.initialize = MagicMock()
    return provider


class TestStreamResult:
    def test_defaults(self):
        r = StreamResult()
        assert r.input_tokens == 0
        assert r.output_tokens == 0

    def test_assignment(self):
        r = StreamResult()
        r.input_tokens = 100
        r.output_tokens = 50
        assert r.input_tokens == 100
        assert r.output_tokens == 50


class TestStreamCompletion:
    @pytest.mark.asyncio
    async def test_routes_to_anthropic(self):
        provider = _make_provider(ProviderType.ANTHROPIC)
        messages = [{"role": "user", "content": "hi"}]

        with patch("animus_forge.tui.streaming._stream_anthropic") as mock_stream:

            async def gen(*args, **kwargs):
                yield "hello"

            mock_stream.return_value = gen()

            chunks = []
            async for chunk in stream_completion(provider, messages):
                chunks.append(chunk)
            assert chunks == ["hello"]

    @pytest.mark.asyncio
    async def test_routes_to_openai(self):
        provider = _make_provider(ProviderType.OPENAI)
        messages = [{"role": "user", "content": "hi"}]

        with patch("animus_forge.tui.streaming._stream_openai") as mock_stream:

            async def gen(*args, **kwargs):
                yield "world"

            mock_stream.return_value = gen()

            chunks = []
            async for chunk in stream_completion(provider, messages):
                chunks.append(chunk)
            assert chunks == ["world"]

    @pytest.mark.asyncio
    async def test_routes_to_ollama(self):
        provider = _make_provider(ProviderType.OLLAMA)
        messages = [{"role": "user", "content": "hi"}]

        with patch("animus_forge.tui.streaming._stream_ollama") as mock_stream:

            async def gen(*args, **kwargs):
                yield "local"

            mock_stream.return_value = gen()

            chunks = []
            async for chunk in stream_completion(provider, messages):
                chunks.append(chunk)
            assert chunks == ["local"]

    @pytest.mark.asyncio
    async def test_routes_to_fallback_for_unknown_type(self):
        provider = _make_provider(ProviderType.VERTEX)
        messages = [{"role": "user", "content": "hi"}]

        with patch("animus_forge.tui.streaming._stream_fallback") as mock_stream:

            async def gen(*args, **kwargs):
                yield "fallback"

            mock_stream.return_value = gen()

            chunks = []
            async for chunk in stream_completion(provider, messages):
                chunks.append(chunk)
            assert chunks == ["fallback"]

    @pytest.mark.asyncio
    async def test_extracts_system_prompt_from_messages(self):
        provider = _make_provider(ProviderType.ANTHROPIC)
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hi"},
        ]

        with patch("animus_forge.tui.streaming._stream_anthropic") as mock_stream:

            async def gen(*args, **kwargs):
                yield "ok"

            mock_stream.return_value = gen()

            chunks = []
            async for chunk in stream_completion(provider, messages):
                chunks.append(chunk)

            # System messages should be extracted and filtered
            call_args = mock_stream.call_args
            filtered_messages = call_args[0][1]  # second positional arg
            assert all(m["role"] != "system" for m in filtered_messages)

    @pytest.mark.asyncio
    async def test_explicit_system_prompt_takes_precedence(self):
        provider = _make_provider(ProviderType.ANTHROPIC)
        messages = [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "hi"},
        ]

        with patch("animus_forge.tui.streaming._stream_anthropic") as mock_stream:

            async def gen(*args, **kwargs):
                yield "ok"

            mock_stream.return_value = gen()

            chunks = []
            async for chunk in stream_completion(provider, messages, system_prompt="explicit"):
                chunks.append(chunk)

            call_args = mock_stream.call_args
            system_prompt = call_args[0][2]  # third positional arg
            assert system_prompt == "explicit"

    @pytest.mark.asyncio
    async def test_populates_result(self):
        provider = _make_provider(ProviderType.ANTHROPIC)
        messages = [{"role": "user", "content": "hi"}]
        result = StreamResult()

        with patch("animus_forge.tui.streaming._stream_anthropic") as mock_stream:

            async def gen(prov, msgs, sys, model, res):
                res.input_tokens = 10
                res.output_tokens = 20
                yield "done"

            mock_stream.side_effect = gen

            async for _ in stream_completion(provider, messages, result=result):
                pass

            assert result.input_tokens == 10
            assert result.output_tokens == 20

    @pytest.mark.asyncio
    async def test_falls_back_on_generic_exception(self):
        provider = _make_provider(ProviderType.ANTHROPIC)
        messages = [{"role": "user", "content": "hi"}]

        with (
            patch(
                "animus_forge.tui.streaming._stream_anthropic",
                side_effect=RuntimeError("oops"),
            ),
            patch("animus_forge.tui.streaming._stream_fallback") as mock_fallback,
        ):

            async def gen(*args, **kwargs):
                yield "recovered"

            mock_fallback.return_value = gen()

            chunks = []
            async for chunk in stream_completion(provider, messages):
                chunks.append(chunk)
            assert chunks == ["recovered"]

    @pytest.mark.asyncio
    async def test_does_not_catch_provider_error(self):
        provider = _make_provider(ProviderType.ANTHROPIC)
        messages = [{"role": "user", "content": "hi"}]

        with patch(
            "animus_forge.tui.streaming._stream_anthropic",
            side_effect=ProviderError("bad key"),
        ):
            with pytest.raises(ProviderError, match="bad key"):
                async for _ in stream_completion(provider, messages):
                    pass

    @pytest.mark.asyncio
    async def test_initializes_provider_if_needed(self):
        provider = _make_provider(ProviderType.VERTEX, initialized=False)
        messages = [{"role": "user", "content": "hi"}]

        with patch("animus_forge.tui.streaming._stream_fallback") as mock_stream:

            async def gen(*args, **kwargs):
                yield "ok"

            mock_stream.return_value = gen()

            async for _ in stream_completion(provider, messages):
                pass

            provider.initialize.assert_called_once()


class TestStreamFallback:
    @pytest.mark.asyncio
    async def test_yields_single_chunk(self):
        provider = MagicMock()
        provider.complete_async = AsyncMock(
            return_value=CompletionResponse(
                content="full response",
                model="test",
                provider="test",
                input_tokens=5,
                output_tokens=10,
            )
        )
        result = StreamResult()
        chunks = []
        async for chunk in _stream_fallback(
            provider,
            [{"role": "user", "content": "hi"}],
            "sys prompt",
            None,
            result,
        ):
            chunks.append(chunk)
        assert chunks == ["full response"]


class TestStreamOllama:
    @pytest.mark.asyncio
    async def test_yields_content_from_chunks(self):
        provider = MagicMock()

        async def mock_stream(request):
            yield StreamChunk(content="hello ", model="llama", provider="ollama")
            yield StreamChunk(content="world", model="llama", provider="ollama", is_final=True)

        provider.complete_stream_async = mock_stream
        result = StreamResult()

        chunks = []
        async for chunk in _stream_ollama(
            provider,
            [{"role": "user", "content": "hi"}],
            None,
            None,
            result,
        ):
            chunks.append(chunk)
        assert chunks == ["hello ", "world"]
