"""Tests for OpenAI API client wrapper."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def mock_settings():
    with patch("animus_forge.api_clients.openai_client.get_settings") as ms:
        ms.return_value.openai_api_key = "sk-test"
        yield ms


class TestOpenAIClient:
    """Tests for OpenAI client sync methods."""

    @patch("animus_forge.api_clients.openai_client.OpenAI")
    def test_init(self, mock_openai_cls, mock_settings):
        from animus_forge.api_clients.openai_client import OpenAIClient

        client = OpenAIClient()
        mock_openai_cls.assert_called_once_with(api_key="sk-test")
        assert client._async_client is None

    @patch("animus_forge.api_clients.openai_client.AsyncOpenAI")
    @patch("animus_forge.api_clients.openai_client.OpenAI")
    def test_async_client_lazy_load(self, mock_openai, mock_async_openai, mock_settings):
        from animus_forge.api_clients.openai_client import OpenAIClient

        client = OpenAIClient()
        assert client._async_client is None

        _ = client.async_client
        mock_async_openai.assert_called_once_with(api_key="sk-test")

        # Second access reuses
        _ = client.async_client
        assert mock_async_openai.call_count == 1

    @patch("animus_forge.api_clients.openai_client.OpenAI")
    def test_generate_completion_with_system(self, mock_openai_cls, mock_settings):
        from animus_forge.api_clients.openai_client import OpenAIClient

        client = OpenAIClient()

        with patch.object(client, "_call_api", return_value="Hello!") as mock_call:
            result = client.generate_completion("Say hi", system_prompt="Be friendly")

            assert result == "Hello!"
            args = mock_call.call_args
            messages = args[0][1]
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

    @patch("animus_forge.api_clients.openai_client.OpenAI")
    def test_generate_completion_no_system(self, mock_openai_cls, mock_settings):
        from animus_forge.api_clients.openai_client import OpenAIClient

        client = OpenAIClient()

        with patch.object(client, "_call_api", return_value="Result") as mock_call:
            result = client.generate_completion("Do something")

            assert result == "Result"
            messages = mock_call.call_args[0][1]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"

    @patch("animus_forge.api_clients.openai_client.OpenAI")
    def test_summarize_text(self, mock_openai_cls, mock_settings):
        from animus_forge.api_clients.openai_client import OpenAIClient

        client = OpenAIClient()

        with patch.object(client, "_call_api", return_value="Summary here"):
            result = client.summarize_text("Long text", max_length=100)
            assert result == "Summary here"

    @patch("animus_forge.api_clients.openai_client.OpenAI")
    def test_generate_sop(self, mock_openai_cls, mock_settings):
        from animus_forge.api_clients.openai_client import OpenAIClient

        client = OpenAIClient()

        with patch.object(client, "_call_api", return_value="SOP doc"):
            result = client.generate_sop("onboarding process")
            assert result == "SOP doc"


class TestOpenAIClientAsync:
    """Tests for OpenAI client async methods."""

    @pytest.mark.asyncio
    @patch("animus_forge.api_clients.openai_client.OpenAI")
    async def test_generate_completion_async(self, mock_openai, mock_settings):
        from animus_forge.api_clients.openai_client import OpenAIClient

        client = OpenAIClient()

        with patch.object(
            client,
            "_call_api_async",
            new_callable=AsyncMock,
            return_value="Async result",
        ) as mock_call:
            result = await client.generate_completion_async("Prompt", system_prompt="sys")
            assert result == "Async result"
            messages = mock_call.call_args[0][1]
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    @patch("animus_forge.api_clients.openai_client.OpenAI")
    async def test_generate_completion_async_no_system(self, mock_openai, mock_settings):
        from animus_forge.api_clients.openai_client import OpenAIClient

        client = OpenAIClient()

        with patch.object(
            client, "_call_api_async", new_callable=AsyncMock, return_value="No sys"
        ) as mock_call:
            result = await client.generate_completion_async("Prompt only")
            assert result == "No sys"
            messages = mock_call.call_args[0][1]
            assert len(messages) == 1

    @pytest.mark.asyncio
    @patch("animus_forge.api_clients.openai_client.OpenAI")
    async def test_summarize_text_async(self, mock_openai, mock_settings):
        from animus_forge.api_clients.openai_client import OpenAIClient

        client = OpenAIClient()

        with patch.object(
            client,
            "_call_api_async",
            new_callable=AsyncMock,
            return_value="Async summary",
        ):
            result = await client.summarize_text_async("Long text")
            assert result == "Async summary"

    @pytest.mark.asyncio
    @patch("animus_forge.api_clients.openai_client.OpenAI")
    async def test_generate_sop_async(self, mock_openai, mock_settings):
        from animus_forge.api_clients.openai_client import OpenAIClient

        client = OpenAIClient()

        with patch.object(
            client, "_call_api_async", new_callable=AsyncMock, return_value="Async SOP"
        ):
            result = await client.generate_sop_async("deployment process")
            assert result == "Async SOP"
