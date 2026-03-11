"""Integration tests for ForgeBackend cognitive calls in the Bootstrap runtime."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from animus_bootstrap.config.schema import (
    AnimusConfig,
    AnimusSection,
    ApiSection,
    ForgeSection,
    GatewaySection,
    IntelligenceSection,
    ProactiveSection,
    ServicesSection,
)
from animus_bootstrap.gateway.cognitive import ForgeBackend
from animus_bootstrap.gateway.cognitive_types import CognitiveResponse
from animus_bootstrap.runtime import AnimusRuntime

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_HTTPX_CLIENT = "animus_bootstrap.gateway.cognitive.httpx.AsyncClient"
_FORGE_URL = "http://forge.local:9999/api/v1/chat"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_config(
    *,
    backend: str = "forge",
    forge_enabled: bool = True,
    forge_host: str = "forge.local",
    forge_port: int = 9999,
    forge_api_key: str = "fk-test-key",
    data_dir: str = "/tmp/animus-forge-test",
    intelligence_enabled: bool = False,
    proactive_enabled: bool = False,
) -> AnimusConfig:
    """Build an AnimusConfig tuned for Forge integration tests."""
    return AnimusConfig(
        animus=AnimusSection(data_dir=data_dir),
        api=ApiSection(anthropic_key=""),
        forge=ForgeSection(
            enabled=forge_enabled,
            host=forge_host,
            port=forge_port,
            api_key=forge_api_key,
        ),
        gateway=GatewaySection(
            default_backend=backend,
            system_prompt="You are Animus.",
        ),
        intelligence=IntelligenceSection(
            enabled=intelligence_enabled,
            memory_backend="sqlite",
            memory_db_path=f"{data_dir}/intelligence.db",
        ),
        proactive=ProactiveSection(enabled=proactive_enabled),
        services=ServicesSection(port=7700),
    )


def _mock_httpx_client(
    *,
    response_json: dict | None = None,
    status_code: int = 200,
    raise_for_status: Exception | None = None,
    post_side_effect: Exception | None = None,
) -> AsyncMock:
    """Build a mock httpx.AsyncClient for use as a context manager."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    if post_side_effect is not None:
        mock_client.post.side_effect = post_side_effect
        return mock_client

    mock_response = MagicMock()
    mock_response.status_code = status_code
    if raise_for_status is not None:
        mock_response.raise_for_status.side_effect = raise_for_status
    else:
        mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = response_json or {}
    mock_client.post.return_value = mock_response
    return mock_client


# ------------------------------------------------------------------
# 1. ForgeBackend initializes with host/port/api_key from config
# ------------------------------------------------------------------


class TestForgeBackendInit:
    """Verify ForgeBackend stores config values correctly."""

    def test_default_values(self) -> None:
        """Defaults produce a localhost:8000 base URL."""
        backend = ForgeBackend()
        assert backend._base_url == "http://localhost:8000"
        assert backend._api_key == ""

    def test_custom_host_port(self) -> None:
        """Custom host/port are reflected in _base_url."""
        backend = ForgeBackend(host="forge.local", port=9999, api_key="fk-abc")
        assert backend._base_url == "http://forge.local:9999"
        assert backend._api_key == "fk-abc"

    def test_config_values_wired(self) -> None:
        """Values from ForgeSection are passed through correctly."""
        cfg = _make_config(
            forge_host="10.0.0.5",
            forge_port=4321,
            forge_api_key="fk-xyz",
        )
        backend = ForgeBackend(
            host=cfg.forge.host,
            port=cfg.forge.port,
            api_key=cfg.forge.api_key,
        )
        assert backend._base_url == "http://10.0.0.5:4321"
        assert backend._api_key == "fk-xyz"


# ------------------------------------------------------------------
# 2. ForgeBackend.generate_response() — happy path
# ------------------------------------------------------------------


class TestForgeBackendGenerate:
    """Verify generate_response sends correct request and parses response."""

    @pytest.mark.asyncio
    async def test_sends_correct_request(self) -> None:
        """POSTs to /api/v1/chat with correct payload and headers."""
        client = _mock_httpx_client(
            response_json={"response": "Hello from Forge"},
        )

        with patch(_HTTPX_CLIENT, return_value=client):
            backend = ForgeBackend(host="forge.local", port=9999, api_key="fk-test")
            messages = [{"role": "user", "content": "Hello"}]
            result = await backend.generate_response(
                messages, system_prompt="Be helpful", max_tokens=512
            )

        assert result == "Hello from Forge"

        # Verify the POST call
        client.post.assert_called_once()
        call_args = client.post.call_args
        assert call_args[0][0] == _FORGE_URL

        payload = call_args[1]["json"]
        assert payload["messages"] == messages
        assert payload["max_tokens"] == 512
        assert payload["system_prompt"] == "Be helpful"

        headers = call_args[1]["headers"]
        assert headers["content-type"] == "application/json"
        assert headers["authorization"] == "Bearer fk-test"

    @pytest.mark.asyncio
    async def test_returns_text_field_fallback(self) -> None:
        """Falls back to 'text' key when 'response' is absent."""
        client = _mock_httpx_client(
            response_json={"text": "Fallback text"},
        )

        with patch(_HTTPX_CLIENT, return_value=client):
            backend = ForgeBackend(host="localhost", port=8000)
            result = await backend.generate_response(
                [{"role": "user", "content": "Hi"}],
            )

        assert result == "Fallback text"

    @pytest.mark.asyncio
    async def test_no_auth_header_without_api_key(self) -> None:
        """Omits authorization header when api_key is empty."""
        client = _mock_httpx_client(
            response_json={"response": "ok"},
        )

        with patch(_HTTPX_CLIENT, return_value=client):
            backend = ForgeBackend(host="localhost", port=8000, api_key="")
            await backend.generate_response(
                [{"role": "user", "content": "Hi"}],
            )

        headers = client.post.call_args[1]["headers"]
        assert "authorization" not in headers

    @pytest.mark.asyncio
    async def test_no_system_prompt_omitted(self) -> None:
        """Omits system_prompt from payload when None."""
        client = _mock_httpx_client(
            response_json={"response": "ok"},
        )

        with patch(_HTTPX_CLIENT, return_value=client):
            backend = ForgeBackend(host="localhost", port=8000)
            await backend.generate_response(
                [{"role": "user", "content": "Hi"}],
                system_prompt=None,
            )

        payload = client.post.call_args[1]["json"]
        assert "system_prompt" not in payload

    @pytest.mark.asyncio
    async def test_generate_structured_wraps_response(self) -> None:
        """generate_structured returns CognitiveResponse wrapping text."""
        client = _mock_httpx_client(
            response_json={"response": "Structured reply"},
        )

        with patch(_HTTPX_CLIENT, return_value=client):
            backend = ForgeBackend(host="localhost", port=8000, api_key="fk-k")
            result = await backend.generate_structured(
                [{"role": "user", "content": "Hi"}],
                system_prompt="test",
            )

        assert isinstance(result, CognitiveResponse)
        assert result.text == "Structured reply"
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"


# ------------------------------------------------------------------
# 3. ForgeBackend.generate_response() handles HTTP errors
# ------------------------------------------------------------------


class TestForgeBackendHTTPErrors:
    """Verify graceful handling of HTTP-level failures."""

    @pytest.mark.asyncio
    async def test_http_500_raises(self) -> None:
        """500 from Forge raises HTTPStatusError."""
        client = _mock_httpx_client(
            status_code=500,
            raise_for_status=httpx.HTTPStatusError(
                "Internal Server Error",
                request=httpx.Request("POST", _FORGE_URL),
                response=httpx.Response(500),
            ),
        )

        with patch(_HTTPX_CLIENT, return_value=client):
            backend = ForgeBackend(host="forge.local", port=9999)
            with pytest.raises(httpx.HTTPStatusError):
                await backend.generate_response(
                    [{"role": "user", "content": "Hi"}],
                )

    @pytest.mark.asyncio
    async def test_http_401_raises(self) -> None:
        """401 Unauthorized from Forge raises HTTPStatusError."""
        client = _mock_httpx_client(
            status_code=401,
            raise_for_status=httpx.HTTPStatusError(
                "Unauthorized",
                request=httpx.Request("POST", _FORGE_URL),
                response=httpx.Response(401),
            ),
        )

        with patch(_HTTPX_CLIENT, return_value=client):
            backend = ForgeBackend(host="forge.local", port=9999, api_key="bad-key")
            with pytest.raises(httpx.HTTPStatusError):
                await backend.generate_response(
                    [{"role": "user", "content": "Hi"}],
                )

    @pytest.mark.asyncio
    async def test_http_429_raises(self) -> None:
        """429 rate limit from Forge raises HTTPStatusError."""
        client = _mock_httpx_client(
            status_code=429,
            raise_for_status=httpx.HTTPStatusError(
                "Too Many Requests",
                request=httpx.Request("POST", _FORGE_URL),
                response=httpx.Response(429),
            ),
        )

        with patch(_HTTPX_CLIENT, return_value=client):
            backend = ForgeBackend(host="forge.local", port=9999)
            with pytest.raises(httpx.HTTPStatusError):
                await backend.generate_response(
                    [{"role": "user", "content": "Hi"}],
                )


# ------------------------------------------------------------------
# 4. ForgeBackend.generate_response() handles connection refused
# ------------------------------------------------------------------


class TestForgeBackendConnectionErrors:
    """Verify graceful handling of network-level failures."""

    @pytest.mark.asyncio
    async def test_connection_refused(self) -> None:
        """ConnectError (connection refused) propagates cleanly."""
        client = _mock_httpx_client(
            post_side_effect=httpx.ConnectError("Connection refused"),
        )

        with patch(_HTTPX_CLIENT, return_value=client):
            backend = ForgeBackend(host="localhost", port=9999)
            with pytest.raises(httpx.ConnectError):
                await backend.generate_response(
                    [{"role": "user", "content": "Hi"}],
                )

    @pytest.mark.asyncio
    async def test_timeout_error(self) -> None:
        """ReadTimeout propagates cleanly."""
        client = _mock_httpx_client(
            post_side_effect=httpx.ReadTimeout("Read timed out"),
        )

        with patch(_HTTPX_CLIENT, return_value=client):
            backend = ForgeBackend(host="localhost", port=9999)
            with pytest.raises(httpx.ReadTimeout):
                await backend.generate_response(
                    [{"role": "user", "content": "Hi"}],
                )

    @pytest.mark.asyncio
    async def test_structured_propagates_connection_error(self) -> None:
        """generate_structured also propagates connection errors."""
        client = _mock_httpx_client(
            post_side_effect=httpx.ConnectError("Connection refused"),
        )

        with patch(_HTTPX_CLIENT, return_value=client):
            backend = ForgeBackend(host="localhost", port=9999)
            with pytest.raises(httpx.ConnectError):
                await backend.generate_structured(
                    [{"role": "user", "content": "Hi"}],
                )


# ------------------------------------------------------------------
# 5. Runtime creates ForgeBackend when forge is enabled
# ------------------------------------------------------------------


class TestRuntimeForgeBackendCreation:
    """Verify Runtime._create_cognitive_backend() wires ForgeBackend."""

    def test_creates_forge_backend_when_enabled(self, tmp_path: Path) -> None:
        """Runtime creates ForgeBackend when forge is enabled."""
        cfg = _make_config(
            backend="forge",
            forge_enabled=True,
            forge_host="forge.example.com",
            forge_port=5555,
            forge_api_key="fk-runtime-test",
            data_dir=str(tmp_path / "data"),
        )
        rt = AnimusRuntime(config=cfg)
        backend = rt._create_cognitive_backend()

        assert isinstance(backend, ForgeBackend)
        assert backend._base_url == "http://forge.example.com:5555"
        assert backend._api_key == "fk-runtime-test"

    def test_cognitive_backend_set_on_start(self, tmp_path: Path) -> None:
        """After start(), cognitive_backend is a ForgeBackend."""
        cfg = _make_config(
            backend="forge",
            forge_enabled=True,
            data_dir=str(tmp_path / "data"),
        )
        rt = AnimusRuntime(config=cfg)

        async def _run() -> None:
            await rt.start()
            try:
                assert isinstance(rt.cognitive_backend, ForgeBackend)
            finally:
                await rt.stop()

        asyncio.run(_run())


# ------------------------------------------------------------------
# 6. Runtime falls back when forge is disabled
# ------------------------------------------------------------------


class TestRuntimeForgeFallback:
    """Verify Runtime falls back to Ollama when forge is unavailable."""

    def test_fallback_when_forge_disabled(self, tmp_path: Path) -> None:
        """backend=forge but forge.enabled=False falls back to Ollama."""
        from animus_bootstrap.gateway.cognitive import OllamaBackend

        cfg = _make_config(
            backend="forge",
            forge_enabled=False,
            data_dir=str(tmp_path / "data"),
        )
        rt = AnimusRuntime(config=cfg)
        backend = rt._create_cognitive_backend()

        assert not isinstance(backend, ForgeBackend)
        assert isinstance(backend, OllamaBackend)

    def test_fallback_when_backend_not_forge(self, tmp_path: Path) -> None:
        """backend=ollama ignores forge settings entirely."""
        from animus_bootstrap.gateway.cognitive import OllamaBackend

        cfg = _make_config(
            backend="ollama",
            forge_enabled=True,
            data_dir=str(tmp_path / "data"),
        )
        rt = AnimusRuntime(config=cfg)
        backend = rt._create_cognitive_backend()

        assert isinstance(backend, OllamaBackend)
        assert not isinstance(backend, ForgeBackend)

    def test_fallback_logs_warning(self, tmp_path: Path) -> None:
        """Falling back to Ollama emits a warning log."""
        cfg = _make_config(
            backend="forge",
            forge_enabled=False,
            data_dir=str(tmp_path / "data"),
        )
        rt = AnimusRuntime(config=cfg)

        with patch("animus_bootstrap.runtime.logger") as mock_log:
            rt._create_cognitive_backend()
            mock_log.warning.assert_called_once()
            warn_args = mock_log.warning.call_args[0]
            assert "forge" in warn_args[1].lower()
