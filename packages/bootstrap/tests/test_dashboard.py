"""Tests for the Animus Bootstrap dashboard application."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from animus_bootstrap.config.schema import AnimusConfig
from animus_bootstrap.dashboard.app import app

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_mock_config() -> AnimusConfig:
    """Return a default AnimusConfig instance for mocking."""
    return AnimusConfig()


def _mock_config_manager() -> MagicMock:
    """Build a mock ConfigManager whose .load() returns defaults."""
    mgr = MagicMock()
    mgr.load.return_value = _make_mock_config()
    return mgr


def _mock_installer(running: bool = False) -> MagicMock:
    """Build a mock AnimusInstaller."""
    inst = MagicMock()
    inst.is_service_running.return_value = running
    return inst


def _mock_httpx_async_client(status_code: int = 200) -> MagicMock:
    """Build a mock for httpx.AsyncClient that works as an async context manager.

    The `_check_forge_status` function uses `async with httpx.AsyncClient(...) as client:`
    so we need __aenter__/__aexit__ to be proper async callables.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = status_code

    mock_client_instance = AsyncMock()
    mock_client_instance.get.return_value = mock_resp

    # AsyncClient(...) returns an object used as `async with ... as client:`
    mock_cls = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__.return_value = mock_client_instance
    ctx.__aexit__.return_value = False
    mock_cls.return_value = ctx

    return mock_cls


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """TestClient with all routers' external deps patched."""
    return TestClient(app)


# ------------------------------------------------------------------
# Home page — GET /
# ------------------------------------------------------------------


class TestHomePage:
    """Tests for the home page router."""

    @patch("animus_bootstrap.dashboard.routers.home.httpx.AsyncClient")
    @patch("animus_bootstrap.dashboard.routers.home.ConfigManager")
    def test_home_returns_200(
        self,
        mock_cm_cls: MagicMock,
        mock_async_client: MagicMock,
        client: TestClient,
    ) -> None:
        """GET / returns 200 with mocked deps."""
        mock_cm_cls.return_value = _mock_config_manager()

        fixed = _mock_httpx_async_client(status_code=200)
        mock_async_client.return_value = fixed.return_value

        resp = client.get("/")
        assert resp.status_code == 200

    @patch("animus_bootstrap.dashboard.routers.home.httpx.AsyncClient")
    @patch("animus_bootstrap.dashboard.routers.home.ConfigManager")
    def test_home_shows_animus(
        self,
        mock_cm_cls: MagicMock,
        mock_async_client: MagicMock,
        client: TestClient,
    ) -> None:
        """GET / response body contains the Animus string."""
        mock_cm_cls.return_value = _mock_config_manager()

        fixed = _mock_httpx_async_client(status_code=200)
        mock_async_client.return_value = fixed.return_value

        resp = client.get("/")
        assert resp.status_code == 200
        assert "Animus" in resp.text

    @patch("animus_bootstrap.dashboard.routers.home.httpx.AsyncClient")
    @patch("animus_bootstrap.dashboard.routers.home.ConfigManager")
    def test_home_runtime_stopped(
        self,
        mock_cm_cls: MagicMock,
        mock_async_client: MagicMock,
        client: TestClient,
    ) -> None:
        """GET / shows 'Stopped' when runtime is not started."""
        mock_cm_cls.return_value = _mock_config_manager()

        fixed = _mock_httpx_async_client(status_code=200)
        mock_async_client.return_value = fixed.return_value

        resp = client.get("/")
        assert "Stopped" in resp.text

    @patch("animus_bootstrap.dashboard.routers.home.httpx.AsyncClient")
    @patch("animus_bootstrap.dashboard.routers.home.ConfigManager")
    def test_home_forge_disconnected(
        self,
        mock_cm_cls: MagicMock,
        mock_async_client: MagicMock,
        client: TestClient,
    ) -> None:
        """GET / shows 'Disconnected' when forge health check fails."""
        mock_cm_cls.return_value = _mock_config_manager()

        fixed = _mock_httpx_async_client(status_code=500)
        mock_async_client.return_value = fixed.return_value

        resp = client.get("/")
        assert "Disconnected" in resp.text

    @patch("animus_bootstrap.dashboard.routers.home.httpx.AsyncClient")
    @patch("animus_bootstrap.dashboard.routers.home.ConfigManager")
    def test_home_forge_connected(
        self,
        mock_cm_cls: MagicMock,
        mock_async_client: MagicMock,
        client: TestClient,
    ) -> None:
        """GET / shows 'Connected' when forge health check succeeds."""
        mock_cm_cls.return_value = _mock_config_manager()

        fixed = _mock_httpx_async_client(status_code=200)
        mock_async_client.return_value = fixed.return_value

        resp = client.get("/")
        assert "Connected" in resp.text


# ------------------------------------------------------------------
# Config page — GET /config and POST /config
# ------------------------------------------------------------------


class TestConfigPage:
    """Tests for the configuration editor router."""

    @patch("animus_bootstrap.dashboard.routers.config.ConfigManager")
    def test_config_page_returns_200(
        self,
        mock_cm_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """GET /config returns 200."""
        mock_cm_cls.return_value = _mock_config_manager()
        resp = client.get("/config")
        assert resp.status_code == 200

    @patch("animus_bootstrap.dashboard.routers.config.ConfigManager")
    def test_config_page_masks_keys(
        self,
        mock_cm_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """GET /config masks API keys in the rendered page."""
        cfg = _make_mock_config()
        cfg.api.anthropic_key = "sk-ant-secret-key-12345678"
        mgr = MagicMock()
        mgr.load.return_value = cfg
        mock_cm_cls.return_value = mgr

        resp = client.get("/config")
        assert resp.status_code == 200
        # The full key must NOT appear; only last 4 chars visible
        assert "sk-ant-secret-key-12345678" not in resp.text

    @patch("animus_bootstrap.dashboard.routers.config.ConfigManager")
    def test_config_post_saves(
        self,
        mock_cm_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """POST /config with form data calls save and redirects."""
        mgr = _mock_config_manager()
        mock_cm_cls.return_value = mgr

        resp = client.post(
            "/config",
            data={
                "identity_name": "TestUser",
                "identity_timezone": "UTC",
                "identity_locale": "en_US",
                "forge_host": "localhost",
                "forge_port": "8000",
                "memory_backend": "sqlite",
                "memory_path": "~/.local/share/animus/memory.db",
                "max_context_tokens": "100000",
                "services_port": "7700",
                "log_level": "info",
                "data_dir": "~/.local/share/animus",
            },
            follow_redirects=False,
        )
        # POST should redirect to /config?saved=1 with 303
        assert resp.status_code == 303
        assert "/config?saved=1" in resp.headers["location"]
        mgr.save.assert_called_once()

    @patch("animus_bootstrap.dashboard.routers.config.ConfigManager")
    def test_config_post_does_not_overwrite_masked_keys(
        self,
        mock_cm_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """POST /config with masked API key string does not overwrite existing key."""
        cfg = _make_mock_config()
        cfg.api.anthropic_key = "original-key-1234"
        mgr = MagicMock()
        mgr.load.return_value = cfg
        mock_cm_cls.return_value = mgr

        resp = client.post(
            "/config",
            data={
                "anthropic_key": "**************1234",  # masked value
                "forge_host": "localhost",
                "forge_port": "8000",
                "memory_backend": "sqlite",
                "memory_path": "~/.local/share/animus/memory.db",
                "max_context_tokens": "100000",
                "services_port": "7700",
                "log_level": "info",
                "data_dir": "~/.local/share/animus",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303
        # The original key should be preserved — save is called with the cfg
        saved_cfg = mgr.save.call_args[0][0]
        assert saved_cfg.api.anthropic_key == "original-key-1234"

    @patch("animus_bootstrap.dashboard.routers.config.ConfigManager")
    def test_config_post_updates_new_api_key(
        self,
        mock_cm_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """POST /config with a real (non-masked) API key does update it."""
        cfg = _make_mock_config()
        cfg.api.anthropic_key = "old-key"
        mgr = MagicMock()
        mgr.load.return_value = cfg
        mock_cm_cls.return_value = mgr

        resp = client.post(
            "/config",
            data={
                "anthropic_key": "brand-new-key-5678",
                "forge_host": "localhost",
                "forge_port": "8000",
                "memory_backend": "sqlite",
                "memory_path": "~/.local/share/animus/memory.db",
                "max_context_tokens": "100000",
                "services_port": "7700",
                "log_level": "info",
                "data_dir": "~/.local/share/animus",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303
        saved_cfg = mgr.save.call_args[0][0]
        assert saved_cfg.api.anthropic_key == "brand-new-key-5678"


# ------------------------------------------------------------------
# Memory page — GET /memory and GET /memory/export
# ------------------------------------------------------------------


class TestMemoryPage:
    """Tests for the memory browser router."""

    def test_memory_page_returns_200(self, client: TestClient) -> None:
        """GET /memory returns 200."""
        resp = client.get("/memory")
        assert resp.status_code == 200

    @patch("animus_bootstrap.dashboard.routers.memory.ConfigManager")
    def test_memory_export(
        self,
        mock_cm_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """GET /memory/export returns JSON with config data."""
        mock_cm_cls.return_value = _mock_config_manager()
        resp = client.get("/memory/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        data = resp.json()
        # Should contain top-level config sections
        assert "animus" in data
        assert "api" in data
        assert "forge" in data
        assert "memory" in data
        assert "identity" in data
        assert "services" in data

    @patch("animus_bootstrap.dashboard.routers.memory.ConfigManager")
    def test_memory_export_contains_defaults(
        self,
        mock_cm_cls: MagicMock,
        client: TestClient,
    ) -> None:
        """GET /memory/export JSON includes default config values."""
        mock_cm_cls.return_value = _mock_config_manager()
        resp = client.get("/memory/export")
        data = resp.json()
        assert data["animus"]["version"] == "0.1.0"
        assert data["memory"]["backend"] == "sqlite"


# ------------------------------------------------------------------
# Logs page — GET /logs and GET /logs/stream
# ------------------------------------------------------------------


class TestLogsPage:
    """Tests for the log viewer router."""

    def test_logs_page_returns_200(self, client: TestClient) -> None:
        """GET /logs returns 200."""
        resp = client.get("/logs")
        assert resp.status_code == 200

    def test_logs_stream_no_file(self, client: TestClient, tmp_path: Path) -> None:
        """GET /logs/stream sends SSE event when no log file exists."""
        # Point _LOG_FILE at a path that doesn't exist
        fake_log = tmp_path / "nonexistent.log"
        with patch("animus_bootstrap.dashboard.routers.logs._LOG_FILE", fake_log):
            resp = client.get("/logs/stream")
        assert resp.status_code == 200
        # SSE response should contain the "No logs yet" event
        assert "No logs yet" in resp.text

    def test_logs_stream_with_mocked_generator(self, client: TestClient) -> None:
        """GET /logs/stream returns SSE events from a mocked generator."""

        async def _fake_tail() -> AsyncGenerator[dict[str, str], None]:
            yield {"event": "log", "data": "Hello from mock"}

        with patch("animus_bootstrap.dashboard.routers.logs._tail_log", side_effect=_fake_tail):
            resp = client.get("/logs/stream")
        assert resp.status_code == 200
        assert "Hello from mock" in resp.text


# ------------------------------------------------------------------
# Update page — GET /update and POST /update/apply
# ------------------------------------------------------------------


class TestUpdatePage:
    """Tests for the update management router."""

    def test_update_page_returns_200(self, client: TestClient) -> None:
        """GET /update returns 200."""
        resp = client.get("/update")
        assert resp.status_code == 200

    def test_update_page_shows_version(self, client: TestClient) -> None:
        """GET /update response contains the current version."""
        import animus_bootstrap

        resp = client.get("/update")
        assert resp.status_code == 200
        assert animus_bootstrap.__version__ in resp.text

    def test_update_apply_placeholder(self, client: TestClient) -> None:
        """POST /update/apply returns JSON placeholder response."""
        resp = client.post("/update/apply")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "info"
        assert "coming soon" in data["message"].lower()


# ------------------------------------------------------------------
# Home helpers — _format_uptime, _get_memory_size, _mask_key
# ------------------------------------------------------------------


class TestFormatUptime:
    """Unit tests for the _format_uptime helper."""

    def test_seconds_only(self) -> None:
        from animus_bootstrap.dashboard.routers.home import _format_uptime

        assert _format_uptime(45) == "45s"

    def test_minutes_and_seconds(self) -> None:
        from animus_bootstrap.dashboard.routers.home import _format_uptime

        assert _format_uptime(125) == "2m 5s"

    def test_hours_minutes_seconds(self) -> None:
        from animus_bootstrap.dashboard.routers.home import _format_uptime

        assert _format_uptime(3661) == "1h 1m 1s"

    def test_days(self) -> None:
        from animus_bootstrap.dashboard.routers.home import _format_uptime

        assert _format_uptime(90061) == "1d 1h 1m 1s"

    def test_zero(self) -> None:
        from animus_bootstrap.dashboard.routers.home import _format_uptime

        assert _format_uptime(0) == "0s"


class TestMaskKey:
    """Unit tests for the _mask_key helper."""

    def test_empty_string(self) -> None:
        from animus_bootstrap.dashboard.routers.config import _mask_key

        assert _mask_key("") == ""

    def test_short_key(self) -> None:
        from animus_bootstrap.dashboard.routers.config import _mask_key

        assert _mask_key("abcd") == "abcd"

    def test_normal_key(self) -> None:
        from animus_bootstrap.dashboard.routers.config import _mask_key

        result = _mask_key("sk-ant-12345678")
        assert result.endswith("5678")
        assert result.startswith("*")
        assert "sk-ant" not in result

    def test_five_chars(self) -> None:
        from animus_bootstrap.dashboard.routers.config import _mask_key

        assert _mask_key("12345") == "*2345"


class TestGetMemorySize:
    """Unit tests for the _get_memory_size helper."""

    def test_no_runtime(self) -> None:
        from animus_bootstrap.dashboard.routers.home import _get_memory_size

        assert _get_memory_size(None) == "N/A"

    def test_no_memory_manager(self) -> None:
        from animus_bootstrap.dashboard.routers.home import _get_memory_size

        runtime = MagicMock()
        runtime.memory_manager = None
        assert _get_memory_size(runtime) == "N/A"

    def test_with_db_file(self, tmp_path: Path) -> None:
        from animus_bootstrap.dashboard.routers.home import _get_memory_size

        db_file = tmp_path / "memory.db"
        db_file.write_bytes(b"x" * 512)
        runtime = MagicMock()
        runtime.memory_manager = MagicMock()
        runtime.config.intelligence.memory_db_path = str(db_file)
        result = _get_memory_size(runtime)
        assert "B" in result


# ------------------------------------------------------------------
# App-level smoke tests
# ------------------------------------------------------------------


class TestAppStructure:
    """Verify the FastAPI app is wired correctly."""

    def test_app_title(self) -> None:
        assert app.title == "Animus Dashboard"

    def test_app_version(self) -> None:
        assert app.version == "0.4.0"

    def test_static_mount_exists(self) -> None:
        """The /static mount should be present."""
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/static" in route_paths

    def test_templates_on_app_state(self) -> None:
        """Templates should be available on app.state."""
        assert hasattr(app.state, "templates")
