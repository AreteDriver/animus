"""
Tests for Phase 4: Integration Suite

Filesystem, Todoist, Google Calendar, Gmail, Webhooks integrations.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from animus.config import (
    AnimusConfig,
    FilesystemConfig,
    GoogleIntegrationConfig,
    IntegrationConfig,
    TodoistConfig,
    WebhookConfig,
)
from animus.integrations import (
    AuthType,
    FilesystemIntegration,
    IntegrationInfo,
    IntegrationManager,
    IntegrationStatus,
    TodoistIntegration,
    WebhookIntegration,
)

# =============================================================================
# Integration Base Tests
# =============================================================================


class TestIntegrationStatus:
    """Tests for IntegrationStatus enum."""

    def test_all_statuses_exist(self):
        assert IntegrationStatus.DISCONNECTED.value == "disconnected"
        assert IntegrationStatus.CONNECTING.value == "connecting"
        assert IntegrationStatus.CONNECTED.value == "connected"
        assert IntegrationStatus.ERROR.value == "error"
        assert IntegrationStatus.EXPIRED.value == "expired"


class TestAuthType:
    """Tests for AuthType enum."""

    def test_all_auth_types_exist(self):
        assert AuthType.NONE.value == "none"
        assert AuthType.API_KEY.value == "api_key"
        assert AuthType.OAUTH2.value == "oauth2"


class TestIntegrationInfo:
    """Tests for IntegrationInfo dataclass."""

    def test_info_creation(self):
        info = IntegrationInfo(
            name="test",
            display_name="Test Integration",
            status=IntegrationStatus.DISCONNECTED,
            auth_type=AuthType.NONE,
        )
        assert info.name == "test"
        assert info.status == IntegrationStatus.DISCONNECTED

    def test_info_to_dict(self):
        info = IntegrationInfo(
            name="test",
            display_name="Test",
            status=IntegrationStatus.CONNECTED,
            auth_type=AuthType.API_KEY,
            capabilities=["tool1", "tool2"],
        )
        data = info.to_dict()
        assert data["name"] == "test"
        assert data["status"] == "connected"
        assert data["auth_type"] == "api_key"
        assert data["capabilities"] == ["tool1", "tool2"]


# =============================================================================
# Integration Manager Tests
# =============================================================================


class TestIntegrationManager:
    """Tests for IntegrationManager."""

    def test_manager_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IntegrationManager(Path(tmpdir))
            assert manager._data_dir == Path(tmpdir)
            assert len(manager._integrations) == 0

    def test_register_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IntegrationManager(Path(tmpdir))
            integration = FilesystemIntegration(Path(tmpdir))
            manager.register(integration)
            assert "filesystem" in manager._integrations

    def test_get_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IntegrationManager(Path(tmpdir))
            integration = FilesystemIntegration(Path(tmpdir))
            manager.register(integration)

            retrieved = manager.get("filesystem")
            assert retrieved is integration

            missing = manager.get("nonexistent")
            assert missing is None

    def test_list_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IntegrationManager(Path(tmpdir))
            manager.register(FilesystemIntegration(Path(tmpdir)))
            manager.register(WebhookIntegration())

            all_integrations = manager.list_all()
            assert len(all_integrations) == 2

    def test_get_status_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IntegrationManager(Path(tmpdir))
            manager.register(FilesystemIntegration(Path(tmpdir)))

            summary = manager.get_status_summary()
            assert summary["total"] == 1
            assert summary["connected"] == 0
            assert "filesystem" in summary["statuses"]


# =============================================================================
# Filesystem Integration Tests
# =============================================================================


class TestFilesystemIntegration:
    """Tests for FilesystemIntegration."""

    def test_integration_properties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FilesystemIntegration(Path(tmpdir))
            assert fs.name == "filesystem"
            assert fs.display_name == "Local Filesystem"
            assert fs.auth_type == AuthType.NONE

    def test_initial_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FilesystemIntegration(Path(tmpdir))
            assert fs.status == IntegrationStatus.DISCONNECTED
            assert fs.is_connected is False

    def test_connect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FilesystemIntegration(Path(tmpdir))
            result = asyncio.run(fs.connect({"paths": [tmpdir]}))
            assert result is True
            assert fs.is_connected is True

    def test_disconnect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FilesystemIntegration(Path(tmpdir))
            asyncio.run(fs.connect({"paths": [tmpdir]}))
            result = asyncio.run(fs.disconnect())
            assert result is True
            assert fs.is_connected is False

    def test_get_tools(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FilesystemIntegration(Path(tmpdir))
            tools = fs.get_tools()
            tool_names = [t.name for t in tools]
            assert "fs_index" in tool_names
            assert "fs_search" in tool_names
            assert "fs_search_content" in tool_names
            assert "fs_read" in tool_names

    def test_index_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "test.txt").write_text("hello")
            (Path(tmpdir) / "test.py").write_text("print('hi')")

            fs = FilesystemIntegration(Path(tmpdir))
            asyncio.run(fs.connect({"paths": []}))

            result = asyncio.run(fs._tool_index(tmpdir))
            assert result.success is True
            assert result.output["files_indexed"] >= 2

    def test_search_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "readme.txt").write_text("readme content")
            (Path(tmpdir) / "config.py").write_text("config = {}")

            fs = FilesystemIntegration(Path(tmpdir))
            asyncio.run(fs.connect({"paths": []}))
            asyncio.run(fs._tool_index(tmpdir))

            # Search by pattern
            result = asyncio.run(fs._tool_search("*.py"))
            assert result.success is True
            assert result.output["count"] >= 1

    def test_read_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("line 1\nline 2\nline 3")

            fs = FilesystemIntegration(Path(tmpdir))
            asyncio.run(fs.connect({"paths": []}))

            result = asyncio.run(fs._tool_read(str(test_file)))
            assert result.success is True
            assert "line 1" in result.output["content"]


# =============================================================================
# Todoist Integration Tests
# =============================================================================


class TestTodoistIntegration:
    """Tests for TodoistIntegration."""

    def test_integration_properties(self):
        todoist = TodoistIntegration()
        assert todoist.name == "todoist"
        assert todoist.display_name == "Todoist"
        assert todoist.auth_type == AuthType.API_KEY

    def test_initial_status(self):
        todoist = TodoistIntegration()
        assert todoist.status == IntegrationStatus.DISCONNECTED
        assert todoist.is_connected is False

    def test_connect_without_api_key(self):
        todoist = TodoistIntegration()
        result = asyncio.run(todoist.connect({}))
        # Should fail without API key
        assert result is False
        assert todoist.status == IntegrationStatus.ERROR

    def test_get_tools(self):
        todoist = TodoistIntegration()
        tools = todoist.get_tools()
        tool_names = [t.name for t in tools]
        assert "todoist_list_tasks" in tool_names
        assert "todoist_create_task" in tool_names
        assert "todoist_complete_task" in tool_names
        assert "todoist_list_projects" in tool_names
        assert "todoist_sync" in tool_names


# =============================================================================
# Webhook Integration Tests
# =============================================================================


class TestWebhookIntegration:
    """Tests for WebhookIntegration."""

    def test_integration_properties(self):
        webhook = WebhookIntegration()
        assert webhook.name == "webhooks"
        assert webhook.display_name == "Webhooks"
        assert webhook.auth_type == AuthType.NONE

    def test_initial_status(self):
        webhook = WebhookIntegration()
        assert webhook.status == IntegrationStatus.DISCONNECTED
        assert webhook.is_connected is False

    def test_get_tools(self):
        webhook = WebhookIntegration()
        tools = webhook.get_tools()
        tool_names = [t.name for t in tools]
        assert "webhook_list_events" in tool_names
        assert "webhook_get_event" in tool_names
        assert "webhook_info" in tool_names


# =============================================================================
# Config Tests
# =============================================================================


class TestIntegrationConfig:
    """Tests for IntegrationConfig and sub-configs."""

    def test_google_config_defaults(self):
        config = GoogleIntegrationConfig()
        assert config.enabled is False
        assert config.client_id is None
        assert config.client_secret is None

    def test_todoist_config_defaults(self):
        config = TodoistConfig()
        assert config.enabled is False
        assert config.api_key is None

    def test_filesystem_config_defaults(self):
        config = FilesystemConfig()
        assert config.enabled is False
        assert config.indexed_paths == []
        assert ".git" in config.exclude_patterns

    def test_webhook_config_defaults(self):
        config = WebhookConfig()
        assert config.enabled is False
        assert config.port == 8421
        assert config.secret is None

    def test_integration_config_defaults(self):
        config = IntegrationConfig()
        assert isinstance(config.google, GoogleIntegrationConfig)
        assert isinstance(config.todoist, TodoistConfig)
        assert isinstance(config.filesystem, FilesystemConfig)
        assert isinstance(config.webhooks, WebhookConfig)

    def test_animus_config_has_integrations(self):
        config = AnimusConfig()
        assert hasattr(config, "integrations")
        assert isinstance(config.integrations, IntegrationConfig)

    def test_config_to_dict_includes_integrations(self):
        config = AnimusConfig()
        data = config.to_dict()
        assert "integrations" in data
        assert "google" in data["integrations"]
        assert "todoist" in data["integrations"]
        assert "filesystem" in data["integrations"]
        assert "webhooks" in data["integrations"]

    def test_config_save_load_preserves_integrations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnimusConfig(data_dir=Path(tmpdir))
            config.integrations.filesystem.enabled = True
            config.integrations.filesystem.indexed_paths = ["/home/test"]
            config.integrations.webhooks.port = 9000
            config.save()

            loaded = AnimusConfig.load(config.config_file)
            assert loaded.integrations.filesystem.enabled is True
            assert loaded.integrations.filesystem.indexed_paths == ["/home/test"]
            assert loaded.integrations.webhooks.port == 9000


# =============================================================================
# Version Tests
# =============================================================================


class TestVersion:
    """Test version is updated for Phase 4+."""

    def test_version_is_0_5_0(self):
        from animus import __version__

        assert __version__ == "0.6.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
