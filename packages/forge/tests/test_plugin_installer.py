"""Tests for plugin installer module.

Comprehensively tests PluginInstaller with filesystem and network ops mocked.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest

from animus_forge.plugins.installer import PluginInstaller
from animus_forge.plugins.models import (
    PluginInstallation,
    PluginInstallRequest,
    PluginSource,
    PluginUpdateRequest,
)
from animus_forge.plugins.registry import PluginRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend():
    """Create a mock database backend."""
    backend = MagicMock()
    backend.transaction.return_value.__enter__ = MagicMock()
    backend.transaction.return_value.__exit__ = MagicMock(return_value=False)
    backend.execute.return_value = []
    return backend


@pytest.fixture
def mock_registry():
    """Create a mock plugin registry."""
    registry = MagicMock(spec=PluginRegistry)
    return registry


@pytest.fixture
def mock_marketplace():
    """Create a mock plugin marketplace."""
    marketplace = MagicMock()
    return marketplace


@pytest.fixture
def installer(tmp_path, mock_backend, mock_registry):
    """Create a PluginInstaller with mocked dependencies."""
    return PluginInstaller(
        backend=mock_backend,
        plugins_dir=tmp_path / "plugins",
        registry=mock_registry,
    )


@pytest.fixture
def installer_with_marketplace(tmp_path, mock_backend, mock_registry, mock_marketplace):
    """Create a PluginInstaller with a marketplace."""
    return PluginInstaller(
        backend=mock_backend,
        plugins_dir=tmp_path / "plugins",
        marketplace=mock_marketplace,
        registry=mock_registry,
    )


def _make_installation(**overrides) -> PluginInstallation:
    """Helper to create a PluginInstallation with defaults."""
    defaults = dict(
        id="inst-1",
        plugin_name="test-plugin",
        version="1.0.0",
        installed_at=datetime(2026, 1, 15, 10, 0),
        enabled=True,
        config={},
        local_path="/tmp/plugins/test-plugin/plugin.py",
        source=PluginSource.LOCAL,
        source_url=None,
        checksum="abc123",
        auto_update=False,
    )
    defaults.update(overrides)
    return PluginInstallation(**defaults)


def _installation_row(installation=None, **overrides):
    """Create a database row tuple from an installation."""
    inst = installation or _make_installation(**overrides)
    return (
        inst.id,
        inst.plugin_name,
        inst.version,
        inst.installed_at.isoformat(),
        inst.updated_at.isoformat() if inst.updated_at else None,
        int(inst.enabled),
        json.dumps(inst.config),
        inst.local_path,
        inst.source.value if isinstance(inst.source, PluginSource) else inst.source,
        inst.source_url,
        inst.checksum,
        int(inst.auto_update),
    )


# ---------------------------------------------------------------------------
# Schema initialization tests
# ---------------------------------------------------------------------------


class TestEnsureSchema:
    """Tests for schema initialization."""

    def test_schema_created_on_init(self, mock_backend, mock_registry, tmp_path):
        PluginInstaller(
            backend=mock_backend,
            plugins_dir=tmp_path / "plugins",
            registry=mock_registry,
        )
        mock_backend.transaction.assert_called_once()

    def test_schema_error_logged(self, mock_backend, mock_registry, tmp_path):
        mock_backend.transaction.side_effect = Exception("DB error")
        # Should not raise, just log the error
        installer = PluginInstaller(
            backend=mock_backend,
            plugins_dir=tmp_path / "plugins",
            registry=mock_registry,
        )
        assert installer is not None


# ---------------------------------------------------------------------------
# Install tests
# ---------------------------------------------------------------------------


class TestInstall:
    """Tests for the install method."""

    def test_install_already_exists(self, installer, mock_backend):
        mock_backend.execute.return_value = [_installation_row()]
        request = PluginInstallRequest(name="test-plugin", source=PluginSource.LOCAL)
        result = installer.install(request)
        assert result is None

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_install_local_success(self, mock_load, installer, tmp_path, mock_backend):
        # No existing installation
        mock_backend.execute.side_effect = [
            [],  # get_installation returns nothing
            None,  # save
            None,  # enable load
        ]

        # Create a source file
        source_file = tmp_path / "source_plugin.py"
        source_file.write_text("class Plugin: version = '1.0.0'")

        mock_plugin = MagicMock()
        mock_plugin.version = "1.0.0"
        mock_load.return_value = mock_plugin

        request = PluginInstallRequest(
            name="my-plugin",
            source=PluginSource.LOCAL,
            source_url=str(source_file),
            version="1.0.0",
        )
        result = installer.install(request)
        assert result is not None
        assert result.plugin_name == "my-plugin"
        assert result.version == "1.0.0"

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_install_local_dir(self, mock_load, installer, tmp_path, mock_backend):
        mock_backend.execute.side_effect = [[], None, None]

        source_dir = tmp_path / "source_dir"
        source_dir.mkdir()
        (source_dir / "plugin.py").write_text("class Plugin: version = '2.0.0'")

        mock_plugin = MagicMock()
        mock_plugin.version = "2.0.0"
        mock_load.return_value = mock_plugin

        request = PluginInstallRequest(
            name="dir-plugin",
            source=PluginSource.LOCAL,
            source_url=str(source_dir),
            version="2.0.0",
        )
        result = installer.install(request)
        assert result is not None

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_install_local_no_source_url(self, mock_load, installer, mock_backend):
        mock_backend.execute.return_value = []
        request = PluginInstallRequest(
            name="no-source",
            source=PluginSource.LOCAL,
        )
        result = installer.install(request)
        assert result is None  # No plugin_path obtained

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_install_validation_fails(self, mock_load, installer, tmp_path, mock_backend):
        mock_backend.execute.return_value = []

        source_file = tmp_path / "bad_plugin.py"
        source_file.write_text("bad code")

        mock_load.return_value = None  # Validation failure

        request = PluginInstallRequest(
            name="bad-plugin",
            source=PluginSource.LOCAL,
            source_url=str(source_file),
        )
        result = installer.install(request)
        assert result is None

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_install_validation_exception(self, mock_load, installer, tmp_path, mock_backend):
        mock_backend.execute.return_value = []

        source_file = tmp_path / "error_plugin.py"
        source_file.write_text("raise Exception")

        mock_load.side_effect = Exception("Load error")

        request = PluginInstallRequest(
            name="error-plugin",
            source=PluginSource.LOCAL,
            source_url=str(source_file),
        )
        result = installer.install(request)
        assert result is None

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_install_save_failure(self, mock_load, installer, tmp_path, mock_backend):
        mock_backend.execute.side_effect = [
            [],  # get_installation
            Exception("DB write error"),  # save fails
        ]

        source_file = tmp_path / "good_plugin.py"
        source_file.write_text("class Plugin: version = '1.0.0'")

        mock_plugin = MagicMock()
        mock_plugin.version = "1.0.0"
        mock_load.return_value = mock_plugin

        request = PluginInstallRequest(
            name="good-plugin",
            source=PluginSource.LOCAL,
            source_url=str(source_file),
            version="1.0.0",
        )
        result = installer.install(request)
        assert result is None  # Save failed

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_install_uses_plugin_version_when_not_specified(
        self, mock_load, installer, tmp_path, mock_backend
    ):
        mock_backend.execute.side_effect = [[], None, None]

        source_file = tmp_path / "versioned.py"
        source_file.write_text("class Plugin: version = '3.2.1'")

        mock_plugin = MagicMock()
        mock_plugin.version = "3.2.1"
        mock_load.return_value = mock_plugin

        request = PluginInstallRequest(
            name="versioned",
            source=PluginSource.LOCAL,
            source_url=str(source_file),
            version=None,
        )
        result = installer.install(request)
        assert result is not None
        assert result.version == "3.2.1"

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_install_disabled(self, mock_load, installer, tmp_path, mock_backend):
        mock_backend.execute.side_effect = [[], None]

        source_file = tmp_path / "disabled.py"
        source_file.write_text("class Plugin: version = '1.0.0'")

        mock_plugin = MagicMock()
        mock_plugin.version = "1.0.0"
        mock_load.return_value = mock_plugin

        request = PluginInstallRequest(
            name="disabled-plugin",
            source=PluginSource.LOCAL,
            source_url=str(source_file),
            version="1.0.0",
            enable=False,
        )
        result = installer.install(request)
        assert result is not None
        assert result.enabled is False
        # load_plugin_from_file called once for validation, NOT for registration
        assert mock_load.call_count == 1

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_install_enable_failure_still_installs(
        self, mock_load, installer, tmp_path, mock_backend
    ):
        mock_backend.execute.side_effect = [[], None]

        source_file = tmp_path / "partial.py"
        source_file.write_text("class Plugin: version = '1.0.0'")

        # First call (validation) succeeds, second call (enable) fails
        mock_plugin = MagicMock()
        mock_plugin.version = "1.0.0"
        mock_load.side_effect = [mock_plugin, Exception("Enable failed")]

        request = PluginInstallRequest(
            name="partial-plugin",
            source=PluginSource.LOCAL,
            source_url=str(source_file),
            version="1.0.0",
        )
        result = installer.install(request)
        # Should still return the installation even if enable fails
        assert result is not None

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_install_marketplace_increments_downloads(
        self,
        mock_load,
        installer_with_marketplace,
        tmp_path,
        mock_backend,
        mock_marketplace,
    ):
        inst = installer_with_marketplace

        mock_listing = MagicMock()
        mock_listing.latest_version = "1.0.0"
        mock_marketplace.get_plugin.return_value = mock_listing

        mock_release = MagicMock()
        mock_release.download_url = "https://example.com/plugin.py"
        mock_release.checksum = "abc123"
        mock_marketplace.get_release.return_value = mock_release

        mock_backend.execute.side_effect = [[], None, None]

        mock_plugin = MagicMock()
        mock_plugin.version = "1.0.0"
        mock_load.return_value = mock_plugin

        # Mock the download
        with patch.object(inst, "_download_and_verify") as mock_dl:
            plugin_path = tmp_path / "plugins" / "mp-plugin" / "plugin.py"
            plugin_path.parent.mkdir(parents=True, exist_ok=True)
            plugin_path.write_text("class Plugin: pass")
            mock_dl.return_value = (plugin_path, "checksum123")

            request = PluginInstallRequest(
                name="mp-plugin",
                source=PluginSource.MARKETPLACE,
                version="1.0.0",
            )
            result = inst.install(request)
            assert result is not None
            mock_marketplace.increment_downloads.assert_called_once_with("mp-plugin")


# ---------------------------------------------------------------------------
# Uninstall tests
# ---------------------------------------------------------------------------


class TestUninstall:
    """Tests for the uninstall method."""

    def test_uninstall_not_installed(self, installer, mock_backend):
        mock_backend.execute.return_value = []
        result = installer.uninstall("nonexistent")
        assert result is False

    def test_uninstall_success(self, installer, mock_backend):
        mock_backend.execute.side_effect = [
            [_installation_row()],  # get_installation
            None,  # delete
        ]
        result = installer.uninstall("test-plugin")
        assert result is True

    def test_uninstall_registry_error_still_succeeds(self, installer, mock_backend, mock_registry):
        mock_backend.execute.side_effect = [
            [_installation_row()],
            None,
        ]
        mock_registry.unregister.side_effect = Exception("Not registered")
        result = installer.uninstall("test-plugin")
        assert result is True

    def test_uninstall_db_delete_error(self, installer, mock_backend):
        mock_backend.execute.side_effect = [
            [_installation_row()],
            Exception("DB error"),
        ]
        result = installer.uninstall("test-plugin")
        assert result is False


# ---------------------------------------------------------------------------
# Update tests
# ---------------------------------------------------------------------------


class TestUpdate:
    """Tests for the update method."""

    def test_update_not_installed(self, installer, mock_backend):
        mock_backend.execute.return_value = []
        request = PluginUpdateRequest(name="missing")
        result = installer.update(request)
        assert result is None

    def test_update_already_at_version(self, installer, mock_backend):
        mock_backend.execute.return_value = [_installation_row(version="1.0.0")]
        request = PluginUpdateRequest(name="test-plugin", version="1.0.0")
        result = installer.update(request)
        assert result is not None
        assert result.version == "1.0.0"

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_update_marketplace_success(
        self,
        mock_load,
        installer_with_marketplace,
        tmp_path,
        mock_backend,
        mock_marketplace,
    ):
        inst = installer_with_marketplace

        # Current installation (must be MARKETPLACE source for update to use marketplace)
        current_row = _installation_row(
            version="1.0.0",
            source=PluginSource.MARKETPLACE,
            local_path=str(tmp_path / "plugins" / "test-plugin" / "plugin.py"),
        )
        mock_backend.execute.side_effect = [
            [current_row],  # get_installation
            None,  # update_installation
        ]

        # Marketplace returns new version
        mock_listing = MagicMock()
        mock_listing.latest_version = "2.0.0"
        mock_marketplace.get_plugin.return_value = mock_listing

        mock_plugin = MagicMock()
        mock_plugin.version = "2.0.0"
        mock_load.return_value = mock_plugin

        with (
            patch.object(inst, "_download_from_marketplace") as mock_dl_mp,
            patch.object(inst, "_backup_plugin") as mock_backup,
            patch("shutil.rmtree"),
        ):
            plugin_path = tmp_path / "plugins" / "test-plugin" / "plugin.py"
            plugin_path.parent.mkdir(parents=True, exist_ok=True)
            plugin_path.write_text("class Plugin: pass")
            mock_dl_mp.return_value = (plugin_path, "newchecksum", "2.0.0")
            mock_backup.return_value = tmp_path / "plugins" / "test-plugin.backup"

            request = PluginUpdateRequest(name="test-plugin")
            result = inst.update(request)
            assert result is not None
            assert result.version == "2.0.0"

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_update_download_failure_restores_backup(
        self,
        mock_load,
        installer_with_marketplace,
        tmp_path,
        mock_backend,
        mock_marketplace,
    ):
        inst = installer_with_marketplace

        current_row = _installation_row(
            version="1.0.0",
            source=PluginSource.MARKETPLACE,
            local_path=str(tmp_path / "plugins" / "test-plugin" / "plugin.py"),
        )
        mock_backend.execute.return_value = [current_row]

        mock_listing = MagicMock()
        mock_listing.latest_version = "2.0.0"
        mock_marketplace.get_plugin.return_value = mock_listing

        with (
            patch.object(inst, "_download_and_verify") as mock_dl,
            patch.object(inst, "_backup_plugin") as mock_backup,
            patch.object(inst, "_restore_backup") as mock_restore,
        ):
            backup_path = tmp_path / "plugins" / "test-plugin.backup"
            backup_path.mkdir(parents=True, exist_ok=True)
            mock_backup.return_value = backup_path
            mock_dl.return_value = (None, None)  # Download fails

            request = PluginUpdateRequest(name="test-plugin")
            result = inst.update(request)
            assert result is None
            mock_restore.assert_called_once()

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_update_validation_failure_restores_backup(
        self,
        mock_load,
        installer_with_marketplace,
        tmp_path,
        mock_backend,
        mock_marketplace,
    ):
        inst = installer_with_marketplace

        current_row = _installation_row(
            version="1.0.0",
            source=PluginSource.MARKETPLACE,
            local_path=str(tmp_path / "plugins" / "test-plugin" / "plugin.py"),
        )
        mock_backend.execute.return_value = [current_row]

        mock_listing = MagicMock()
        mock_listing.latest_version = "2.0.0"
        mock_marketplace.get_plugin.return_value = mock_listing

        mock_load.return_value = None  # Validation fails

        with (
            patch.object(inst, "_download_and_verify") as mock_dl,
            patch.object(inst, "_backup_plugin") as mock_backup,
            patch.object(inst, "_restore_backup") as mock_restore,
        ):
            plugin_path = tmp_path / "plugins" / "test-plugin" / "plugin.py"
            plugin_path.parent.mkdir(parents=True, exist_ok=True)
            plugin_path.write_text("bad")
            mock_backup.return_value = tmp_path / "plugins" / "test-plugin.backup"
            mock_dl.return_value = (plugin_path, "hash")

            request = PluginUpdateRequest(name="test-plugin")
            result = inst.update(request)
            assert result is None
            mock_restore.assert_called_once()

    def test_update_with_config_override(self, installer, mock_backend):
        current_row = _installation_row(version="1.0.0")
        mock_backend.execute.return_value = [current_row]

        request = PluginUpdateRequest(
            name="test-plugin",
            version="1.0.0",  # Same version, no update needed
            config={"key": "value"},
        )
        result = installer.update(request)
        # Same version returns existing installation
        assert result is not None


# ---------------------------------------------------------------------------
# Enable/Disable tests
# ---------------------------------------------------------------------------


class TestEnableDisable:
    """Tests for enable and disable methods."""

    def test_enable_not_installed(self, installer, mock_backend):
        mock_backend.execute.return_value = []
        assert installer.enable("missing") is False

    def test_enable_already_enabled(self, installer, mock_backend):
        mock_backend.execute.return_value = [_installation_row(enabled=True)]
        assert installer.enable("test-plugin") is True

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_enable_success(self, mock_load, installer, mock_backend):
        mock_backend.execute.side_effect = [
            [_installation_row(enabled=False)],
            None,  # UPDATE query
        ]
        mock_plugin = MagicMock()
        mock_load.return_value = mock_plugin

        assert installer.enable("test-plugin") is True

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_enable_load_returns_none(self, mock_load, installer, mock_backend):
        mock_backend.execute.return_value = [_installation_row(enabled=False)]
        mock_load.return_value = None

        assert installer.enable("test-plugin") is False

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_enable_load_exception(self, mock_load, installer, mock_backend):
        mock_backend.execute.return_value = [_installation_row(enabled=False)]
        mock_load.side_effect = Exception("Load error")

        assert installer.enable("test-plugin") is False

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_enable_with_config_override(self, mock_load, installer, mock_backend):
        mock_backend.execute.side_effect = [
            [_installation_row(enabled=False, config={"old": "val"})],
            None,
        ]
        mock_load.return_value = MagicMock()

        result = installer.enable("test-plugin", config={"new": "val"})
        assert result is True
        # Check the config passed to load_plugin_from_file
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["config"] == {"new": "val"}

    def test_disable_not_installed(self, installer, mock_backend):
        mock_backend.execute.return_value = []
        assert installer.disable("missing") is False

    def test_disable_already_disabled(self, installer, mock_backend):
        mock_backend.execute.return_value = [_installation_row(enabled=False)]
        assert installer.disable("test-plugin") is True

    def test_disable_success(self, installer, mock_backend):
        mock_backend.execute.side_effect = [
            [_installation_row(enabled=True)],
            None,  # UPDATE
        ]
        assert installer.disable("test-plugin") is True

    def test_disable_registry_error_still_succeeds(self, installer, mock_backend, mock_registry):
        mock_backend.execute.side_effect = [
            [_installation_row(enabled=True)],
            None,
        ]
        mock_registry.unregister.side_effect = Exception("Not registered")
        assert installer.disable("test-plugin") is True

    def test_disable_db_error(self, installer, mock_backend):
        mock_backend.execute.side_effect = [
            [_installation_row(enabled=True)],
            Exception("DB error"),
        ]
        assert installer.disable("test-plugin") is False


# ---------------------------------------------------------------------------
# Get/List tests
# ---------------------------------------------------------------------------


class TestGetAndList:
    """Tests for get_installation and list_installations."""

    def test_get_installation_found(self, installer, mock_backend):
        row = _installation_row()
        mock_backend.execute.return_value = [row]
        result = installer.get_installation("test-plugin")
        assert result is not None
        assert result.plugin_name == "test-plugin"
        assert result.version == "1.0.0"

    def test_get_installation_not_found(self, installer, mock_backend):
        mock_backend.execute.return_value = []
        result = installer.get_installation("missing")
        assert result is None

    def test_get_installation_db_error(self, installer, mock_backend):
        mock_backend.execute.side_effect = Exception("DB error")
        result = installer.get_installation("test-plugin")
        assert result is None

    def test_list_installations_all(self, installer, mock_backend):
        rows = [
            _installation_row(id="1", plugin_name="plugin-a"),
            _installation_row(id="2", plugin_name="plugin-b"),
        ]
        mock_backend.execute.return_value = rows
        results = installer.list_installations()
        assert len(results) == 2
        assert results[0].plugin_name == "plugin-a"
        assert results[1].plugin_name == "plugin-b"

    def test_list_installations_enabled_only(self, installer, mock_backend):
        mock_backend.execute.return_value = [
            _installation_row(enabled=True),
        ]
        results = installer.list_installations(enabled_only=True)
        assert len(results) == 1

    def test_list_installations_empty(self, installer, mock_backend):
        mock_backend.execute.return_value = []
        results = installer.list_installations()
        assert results == []

    def test_list_installations_db_error(self, installer, mock_backend):
        mock_backend.execute.side_effect = Exception("DB error")
        results = installer.list_installations()
        assert results == []


# ---------------------------------------------------------------------------
# Load enabled plugins
# ---------------------------------------------------------------------------


class TestLoadEnabledPlugins:
    """Tests for load_enabled_plugins."""

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_load_enabled_all_succeed(self, mock_load, installer, mock_backend):
        rows = [
            _installation_row(id="1", plugin_name="a"),
            _installation_row(id="2", plugin_name="b"),
        ]
        mock_backend.execute.return_value = rows
        mock_load.return_value = MagicMock()

        count = installer.load_enabled_plugins()
        assert count == 2

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_load_enabled_partial_failure(self, mock_load, installer, mock_backend):
        rows = [
            _installation_row(id="1", plugin_name="good"),
            _installation_row(id="2", plugin_name="bad"),
        ]
        mock_backend.execute.return_value = rows
        mock_load.side_effect = [MagicMock(), Exception("Load failed")]

        count = installer.load_enabled_plugins()
        assert count == 1

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_load_enabled_returns_none(self, mock_load, installer, mock_backend):
        rows = [_installation_row()]
        mock_backend.execute.return_value = rows
        mock_load.return_value = None

        count = installer.load_enabled_plugins()
        assert count == 0

    @patch("animus_forge.plugins.installer.load_plugin_from_file")
    def test_load_enabled_empty(self, mock_load, installer, mock_backend):
        mock_backend.execute.return_value = []
        count = installer.load_enabled_plugins()
        assert count == 0


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


class TestDownloadHelpers:
    """Tests for download-related private methods."""

    def test_download_from_marketplace_no_marketplace(self, installer):
        result = installer._download_from_marketplace("test", None)
        assert result == (None, None, None)

    def test_download_from_marketplace_no_listing(
        self, installer_with_marketplace, mock_marketplace
    ):
        mock_marketplace.get_plugin.return_value = None
        result = installer_with_marketplace._download_from_marketplace("test", None)
        assert result == (None, None, None)

    def test_download_from_marketplace_no_release(
        self, installer_with_marketplace, mock_marketplace
    ):
        mock_listing = MagicMock()
        mock_listing.latest_version = "1.0.0"
        mock_marketplace.get_plugin.return_value = mock_listing
        mock_marketplace.get_release.return_value = None

        result = installer_with_marketplace._download_from_marketplace("test", None)
        assert result == (None, None, None)

    def test_download_from_github_no_url(self, installer):
        result = installer._download_from_github(None, "test")
        assert result == (None, None)

    def test_download_from_github_invalid_host(self, installer):
        result = installer._download_from_github("https://evil.com/repo", "test")
        assert result == (None, None)

    def test_download_from_github_valid_url(self, installer):
        with patch.object(installer, "_download_and_verify") as mock_dl:
            mock_dl.return_value = (Path("/tmp/plugin.py"), "checksum")
            result = installer._download_from_github("https://github.com/user/repo", "test")
            assert result == (Path("/tmp/plugin.py"), "checksum")
            # Should convert to raw URL
            call_url = mock_dl.call_args[0][0]
            assert urlparse(call_url).hostname == "raw.githubusercontent.com"

    def test_download_from_github_blob_url(self, installer):
        with patch.object(installer, "_download_and_verify") as mock_dl:
            mock_dl.return_value = (Path("/tmp/plugin.py"), "checksum")
            installer._download_from_github(
                "https://github.com/user/repo/blob/main/plugin.py", "test"
            )
            call_url = mock_dl.call_args[0][0]
            assert "/raw/" in call_url

    def test_download_from_github_raw_url(self, installer):
        with patch.object(installer, "_download_and_verify") as mock_dl:
            mock_dl.return_value = (Path("/tmp/plugin.py"), "checksum")
            installer._download_from_github(
                "https://raw.githubusercontent.com/user/repo/main/plugin.py", "test"
            )
            call_url = mock_dl.call_args[0][0]
            assert urlparse(call_url).hostname == "raw.githubusercontent.com"

    def test_download_from_url_none(self, installer):
        result = installer._download_from_url(None, "test")
        assert result == (None, None)

    def test_download_from_url_valid(self, installer):
        with patch.object(installer, "_download_and_verify") as mock_dl:
            mock_dl.return_value = (Path("/tmp/plugin.py"), "checksum")
            result = installer._download_from_url("https://example.com/plugin.py", "test")
            assert result == (Path("/tmp/plugin.py"), "checksum")


# ---------------------------------------------------------------------------
# Download and verify
# ---------------------------------------------------------------------------


class TestDownloadAndVerify:
    """Tests for _download_and_verify."""

    @patch("animus_forge.plugins.installer.urlretrieve")
    def test_download_success(self, mock_retrieve, installer, tmp_path):
        # Create a temp file to simulate download
        def fake_retrieve(url, filename):
            Path(filename).write_text("plugin content")

        mock_retrieve.side_effect = fake_retrieve

        with patch.object(installer, "_compute_checksum", return_value="abc123"):
            result = installer._download_and_verify(
                "https://example.com/plugin.py", "test-plugin", None
            )
            assert result[0] is not None
            assert result[1] == "abc123"

    @patch("animus_forge.plugins.installer.urlretrieve")
    def test_download_checksum_mismatch(self, mock_retrieve, installer, tmp_path):
        def fake_retrieve(url, filename):
            Path(filename).write_text("plugin content")

        mock_retrieve.side_effect = fake_retrieve

        with patch.object(installer, "_compute_checksum", return_value="wrong"):
            result = installer._download_and_verify(
                "https://example.com/plugin.py", "test-plugin", "expected"
            )
            assert result == (None, None)

    @patch("animus_forge.plugins.installer.urlretrieve")
    def test_download_exception(self, mock_retrieve, installer):
        mock_retrieve.side_effect = Exception("Network error")
        result = installer._download_and_verify(
            "https://example.com/plugin.py", "test-plugin", None
        )
        assert result == (None, None)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


class TestFileHelpers:
    """Tests for filesystem helper methods."""

    def test_copy_local_plugin_file(self, installer, tmp_path):
        source = tmp_path / "source.py"
        source.write_text("class Plugin: pass")
        result = installer._copy_local_plugin(str(source), "test-plugin")
        assert result is not None
        assert result.exists()

    def test_copy_local_plugin_dir(self, installer, tmp_path):
        source_dir = tmp_path / "source_dir"
        source_dir.mkdir()
        (source_dir / "plugin.py").write_text("class Plugin: pass")
        result = installer._copy_local_plugin(str(source_dir), "test-plugin")
        assert result is not None

    def test_copy_local_plugin_not_exists(self, installer):
        result = installer._copy_local_plugin("/nonexistent/path", "test")
        assert result is None

    def test_compute_checksum(self, installer, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("hello world")
        checksum = installer._compute_checksum(test_file)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex length

    def test_compute_checksum_deterministic(self, installer, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("deterministic content")
        checksum1 = installer._compute_checksum(test_file)
        checksum2 = installer._compute_checksum(test_file)
        assert checksum1 == checksum2

    def test_cleanup_plugin_dir(self, installer, tmp_path):
        plugin_dir = installer.plugins_dir / "test-plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.py").write_text("content")
        installer._cleanup_plugin_dir("test-plugin")
        assert not plugin_dir.exists()

    def test_cleanup_plugin_dir_not_exists(self, installer):
        # Should not raise
        installer._cleanup_plugin_dir("nonexistent")

    def test_backup_plugin(self, installer, tmp_path):
        plugin_dir = installer.plugins_dir / "test-plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.py").write_text("original")
        backup = installer._backup_plugin("test-plugin")
        assert backup is not None
        assert backup.exists()

    def test_backup_plugin_not_exists(self, installer):
        result = installer._backup_plugin("nonexistent")
        assert result is None

    def test_restore_backup(self, installer, tmp_path):
        plugin_dir = installer.plugins_dir / "test-plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.py").write_text("current")

        backup_dir = installer.plugins_dir / "test-plugin.backup"
        backup_dir.mkdir(parents=True)
        (backup_dir / "plugin.py").write_text("backup")

        installer._restore_backup("test-plugin", backup_dir)
        assert plugin_dir.exists()
        assert (plugin_dir / "plugin.py").read_text() == "backup"

    def test_restore_backup_none(self, installer):
        installer._restore_backup("test-plugin", None)
        # Should not raise

    def test_restore_backup_missing_path(self, installer, tmp_path):
        installer._restore_backup("test-plugin", tmp_path / "nonexistent")
        # Should not raise


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


class TestDatabaseHelpers:
    """Tests for database helper methods."""

    def test_save_installation(self, installer, mock_backend):
        mock_backend.execute.return_value = None
        inst = _make_installation()
        result = installer._save_installation(inst)
        assert result is True

    def test_save_installation_failure(self, installer, mock_backend):
        mock_backend.execute.side_effect = Exception("DB error")
        inst = _make_installation()
        result = installer._save_installation(inst)
        assert result is False

    def test_update_installation(self, installer, mock_backend):
        mock_backend.execute.return_value = None
        inst = _make_installation(updated_at=datetime(2026, 2, 1))
        result = installer._update_installation(inst)
        assert result is True

    def test_update_installation_failure(self, installer, mock_backend):
        mock_backend.execute.side_effect = Exception("DB error")
        inst = _make_installation()
        result = installer._update_installation(inst)
        assert result is False

    def test_row_to_installation(self, installer):
        row = _installation_row()
        result = installer._row_to_installation(row)
        assert isinstance(result, PluginInstallation)
        assert result.plugin_name == "test-plugin"
        assert result.version == "1.0.0"
        assert result.enabled is True

    def test_row_to_installation_null_fields(self, installer):
        row = (
            "id-1",
            "plugin",
            "1.0.0",
            None,  # installed_at
            None,  # updated_at
            0,  # enabled
            None,  # config
            "/path",
            None,  # source
            None,  # source_url
            None,  # checksum
            0,  # auto_update
        )
        result = installer._row_to_installation(row)
        assert result.updated_at is None
        assert result.config == {}
        assert result.source == PluginSource.LOCAL
        assert result.enabled is False

    def test_row_to_installation_with_config_json(self, installer):
        row = _installation_row(config={"key": "value"})
        result = installer._row_to_installation(row)
        assert result.config == {"key": "value"}
