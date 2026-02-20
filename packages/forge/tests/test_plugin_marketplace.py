"""Tests for plugin marketplace functionality."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from animus_forge.plugins.models import (
    PluginCategory,
    PluginInstallation,
    PluginInstallRequest,
    PluginListing,
    PluginMetadata,
    PluginRelease,
    PluginSearchResult,
    PluginSource,
    PluginUpdateRequest,
)


class TestPluginModels:
    """Tests for plugin marketplace models."""

    def test_plugin_category_values(self):
        """Test all plugin categories are defined."""
        expected = [
            "integration",
            "data_transform",
            "monitoring",
            "security",
            "workflow",
            "ai_provider",
            "storage",
            "notification",
            "analytics",
            "other",
        ]
        actual = [c.value for c in PluginCategory]
        for cat in expected:
            assert cat in actual

    def test_plugin_source_values(self):
        """Test all plugin sources are defined."""
        expected = ["marketplace", "local", "github", "pypi", "url"]
        actual = [s.value for s in PluginSource]
        for src in expected:
            assert src in actual

    def test_plugin_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            description="A test plugin",
            author="Test Author",
            category=PluginCategory.INTEGRATION,
            tags=["test", "example"],
            requirements=["requests>=2.0"],
            provides_handlers=["custom_step"],
        )
        assert metadata.name == "test-plugin"
        assert metadata.version == "1.0.0"
        assert metadata.category == PluginCategory.INTEGRATION
        assert "test" in metadata.tags
        assert "custom_step" in metadata.provides_handlers

    def test_plugin_metadata_defaults(self):
        """Test plugin metadata default values."""
        metadata = PluginMetadata(name="minimal", version="0.1.0")
        assert metadata.description == ""
        assert metadata.author == ""
        assert metadata.license == "MIT"
        assert metadata.category == PluginCategory.OTHER
        assert metadata.tags == []
        assert metadata.entry_point == "Plugin"

    def test_plugin_release_creation(self):
        """Test creating plugin release."""
        release = PluginRelease(
            id="release-123",
            plugin_name="test-plugin",
            version="1.0.0",
            download_url="https://example.com/plugin.py",
            checksum="abc123",
            changelog="Initial release",
        )
        assert release.plugin_name == "test-plugin"
        assert release.version == "1.0.0"
        assert release.checksum == "abc123"

    def test_plugin_listing_creation(self):
        """Test creating plugin listing."""
        listing = PluginListing(
            id="listing-123",
            name="awesome-plugin",
            display_name="Awesome Plugin",
            description="Does awesome things",
            author="Developer",
            category=PluginCategory.WORKFLOW,
            latest_version="2.0.0",
            downloads=1000,
            rating=4.5,
            verified=True,
        )
        assert listing.name == "awesome-plugin"
        assert listing.downloads == 1000
        assert listing.rating == 4.5
        assert listing.verified is True

    def test_plugin_listing_defaults(self):
        """Test plugin listing default values."""
        listing = PluginListing(
            id="id",
            name="plugin",
            display_name="Plugin",
            latest_version="1.0.0",
        )
        assert listing.downloads == 0
        assert listing.rating == 0.0
        assert listing.review_count == 0
        assert listing.verified is False
        assert listing.featured is False

    def test_plugin_installation_creation(self):
        """Test creating plugin installation."""
        installation = PluginInstallation(
            id="inst-123",
            plugin_name="my-plugin",
            version="1.0.0",
            local_path="/path/to/plugin",
            source=PluginSource.MARKETPLACE,
            enabled=True,
            config={"key": "value"},
        )
        assert installation.plugin_name == "my-plugin"
        assert installation.enabled is True
        assert installation.config == {"key": "value"}

    def test_plugin_search_result(self):
        """Test search result structure."""
        result = PluginSearchResult(
            query="test",
            total=100,
            page=2,
            per_page=20,
            results=[],
        )
        assert result.query == "test"
        assert result.total == 100
        assert result.page == 2

    def test_plugin_install_request(self):
        """Test install request."""
        request = PluginInstallRequest(
            name="plugin",
            version="1.0.0",
            source=PluginSource.GITHUB,
            source_url="https://github.com/user/repo",
            config={"api_key": "secret"},
        )
        assert request.name == "plugin"
        assert request.source == PluginSource.GITHUB
        assert request.enable is True  # default

    def test_plugin_update_request(self):
        """Test update request."""
        request = PluginUpdateRequest(
            name="plugin",
            version="2.0.0",
            config={"new_setting": True},
        )
        assert request.name == "plugin"
        assert request.version == "2.0.0"


class TestPluginMarketplace:
    """Tests for PluginMarketplace class."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock database backend."""
        backend = MagicMock()
        backend.execute.return_value = []
        backend.transaction.return_value.__enter__ = MagicMock()
        backend.transaction.return_value.__exit__ = MagicMock()
        return backend

    @pytest.fixture
    def marketplace(self, mock_backend):
        """Create marketplace with mock backend."""
        from animus_forge.plugins.marketplace import PluginMarketplace

        return PluginMarketplace(mock_backend)

    def test_marketplace_init(self, marketplace, mock_backend):
        """Test marketplace initialization creates schema."""
        assert marketplace.backend == mock_backend
        mock_backend.transaction.assert_called()

    def test_search_empty_results(self, marketplace, mock_backend):
        """Test search with no results."""
        mock_backend.execute.return_value = []
        result = marketplace.search("nonexistent")
        assert result.query == "nonexistent"
        assert result.total == 0
        assert result.results == []

    def test_search_with_category(self, marketplace, mock_backend):
        """Test search filtered by category."""
        mock_backend.execute.return_value = []
        result = marketplace.search("test", category=PluginCategory.INTEGRATION)
        assert result.query == "test"
        # Verify category filter was applied (check SQL params)
        calls = mock_backend.execute.call_args_list
        assert len(calls) >= 1

    def test_get_featured(self, marketplace, mock_backend):
        """Test getting featured plugins."""
        mock_backend.execute.return_value = []
        result = marketplace.get_featured(limit=5)
        assert result == []
        # Verify SQL was called
        mock_backend.execute.assert_called()

    def test_get_popular(self, marketplace, mock_backend):
        """Test getting popular plugins."""
        mock_backend.execute.return_value = []
        result = marketplace.get_popular(limit=10)
        assert result == []

    def test_get_plugin_not_found(self, marketplace, mock_backend):
        """Test getting non-existent plugin."""
        mock_backend.execute.return_value = []
        result = marketplace.get_plugin("nonexistent")
        assert result is None

    def test_get_releases(self, marketplace, mock_backend):
        """Test getting plugin releases."""
        mock_backend.execute.return_value = []
        result = marketplace.get_releases("plugin")
        assert result == []

    def test_add_plugin(self, marketplace, mock_backend):
        """Test adding plugin to catalog."""
        listing = PluginListing(
            id=str(uuid.uuid4()),
            name="new-plugin",
            display_name="New Plugin",
            latest_version="1.0.0",
        )
        result = marketplace.add_plugin(listing)
        assert result is True
        mock_backend.execute.assert_called()

    def test_add_release(self, marketplace, mock_backend):
        """Test adding release to catalog."""
        release = PluginRelease(
            id=str(uuid.uuid4()),
            plugin_name="plugin",
            version="1.0.0",
            download_url="https://example.com/plugin.py",
            checksum="abc123",
        )
        result = marketplace.add_release(release)
        assert result is True

    def test_update_plugin(self, marketplace, mock_backend):
        """Test updating plugin."""
        result = marketplace.update_plugin("plugin", downloads=100, rating=4.5)
        assert result is True

    def test_increment_downloads(self, marketplace, mock_backend):
        """Test incrementing download count."""
        result = marketplace.increment_downloads("plugin")
        assert result is True

    def test_get_categories(self, marketplace, mock_backend):
        """Test getting category counts."""
        mock_backend.execute.return_value = [("integration", 5), ("workflow", 3)]
        result = marketplace.get_categories()
        assert result == {"integration": 5, "workflow": 3}


class TestPluginInstaller:
    """Tests for PluginInstaller class."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock database backend."""
        backend = MagicMock()
        backend.execute.return_value = []
        backend.transaction.return_value.__enter__ = MagicMock()
        backend.transaction.return_value.__exit__ = MagicMock()
        return backend

    @pytest.fixture
    def temp_plugins_dir(self):
        """Create temporary plugins directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def installer(self, mock_backend, temp_plugins_dir):
        """Create installer with mock backend."""
        from animus_forge.plugins.installer import PluginInstaller

        return PluginInstaller(mock_backend, temp_plugins_dir)

    def test_installer_init(self, installer, mock_backend, temp_plugins_dir):
        """Test installer initialization."""
        assert installer.backend == mock_backend
        assert installer.plugins_dir == temp_plugins_dir
        assert temp_plugins_dir.exists()

    def test_get_installation_not_found(self, installer, mock_backend):
        """Test getting non-existent installation."""
        mock_backend.execute.return_value = []
        result = installer.get_installation("nonexistent")
        assert result is None

    def test_list_installations_empty(self, installer, mock_backend):
        """Test listing with no installations."""
        mock_backend.execute.return_value = []
        result = installer.list_installations()
        assert result == []

    def test_list_installations_enabled_only(self, installer, mock_backend):
        """Test listing enabled installations only."""
        mock_backend.execute.return_value = []
        result = installer.list_installations(enabled_only=True)
        assert result == []
        # Verify SQL includes WHERE enabled = 1
        call_args = mock_backend.execute.call_args
        assert "enabled = 1" in call_args[0][0]

    def test_uninstall_not_installed(self, installer, mock_backend):
        """Test uninstalling non-existent plugin."""
        mock_backend.execute.return_value = []
        result = installer.uninstall("nonexistent")
        assert result is False

    def test_enable_not_installed(self, installer, mock_backend):
        """Test enabling non-existent plugin."""
        mock_backend.execute.return_value = []
        result = installer.enable("nonexistent")
        assert result is False

    def test_disable_not_installed(self, installer, mock_backend):
        """Test disabling non-existent plugin."""
        mock_backend.execute.return_value = []
        result = installer.disable("nonexistent")
        assert result is False

    def test_compute_checksum(self, installer, temp_plugins_dir):
        """Test computing file checksum."""
        test_file = temp_plugins_dir / "test.txt"
        test_file.write_text("test content")
        checksum = installer._compute_checksum(test_file)
        assert len(checksum) == 64  # SHA256 hex length
        # Same content should produce same hash
        assert checksum == installer._compute_checksum(test_file)

    def test_cleanup_plugin_dir(self, installer, temp_plugins_dir):
        """Test cleaning up plugin directory."""
        plugin_dir = temp_plugins_dir / "test-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.py").write_text("# plugin")
        installer._cleanup_plugin_dir("test-plugin")
        assert not plugin_dir.exists()

    def test_backup_and_restore(self, installer, temp_plugins_dir):
        """Test backup and restore functionality."""
        # Create plugin directory
        plugin_dir = temp_plugins_dir / "test-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.py").write_text("original content")

        # Backup
        backup_path = installer._backup_plugin("test-plugin")
        assert backup_path is not None
        assert backup_path.exists()

        # Modify original
        (plugin_dir / "plugin.py").write_text("modified content")

        # Restore
        installer._restore_backup("test-plugin", backup_path)
        assert (plugin_dir / "plugin.py").read_text() == "original content"

    def test_load_enabled_plugins(self, installer, mock_backend):
        """Test loading enabled plugins."""
        mock_backend.execute.return_value = []
        loaded = installer.load_enabled_plugins()
        assert loaded == 0


class TestPluginIntegration:
    """Integration tests for plugin marketplace."""

    def test_metadata_to_json_roundtrip(self):
        """Test metadata serialization roundtrip."""
        metadata = PluginMetadata(
            name="test",
            version="1.0.0",
            description="Test plugin",
            author="Author",
            tags=["tag1", "tag2"],
        )
        json_str = metadata.model_dump_json()
        restored = PluginMetadata.model_validate_json(json_str)
        assert restored.name == metadata.name
        assert restored.tags == metadata.tags

    def test_listing_with_releases(self):
        """Test listing with multiple releases."""
        releases = [
            PluginRelease(
                id="r1",
                plugin_name="plugin",
                version="1.0.0",
                download_url="url1",
                checksum="hash1",
            ),
            PluginRelease(
                id="r2",
                plugin_name="plugin",
                version="1.1.0",
                download_url="url2",
                checksum="hash2",
            ),
        ]
        listing = PluginListing(
            id="l1",
            name="plugin",
            display_name="Plugin",
            latest_version="1.1.0",
            releases=releases,
        )
        assert len(listing.releases) == 2
        assert listing.releases[0].version == "1.0.0"

    def test_installation_with_config(self):
        """Test installation with complex config."""
        installation = PluginInstallation(
            id="i1",
            plugin_name="plugin",
            version="1.0.0",
            local_path="/path",
            config={
                "api_key": "secret",
                "settings": {"nested": True, "count": 5},
            },
        )
        assert installation.config["api_key"] == "secret"
        assert installation.config["settings"]["nested"] is True
