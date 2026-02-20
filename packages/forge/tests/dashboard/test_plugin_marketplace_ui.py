"""Tests for the plugin marketplace UI."""

from unittest.mock import MagicMock, patch

import pytest

from animus_forge.dashboard.plugin_marketplace import (
    CATEGORY_CONFIG,
    SAMPLE_INSTALLED,
    SAMPLE_PLUGINS,
    _get_filtered_plugins,
    _get_installed_version,
    _init_marketplace_state,
    _is_installed,
    _render_category_sidebar,
    _render_installed_plugins,
    _render_plugin_card,
    _render_plugin_details,
    _render_rating_stars,
    render_plugin_marketplace,
)
from animus_forge.plugins.models import PluginCategory


class TestCategoryConfig:
    """Test category configuration."""

    def test_all_categories_have_config(self):
        """All plugin categories should have config."""
        for category in PluginCategory:
            assert category in CATEGORY_CONFIG, f"Missing config for {category}"

    def test_config_has_required_fields(self):
        """Each config should have required fields."""
        for category, config in CATEGORY_CONFIG.items():
            assert "icon" in config, f"{category} missing icon"
            assert "color" in config, f"{category} missing color"
            assert "label" in config, f"{category} missing label"


class TestSamplePlugins:
    """Test sample plugin data."""

    def test_sample_plugins_exist(self):
        """Should have sample plugins for demonstration."""
        assert len(SAMPLE_PLUGINS) >= 5

    def test_sample_plugin_structure(self):
        """Each sample plugin should have required fields."""
        required_fields = [
            "id",
            "name",
            "display_name",
            "description",
            "author",
            "category",
            "tags",
            "downloads",
            "rating",
            "latest_version",
        ]
        for plugin in SAMPLE_PLUGINS:
            for field in required_fields:
                assert field in plugin, f"Plugin {plugin.get('name', '?')} missing {field}"

    def test_sample_plugins_have_valid_categories(self):
        """All sample plugins should have valid categories."""
        for plugin in SAMPLE_PLUGINS:
            assert isinstance(plugin["category"], PluginCategory)

    def test_sample_plugins_have_valid_ratings(self):
        """Ratings should be between 0 and 5."""
        for plugin in SAMPLE_PLUGINS:
            assert 0 <= plugin["rating"] <= 5


class TestMarketplaceState:
    """Test marketplace state management."""

    def test_init_marketplace_state(self, mock_session_state):
        """Should initialize all required state variables."""
        _init_marketplace_state()

        assert hasattr(mock_session_state, "marketplace_search")
        assert hasattr(mock_session_state, "marketplace_category")
        assert hasattr(mock_session_state, "marketplace_selected_plugin")
        assert hasattr(mock_session_state, "installed_plugins")

    def test_get_filtered_plugins_no_filter(self, mock_session_state):
        """Should return all plugins with no filter."""
        mock_session_state.marketplace_search = ""
        mock_session_state.marketplace_category = "all"

        filtered = _get_filtered_plugins()
        assert len(filtered) == len(SAMPLE_PLUGINS)

    def test_get_filtered_plugins_by_category(self, mock_session_state):
        """Should filter plugins by category."""
        mock_session_state.marketplace_search = ""
        mock_session_state.marketplace_category = "integration"

        filtered = _get_filtered_plugins()
        for plugin in filtered:
            assert plugin["category"].value == "integration"

    def test_get_filtered_plugins_by_search(self, mock_session_state):
        """Should filter plugins by search query."""
        mock_session_state.marketplace_search = "github"
        mock_session_state.marketplace_category = "all"

        filtered = _get_filtered_plugins()
        assert len(filtered) >= 1
        # At least one result should contain "github"
        assert any("github" in p["name"].lower() for p in filtered)


class TestInstallationHelpers:
    """Test installation helper functions."""

    def test_is_installed_true(self, mock_session_state):
        """Should return True for installed plugins."""
        mock_session_state.installed_plugins = {"test-plugin": {"version": "1.0.0"}}
        assert _is_installed("test-plugin") is True

    def test_is_installed_false(self, mock_session_state):
        """Should return False for non-installed plugins."""
        mock_session_state.installed_plugins = {}
        assert _is_installed("test-plugin") is False

    def test_get_installed_version(self, mock_session_state):
        """Should return installed version."""
        mock_session_state.installed_plugins = {"test-plugin": {"version": "2.1.0"}}
        assert _get_installed_version("test-plugin") == "2.1.0"

    def test_get_installed_version_not_installed(self, mock_session_state):
        """Should return None for non-installed plugins."""
        mock_session_state.installed_plugins = {}
        assert _get_installed_version("test-plugin") is None


class TestRatingStars:
    """Test rating stars rendering."""

    def test_render_rating_stars_full(self):
        """Should render full stars correctly."""
        result = _render_rating_stars(5.0)
        assert "★★★★★" in result
        assert "(5.0)" in result

    def test_render_rating_stars_partial(self):
        """Should handle partial ratings."""
        result = _render_rating_stars(4.5)
        assert "★★★★" in result
        assert "½" in result
        assert "(4.5)" in result

    def test_render_rating_stars_zero(self):
        """Should handle zero rating."""
        result = _render_rating_stars(0.0)
        assert "☆☆☆☆☆" in result
        assert "(0.0)" in result


@pytest.fixture
def mock_session_state(monkeypatch):
    """Mock Streamlit session state."""

    class MockSessionState:
        def __init__(self):
            self.marketplace_search = ""
            self.marketplace_category = "all"
            self.marketplace_selected_plugin = None
            self.installed_plugins = {}

        def __contains__(self, key):
            return hasattr(self, key)

    mock_state = MockSessionState()

    import streamlit as st

    monkeypatch.setattr(st, "session_state", mock_state)

    return mock_state


class _RerunCalled(Exception):
    """Sentinel exception to simulate st.rerun() halting execution."""


def _make_col_mocks(n=3):
    """Create context-manager column mocks for st.columns."""
    cols = [MagicMock() for _ in range(n)]
    for cm in cols:
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
    return cols


def _columns_side_effect(spec):
    """Dynamic st.columns mock that returns correct number of column mocks."""
    n = len(spec) if isinstance(spec, (list, tuple)) else spec
    return _make_col_mocks(n)


class TestRenderPluginCardActions:
    """Test button actions inside _render_plugin_card."""

    def test_details_button_selects_plugin(self, mock_session_state):
        """Details button should set marketplace_selected_plugin and rerun."""
        mock_session_state.installed_plugins = {}
        plugin = SAMPLE_PLUGINS[0]

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(3)
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.button.side_effect = lambda label, **kwargs: kwargs.get("key", "").startswith(
                "details_"
            )

            with pytest.raises(_RerunCalled):
                _render_plugin_card(plugin)

            assert mock_session_state.marketplace_selected_plugin == plugin["name"]

    def test_disable_button_for_enabled_plugin(self, mock_session_state):
        """Disable button should set enabled=False, toast, and rerun."""
        plugin = SAMPLE_PLUGINS[0]
        mock_session_state.installed_plugins = {
            plugin["name"]: {
                "plugin_name": plugin["name"],
                "version": plugin["latest_version"],
                "enabled": True,
            }
        }

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(3)
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.button.side_effect = lambda label, **kwargs: kwargs.get("key", "").startswith(
                "disable_"
            )

            with pytest.raises(_RerunCalled):
                _render_plugin_card(plugin)

            assert mock_session_state.installed_plugins[plugin["name"]]["enabled"] is False
            mock_st.toast.assert_called_once()

    def test_enable_button_for_disabled_plugin(self, mock_session_state):
        """Enable button should set enabled=True, toast, and rerun."""
        plugin = SAMPLE_PLUGINS[0]
        mock_session_state.installed_plugins = {
            plugin["name"]: {
                "plugin_name": plugin["name"],
                "version": plugin["latest_version"],
                "enabled": False,
            }
        }

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(3)
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.button.side_effect = lambda label, **kwargs: kwargs.get("key", "").startswith(
                "enable_"
            )

            with pytest.raises(_RerunCalled):
                _render_plugin_card(plugin)

            assert mock_session_state.installed_plugins[plugin["name"]]["enabled"] is True
            mock_st.toast.assert_called_once()

    def test_install_button_for_not_installed_plugin(self, mock_session_state):
        """Install button should create installation entry, toast, and rerun."""
        plugin = SAMPLE_PLUGINS[0]
        mock_session_state.installed_plugins = {}

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(3)
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.button.side_effect = lambda label, **kwargs: kwargs.get("key", "").startswith(
                "install_"
            )

            with pytest.raises(_RerunCalled):
                _render_plugin_card(plugin)

            assert plugin["name"] in mock_session_state.installed_plugins
            installed = mock_session_state.installed_plugins[plugin["name"]]
            assert installed["version"] == plugin["latest_version"]
            assert installed["enabled"] is True
            assert installed["source"] == "marketplace"
            mock_st.toast.assert_called_once()


class TestRenderPluginDetailsActions:
    """Test button actions inside _render_plugin_details."""

    def test_back_button_clears_selection(self, mock_session_state):
        """Back button should clear selected_plugin and rerun."""
        plugin = SAMPLE_PLUGINS[0]
        mock_session_state.marketplace_selected_plugin = plugin["name"]
        mock_session_state.installed_plugins = {}

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(2)
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.button.side_effect = lambda label, **kwargs: "Back" in str(label)

            with pytest.raises(_RerunCalled):
                _render_plugin_details(plugin["name"])

            assert mock_session_state.marketplace_selected_plugin is None

    def test_update_button_updates_version(self, mock_session_state):
        """Update button should update version in installed_plugins."""
        plugin = SAMPLE_PLUGINS[0]
        old_version = "1.0.0"
        mock_session_state.installed_plugins = {
            plugin["name"]: {
                "plugin_name": plugin["name"],
                "version": old_version,
                "enabled": True,
            }
        }

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(2)
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.button.side_effect = lambda label, **kwargs: "Update" in str(label)

            with pytest.raises(_RerunCalled):
                _render_plugin_details(plugin["name"])

            assert (
                mock_session_state.installed_plugins[plugin["name"]]["version"]
                == plugin["latest_version"]
            )
            mock_st.toast.assert_called()

    def test_uninstall_button_removes_plugin(self, mock_session_state):
        """Uninstall button should remove plugin from installed_plugins."""
        plugin = SAMPLE_PLUGINS[0]
        mock_session_state.installed_plugins = {
            plugin["name"]: {
                "plugin_name": plugin["name"],
                "version": plugin["latest_version"],
                "enabled": True,
            }
        }

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(2)
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.button.side_effect = lambda label, **kwargs: "Uninstall" in str(label)

            with pytest.raises(_RerunCalled):
                _render_plugin_details(plugin["name"])

            assert plugin["name"] not in mock_session_state.installed_plugins
            mock_st.toast.assert_called()

    def test_install_button_installs_plugin(self, mock_session_state):
        """Install Plugin button should create installation entry."""
        plugin = SAMPLE_PLUGINS[0]
        mock_session_state.installed_plugins = {}

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(2)
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.button.side_effect = lambda label, **kwargs: "Install" in str(label)

            with pytest.raises(_RerunCalled):
                _render_plugin_details(plugin["name"])

            assert plugin["name"] in mock_session_state.installed_plugins
            installed = mock_session_state.installed_plugins[plugin["name"]]
            assert installed["version"] == plugin["latest_version"]
            assert installed["enabled"] is True
            mock_st.toast.assert_called()


class TestRenderInstalledPluginsActions:
    """Test button actions inside _render_installed_plugins."""

    def _setup_installed(self, mock_session_state, enabled=True, old_version=False):
        """Set up installed_plugins with a known plugin."""
        plugin = SAMPLE_PLUGINS[0]
        version = "1.0.0" if old_version else plugin["latest_version"]
        mock_session_state.installed_plugins = {
            plugin["name"]: {
                "id": f"inst-{plugin['name']}",
                "plugin_name": plugin["name"],
                "version": version,
                "enabled": enabled,
            }
        }
        return plugin

    def test_disable_installed_plugin(self, mock_session_state):
        """Disable button in installed list should set enabled=False."""
        plugin = self._setup_installed(mock_session_state, enabled=True)

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(3)
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.button.side_effect = lambda label, **kwargs: kwargs.get("key", "").startswith(
                "inst_disable_"
            )

            with pytest.raises(_RerunCalled):
                _render_installed_plugins()

            assert mock_session_state.installed_plugins[plugin["name"]]["enabled"] is False

    def test_enable_installed_plugin(self, mock_session_state):
        """Enable button in installed list should set enabled=True."""
        plugin = self._setup_installed(mock_session_state, enabled=False)

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(3)
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.button.side_effect = lambda label, **kwargs: kwargs.get("key", "").startswith(
                "inst_enable_"
            )

            with pytest.raises(_RerunCalled):
                _render_installed_plugins()

            assert mock_session_state.installed_plugins[plugin["name"]]["enabled"] is True

    def test_update_installed_plugin(self, mock_session_state):
        """Update button in installed list should update version."""
        plugin = self._setup_installed(mock_session_state, enabled=True, old_version=True)

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(3)
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.button.side_effect = lambda label, **kwargs: kwargs.get("key", "").startswith(
                "inst_update_"
            )

            with pytest.raises(_RerunCalled):
                _render_installed_plugins()

            assert (
                mock_session_state.installed_plugins[plugin["name"]]["version"]
                == plugin["latest_version"]
            )
            mock_st.toast.assert_called()

    def test_uninstall_installed_plugin(self, mock_session_state):
        """Uninstall button in installed list should remove plugin."""
        plugin = self._setup_installed(mock_session_state, enabled=True)

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(3)
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.button.side_effect = lambda label, **kwargs: kwargs.get("key", "").startswith(
                "inst_uninstall_"
            )

            with pytest.raises(_RerunCalled):
                _render_installed_plugins()

            assert plugin["name"] not in mock_session_state.installed_plugins
            mock_st.toast.assert_called()


class TestRenderCategorySidebar:
    """Test button actions inside _render_category_sidebar."""

    def test_all_category_button(self, mock_session_state):
        """All category button should set category to 'all' and rerun."""
        mock_session_state.marketplace_category = "integration"

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.button.side_effect = lambda label, **kwargs: kwargs.get("key", "") == "cat_all"

            with pytest.raises(_RerunCalled):
                _render_category_sidebar()

            assert mock_session_state.marketplace_category == "all"

    def test_individual_category_button(self, mock_session_state):
        """Category button should set category value and rerun."""
        mock_session_state.marketplace_category = "all"
        target_category = PluginCategory.INTEGRATION

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.button.side_effect = lambda label, **kwargs: (
                kwargs.get("key", "") == f"cat_{target_category.value}"
            )

            with pytest.raises(_RerunCalled):
                _render_category_sidebar()

            assert mock_session_state.marketplace_category == target_category.value


class TestRenderPluginMarketplaceSearch:
    """Test search input in render_plugin_marketplace."""

    def test_search_input_updates_state(self, mock_session_state):
        """Search input change should update state and rerun."""
        mock_session_state.marketplace_search = ""
        mock_session_state.marketplace_category = "all"
        mock_session_state.marketplace_selected_plugin = None
        mock_session_state.installed_plugins = {p["plugin_name"]: p for p in SAMPLE_INSTALLED}

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.side_effect = _columns_side_effect
            mock_st.button.return_value = False
            mock_st.rerun.side_effect = _RerunCalled
            mock_st.text_input.return_value = "github"

            with pytest.raises(_RerunCalled):
                render_plugin_marketplace()

            assert mock_session_state.marketplace_search == "github"


class TestInitMarketplaceStateDefaults:
    """Test _init_marketplace_state when attributes are missing."""

    def test_init_sets_defaults_on_empty_state(self, monkeypatch):
        """Should set all defaults when session_state has no marketplace attrs."""

        class EmptyState:
            def __contains__(self, key):
                return hasattr(self, key)

        empty = EmptyState()
        import streamlit as st

        monkeypatch.setattr(st, "session_state", empty)

        _init_marketplace_state()

        assert empty.marketplace_search == ""
        assert empty.marketplace_category == "all"
        assert empty.marketplace_selected_plugin is None
        assert isinstance(empty.installed_plugins, dict)
        assert len(empty.installed_plugins) > 0


class TestRenderPluginDetailsNoAction:
    """Test _render_plugin_details paths not involving button clicks."""

    def test_details_render_full_page_no_buttons(self, mock_session_state):
        """Rendering details without clicking any button covers the full page."""
        plugin = SAMPLE_PLUGINS[0]
        mock_session_state.installed_plugins = {}

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(2)
            mock_st.button.return_value = False

            _render_plugin_details(plugin["name"])

            # Verify the page rendered fully (about section, tags, badges)
            markdown_calls = [str(c) for c in mock_st.markdown.call_args_list]
            assert len(markdown_calls) > 0

    def test_details_render_installed_with_update_no_click(self, mock_session_state):
        """Rendering details for installed plugin with update available."""
        plugin = SAMPLE_PLUGINS[0]
        mock_session_state.installed_plugins = {
            plugin["name"]: {
                "plugin_name": plugin["name"],
                "version": "1.0.0",
                "enabled": True,
            }
        }

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(2)
            mock_st.button.return_value = False

            _render_plugin_details(plugin["name"])

            mock_st.success.assert_called_once()

    def test_details_no_badges_plugin(self, mock_session_state):
        """Plugin without verified/featured should show 'No special badges'."""
        # s3-storage has verified=False and featured=False
        plugin = next(p for p in SAMPLE_PLUGINS if p["name"] == "s3-storage")
        mock_session_state.installed_plugins = {}

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(2)
            mock_st.button.return_value = False

            _render_plugin_details(plugin["name"])

            mock_st.info.assert_called_once_with("No special badges")

    def test_plugin_not_found(self, mock_session_state):
        """Should show error when plugin name doesn't match any sample."""
        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state

            _render_plugin_details("nonexistent-plugin")

            mock_st.error.assert_called_once_with("Plugin not found")


class TestRenderInstalledPluginsEdgeCases:
    """Test edge cases in _render_installed_plugins."""

    def test_empty_installed_plugins(self, mock_session_state):
        """Should render empty state when no plugins installed."""
        mock_session_state.installed_plugins = {}

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state

            _render_installed_plugins()

            # Verify the "no plugins" message was rendered
            assert mock_st.markdown.call_count >= 2  # header + empty message

    def test_unknown_plugin_in_installed_skipped(self, mock_session_state):
        """Installed plugin not in SAMPLE_PLUGINS should be skipped."""
        mock_session_state.installed_plugins = {
            "unknown-plugin": {
                "id": "inst-unknown",
                "plugin_name": "unknown-plugin",
                "version": "1.0.0",
                "enabled": True,
            }
        }

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state

            _render_installed_plugins()

            # Only the header markdown should be called, no plugin cards
            mock_st.columns.assert_not_called()


class TestRenderPluginMarketplaceFullRender:
    """Test render_plugin_marketplace paths beyond search."""

    def test_renders_selected_plugin_details(self, mock_session_state):
        """When a plugin is selected, should render details view."""
        plugin = SAMPLE_PLUGINS[0]
        mock_session_state.marketplace_search = ""
        mock_session_state.marketplace_category = "all"
        mock_session_state.marketplace_selected_plugin = plugin["name"]
        mock_session_state.installed_plugins = {}

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.side_effect = _columns_side_effect
            mock_st.button.return_value = False

            render_plugin_marketplace()

            mock_st.title.assert_called_once()
            mock_st.error.assert_not_called()

    def test_renders_main_marketplace_view(self, mock_session_state):
        """Full marketplace render without search change covers main content."""
        mock_session_state.marketplace_search = ""
        mock_session_state.marketplace_category = "all"
        mock_session_state.marketplace_selected_plugin = None
        mock_session_state.installed_plugins = {p["plugin_name"]: p for p in SAMPLE_INSTALLED}

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.side_effect = _columns_side_effect
            mock_st.button.return_value = False
            mock_st.text_input.return_value = ""

            render_plugin_marketplace()

            mock_st.title.assert_called_once()
            assert mock_st.markdown.call_count > 0
            mock_st.divider.assert_called()

    def test_renders_no_results_message(self, mock_session_state):
        """Should render no-results message when filter matches nothing."""
        mock_session_state.marketplace_search = "zzz_nonexistent_query_zzz"
        mock_session_state.marketplace_category = "all"
        mock_session_state.marketplace_selected_plugin = None
        mock_session_state.installed_plugins = {}

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.side_effect = _columns_side_effect
            mock_st.button.return_value = False
            mock_st.text_input.return_value = "zzz_nonexistent_query_zzz"

            render_plugin_marketplace()

            found_no_results = any(
                "No Plugins Found" in str(c) for c in mock_st.markdown.call_args_list
            )
            assert found_no_results


class TestRenderPluginCardUpdateBadge:
    """Test _render_plugin_card update-available badge path."""

    def test_update_available_badge_shown(self, mock_session_state):
        """Should show update badge when installed version differs from latest."""
        plugin = SAMPLE_PLUGINS[0]
        mock_session_state.installed_plugins = {
            plugin["name"]: {
                "plugin_name": plugin["name"],
                "version": "1.0.0",  # Older than latest
                "enabled": True,
            }
        }

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.columns.return_value = _make_col_mocks(3)
            mock_st.button.return_value = False

            _render_plugin_card(plugin)

            # Verify "Update Available" badge in markdown
            found_update = any(
                "Update Available" in str(c) for c in mock_st.markdown.call_args_list
            )
            assert found_update


class TestRenderCategorySidebarSkipEmpty:
    """Test _render_category_sidebar skipping empty categories."""

    def test_empty_categories_skipped(self, mock_session_state):
        """Categories with 0 plugins should not render buttons."""
        mock_session_state.marketplace_category = "all"

        with patch("animus_forge.dashboard.plugin_marketplace.st") as mock_st:
            mock_st.session_state = mock_session_state
            mock_st.button.return_value = False

            _render_category_sidebar()

            # Count button calls - should NOT include categories with 0 plugins
            button_keys = [
                call.kwargs.get("key", "")
                for call in mock_st.button.call_args_list
                if call.kwargs.get("key", "").startswith("cat_")
            ]
            # Should have "cat_all" + only categories that have plugins
            from collections import Counter

            category_counts = Counter(p["category"].value for p in SAMPLE_PLUGINS)
            expected_cat_buttons = 1 + len(category_counts)  # "all" + populated
            assert len(button_keys) == expected_cat_buttons
