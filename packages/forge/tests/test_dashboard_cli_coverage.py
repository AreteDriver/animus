"""Tests for dashboard plugin_marketplace, CLI calendar_cmd, bot, and browser commands.

Covers missed lines in:
- src/animus_forge/dashboard/plugin_marketplace.py (168 missed lines)
- src/animus_forge/cli/commands/calendar_cmd.py (168 missed lines)
- src/animus_forge/cli/commands/bot.py (136 missed lines)
- src/animus_forge/cli/commands/browser.py (99 missed lines)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from animus_forge.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Streamlit mock helpers (same pattern as test_dashboard.py / test_cost_dashboard.py)
# ---------------------------------------------------------------------------


def _create_context_manager():
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_cm)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm


def _create_columns(n):
    count = n if isinstance(n, int) else len(n)
    return [_create_context_manager() for _ in range(count)]


def _create_tabs(labels):
    return [_create_context_manager() for _ in labels]


def _create_expander(label, **kwargs):
    return _create_context_manager()


class SessionState(dict):
    """Dict-like session state that also supports attribute access."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value):
        self[name] = value

    def __delattr__(self, name: str):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)


_MARKETPLACE_MODULES = [
    "animus_forge.dashboard.plugin_marketplace",
]


@pytest.fixture()
def mock_streamlit():
    """Mock streamlit for plugin marketplace tests."""
    for mod_name in list(sys.modules.keys()):
        if mod_name in _MARKETPLACE_MODULES:
            del sys.modules[mod_name]

    mock_st = MagicMock()
    mock_st.session_state = SessionState()
    mock_st.cache_resource = lambda f: f
    mock_st.columns.side_effect = _create_columns
    mock_st.tabs.side_effect = _create_tabs
    mock_st.expander.side_effect = _create_expander
    mock_st.button.return_value = False  # Default: no buttons pressed
    # text_input must return a real string for the search filter
    mock_st.text_input.return_value = ""

    mock_submodules = {
        "streamlit": mock_st,
        "streamlit.emojis": MagicMock(),
        "streamlit.components": MagicMock(),
        "streamlit.components.v1": MagicMock(),
        "streamlit.runtime": MagicMock(),
        "streamlit.runtime.scriptrunner": MagicMock(),
        "streamlit.web": MagicMock(),
        "streamlit.delta_generator": MagicMock(),
    }

    with patch.dict(sys.modules, mock_submodules):
        yield mock_st

    # Clean up cached module
    for mod_name in list(sys.modules.keys()):
        if mod_name in _MARKETPLACE_MODULES:
            sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# CalendarEvent helper for calendar CLI tests
# ---------------------------------------------------------------------------


@dataclass
class FakeCalendarEvent:
    id: str | None = None
    summary: str = ""
    description: str = ""
    location: str = ""
    start: datetime | None = None
    end: datetime | None = None
    all_day: bool = False
    html_link: str = ""
    attendees: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Async context manager helper for browser CLI tests
# ---------------------------------------------------------------------------


@dataclass
class FakePageResult:
    success: bool = True
    data: dict | None = None
    error: str | None = None
    screenshot_path: str | None = None
    url: str | None = None
    title: str | None = None


class AsyncBrowserMock:
    """A proper async context manager mock for BrowserAutomation."""

    def __init__(self):
        self._navigate_result: FakePageResult | None = None
        self._extract_result: FakePageResult | None = None
        self._screenshot_result: FakePageResult | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def navigate(self, url, **kwargs):
        return self._navigate_result

    async def extract_content(self, **kwargs):
        return self._extract_result

    async def screenshot(self, path=None, full_page=False, **kwargs):
        return self._screenshot_result


def _make_browser_mock(
    nav_result: FakePageResult | None = None,
    extract_result: FakePageResult | None = None,
    screenshot_result: FakePageResult | None = None,
):
    """Create a browser mock with the specified results and a patched module."""
    browser = AsyncBrowserMock()
    browser._navigate_result = nav_result
    browser._extract_result = extract_result
    browser._screenshot_result = screenshot_result

    mock_module = MagicMock()
    mock_module.BrowserAutomation.return_value = browser
    mock_module.BrowserConfig.return_value = MagicMock()
    return mock_module


# ==========================================================================
#  PLUGIN MARKETPLACE TESTS
# ==========================================================================


class TestRenderPluginCard:
    """Test _render_plugin_card rendering."""

    def test_render_plugin_card_not_installed(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            SAMPLE_PLUGINS,
            _init_marketplace_state,
            _render_plugin_card,
        )

        _init_marketplace_state()
        mock_streamlit.session_state.installed_plugins = {}
        _render_plugin_card(SAMPLE_PLUGINS[0])
        mock_streamlit.markdown.assert_called()

    def test_render_plugin_card_installed_enabled(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            SAMPLE_PLUGINS,
            _init_marketplace_state,
            _render_plugin_card,
        )

        _init_marketplace_state()
        plugin = SAMPLE_PLUGINS[0]
        mock_streamlit.session_state.installed_plugins = {
            plugin["name"]: {
                "plugin_name": plugin["name"],
                "version": plugin["latest_version"],
                "enabled": True,
            }
        }
        _render_plugin_card(plugin)
        mock_streamlit.markdown.assert_called()

    def test_render_plugin_card_installed_disabled(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            SAMPLE_PLUGINS,
            _init_marketplace_state,
            _render_plugin_card,
        )

        _init_marketplace_state()
        plugin = SAMPLE_PLUGINS[0]
        mock_streamlit.session_state.installed_plugins = {
            plugin["name"]: {
                "plugin_name": plugin["name"],
                "version": plugin["latest_version"],
                "enabled": False,
            }
        }
        _render_plugin_card(plugin)
        mock_streamlit.markdown.assert_called()

    def test_render_plugin_card_update_available(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            SAMPLE_PLUGINS,
            _init_marketplace_state,
            _render_plugin_card,
        )

        _init_marketplace_state()
        plugin = SAMPLE_PLUGINS[0]
        mock_streamlit.session_state.installed_plugins = {
            plugin["name"]: {
                "plugin_name": plugin["name"],
                "version": "0.0.1",
                "enabled": True,
            }
        }
        _render_plugin_card(plugin)
        calls = [str(c) for c in mock_streamlit.markdown.call_args_list]
        assert any("Update Available" in c for c in calls)

    def test_render_plugin_card_not_verified_not_featured(self, mock_streamlit):
        """Plugin with no verified/featured badges."""
        from animus_forge.dashboard.plugin_marketplace import (
            SAMPLE_PLUGINS,
            _init_marketplace_state,
            _render_plugin_card,
        )

        _init_marketplace_state()
        mock_streamlit.session_state.installed_plugins = {}
        plugin = next(p for p in SAMPLE_PLUGINS if not p.get("verified") and not p.get("featured"))
        _render_plugin_card(plugin)
        mock_streamlit.markdown.assert_called()


class TestRenderPluginDetails:
    """Test _render_plugin_details rendering."""

    def test_render_details_not_found(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            _init_marketplace_state,
            _render_plugin_details,
        )

        _init_marketplace_state()
        _render_plugin_details("nonexistent-plugin")
        mock_streamlit.error.assert_called_once()

    def test_render_details_not_installed(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            _init_marketplace_state,
            _render_plugin_details,
        )

        _init_marketplace_state()
        mock_streamlit.session_state.installed_plugins = {}
        _render_plugin_details("github-integration")
        mock_streamlit.markdown.assert_called()

    def test_render_details_installed_same_version(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            SAMPLE_PLUGINS,
            _init_marketplace_state,
            _render_plugin_details,
        )

        _init_marketplace_state()
        plugin = next(p for p in SAMPLE_PLUGINS if p["name"] == "github-integration")
        mock_streamlit.session_state.installed_plugins = {
            plugin["name"]: {
                "plugin_name": plugin["name"],
                "version": plugin["latest_version"],
                "enabled": True,
            }
        }
        _render_plugin_details("github-integration")
        mock_streamlit.success.assert_called()

    def test_render_details_installed_update_available(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            _init_marketplace_state,
            _render_plugin_details,
        )

        _init_marketplace_state()
        mock_streamlit.session_state.installed_plugins = {
            "github-integration": {
                "plugin_name": "github-integration",
                "version": "0.0.1",
                "enabled": True,
            }
        }
        _render_plugin_details("github-integration")
        mock_streamlit.success.assert_called()

    def test_render_details_verified_and_featured_badges(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            _init_marketplace_state,
            _render_plugin_details,
        )

        _init_marketplace_state()
        mock_streamlit.session_state.installed_plugins = {}
        _render_plugin_details("github-integration")  # verified=True, featured=True
        calls = [str(c) for c in mock_streamlit.markdown.call_args_list]
        assert any("Verified" in c for c in calls)

    def test_render_details_no_badges(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            _init_marketplace_state,
            _render_plugin_details,
        )

        _init_marketplace_state()
        mock_streamlit.session_state.installed_plugins = {}
        _render_plugin_details("s3-storage")
        mock_streamlit.info.assert_called()


class TestRenderInstalledPlugins:
    """Test _render_installed_plugins rendering."""

    def test_render_installed_empty(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            _init_marketplace_state,
            _render_installed_plugins,
        )

        _init_marketplace_state()
        mock_streamlit.session_state.installed_plugins = {}
        _render_installed_plugins()
        calls = [str(c) for c in mock_streamlit.markdown.call_args_list]
        assert any("No Plugins Installed" in c for c in calls)

    def test_render_installed_with_plugins(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            _init_marketplace_state,
            _render_installed_plugins,
        )

        _init_marketplace_state()
        _render_installed_plugins()
        mock_streamlit.markdown.assert_called()

    def test_render_installed_with_disabled_plugin(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            _init_marketplace_state,
            _render_installed_plugins,
        )

        _init_marketplace_state()
        mock_streamlit.session_state.installed_plugins = {
            "github-integration": {
                "plugin_name": "github-integration",
                "version": "2.1.0",
                "enabled": False,
            }
        }
        _render_installed_plugins()
        calls = [str(c) for c in mock_streamlit.markdown.call_args_list]
        assert any("Disabled" in c for c in calls)

    def test_render_installed_with_update_available(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            _init_marketplace_state,
            _render_installed_plugins,
        )

        _init_marketplace_state()
        mock_streamlit.session_state.installed_plugins = {
            "github-integration": {
                "plugin_name": "github-integration",
                "version": "0.0.1",
                "enabled": True,
            }
        }
        _render_installed_plugins()
        calls = [str(c) for c in mock_streamlit.markdown.call_args_list]
        assert any("Update available" in c for c in calls)

    def test_render_installed_unknown_plugin_skipped(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            _init_marketplace_state,
            _render_installed_plugins,
        )

        _init_marketplace_state()
        mock_streamlit.session_state.installed_plugins = {
            "nonexistent-plugin": {
                "plugin_name": "nonexistent-plugin",
                "version": "1.0.0",
                "enabled": True,
            }
        }
        _render_installed_plugins()


class TestRenderCategorySidebar:
    """Test _render_category_sidebar rendering."""

    def test_render_category_sidebar(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            _init_marketplace_state,
            _render_category_sidebar,
        )

        _init_marketplace_state()
        _render_category_sidebar()
        assert mock_streamlit.button.called

    def test_render_category_sidebar_with_selected(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import (
            _init_marketplace_state,
            _render_category_sidebar,
        )

        _init_marketplace_state()
        mock_streamlit.session_state.marketplace_category = "integration"
        _render_category_sidebar()
        assert mock_streamlit.button.called


class TestRenderPluginMarketplace:
    """Test render_plugin_marketplace main entry point."""

    def test_render_marketplace_main_view(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import render_plugin_marketplace

        # text_input must return a string matching current search state
        mock_streamlit.text_input.return_value = ""
        render_plugin_marketplace()
        mock_streamlit.title.assert_called_once()
        mock_streamlit.markdown.assert_called()

    def test_render_marketplace_with_selected_plugin(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import render_plugin_marketplace

        mock_streamlit.session_state["marketplace_selected_plugin"] = "github-integration"
        mock_streamlit.session_state["marketplace_search"] = ""
        mock_streamlit.session_state["marketplace_category"] = "all"
        mock_streamlit.session_state["installed_plugins"] = {}
        render_plugin_marketplace()
        mock_streamlit.title.assert_called_once()

    def test_render_marketplace_no_results(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import render_plugin_marketplace

        mock_streamlit.session_state["marketplace_search"] = "zzzzzzznotfound"
        mock_streamlit.session_state["marketplace_category"] = "all"
        mock_streamlit.session_state["marketplace_selected_plugin"] = None
        mock_streamlit.session_state["installed_plugins"] = {}
        # text_input returns the search value — but the code compares
        # the return value to the session state; return same value to avoid rerun
        mock_streamlit.text_input.return_value = "zzzzzzznotfound"
        render_plugin_marketplace()
        calls = [str(c) for c in mock_streamlit.markdown.call_args_list]
        assert any("No Plugins Found" in c for c in calls)

    def test_render_marketplace_with_category_filter(self, mock_streamlit):
        from animus_forge.dashboard.plugin_marketplace import render_plugin_marketplace

        mock_streamlit.session_state["marketplace_search"] = ""
        mock_streamlit.session_state["marketplace_category"] = "integration"
        mock_streamlit.session_state["marketplace_selected_plugin"] = None
        mock_streamlit.session_state["installed_plugins"] = {}
        mock_streamlit.text_input.return_value = ""
        render_plugin_marketplace()
        mock_streamlit.title.assert_called_once()

    def test_render_marketplace_featured_section(self, mock_streamlit):
        """Featured section shows when no filters applied."""
        from animus_forge.dashboard.plugin_marketplace import render_plugin_marketplace

        mock_streamlit.session_state["marketplace_search"] = ""
        mock_streamlit.session_state["marketplace_category"] = "all"
        mock_streamlit.session_state["marketplace_selected_plugin"] = None
        mock_streamlit.session_state["installed_plugins"] = {}
        mock_streamlit.text_input.return_value = ""
        render_plugin_marketplace()
        calls = [str(c) for c in mock_streamlit.markdown.call_args_list]
        assert any("Featured" in c for c in calls)


# ==========================================================================
#  CALENDAR CLI TESTS
# ==========================================================================


class TestCalendarList:
    """Test calendar list command."""

    def test_list_import_error(self):
        """calendar list fails gracefully when CalendarClient is missing."""
        with patch.dict(sys.modules, {"animus_forge.api_clients": None}):
            result = runner.invoke(app, ["calendar", "list"])
            assert result.exit_code != 0

    def test_list_auth_failure(self):
        mock_client = MagicMock()
        mock_client.authenticate.return_value = False

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "list"])
            assert result.exit_code != 0

    def test_list_events_success(self):
        now = datetime.now(UTC)
        events = [
            FakeCalendarEvent(
                id="1",
                summary="Meeting",
                start=now,
                end=now + timedelta(hours=1),
                location="Room A",
            ),
            FakeCalendarEvent(
                id="2",
                summary="Lunch",
                start=now + timedelta(hours=2),
                end=now + timedelta(hours=3),
                all_day=True,
            ),
        ]

        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.list_events.return_value = events

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "list"])
            assert result.exit_code == 0
            assert "Meeting" in result.output

    def test_list_events_empty(self):
        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.list_events.return_value = []

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "list"])
            assert result.exit_code == 0
            assert "No events" in result.output

    def test_list_events_json_output(self):
        now = datetime.now(UTC)
        events = [
            FakeCalendarEvent(
                id="1",
                summary="Meeting",
                start=now,
                end=now + timedelta(hours=1),
                location="Room A",
            ),
        ]

        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.list_events.return_value = events

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "list", "--json"])
            assert result.exit_code == 0
            assert "Meeting" in result.output

    def test_list_events_with_location(self):
        now = datetime.now(UTC)
        events = [
            FakeCalendarEvent(
                id="1",
                summary="Offsite",
                start=now,
                end=now + timedelta(hours=1),
                location="123 Main St",
            ),
        ]

        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.list_events.return_value = events

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "list"])
            assert result.exit_code == 0
            assert "123 Main St" in result.output


class TestCalendarToday:
    """Test calendar today command."""

    def test_today_auth_failure(self):
        mock_client = MagicMock()
        mock_client.authenticate.return_value = False

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "today"])
            assert result.exit_code != 0

    def test_today_no_events(self):
        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.get_upcoming_today.return_value = []

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "today"])
            assert result.exit_code == 0
            assert "No more events" in result.output

    def test_today_with_events(self):
        now = datetime.now(UTC)
        events = [
            FakeCalendarEvent(
                id="1", summary="Standup", start=now, end=now + timedelta(minutes=15)
            ),
            FakeCalendarEvent(
                id="2",
                summary="All Day Event",
                start=now,
                end=now + timedelta(days=1),
                all_day=True,
            ),
        ]

        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.get_upcoming_today.return_value = events

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "today"])
            assert result.exit_code == 0
            assert "Standup" in result.output

    def test_today_json_output(self):
        now = datetime.now(UTC)
        events = [
            FakeCalendarEvent(
                id="1", summary="Standup", start=now, end=now + timedelta(minutes=15)
            ),
        ]

        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.get_upcoming_today.return_value = events

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "today", "--json"])
            assert result.exit_code == 0
            assert "Standup" in result.output

    def test_today_import_error(self):
        with patch.dict(sys.modules, {"animus_forge.api_clients": None}):
            result = runner.invoke(app, ["calendar", "today"])
            assert result.exit_code != 0


class TestCalendarAdd:
    """Test calendar add command."""

    def test_add_import_error(self):
        with patch.dict(sys.modules, {"animus_forge.api_clients": None}):
            result = runner.invoke(
                app, ["calendar", "add", "Test Event", "--start", "2024-01-15 14:00"]
            )
            assert result.exit_code != 0

    def test_add_auth_failure(self):
        mock_client = MagicMock()
        mock_client.authenticate.return_value = False

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        mock_module.CalendarEvent = FakeCalendarEvent
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "add", "Test", "--start", "2024-01-15 14:00"])
            assert result.exit_code != 0

    def test_add_no_start_time(self):
        mock_client = MagicMock()
        mock_client.authenticate.return_value = True

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        mock_module.CalendarEvent = FakeCalendarEvent
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "add", "Test Event"])
            assert result.exit_code != 0

    def test_add_event_success(self):
        result_event = FakeCalendarEvent(
            id="new-1",
            summary="Team Meeting",
            html_link="https://calendar.google.com/event/1",
        )

        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.create_event.return_value = result_event

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        mock_module.CalendarEvent = FakeCalendarEvent
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(
                app,
                ["calendar", "add", "Team Meeting", "--start", "2024-01-15 14:00"],
            )
            assert result.exit_code == 0
            assert "Event created" in result.output

    def test_add_event_with_all_day(self):
        result_event = FakeCalendarEvent(id="new-2", summary="Vacation")

        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.create_event.return_value = result_event

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        mock_module.CalendarEvent = FakeCalendarEvent
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(
                app,
                ["calendar", "add", "Vacation", "--start", "2024-01-20", "--all-day"],
            )
            assert result.exit_code == 0

    def test_add_event_creation_fails(self):
        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.create_event.return_value = None

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        mock_module.CalendarEvent = FakeCalendarEvent
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(
                app,
                ["calendar", "add", "Test", "--start", "2024-01-15 14:00"],
            )
            assert result.exit_code != 0

    def test_add_quick_event_success(self):
        result_event = FakeCalendarEvent(
            id="q-1",
            summary="Doctor at 3pm",
            start=datetime.now(UTC),
            html_link="https://calendar.google.com/event/q1",
        )

        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.quick_add.return_value = result_event

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        mock_module.CalendarEvent = FakeCalendarEvent
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(
                app,
                ["calendar", "add", "Doctor", "--quick", "--start", "3pm"],
            )
            assert result.exit_code == 0
            assert "Event created" in result.output

    def test_add_quick_event_with_location(self):
        result_event = FakeCalendarEvent(
            id="q-2",
            summary="Doctor at hospital",
            start=datetime.now(UTC),
            html_link="",
        )

        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.quick_add.return_value = result_event

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        mock_module.CalendarEvent = FakeCalendarEvent
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(
                app,
                [
                    "calendar",
                    "add",
                    "Doctor",
                    "--quick",
                    "--location",
                    "Hospital",
                ],
            )
            assert result.exit_code == 0

    def test_add_quick_event_fails(self):
        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.quick_add.return_value = None

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        mock_module.CalendarEvent = FakeCalendarEvent
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(
                app,
                ["calendar", "add", "Test", "--quick"],
            )
            assert result.exit_code == 0  # prints failure but doesn't exit 1

    def test_add_invalid_date_format(self):
        mock_client = MagicMock()
        mock_client.authenticate.return_value = True

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        mock_module.CalendarEvent = FakeCalendarEvent
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(
                app,
                ["calendar", "add", "Test", "--start", "not-a-date"],
            )
            assert result.exit_code != 0

    def test_add_event_with_location_and_description(self):
        result_event = FakeCalendarEvent(
            id="loc-1",
            summary="Meeting",
            html_link="https://calendar.google.com/event/loc1",
        )

        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.create_event.return_value = result_event

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        mock_module.CalendarEvent = FakeCalendarEvent
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(
                app,
                [
                    "calendar",
                    "add",
                    "Meeting",
                    "--start",
                    "01/15/2024 14:00",
                    "--location",
                    "Office",
                    "--desc",
                    "Weekly sync",
                    "--duration",
                    "30",
                ],
            )
            assert result.exit_code == 0

    def test_add_event_mm_dd_yyyy_format(self):
        """Test date parsing with MM/DD/YYYY format."""
        result_event = FakeCalendarEvent(id="fmt-1", summary="Test")

        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.create_event.return_value = result_event

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        mock_module.CalendarEvent = FakeCalendarEvent
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(
                app,
                ["calendar", "add", "Test", "--start", "01/15/2024"],
            )
            assert result.exit_code == 0


class TestCalendarDelete:
    """Test calendar delete command."""

    def test_delete_import_error(self):
        with patch.dict(sys.modules, {"animus_forge.api_clients": None}):
            result = runner.invoke(app, ["calendar", "delete", "event-123"])
            assert result.exit_code != 0

    def test_delete_auth_failure(self):
        mock_client = MagicMock()
        mock_client.authenticate.return_value = False

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "delete", "event-123"])
            assert result.exit_code != 0

    def test_delete_event_not_found(self):
        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.get_event.return_value = None

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "delete", "event-123"])
            assert result.exit_code != 0

    def test_delete_event_force(self):
        event = FakeCalendarEvent(id="del-1", summary="Test Event")
        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.get_event.return_value = event
        mock_client.delete_event.return_value = True

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "delete", "del-1", "--force"])
            assert result.exit_code == 0
            assert "deleted" in result.output

    def test_delete_event_confirm_yes(self):
        event = FakeCalendarEvent(
            id="del-2",
            summary="Test Event",
            start=datetime.now(UTC),
        )
        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.get_event.return_value = event
        mock_client.delete_event.return_value = True

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "delete", "del-2"], input="y\n")
            assert result.exit_code == 0

    def test_delete_event_confirm_no(self):
        event = FakeCalendarEvent(id="del-3", summary="Test Event")
        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.get_event.return_value = event

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "delete", "del-3"], input="n\n")
            assert result.exit_code != 0  # Abort

    def test_delete_event_fails(self):
        event = FakeCalendarEvent(id="del-4", summary="Test Event")
        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.get_event.return_value = event
        mock_client.delete_event.return_value = False

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "delete", "del-4", "--force"])
            assert result.exit_code != 0


class TestCalendarBusy:
    """Test calendar busy command."""

    def test_busy_import_error(self):
        with patch.dict(sys.modules, {"animus_forge.api_clients": None}):
            result = runner.invoke(app, ["calendar", "busy"])
            assert result.exit_code != 0

    def test_busy_auth_failure(self):
        mock_client = MagicMock()
        mock_client.authenticate.return_value = False

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "busy"])
            assert result.exit_code != 0

    def test_busy_no_periods(self):
        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.check_availability.return_value = []

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "busy"])
            assert result.exit_code == 0
            assert "No busy periods" in result.output

    def test_busy_with_periods(self):
        periods = [
            {"start": "2024-01-15T10:00:00Z", "end": "2024-01-15T11:00:00Z"},
            {"start": "2024-01-15T14:00:00Z", "end": "2024-01-15T15:00:00Z"},
        ]

        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.check_availability.return_value = periods

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "busy"])
            assert result.exit_code == 0
            assert "Busy Periods" in result.output

    def test_busy_with_invalid_date_periods(self):
        """Test graceful handling of unparseable date strings."""
        periods = [
            {"start": "not-a-date", "end": "also-not-a-date"},
        ]

        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.check_availability.return_value = periods

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "busy"])
            assert result.exit_code == 0
            assert "not-a-date" in result.output

    def test_busy_with_empty_period_fields(self):
        """Test period with empty start/end."""
        periods = [
            {"start": "", "end": ""},
        ]

        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.check_availability.return_value = periods

        mock_module = MagicMock()
        mock_module.CalendarClient.return_value = mock_client
        with patch.dict(sys.modules, {"animus_forge.api_clients": mock_module}):
            result = runner.invoke(app, ["calendar", "busy"])
            assert result.exit_code == 0


# ==========================================================================

# ==========================================================================
#  BROWSER CLI TESTS
# ==========================================================================


class TestBrowserNavigate:
    """Test browser navigate command."""

    def test_navigate_import_error(self):
        with patch.dict(sys.modules, {"animus_forge.browser": None}):
            result = runner.invoke(app, ["browser", "navigate", "https://example.com"])
            assert result.exit_code != 0

    def test_navigate_success(self):
        mock_module = _make_browser_mock(
            nav_result=FakePageResult(success=True, title="Example", url="https://example.com"),
        )
        with patch.dict(sys.modules, {"animus_forge.browser": mock_module}):
            result = runner.invoke(app, ["browser", "navigate", "https://example.com"])
            assert result.exit_code == 0
            assert "Example" in result.output

    def test_navigate_failure(self):
        mock_module = _make_browser_mock(
            nav_result=FakePageResult(success=False, error="Timeout"),
        )
        with patch.dict(sys.modules, {"animus_forge.browser": mock_module}):
            result = runner.invoke(app, ["browser", "navigate", "https://example.com"])
            assert result.exit_code != 0

    def test_navigate_with_extract(self):
        mock_module = _make_browser_mock(
            nav_result=FakePageResult(success=True, title="Example", url="https://example.com"),
            extract_result=FakePageResult(
                success=True,
                data={
                    "text": "Hello World " * 100,
                    "links": [{"text": "Link", "href": "https://example.com/link"}],
                },
            ),
        )
        with patch.dict(sys.modules, {"animus_forge.browser": mock_module}):
            result = runner.invoke(
                app,
                ["browser", "navigate", "https://example.com", "--extract"],
            )
            assert result.exit_code == 0
            assert "Hello World" in result.output

    def test_navigate_with_screenshot(self):
        mock_module = _make_browser_mock(
            nav_result=FakePageResult(success=True, title="Example", url="https://example.com"),
            screenshot_result=FakePageResult(success=True, screenshot_path="/tmp/screenshot.png"),
        )
        with patch.dict(sys.modules, {"animus_forge.browser": mock_module}):
            result = runner.invoke(
                app,
                ["browser", "navigate", "https://example.com", "--screenshot"],
            )
            assert result.exit_code == 0
            assert "screenshot" in result.output.lower()


class TestBrowserScreenshot:
    """Test browser screenshot command."""

    def test_screenshot_import_error(self):
        with patch.dict(sys.modules, {"animus_forge.browser": None}):
            result = runner.invoke(app, ["browser", "screenshot", "https://example.com"])
            assert result.exit_code != 0

    def test_screenshot_success(self):
        mock_module = _make_browser_mock(
            nav_result=FakePageResult(success=True, title="Example", url="https://example.com"),
            screenshot_result=FakePageResult(success=True, screenshot_path="/tmp/screenshot.png"),
        )
        with patch.dict(sys.modules, {"animus_forge.browser": mock_module}):
            result = runner.invoke(app, ["browser", "screenshot", "https://example.com"])
            assert result.exit_code == 0
            assert "Screenshot saved" in result.output

    def test_screenshot_nav_failure(self):
        mock_module = _make_browser_mock(
            nav_result=FakePageResult(success=False, error="404"),
        )
        with patch.dict(sys.modules, {"animus_forge.browser": mock_module}):
            result = runner.invoke(app, ["browser", "screenshot", "https://example.com"])
            assert result.exit_code != 0

    def test_screenshot_capture_failure(self):
        mock_module = _make_browser_mock(
            nav_result=FakePageResult(success=True, title="Example"),
            screenshot_result=FakePageResult(success=False, error="Capture failed"),
        )
        with patch.dict(sys.modules, {"animus_forge.browser": mock_module}):
            result = runner.invoke(app, ["browser", "screenshot", "https://example.com"])
            assert result.exit_code != 0

    def test_screenshot_with_output_path(self):
        mock_module = _make_browser_mock(
            nav_result=FakePageResult(success=True, title="Example"),
            screenshot_result=FakePageResult(success=True, screenshot_path="/tmp/custom.png"),
        )
        with patch.dict(sys.modules, {"animus_forge.browser": mock_module}):
            result = runner.invoke(
                app,
                [
                    "browser",
                    "screenshot",
                    "https://example.com",
                    "-o",
                    "/tmp/custom.png",
                    "--full",
                ],
            )
            assert result.exit_code == 0


class TestBrowserExtract:
    """Test browser extract command."""

    def test_extract_import_error(self):
        with patch.dict(sys.modules, {"animus_forge.browser": None}):
            result = runner.invoke(app, ["browser", "extract", "https://example.com"])
            assert result.exit_code != 0

    def test_extract_nav_failure(self):
        mock_module = _make_browser_mock(
            nav_result=FakePageResult(success=False, error="Connection refused"),
        )
        with patch.dict(sys.modules, {"animus_forge.browser": mock_module}):
            result = runner.invoke(app, ["browser", "extract", "https://example.com"])
            assert result.exit_code != 0

    def test_extract_extraction_failure(self):
        mock_module = _make_browser_mock(
            nav_result=FakePageResult(success=True, title="Example"),
            extract_result=FakePageResult(success=False, error="No content"),
        )
        with patch.dict(sys.modules, {"animus_forge.browser": mock_module}):
            result = runner.invoke(app, ["browser", "extract", "https://example.com"])
            assert result.exit_code != 0

    def test_extract_success_plain(self):
        mock_module = _make_browser_mock(
            nav_result=FakePageResult(success=True, title="Example"),
            extract_result=FakePageResult(
                success=True,
                data={
                    "title": "Example Page",
                    "url": "https://example.com",
                    "text": "Sample content here " * 100,
                    "links": [
                        {"text": "Link One", "href": "https://example.com/one"},
                        {"text": "Link Two", "href": "https://example.com/two"},
                    ],
                    "tables": [{"data": "table1"}],
                },
            ),
        )
        with patch.dict(sys.modules, {"animus_forge.browser": mock_module}):
            result = runner.invoke(app, ["browser", "extract", "https://example.com"])
            assert result.exit_code == 0
            assert "Example Page" in result.output
            assert "Link One" in result.output
            assert "Tables" in result.output

    def test_extract_success_json(self):
        mock_module = _make_browser_mock(
            nav_result=FakePageResult(success=True, title="Example"),
            extract_result=FakePageResult(
                success=True,
                data={
                    "title": "Example",
                    "url": "https://example.com",
                    "text": "content",
                },
            ),
        )
        with patch.dict(sys.modules, {"animus_forge.browser": mock_module}):
            result = runner.invoke(app, ["browser", "extract", "https://example.com", "--json"])
            assert result.exit_code == 0
            assert "Example" in result.output

    def test_extract_with_selector(self):
        mock_module = _make_browser_mock(
            nav_result=FakePageResult(success=True, title="Example"),
            extract_result=FakePageResult(
                success=True,
                data={
                    "title": "Example",
                    "url": "https://example.com",
                    "text": "Selected content",
                },
            ),
        )
        with patch.dict(sys.modules, {"animus_forge.browser": mock_module}):
            result = runner.invoke(
                app,
                [
                    "browser",
                    "extract",
                    "https://example.com",
                    "--selector",
                    "main article",
                ],
            )
            assert result.exit_code == 0

    def test_extract_no_links_no_tables(self):
        """Extract with no links and no tables in data."""
        mock_module = _make_browser_mock(
            nav_result=FakePageResult(success=True, title="Example"),
            extract_result=FakePageResult(
                success=True,
                data={
                    "title": "Example",
                    "url": "https://example.com",
                    "text": "Plain text only",
                },
            ),
        )
        with patch.dict(sys.modules, {"animus_forge.browser": mock_module}):
            result = runner.invoke(
                app,
                [
                    "browser",
                    "extract",
                    "https://example.com",
                    "--no-links",
                    "--no-tables",
                ],
            )
            assert result.exit_code == 0
