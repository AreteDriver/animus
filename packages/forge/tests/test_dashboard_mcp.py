"""Tests for MCP dashboard page."""

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Streamlit mock helpers (same pattern as test_cost_dashboard.py)
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


@pytest.fixture(autouse=True)
def mock_streamlit():
    mock_st = MagicMock()

    class SessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

        def __setattr__(self, name, value):
            self[name] = value

        def __delattr__(self, name):
            try:
                del self[name]
            except KeyError:
                raise AttributeError(name)

    mock_st.session_state = SessionState()
    mock_st.cache_resource = lambda f: f
    mock_st.columns.side_effect = _create_columns
    mock_st.tabs.side_effect = _create_tabs
    mock_st.expander.side_effect = _create_expander

    # Mock st.form as context manager
    mock_st.form.return_value = _create_context_manager()

    mod_key = "animus_forge.dashboard.mcp_page"
    cached = sys.modules.pop(mod_key, None)

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        yield mock_st

    sys.modules.pop(mod_key, None)
    if cached is not None:
        sys.modules[mod_key] = cached


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_server(
    *,
    name: str = "Test Server",
    url: str = "http://localhost:8080/mcp",
    tools: list | None = None,
    resources: list | None = None,
    status: str = "connected",
    auth_type: str = "none",
    server_id: str = "srv-1",
):
    """Build a mock MCPServer."""
    server = MagicMock()
    server.id = server_id
    server.name = name
    server.url = url
    server.type = "sse"
    server.status = status
    server.authType = auth_type
    server.description = "A test server"
    server.lastConnected = datetime(2025, 6, 1, 12, 0)
    server.error = None
    server.tools = tools or []
    server.resources = resources or []
    return server


def _make_tool(name: str = "list_repos", description: str = "List repos"):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = {
        "type": "object",
        "properties": {"repo": {"type": "string"}},
        "required": ["repo"],
    }
    return tool


def _make_resource(uri: str = "file:///tmp/readme.md", name: str = "readme"):
    resource = MagicMock()
    resource.uri = uri
    resource.name = name
    resource.description = "A resource"
    resource.mimeType = "text/markdown"
    return resource


def _make_manager(servers=None):
    """Create a mock MCPConnectorManager."""
    manager = MagicMock()
    manager.list_servers.return_value = servers or []
    manager.get_server.return_value = servers[0] if servers else None
    manager.test_connection.return_value = MagicMock(
        success=True, tools=[], resources=[], error=None
    )
    manager.delete_server.return_value = True
    manager.create_server.return_value = _make_server()
    return manager


# ---------------------------------------------------------------------------
# Tests: render_mcp_page
# ---------------------------------------------------------------------------


class TestRenderMcpPage:
    def test_renders_title(self, mock_streamlit):
        manager = _make_manager()
        with patch("animus_forge.dashboard.mcp_page._get_manager", return_value=manager):
            from animus_forge.dashboard.mcp_page import render_mcp_page

            render_mcp_page()

            mock_streamlit.title.assert_called_once()
            assert "MCP" in mock_streamlit.title.call_args[0][0]

    def test_creates_three_tabs(self, mock_streamlit):
        manager = _make_manager()
        with patch("animus_forge.dashboard.mcp_page._get_manager", return_value=manager):
            from animus_forge.dashboard.mcp_page import render_mcp_page

            render_mcp_page()

            mock_streamlit.tabs.assert_called_once()
            labels = mock_streamlit.tabs.call_args[0][0]
            assert len(labels) == 3

    def test_handles_missing_mcp_module(self, mock_streamlit):
        with patch("animus_forge.dashboard.mcp_page.MCP_AVAILABLE", False):
            from animus_forge.dashboard.mcp_page import render_mcp_page

            render_mcp_page()

            mock_streamlit.error.assert_called()

    def test_handles_manager_init_failure(self, mock_streamlit):
        with patch("animus_forge.dashboard.mcp_page._get_manager", return_value=None):
            from animus_forge.dashboard.mcp_page import render_mcp_page

            render_mcp_page()

            mock_streamlit.error.assert_called()


# ---------------------------------------------------------------------------
# Tests: Server Management tab
# ---------------------------------------------------------------------------


class TestServerManagement:
    def test_shows_empty_state(self, mock_streamlit):
        manager = _make_manager(servers=[])
        from animus_forge.dashboard.mcp_page import _render_server_management

        _render_server_management(manager)

        mock_streamlit.info.assert_called()

    def test_lists_servers(self, mock_streamlit):
        server = _make_server()
        manager = _make_manager(servers=[server])
        from animus_forge.dashboard.mcp_page import _render_server_management

        _render_server_management(manager)

        mock_streamlit.expander.assert_called()
        call_arg = mock_streamlit.expander.call_args[0][0]
        assert "Test Server" in call_arg

    def test_test_connection_success(self, mock_streamlit):
        server = _make_server()
        manager = _make_manager(servers=[server])
        mock_streamlit.button.return_value = True
        manager.test_connection.return_value = MagicMock(
            success=True, tools=[_make_tool()], resources=[], error=None
        )

        from animus_forge.dashboard.mcp_page import _render_server_row

        _render_server_row(manager, server)

        # Button was called (test and delete)
        assert mock_streamlit.button.call_count >= 1

    def test_test_connection_failure(self, mock_streamlit):
        server = _make_server()
        manager = _make_manager(servers=[server])
        mock_streamlit.button.return_value = True
        manager.test_connection.return_value = MagicMock(
            success=False, tools=[], resources=[], error="Connection refused"
        )

        from animus_forge.dashboard.mcp_page import _render_server_row

        _render_server_row(manager, server)

        mock_streamlit.error.assert_called()

    def test_server_with_error_shows_warning(self, mock_streamlit):
        server = _make_server()
        server.error = "Timeout"
        manager = _make_manager(servers=[server])
        mock_streamlit.button.return_value = False

        from animus_forge.dashboard.mcp_page import _render_server_row

        _render_server_row(manager, server)

        mock_streamlit.warning.assert_called()


# ---------------------------------------------------------------------------
# Tests: Add Server form
# ---------------------------------------------------------------------------


class TestAddServerForm:
    def test_form_renders(self, mock_streamlit):
        manager = _make_manager()
        from animus_forge.dashboard.mcp_page import _render_add_server_form

        _render_add_server_form(manager)

        mock_streamlit.subheader.assert_called()
        mock_streamlit.form.assert_called_once_with("add_mcp_server", clear_on_submit=True)


# ---------------------------------------------------------------------------
# Tests: Tool Discovery tab
# ---------------------------------------------------------------------------


class TestToolDiscovery:
    def test_no_servers_shows_info(self, mock_streamlit):
        manager = _make_manager(servers=[])
        from animus_forge.dashboard.mcp_page import _render_tool_discovery

        _render_tool_discovery(manager)

        mock_streamlit.info.assert_called()

    def test_shows_server_selector(self, mock_streamlit):
        server = _make_server(tools=[_make_tool()])
        manager = _make_manager(servers=[server])
        mock_streamlit.selectbox.return_value = server.id
        mock_streamlit.button.return_value = False

        from animus_forge.dashboard.mcp_page import _render_tool_discovery

        _render_tool_discovery(manager)

        mock_streamlit.selectbox.assert_called()

    def test_discover_button_triggers_connection_test(self, mock_streamlit):
        tool = _make_tool()
        server = _make_server(tools=[tool])
        manager = _make_manager(servers=[server])
        mock_streamlit.selectbox.return_value = server.id
        mock_streamlit.button.return_value = True
        manager.test_connection.return_value = MagicMock(
            success=True, tools=[tool], resources=[], error=None
        )

        from animus_forge.dashboard.mcp_page import _render_tool_discovery

        _render_tool_discovery(manager)

        manager.test_connection.assert_called_once_with(server.id)


# ---------------------------------------------------------------------------
# Tests: Discovered tools / resources rendering
# ---------------------------------------------------------------------------


class TestDiscoveredTools:
    def test_empty_tools(self, mock_streamlit):
        from animus_forge.dashboard.mcp_page import _render_discovered_tools

        _render_discovered_tools([])

        mock_streamlit.info.assert_called()

    def test_renders_tool_cards(self, mock_streamlit):
        tools = [_make_tool("list_repos"), _make_tool("create_pr")]
        from animus_forge.dashboard.mcp_page import _render_discovered_tools

        _render_discovered_tools(tools)

        assert mock_streamlit.expander.call_count == 2
        mock_streamlit.subheader.assert_called()


class TestDiscoveredResources:
    def test_empty_resources_returns_early(self, mock_streamlit):
        from animus_forge.dashboard.mcp_page import _render_discovered_resources

        _render_discovered_resources([])

        mock_streamlit.subheader.assert_not_called()

    def test_renders_resource_cards(self, mock_streamlit):
        resources = [_make_resource()]
        from animus_forge.dashboard.mcp_page import _render_discovered_resources

        _render_discovered_resources(resources)

        mock_streamlit.expander.assert_called()
        mock_streamlit.subheader.assert_called()


# ---------------------------------------------------------------------------
# Tests: Tool Execution tab
# ---------------------------------------------------------------------------


class TestToolExecution:
    def test_no_servers_shows_info(self, mock_streamlit):
        manager = _make_manager(servers=[])
        from animus_forge.dashboard.mcp_page import _render_tool_execution

        _render_tool_execution(manager)

        mock_streamlit.info.assert_called()

    def test_no_servers_with_tools_shows_info(self, mock_streamlit):
        server = _make_server(tools=[])
        manager = _make_manager(servers=[server])
        from animus_forge.dashboard.mcp_page import _render_tool_execution

        _render_tool_execution(manager)

        mock_streamlit.info.assert_called()

    def test_shows_tool_selector(self, mock_streamlit):
        tool = _make_tool()
        server = _make_server(tools=[tool])
        manager = _make_manager(servers=[server])
        mock_streamlit.selectbox.side_effect = [server.id, tool.name]
        mock_streamlit.button.return_value = False

        from animus_forge.dashboard.mcp_page import _render_tool_execution

        _render_tool_execution(manager)

        assert mock_streamlit.selectbox.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Dynamic form
# ---------------------------------------------------------------------------


class TestDynamicForm:
    def test_empty_schema(self, mock_streamlit):
        tool = MagicMock()
        tool.name = "no_args"
        tool.inputSchema = {}

        from animus_forge.dashboard.mcp_page import _render_dynamic_form

        result = _render_dynamic_form(tool)

        assert result == {}
        mock_streamlit.info.assert_called()

    def test_string_property(self, mock_streamlit):
        tool = MagicMock()
        tool.name = "test_tool"
        tool.inputSchema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        mock_streamlit.text_input.return_value = "hello"

        from animus_forge.dashboard.mcp_page import _render_dynamic_form

        result = _render_dynamic_form(tool)

        assert result == {"name": "hello"}

    def test_enum_property(self, mock_streamlit):
        tool = MagicMock()
        tool.name = "test_tool"
        tool.inputSchema = {
            "type": "object",
            "properties": {"state": {"type": "string", "enum": ["open", "closed"]}},
        }
        mock_streamlit.selectbox.return_value = "open"

        from animus_forge.dashboard.mcp_page import _render_dynamic_form

        result = _render_dynamic_form(tool)

        assert result == {"state": "open"}

    def test_boolean_property(self, mock_streamlit):
        tool = MagicMock()
        tool.name = "test_tool"
        tool.inputSchema = {
            "type": "object",
            "properties": {"verbose": {"type": "boolean"}},
        }
        mock_streamlit.checkbox.return_value = True

        from animus_forge.dashboard.mcp_page import _render_dynamic_form

        result = _render_dynamic_form(tool)

        assert result == {"verbose": True}

    def test_integer_property(self, mock_streamlit):
        tool = MagicMock()
        tool.name = "test_tool"
        tool.inputSchema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
        mock_streamlit.number_input.return_value = 42

        from animus_forge.dashboard.mcp_page import _render_dynamic_form

        result = _render_dynamic_form(tool)

        assert result == {"count": 42}

    def test_number_property(self, mock_streamlit):
        tool = MagicMock()
        tool.name = "test_tool"
        tool.inputSchema = {
            "type": "object",
            "properties": {"ratio": {"type": "number"}},
        }
        mock_streamlit.number_input.return_value = 3.14

        from animus_forge.dashboard.mcp_page import _render_dynamic_form

        result = _render_dynamic_form(tool)

        assert result == {"ratio": 3.14}

    def test_object_property_valid_json(self, mock_streamlit):
        tool = MagicMock()
        tool.name = "test_tool"
        tool.inputSchema = {
            "type": "object",
            "properties": {"config": {"type": "object"}},
        }
        mock_streamlit.text_area.return_value = '{"key": "val"}'

        from animus_forge.dashboard.mcp_page import _render_dynamic_form

        result = _render_dynamic_form(tool)

        assert result == {"config": {"key": "val"}}

    def test_object_property_invalid_json(self, mock_streamlit):
        tool = MagicMock()
        tool.name = "test_tool"
        tool.inputSchema = {
            "type": "object",
            "properties": {"config": {"type": "object"}},
        }
        mock_streamlit.text_area.return_value = "not json"

        from animus_forge.dashboard.mcp_page import _render_dynamic_form

        result = _render_dynamic_form(tool)

        # Falls back to raw string
        assert result == {"config": "not json"}
        mock_streamlit.warning.assert_called()

    def test_required_field_label(self, mock_streamlit):
        tool = MagicMock()
        tool.name = "test_tool"
        tool.inputSchema = {
            "type": "object",
            "properties": {"repo": {"type": "string"}},
            "required": ["repo"],
        }
        mock_streamlit.text_input.return_value = "my-repo"

        from animus_forge.dashboard.mcp_page import _render_dynamic_form

        _render_dynamic_form(tool)

        # Label should have asterisk for required field
        label = mock_streamlit.text_input.call_args[0][0]
        assert "*" in label

    def test_empty_string_excluded(self, mock_streamlit):
        tool = MagicMock()
        tool.name = "test_tool"
        tool.inputSchema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        mock_streamlit.text_input.return_value = ""

        from animus_forge.dashboard.mcp_page import _render_dynamic_form

        result = _render_dynamic_form(tool)

        assert result == {}
