"""Tests for the MCP CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from animus_forge.cli.main import app
from animus_forge.mcp.models import (
    MCPConnectionTestResult,
    MCPResource,
    MCPServer,
    MCPTool,
)

runner = CliRunner()

# Patch target for _get_manager helper
_MANAGER_PATCH = "animus_forge.cli.commands.mcp._get_manager"


def _make_server(
    server_id: str = "abc-123",
    name: str = "test-server",
    url: str = "http://localhost:8080",
    server_type: str = "sse",
    status: str = "connected",
    tools: list | None = None,
    resources: list | None = None,
) -> MCPServer:
    """Build a mock MCPServer."""
    return MCPServer(
        id=server_id,
        name=name,
        url=url,
        type=server_type,
        status=status,
        authType="none",
        tools=tools or [],
        resources=resources or [],
    )


SAMPLE_TOOLS = [
    MCPTool(name="list_repos", description="List repositories", inputSchema={}),
    MCPTool(name="create_pr", description="Create a pull request", inputSchema={}),
]

SAMPLE_RESOURCES = [
    MCPResource(uri="file:///tmp/data.json", name="data.json", mimeType="application/json"),
]

SAMPLE_SERVER = _make_server(tools=SAMPLE_TOOLS)


# =============================================================================
# mcp list
# =============================================================================


class TestMcpList:
    @patch(_MANAGER_PATCH)
    def test_list_shows_servers(self, mock_get):
        mgr = MagicMock()
        mgr.list_servers.return_value = [SAMPLE_SERVER]
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "list"])
        assert result.exit_code == 0
        assert "MCP Servers" in result.output
        assert "test-server" in result.output
        assert "abc-123" in result.output[:80] or "abc" in result.output

    @patch(_MANAGER_PATCH)
    def test_list_empty(self, mock_get):
        mgr = MagicMock()
        mgr.list_servers.return_value = []
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "list"])
        assert result.exit_code == 0
        assert "No MCP servers registered" in result.output

    @patch(_MANAGER_PATCH)
    def test_list_json(self, mock_get):
        mgr = MagicMock()
        mgr.list_servers.return_value = [SAMPLE_SERVER]
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "list", "--json"])
        assert result.exit_code == 0
        data = _parse_json(result.output)
        assert isinstance(data, list)
        assert data[0]["name"] == "test-server"

    @patch(_MANAGER_PATCH)
    def test_list_error(self, mock_get):
        mock_get.side_effect = Exception("DB error")
        result = runner.invoke(app, ["mcp", "list"])
        assert result.exit_code == 1
        assert "Error" in result.output


# =============================================================================
# mcp add
# =============================================================================


class TestMcpAdd:
    @patch(_MANAGER_PATCH)
    def test_add_basic(self, mock_get):
        mgr = MagicMock()
        mgr.create_server.return_value = SAMPLE_SERVER
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "add", "my-server", "http://localhost:9090"])
        assert result.exit_code == 0
        assert "Server registered" in result.output
        mgr.create_server.assert_called_once()

    @patch(_MANAGER_PATCH)
    def test_add_with_options(self, mock_get):
        mgr = MagicMock()
        mgr.create_server.return_value = SAMPLE_SERVER
        mock_get.return_value = mgr
        result = runner.invoke(
            app,
            [
                "mcp",
                "add",
                "my-server",
                "http://localhost:9090",
                "--type",
                "stdio",
                "--auth",
                "bearer",
                "--description",
                "My test server",
            ],
        )
        assert result.exit_code == 0
        assert "Server registered" in result.output

    @patch(_MANAGER_PATCH)
    def test_add_error(self, mock_get):
        mgr = MagicMock()
        mgr.create_server.side_effect = Exception("Validation error")
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "add", "bad", "http://x"])
        assert result.exit_code == 1
        assert "Error" in result.output


# =============================================================================
# mcp remove
# =============================================================================


class TestMcpRemove:
    @patch(_MANAGER_PATCH)
    def test_remove_success(self, mock_get):
        mgr = MagicMock()
        mgr.delete_server.return_value = True
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "remove", "abc-123"])
        assert result.exit_code == 0
        assert "Server removed" in result.output

    @patch(_MANAGER_PATCH)
    def test_remove_not_found(self, mock_get):
        mgr = MagicMock()
        mgr.delete_server.return_value = False
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "remove", "nonexistent"])
        assert result.exit_code == 1
        assert "Server not found" in result.output

    @patch(_MANAGER_PATCH)
    def test_remove_error(self, mock_get):
        mgr = MagicMock()
        mgr.delete_server.side_effect = Exception("DB error")
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "remove", "abc-123"])
        assert result.exit_code == 1
        assert "Error" in result.output


# =============================================================================
# mcp test
# =============================================================================


class TestMcpTest:
    @patch(_MANAGER_PATCH)
    def test_connection_success(self, mock_get):
        mgr = MagicMock()
        mgr.test_connection.return_value = MCPConnectionTestResult(
            success=True, tools=SAMPLE_TOOLS, resources=SAMPLE_RESOURCES
        )
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "test", "abc-123"])
        assert result.exit_code == 0
        assert "Connection successful" in result.output
        assert "Tools discovered: 2" in result.output
        assert "Resources discovered: 1" in result.output

    @patch(_MANAGER_PATCH)
    def test_connection_failure(self, mock_get):
        mgr = MagicMock()
        mgr.test_connection.return_value = MCPConnectionTestResult(
            success=False, error="Connection refused"
        )
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "test", "abc-123"])
        assert result.exit_code == 1
        assert "Connection failed" in result.output
        assert "Connection refused" in result.output

    @patch(_MANAGER_PATCH)
    def test_connection_error(self, mock_get):
        mock_get.side_effect = Exception("DB error")
        result = runner.invoke(app, ["mcp", "test", "abc-123"])
        assert result.exit_code == 1
        assert "Error" in result.output


# =============================================================================
# mcp discover
# =============================================================================


class TestMcpDiscover:
    @patch(_MANAGER_PATCH)
    def test_discover_success(self, mock_get):
        mgr = MagicMock()
        mgr.test_connection.return_value = MCPConnectionTestResult(
            success=True, tools=SAMPLE_TOOLS, resources=SAMPLE_RESOURCES
        )
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "discover", "abc-123"])
        assert result.exit_code == 0
        assert "Discovered Tools" in result.output
        assert "list_repos" in result.output
        assert "Discovered Resources" in result.output
        assert "data.json" in result.output

    @patch(_MANAGER_PATCH)
    def test_discover_no_tools(self, mock_get):
        mgr = MagicMock()
        mgr.test_connection.return_value = MCPConnectionTestResult(
            success=True, tools=[], resources=[]
        )
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "discover", "abc-123"])
        assert result.exit_code == 0
        assert "No tools discovered" in result.output

    @patch(_MANAGER_PATCH)
    def test_discover_failure(self, mock_get):
        mgr = MagicMock()
        mgr.test_connection.return_value = MCPConnectionTestResult(
            success=False, error="Server not found"
        )
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "discover", "abc-123"])
        assert result.exit_code == 1
        assert "Discovery failed" in result.output

    @patch(_MANAGER_PATCH)
    def test_discover_json(self, mock_get):
        mgr = MagicMock()
        mgr.test_connection.return_value = MCPConnectionTestResult(
            success=True, tools=SAMPLE_TOOLS, resources=SAMPLE_RESOURCES
        )
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "discover", "abc-123", "--json"])
        assert result.exit_code == 0
        data = _parse_json(result.output)
        assert "tools" in data
        assert len(data["tools"]) == 2
        assert "resources" in data
        assert len(data["resources"]) == 1

    @patch(_MANAGER_PATCH)
    def test_discover_error(self, mock_get):
        mock_get.side_effect = Exception("DB error")
        result = runner.invoke(app, ["mcp", "discover", "abc-123"])
        assert result.exit_code == 1
        assert "Error" in result.output


# =============================================================================
# mcp tools
# =============================================================================


class TestMcpTools:
    @patch(_MANAGER_PATCH)
    def test_tools_shows_table(self, mock_get):
        mgr = MagicMock()
        mgr.get_tools.return_value = SAMPLE_TOOLS
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "tools", "abc-123"])
        assert result.exit_code == 0
        assert "Cached Tools" in result.output
        assert "list_repos" in result.output
        assert "create_pr" in result.output

    @patch(_MANAGER_PATCH)
    def test_tools_empty(self, mock_get):
        mgr = MagicMock()
        mgr.get_tools.return_value = []
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "tools", "abc-123"])
        assert result.exit_code == 0
        assert "No cached tools" in result.output

    @patch(_MANAGER_PATCH)
    def test_tools_json(self, mock_get):
        mgr = MagicMock()
        mgr.get_tools.return_value = SAMPLE_TOOLS
        mock_get.return_value = mgr
        result = runner.invoke(app, ["mcp", "tools", "abc-123", "--json"])
        assert result.exit_code == 0
        data = _parse_json(result.output)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "list_repos"

    @patch(_MANAGER_PATCH)
    def test_tools_error(self, mock_get):
        mock_get.side_effect = Exception("DB error")
        result = runner.invoke(app, ["mcp", "tools", "abc-123"])
        assert result.exit_code == 1
        assert "Error" in result.output


# =============================================================================
# Helpers
# =============================================================================


def _parse_json(output: str) -> dict | list:
    """Parse JSON from CLI output, stripping Rich markup."""
    import json

    # The json output goes to stdout via print(), but Rich output
    # also goes to the runner's output. Find the JSON portion.
    # JSON starts with '[' or '{'.
    for i, ch in enumerate(output):
        if ch in ("{", "["):
            return json.loads(output[i:])
    raise ValueError(f"No JSON found in output: {output!r}")
