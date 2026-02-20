"""Tests for MCP protocol client (animus_forge.mcp.client)."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# HAS_MCP_SDK flag
# ---------------------------------------------------------------------------


class TestHasMCPSDK:
    """Test the MCP SDK availability flag."""

    def test_flag_reflects_sdk_availability(self):
        """HAS_MCP_SDK should reflect whether the mcp package is importable."""
        from animus_forge.mcp.client import HAS_MCP_SDK

        # Since mcp is not installed in test env, should be False
        assert isinstance(HAS_MCP_SDK, bool)

    def test_flag_false_when_sdk_missing(self):
        """HAS_MCP_SDK should be False when mcp package is not installed."""
        with patch.dict(
            sys.modules, {"mcp": None, "mcp.client.stdio": None, "mcp.client.sse": None}
        ):
            import animus_forge.mcp.client as client_mod

            reloaded = importlib.reload(client_mod)
            assert reloaded.HAS_MCP_SDK is False

    def test_flag_true_when_sdk_present(self):
        """HAS_MCP_SDK should be True when mcp package is available."""
        mock_mcp = MagicMock()
        mock_stdio = MagicMock()
        mock_sse = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "mcp": mock_mcp,
                "mcp.client": MagicMock(),
                "mcp.client.stdio": mock_stdio,
                "mcp.client.sse": mock_sse,
            },
        ):
            import animus_forge.mcp.client as client_mod

            reloaded = importlib.reload(client_mod)
            assert reloaded.HAS_MCP_SDK is True


# ---------------------------------------------------------------------------
# MCPClientError
# ---------------------------------------------------------------------------


class TestMCPClientError:
    """Test MCPClientError exception."""

    def test_is_exception(self):
        from animus_forge.mcp.client import MCPClientError

        assert issubclass(MCPClientError, Exception)

    def test_message(self):
        from animus_forge.mcp.client import MCPClientError

        err = MCPClientError("test error")
        assert str(err) == "test error"


# ---------------------------------------------------------------------------
# call_mcp_tool
# ---------------------------------------------------------------------------


class TestCallMCPTool:
    """Test the sync call_mcp_tool wrapper."""

    def test_raises_when_sdk_missing(self):
        """Should raise MCPClientError when SDK is not installed."""
        from animus_forge.mcp.client import MCPClientError, call_mcp_tool

        with patch("animus_forge.mcp.client.HAS_MCP_SDK", False):
            with pytest.raises(MCPClientError, match="MCP SDK not installed"):
                call_mcp_tool(
                    server_type="sse",
                    server_url="https://example.com/mcp",
                    tool_name="test_tool",
                    arguments={"key": "val"},
                )

    def test_raises_on_unsupported_type(self):
        """Should raise MCPClientError for unsupported server types."""
        from animus_forge.mcp.client import MCPClientError, call_mcp_tool

        with patch("animus_forge.mcp.client.HAS_MCP_SDK", True):
            with pytest.raises(MCPClientError, match="Unsupported MCP server type"):
                call_mcp_tool(
                    server_type="websocket",
                    server_url="ws://localhost:3000",
                    tool_name="test",
                    arguments={},
                )

    def test_stdio_dispatch(self):
        """Should dispatch to _call_tool_stdio for stdio type."""
        from animus_forge.mcp.client import call_mcp_tool

        mock_result = {"content": "hello", "is_error": False}
        with patch("animus_forge.mcp.client.HAS_MCP_SDK", True):
            with patch("animus_forge.mcp.client.asyncio.run", return_value=mock_result) as mock_run:
                result = call_mcp_tool(
                    server_type="stdio",
                    server_url="npx server --arg1",
                    tool_name="read_file",
                    arguments={"path": "/tmp/test.txt"},
                )
                assert result == mock_result
                mock_run.assert_called_once()

    def test_sse_dispatch(self):
        """Should dispatch to _call_tool_sse for sse type."""
        from animus_forge.mcp.client import call_mcp_tool

        mock_result = {"content": "data", "is_error": False}
        with patch("animus_forge.mcp.client.HAS_MCP_SDK", True):
            with patch("animus_forge.mcp.client.asyncio.run", return_value=mock_result) as mock_run:
                result = call_mcp_tool(
                    server_type="sse",
                    server_url="https://example.com/sse",
                    tool_name="search",
                    arguments={"query": "test"},
                    headers={"Authorization": "Bearer tok"},
                )
                assert result == mock_result
                mock_run.assert_called_once()

    def test_stdio_command_parsing(self):
        """Should correctly parse stdio command string."""
        from animus_forge.mcp.client import call_mcp_tool

        mock_result = {"content": "ok", "is_error": False}
        with patch("animus_forge.mcp.client.HAS_MCP_SDK", True):
            with patch("animus_forge.mcp.client.asyncio.run", return_value=mock_result):
                # Single command, no args
                result = call_mcp_tool(
                    server_type="stdio",
                    server_url="myserver",
                    tool_name="tool",
                    arguments={},
                )
                assert result["content"] == "ok"

    def test_wraps_unexpected_errors(self):
        """Should wrap unexpected errors in MCPClientError."""
        from animus_forge.mcp.client import MCPClientError, call_mcp_tool

        with patch("animus_forge.mcp.client.HAS_MCP_SDK", True):
            with patch(
                "animus_forge.mcp.client.asyncio.run",
                side_effect=ConnectionError("refused"),
            ):
                with pytest.raises(MCPClientError, match="MCP tool call failed"):
                    call_mcp_tool(
                        server_type="sse",
                        server_url="https://dead.server/mcp",
                        tool_name="tool",
                        arguments={},
                    )


# ---------------------------------------------------------------------------
# _extract_content
# ---------------------------------------------------------------------------


class TestExtractContent:
    """Test content extraction from MCP results."""

    def test_empty_content(self):
        from animus_forge.mcp.client import _extract_content

        mock_result = MagicMock()
        mock_result.content = []
        assert _extract_content(mock_result) == ""

    def test_none_content(self):
        from animus_forge.mcp.client import _extract_content

        mock_result = MagicMock()
        mock_result.content = None
        assert _extract_content(mock_result) == ""

    def test_text_blocks(self):
        from animus_forge.mcp.client import _extract_content

        block1 = MagicMock()
        block1.text = "Hello"
        block2 = MagicMock()
        block2.text = "World"
        mock_result = MagicMock()
        mock_result.content = [block1, block2]
        assert _extract_content(mock_result) == "Hello\nWorld"

    def test_non_text_blocks(self):
        from animus_forge.mcp.client import _extract_content

        class RawBlock:
            def __str__(self):
                return "raw-block"

        mock_result = MagicMock()
        mock_result.content = [RawBlock()]
        assert _extract_content(mock_result) == "raw-block"


# ---------------------------------------------------------------------------
# _normalize_discovery
# ---------------------------------------------------------------------------


class TestNormalizeDiscovery:
    """Test normalization of MCP list_tools / list_resources results."""

    def test_empty_results(self):
        from animus_forge.mcp.client import _normalize_discovery

        tools_result = MagicMock()
        tools_result.tools = []
        resources_result = MagicMock()
        resources_result.resources = []
        result = _normalize_discovery(tools_result, resources_result)
        assert result == {"tools": [], "resources": []}

    def test_with_tools(self):
        from animus_forge.mcp.client import _normalize_discovery

        tool = MagicMock()
        tool.name = "read_file"
        tool.description = "Read a file"
        tool.inputSchema = {"type": "object"}
        tools_result = MagicMock()
        tools_result.tools = [tool]
        resources_result = MagicMock()
        resources_result.resources = []

        result = _normalize_discovery(tools_result, resources_result)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "read_file"
        assert result["tools"][0]["description"] == "Read a file"
        assert result["tools"][0]["inputSchema"] == {"type": "object"}

    def test_with_resources(self):
        from animus_forge.mcp.client import _normalize_discovery

        res = MagicMock()
        res.uri = "file:///tmp"
        res.name = "tmp"
        res.mimeType = "text/plain"
        res.description = "Temp dir"
        tools_result = MagicMock()
        tools_result.tools = []
        resources_result = MagicMock()
        resources_result.resources = [res]

        result = _normalize_discovery(tools_result, resources_result)
        assert len(result["resources"]) == 1
        assert result["resources"][0]["uri"] == "file:///tmp"
        assert result["resources"][0]["mimeType"] == "text/plain"

    def test_missing_attributes_default(self):
        from animus_forge.mcp.client import _normalize_discovery

        tool = MagicMock(spec=[])
        tool.name = "minimal"
        tools_result = MagicMock()
        tools_result.tools = [tool]
        resources_result = MagicMock()
        resources_result.resources = []

        result = _normalize_discovery(tools_result, resources_result)
        assert result["tools"][0]["name"] == "minimal"
        assert result["tools"][0]["description"] == ""
        assert result["tools"][0]["inputSchema"] == {}


# ---------------------------------------------------------------------------
# discover_tools
# ---------------------------------------------------------------------------


class TestDiscoverTools:
    """Test the sync discover_tools wrapper."""

    def test_raises_when_sdk_missing(self):
        from animus_forge.mcp.client import MCPClientError, discover_tools

        with patch("animus_forge.mcp.client.HAS_MCP_SDK", False):
            with pytest.raises(MCPClientError, match="MCP SDK not installed"):
                discover_tools(server_type="sse", server_url="https://example.com")

    def test_unsupported_type(self):
        from animus_forge.mcp.client import MCPClientError, discover_tools

        with patch("animus_forge.mcp.client.HAS_MCP_SDK", True):
            with pytest.raises(MCPClientError, match="Unsupported MCP server type"):
                discover_tools(server_type="websocket", server_url="ws://localhost")

    def test_stdio_dispatch(self):
        from animus_forge.mcp.client import discover_tools

        mock_result = {"tools": [], "resources": []}
        with patch("animus_forge.mcp.client.HAS_MCP_SDK", True):
            with patch("animus_forge.mcp.client.asyncio.run", return_value=mock_result) as mock_run:
                result = discover_tools(server_type="stdio", server_url="npx mcp-server")
                assert result == mock_result
                mock_run.assert_called_once()

    def test_sse_dispatch(self):
        from animus_forge.mcp.client import discover_tools

        mock_result = {"tools": [{"name": "t1"}], "resources": []}
        with patch("animus_forge.mcp.client.HAS_MCP_SDK", True):
            with patch("animus_forge.mcp.client.asyncio.run", return_value=mock_result) as mock_run:
                result = discover_tools(
                    server_type="sse",
                    server_url="https://example.com/sse",
                    headers={"Authorization": "Bearer tok"},
                )
                assert result == mock_result
                mock_run.assert_called_once()

    def test_wraps_unexpected_errors(self):
        from animus_forge.mcp.client import MCPClientError, discover_tools

        with patch("animus_forge.mcp.client.HAS_MCP_SDK", True):
            with patch(
                "animus_forge.mcp.client.asyncio.run",
                side_effect=ConnectionError("refused"),
            ):
                with pytest.raises(MCPClientError, match="MCP discovery failed"):
                    discover_tools(server_type="sse", server_url="https://dead.server/mcp")
