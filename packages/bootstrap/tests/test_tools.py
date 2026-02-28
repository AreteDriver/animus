"""Comprehensive tests for the tool executor, permissions, built-ins, and MCP bridge."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from animus_bootstrap.intelligence.tools.builtin import get_all_builtin_tools
from animus_bootstrap.intelligence.tools.builtin.filesystem import (
    _file_read,
    _file_write,
    _validate_path,
    get_filesystem_tools,
)
from animus_bootstrap.intelligence.tools.builtin.gateway_tools import (
    _send_message,
    clear_sent_messages,
    get_gateway_tools,
    get_sent_messages,
    set_gateway_router,
)
from animus_bootstrap.intelligence.tools.builtin.memory_tools import (
    _recall_memory,
    _set_reminder,
    _store_memory,
    clear_memory_stores,
    get_memory_tools,
    get_pending_reminders,
    get_stored_memories,
    set_memory_manager,
)
from animus_bootstrap.intelligence.tools.builtin.system import get_system_tools
from animus_bootstrap.intelligence.tools.builtin.web import get_web_tools
from animus_bootstrap.intelligence.tools.executor import (
    ToolDefinition,
    ToolExecutor,
    ToolResult,
)
from animus_bootstrap.intelligence.tools.mcp_bridge import MCPToolBridge
from animus_bootstrap.intelligence.tools.permissions import (
    PermissionLevel,
    ToolApprovalRequired,
    ToolPermissionManager,
)

# ======================================================================
# Helpers
# ======================================================================


async def _echo_handler(text: str = "hello") -> str:
    return f"echo: {text}"


async def _slow_handler() -> str:
    await asyncio.sleep(10)
    return "done"


async def _failing_handler() -> str:
    raise RuntimeError("tool broke")


def _make_tool(
    name: str = "echo",
    permission: str = "auto",
    handler: object | None = None,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Test tool: {name}",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=handler or _echo_handler,  # type: ignore[arg-type]
        category="test",
        permission=permission,
    )


# ======================================================================
# TestToolDefinition
# ======================================================================


class TestToolDefinition:
    def test_required_fields(self) -> None:
        td = _make_tool("my_tool")
        assert td.name == "my_tool"
        assert td.description == "Test tool: my_tool"
        assert td.parameters["type"] == "object"
        assert callable(td.handler)

    def test_defaults(self) -> None:
        td = _make_tool()
        assert td.category == "test"
        assert td.permission == "auto"

    def test_custom_category(self) -> None:
        td = ToolDefinition(
            name="x",
            description="d",
            parameters={},
            handler=_echo_handler,
            category="web",
            permission="approve",
        )
        assert td.category == "web"
        assert td.permission == "approve"


# ======================================================================
# TestToolResult
# ======================================================================


class TestToolResult:
    def test_required_fields(self) -> None:
        now = datetime.now(UTC)
        tr = ToolResult(
            id="abc",
            tool_name="echo",
            success=True,
            output="hello",
            duration_ms=1.5,
            timestamp=now,
        )
        assert tr.id == "abc"
        assert tr.tool_name == "echo"
        assert tr.success is True
        assert tr.output == "hello"
        assert tr.duration_ms == 1.5
        assert tr.timestamp == now

    def test_default_arguments(self) -> None:
        now = datetime.now(UTC)
        tr = ToolResult(
            id="x", tool_name="t", success=True, output="", duration_ms=0, timestamp=now
        )
        assert tr.arguments == {}

    def test_custom_arguments(self) -> None:
        now = datetime.now(UTC)
        tr = ToolResult(
            id="x",
            tool_name="t",
            success=True,
            output="",
            duration_ms=0,
            timestamp=now,
            arguments={"key": "val"},
        )
        assert tr.arguments == {"key": "val"}


# ======================================================================
# TestToolExecutor
# ======================================================================


class TestToolExecutor:
    def test_register_and_list(self) -> None:
        ex = ToolExecutor()
        tool = _make_tool("a")
        ex.register(tool)
        assert len(ex.list_tools()) == 1
        assert ex.list_tools()[0].name == "a"

    def test_register_duplicate_raises(self) -> None:
        ex = ToolExecutor()
        ex.register(_make_tool("dup"))
        with pytest.raises(ValueError, match="already registered"):
            ex.register(_make_tool("dup"))

    def test_unregister_removes_tool(self) -> None:
        ex = ToolExecutor()
        ex.register(_make_tool("rm"))
        ex.unregister("rm")
        assert ex.list_tools() == []

    def test_unregister_nonexistent_is_noop(self) -> None:
        ex = ToolExecutor()
        ex.unregister("ghost")  # Should not raise

    def test_get_tool_by_name(self) -> None:
        ex = ToolExecutor()
        ex.register(_make_tool("find_me"))
        assert ex.get_tool("find_me") is not None
        assert ex.get_tool("find_me").name == "find_me"  # type: ignore[union-attr]

    def test_get_tool_returns_none_for_unknown(self) -> None:
        ex = ToolExecutor()
        assert ex.get_tool("nonexistent") is None

    def test_get_schemas_anthropic_format(self) -> None:
        ex = ToolExecutor()
        ex.register(_make_tool("s1"))
        ex.register(_make_tool("s2"))
        schemas = ex.get_schemas()
        assert len(schemas) == 2
        for s in schemas:
            assert "name" in s
            assert "description" in s
            assert "input_schema" in s

    def test_get_schemas_empty(self) -> None:
        ex = ToolExecutor()
        assert ex.get_schemas() == []

    @pytest.mark.asyncio()
    async def test_execute_calls_handler(self) -> None:
        ex = ToolExecutor()
        ex.register(_make_tool("echo"))
        result = await ex.execute("echo", {"text": "world"})
        assert result.success is True
        assert result.output == "echo: world"
        assert result.tool_name == "echo"
        assert result.arguments == {"text": "world"}
        assert result.duration_ms >= 0

    @pytest.mark.asyncio()
    async def test_execute_records_in_history(self) -> None:
        ex = ToolExecutor()
        ex.register(_make_tool("h"))
        await ex.execute("h", {"text": "a"})
        await ex.execute("h", {"text": "b"})
        history = ex.get_history()
        assert len(history) == 2
        assert history[0].arguments == {"text": "a"}
        assert history[1].arguments == {"text": "b"}

    @pytest.mark.asyncio()
    async def test_execute_unknown_tool(self) -> None:
        ex = ToolExecutor()
        result = await ex.execute("missing", {"text": "x"})
        assert result.success is False
        assert "Unknown tool" in result.output

    @pytest.mark.asyncio()
    async def test_execute_denied_permission(self) -> None:
        pm = ToolPermissionManager(default=PermissionLevel.DENY)
        ex = ToolExecutor(permission_manager=pm)
        ex.register(_make_tool("blocked"))
        result = await ex.execute("blocked", {"text": "x"})
        assert result.success is False
        assert "denied" in result.output

    @pytest.mark.asyncio()
    async def test_execute_approve_permission_returns_error(self) -> None:
        pm = ToolPermissionManager(default=PermissionLevel.APPROVE)
        ex = ToolExecutor(permission_manager=pm)
        ex.register(_make_tool("needs_approval"))
        result = await ex.execute("needs_approval", {"text": "x"})
        assert result.success is False
        assert "approval" in result.output

    @pytest.mark.asyncio()
    async def test_execute_tool_with_approve_permission_field(self) -> None:
        ex = ToolExecutor()
        tool = _make_tool("gated", permission="approve")
        ex.register(tool)
        result = await ex.execute("gated", {"text": "x"})
        assert result.success is False
        assert "approval" in result.output

    @pytest.mark.asyncio()
    async def test_execute_approve_calls_callback_and_proceeds(self) -> None:
        callback = AsyncMock(return_value=True)
        pm = ToolPermissionManager(default=PermissionLevel.APPROVE)
        ex = ToolExecutor(permission_manager=pm, approval_callback=callback)
        ex.register(_make_tool("gated"))
        result = await ex.execute("gated", {"text": "hi"})
        assert result.success is True
        assert result.output == "echo: hi"
        callback.assert_called_once_with("gated", {"text": "hi"})

    @pytest.mark.asyncio()
    async def test_execute_approve_callback_denies(self) -> None:
        callback = AsyncMock(return_value=False)
        pm = ToolPermissionManager(default=PermissionLevel.APPROVE)
        ex = ToolExecutor(permission_manager=pm, approval_callback=callback)
        ex.register(_make_tool("gated"))
        result = await ex.execute("gated", {"text": "x"})
        assert result.success is False
        assert "denied by user" in result.output

    @pytest.mark.asyncio()
    async def test_execute_approve_callback_exception_denies(self) -> None:
        callback = AsyncMock(side_effect=RuntimeError("callback broke"))
        pm = ToolPermissionManager(default=PermissionLevel.APPROVE)
        ex = ToolExecutor(permission_manager=pm, approval_callback=callback)
        ex.register(_make_tool("gated"))
        result = await ex.execute("gated", {"text": "x"})
        assert result.success is False
        assert "denied by user" in result.output

    @pytest.mark.asyncio()
    async def test_set_approval_callback_after_init(self) -> None:
        callback = AsyncMock(return_value=True)
        ex = ToolExecutor()
        ex.register(_make_tool("gated", permission="approve"))
        # Without callback -> fails
        result = await ex.execute("gated", {"text": "x"})
        assert result.success is False
        # Set callback -> succeeds
        ex.set_approval_callback(callback)
        result = await ex.execute("gated", {"text": "y"})
        assert result.success is True
        # Clear callback -> fails again
        ex.set_approval_callback(None)
        result = await ex.execute("gated", {"text": "z"})
        assert result.success is False

    @pytest.mark.asyncio()
    async def test_execute_with_timeout(self) -> None:
        ex = ToolExecutor(timeout_seconds=0.1)
        ex.register(_make_tool("slow", handler=_slow_handler))
        result = await ex.execute("slow", {})
        assert result.success is False
        assert "timed out" in result.output

    @pytest.mark.asyncio()
    async def test_execute_handler_raises(self) -> None:
        ex = ToolExecutor()
        ex.register(_make_tool("fail", handler=_failing_handler))
        result = await ex.execute("fail", {})
        assert result.success is False
        assert "tool broke" in result.output

    @pytest.mark.asyncio()
    async def test_execute_batch_sequential(self) -> None:
        ex = ToolExecutor(max_calls_per_turn=10)
        ex.register(_make_tool("b"))
        calls = [
            {"name": "b", "arguments": {"text": "1"}},
            {"name": "b", "arguments": {"text": "2"}},
            {"name": "b", "arguments": {"text": "3"}},
        ]
        results = await ex.execute_batch(calls)
        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio()
    async def test_execute_batch_respects_max_calls(self) -> None:
        ex = ToolExecutor(max_calls_per_turn=2)
        ex.register(_make_tool("c"))
        calls = [
            {"name": "c", "arguments": {"text": "1"}},
            {"name": "c", "arguments": {"text": "2"}},
            {"name": "c", "arguments": {"text": "3"}},
            {"name": "c", "arguments": {"text": "4"}},
        ]
        results = await ex.execute_batch(calls)
        assert len(results) == 2

    @pytest.mark.asyncio()
    async def test_execute_batch_empty(self) -> None:
        ex = ToolExecutor()
        results = await ex.execute_batch([])
        assert results == []

    @pytest.mark.asyncio()
    async def test_execute_batch_with_unknown_tool(self) -> None:
        ex = ToolExecutor()
        calls = [{"name": "nope", "arguments": {}}]
        results = await ex.execute_batch(calls)
        assert len(results) == 1
        assert not results[0].success

    def test_get_history_limit(self) -> None:
        ex = ToolExecutor()
        # Manually inject history entries
        now = datetime.now(UTC)
        for i in range(10):
            ex._history.append(
                ToolResult(
                    id=str(i),
                    tool_name="t",
                    success=True,
                    output=str(i),
                    duration_ms=0,
                    timestamp=now,
                )
            )
        assert len(ex.get_history(limit=3)) == 3
        assert ex.get_history(limit=3)[0].id == "7"

    def test_clear_history(self) -> None:
        ex = ToolExecutor()
        now = datetime.now(UTC)
        ex._history.append(
            ToolResult(id="1", tool_name="t", success=True, output="", duration_ms=0, timestamp=now)
        )
        ex.clear_history()
        assert ex.get_history() == []

    @pytest.mark.asyncio()
    async def test_execute_result_has_timestamp(self) -> None:
        ex = ToolExecutor()
        ex.register(_make_tool("ts"))
        before = datetime.now(UTC)
        result = await ex.execute("ts", {"text": "x"})
        after = datetime.now(UTC)
        assert before <= result.timestamp <= after

    @pytest.mark.asyncio()
    async def test_execute_result_has_uuid_id(self) -> None:
        ex = ToolExecutor()
        ex.register(_make_tool("uid"))
        result = await ex.execute("uid", {"text": "x"})
        # UUID4 format: 8-4-4-4-12 hex chars
        assert len(result.id) == 36
        assert result.id.count("-") == 4

    @pytest.mark.asyncio()
    async def test_execute_duration_is_positive(self) -> None:
        ex = ToolExecutor()
        ex.register(_make_tool("dur"))
        result = await ex.execute("dur", {"text": "x"})
        assert result.duration_ms >= 0


# ======================================================================
# TestPermissionLevel
# ======================================================================


class TestPermissionLevel:
    def test_auto_value(self) -> None:
        assert PermissionLevel.AUTO == "auto"
        assert PermissionLevel.AUTO.value == "auto"

    def test_approve_value(self) -> None:
        assert PermissionLevel.APPROVE == "approve"

    def test_deny_value(self) -> None:
        assert PermissionLevel.DENY == "deny"

    def test_is_str_subclass(self) -> None:
        assert isinstance(PermissionLevel.AUTO, str)

    def test_from_string(self) -> None:
        assert PermissionLevel("auto") == PermissionLevel.AUTO
        assert PermissionLevel("approve") == PermissionLevel.APPROVE
        assert PermissionLevel("deny") == PermissionLevel.DENY


# ======================================================================
# TestToolPermissionManager
# ======================================================================


class TestToolPermissionManager:
    def test_default_permission(self) -> None:
        pm = ToolPermissionManager()
        assert pm.get_permission("anything") == PermissionLevel.AUTO

    def test_custom_default(self) -> None:
        pm = ToolPermissionManager(default=PermissionLevel.DENY)
        assert pm.get_permission("anything") == PermissionLevel.DENY

    def test_set_and_get_override(self) -> None:
        pm = ToolPermissionManager()
        pm.set_permission("shell", PermissionLevel.APPROVE)
        assert pm.get_permission("shell") == PermissionLevel.APPROVE

    def test_remove_override_falls_back_to_default(self) -> None:
        pm = ToolPermissionManager(default=PermissionLevel.AUTO)
        pm.set_permission("x", PermissionLevel.DENY)
        pm.remove_override("x")
        assert pm.get_permission("x") == PermissionLevel.AUTO

    def test_remove_override_nonexistent_is_noop(self) -> None:
        pm = ToolPermissionManager()
        pm.remove_override("ghost")  # Should not raise

    def test_list_overrides(self) -> None:
        pm = ToolPermissionManager()
        pm.set_permission("a", PermissionLevel.APPROVE)
        pm.set_permission("b", PermissionLevel.DENY)
        overrides = pm.list_overrides()
        assert overrides == {"a": PermissionLevel.APPROVE, "b": PermissionLevel.DENY}

    def test_list_overrides_empty(self) -> None:
        pm = ToolPermissionManager()
        assert pm.list_overrides() == {}

    def test_check_returns_true_for_auto(self) -> None:
        pm = ToolPermissionManager()
        assert pm.check("auto_tool") is True

    def test_check_raises_for_approve(self) -> None:
        pm = ToolPermissionManager()
        pm.set_permission("gated", PermissionLevel.APPROVE)
        with pytest.raises(ToolApprovalRequired) as exc_info:
            pm.check("gated")
        assert exc_info.value.tool_name == "gated"

    def test_check_returns_false_for_deny(self) -> None:
        pm = ToolPermissionManager()
        pm.set_permission("blocked", PermissionLevel.DENY)
        assert pm.check("blocked") is False

    def test_approval_exception_message(self) -> None:
        exc = ToolApprovalRequired("my_tool")
        assert "my_tool" in str(exc)
        assert "approval" in str(exc).lower()


# ======================================================================
# TestBuiltinTools — Web
# ======================================================================


class TestWebTools:
    def test_web_search_tool_exists(self) -> None:
        tools = get_web_tools()
        names = [t.name for t in tools]
        assert "web_search" in names

    def test_web_fetch_tool_exists(self) -> None:
        tools = get_web_tools()
        names = [t.name for t in tools]
        assert "web_fetch" in names

    def test_web_fetch_has_correct_schema(self) -> None:
        tools = get_web_tools()
        fetch = next(t for t in tools if t.name == "web_fetch")
        assert fetch.parameters["type"] == "object"
        assert "url" in fetch.parameters["properties"]

    def test_web_search_has_correct_schema(self) -> None:
        tools = get_web_tools()
        search = next(t for t in tools if t.name == "web_search")
        assert "query" in search.parameters["properties"]

    def test_web_tools_are_web_category(self) -> None:
        tools = get_web_tools()
        assert all(t.category == "web" for t in tools)

    @pytest.mark.asyncio()
    async def test_web_search_without_ddg(self) -> None:
        with patch("animus_bootstrap.intelligence.tools.builtin.web.HAS_DUCKDUCKGO", False):
            result = await get_web_tools()[0].handler(query="test")
            assert "Would search for" in result

    @pytest.mark.asyncio()
    async def test_web_fetch_strips_html(self) -> None:
        import httpx

        _req = httpx.Request("GET", "https://example.com")
        mock_response = httpx.Response(
            200, text="<html><body><p>Hello World</p></body></html>", request=_req
        )
        with patch("httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.get.return_value = mock_response
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            fetch = next(t for t in get_web_tools() if t.name == "web_fetch")
            result = await fetch.handler(url="https://example.com")
            assert "Hello World" in result
            assert "<p>" not in result

    @pytest.mark.asyncio()
    async def test_web_fetch_truncates_to_2000(self) -> None:
        import httpx

        _req = httpx.Request("GET", "https://example.com")
        mock_response = httpx.Response(200, text="A" * 5000, request=_req)
        with patch("httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.get.return_value = mock_response
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            fetch = next(t for t in get_web_tools() if t.name == "web_fetch")
            result = await fetch.handler(url="https://example.com")
            assert len(result) <= 2000

    @pytest.mark.asyncio()
    async def test_web_fetch_handles_http_error(self) -> None:
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

            fetch = next(t for t in get_web_tools() if t.name == "web_fetch")
            result = await fetch.handler(url="https://down.example.com")
            assert "Fetch failed" in result


# ======================================================================
# TestBuiltinTools — Filesystem
# ======================================================================


class TestFilesystemTools:
    def test_file_read_tool_exists(self) -> None:
        tools = get_filesystem_tools()
        names = [t.name for t in tools]
        assert "file_read" in names

    def test_file_write_tool_exists(self) -> None:
        tools = get_filesystem_tools()
        names = [t.name for t in tools]
        assert "file_write" in names

    def test_filesystem_tools_category(self) -> None:
        tools = get_filesystem_tools()
        assert all(t.category == "filesystem" for t in tools)

    @pytest.mark.asyncio()
    async def test_file_read_reads_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello content", encoding="utf-8")
        with patch(
            "animus_bootstrap.intelligence.tools.builtin.filesystem._DEFAULT_ALLOWED_ROOTS",
            [str(tmp_path)],
        ):
            result = await _file_read(str(f))
        assert result == "hello content"

    @pytest.mark.asyncio()
    async def test_file_read_file_not_found(self, tmp_path: Path) -> None:
        with patch(
            "animus_bootstrap.intelligence.tools.builtin.filesystem._DEFAULT_ALLOWED_ROOTS",
            [str(tmp_path)],
        ):
            result = await _file_read(str(tmp_path / "nonexistent.txt"))
        assert "not found" in result.lower() or "File not found" in result

    @pytest.mark.asyncio()
    async def test_file_read_rejects_outside_sandbox(self) -> None:
        await _file_read("/etc/shadow")
        # /etc/shadow may or may not be under home; path validation depends
        # on whether /etc is under ~. Force a restrictive sandbox:
        tools = get_filesystem_tools(allowed_roots=["/tmp/safe_only"])
        read_tool = next(t for t in tools if t.name == "file_read")
        result = await read_tool.handler(path="/etc/passwd")
        assert "Permission denied" in result

    @pytest.mark.asyncio()
    async def test_file_write_writes_file(self, tmp_path: Path) -> None:
        target = tmp_path / "output.txt"
        with patch(
            "animus_bootstrap.intelligence.tools.builtin.filesystem._DEFAULT_ALLOWED_ROOTS",
            [str(tmp_path)],
        ):
            result = await _file_write(str(target), "new content")
        assert "Successfully wrote" in result
        assert target.read_text(encoding="utf-8") == "new content"

    @pytest.mark.asyncio()
    async def test_file_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "dir" / "file.txt"
        with patch(
            "animus_bootstrap.intelligence.tools.builtin.filesystem._DEFAULT_ALLOWED_ROOTS",
            [str(tmp_path)],
        ):
            result = await _file_write(str(target), "nested")
        assert "Successfully wrote" in result
        assert target.exists()

    @pytest.mark.asyncio()
    async def test_file_write_rejects_outside_sandbox(self) -> None:
        tools = get_filesystem_tools(allowed_roots=["/tmp/safe_only"])
        write_tool = next(t for t in tools if t.name == "file_write")
        result = await write_tool.handler(path="/etc/evil.txt", content="bad")
        assert "Permission denied" in result

    def test_validate_path_valid(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.touch()
        result = _validate_path(str(f), [str(tmp_path)])
        assert result == f.resolve()

    def test_validate_path_rejects_escape(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="outside allowed"):
            _validate_path("/etc/passwd", [str(tmp_path)])

    def test_validate_path_resolves_symlinks(self, tmp_path: Path) -> None:
        real = tmp_path / "real.txt"
        real.touch()
        link = tmp_path / "link.txt"
        link.symlink_to(real)
        result = _validate_path(str(link), [str(tmp_path)])
        assert result == real.resolve()

    def test_validate_path_rejects_symlink_escape(self, tmp_path: Path) -> None:
        link = tmp_path / "escape"
        link.symlink_to("/etc")
        with pytest.raises(ValueError, match="outside allowed"):
            _validate_path(str(link / "passwd"), [str(tmp_path)])

    @pytest.mark.asyncio()
    async def test_file_read_directory_error(self, tmp_path: Path) -> None:
        with patch(
            "animus_bootstrap.intelligence.tools.builtin.filesystem._DEFAULT_ALLOWED_ROOTS",
            [str(tmp_path)],
        ):
            result = await _file_read(str(tmp_path))
        assert "directory" in result.lower() or "Directory" in result

    @pytest.mark.asyncio()
    async def test_custom_allowed_roots(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("custom root", encoding="utf-8")
        tools = get_filesystem_tools(allowed_roots=[str(tmp_path)])
        read_tool = next(t for t in tools if t.name == "file_read")
        result = await read_tool.handler(path=str(f))
        assert result == "custom root"


# ======================================================================
# TestBuiltinTools — System
# ======================================================================


class TestSystemTools:
    def test_shell_exec_has_approve_permission(self) -> None:
        tools = get_system_tools()
        shell = tools[0]
        assert shell.name == "shell_exec"
        assert shell.permission == "approve"

    def test_shell_exec_schema(self) -> None:
        tools = get_system_tools()
        shell = tools[0]
        assert "command" in shell.parameters["properties"]

    def test_shell_exec_category(self) -> None:
        tools = get_system_tools()
        assert tools[0].category == "system"

    @pytest.mark.asyncio()
    async def test_shell_exec_registered_requires_approval(self) -> None:
        ex = ToolExecutor()
        for t in get_system_tools():
            ex.register(t)
        result = await ex.execute("shell_exec", {"command": "echo hi"})
        assert result.success is False
        assert "approval" in result.output


# ======================================================================
# TestBuiltinTools — Gateway
# ======================================================================


class TestGatewayTools:
    def setup_method(self) -> None:
        clear_sent_messages()
        set_gateway_router(None)

    def test_send_message_tool_exists(self) -> None:
        tools = get_gateway_tools()
        assert any(t.name == "send_message" for t in tools)

    def test_send_message_schema(self) -> None:
        tools = get_gateway_tools()
        sm = next(t for t in tools if t.name == "send_message")
        props = sm.parameters["properties"]
        assert "channel" in props
        assert "text" in props

    @pytest.mark.asyncio()
    async def test_send_message_stores_message(self) -> None:
        result = await _send_message("telegram", "hello world")
        assert "queued" in result.lower() or "Message queued" in result
        msgs = get_sent_messages()
        assert len(msgs) == 1
        assert msgs[0]["channel"] == "telegram"
        assert msgs[0]["text"] == "hello world"

    @pytest.mark.asyncio()
    async def test_send_message_multiple(self) -> None:
        await _send_message("discord", "msg1")
        await _send_message("slack", "msg2")
        assert len(get_sent_messages()) == 2

    @pytest.mark.asyncio()
    async def test_send_message_via_router(self) -> None:
        router = AsyncMock()
        router.broadcast = AsyncMock()
        set_gateway_router(router)
        result = await _send_message("telegram", "hello via router")
        assert "sent" in result.lower()
        router.broadcast.assert_called_once_with("hello via router", ["telegram"])
        # Should NOT be in fallback list when router works
        assert get_sent_messages() == []

    @pytest.mark.asyncio()
    async def test_send_message_router_failure_falls_back(self) -> None:
        router = AsyncMock()
        router.broadcast = AsyncMock(side_effect=RuntimeError("channel down"))
        set_gateway_router(router)
        result = await _send_message("discord", "fallback msg")
        assert "queued" in result.lower()
        assert len(get_sent_messages()) == 1

    def test_clear_sent_messages(self) -> None:
        asyncio.get_event_loop().run_until_complete(_send_message("ch", "m"))
        clear_sent_messages()
        assert get_sent_messages() == []


# ======================================================================
# TestBuiltinTools — Memory
# ======================================================================


class TestMemoryTools:
    def setup_method(self) -> None:
        clear_memory_stores()
        set_memory_manager(None)

    def test_store_memory_tool_exists(self) -> None:
        tools = get_memory_tools()
        assert any(t.name == "store_memory" for t in tools)

    def test_recall_memory_tool_exists(self) -> None:
        tools = get_memory_tools()
        assert any(t.name == "recall_memory" for t in tools)

    def test_set_reminder_tool_exists(self) -> None:
        tools = get_memory_tools()
        assert any(t.name == "set_reminder" for t in tools)

    def test_store_memory_schema(self) -> None:
        tools = get_memory_tools()
        sm = next(t for t in tools if t.name == "store_memory")
        assert "content" in sm.parameters["properties"]
        assert "memory_type" in sm.parameters["properties"]

    def test_set_reminder_schema(self) -> None:
        tools = get_memory_tools()
        sr = next(t for t in tools if t.name == "set_reminder")
        assert "message" in sr.parameters["properties"]
        assert "delay_minutes" in sr.parameters["properties"]

    @pytest.mark.asyncio()
    async def test_store_memory_stores(self) -> None:
        result = await _store_memory("important fact", "semantic")
        assert "Stored" in result
        memories = get_stored_memories()
        assert len(memories) == 1
        assert memories[0]["content"] == "important fact"
        assert memories[0]["memory_type"] == "semantic"

    @pytest.mark.asyncio()
    async def test_store_memory_default_type(self) -> None:
        result = await _store_memory("another fact")
        assert "semantic" in result
        assert get_stored_memories()[0]["memory_type"] == "semantic"

    @pytest.mark.asyncio()
    async def test_set_reminder_stores(self) -> None:
        result = await _set_reminder("wake up", 30)
        assert "Reminder set" in result
        reminders = get_pending_reminders()
        assert len(reminders) == 1
        assert reminders[0]["message"] == "wake up"
        assert reminders[0]["delay_minutes"] == 30

    @pytest.mark.asyncio()
    async def test_set_reminder_includes_scheduled_time(self) -> None:
        result = await _set_reminder("meeting", 60)
        # scheduled_for should be an ISO timestamp
        assert "T" in result
        reminders = get_pending_reminders()
        assert "scheduled_for" in reminders[0]

    def test_clear_memory_stores(self) -> None:
        asyncio.get_event_loop().run_until_complete(_store_memory("x"))
        asyncio.get_event_loop().run_until_complete(_set_reminder("y", 5))
        clear_memory_stores()
        assert get_stored_memories() == []
        assert get_pending_reminders() == []

    @pytest.mark.asyncio()
    async def test_recall_memory_fallback_found(self) -> None:
        await _store_memory("The sky is blue", "semantic")
        await _store_memory("How to bake bread", "procedural")
        result = await _recall_memory("sky")
        assert "Found 1" in result
        assert "sky is blue" in result

    @pytest.mark.asyncio()
    async def test_recall_memory_fallback_not_found(self) -> None:
        result = await _recall_memory("nonexistent topic")
        assert "No memories found" in result

    @pytest.mark.asyncio()
    async def test_recall_memory_fallback_filter_type(self) -> None:
        await _store_memory("fact A", "semantic")
        await _store_memory("experience B", "episodic")
        result = await _recall_memory("A", memory_type="semantic")
        assert "Found 1" in result

    @pytest.mark.asyncio()
    async def test_store_memory_with_backend(self) -> None:
        backend = AsyncMock()
        backend.store = AsyncMock()
        mgr = AsyncMock()
        mgr._backend = backend
        set_memory_manager(mgr)
        result = await _store_memory("test fact", "semantic")
        assert "Stored" in result
        backend.store.assert_called_once_with("semantic", "test fact", {"source": "tool"})

    @pytest.mark.asyncio()
    async def test_recall_memory_with_backend(self) -> None:
        mgr = AsyncMock()
        mgr.search = AsyncMock(
            return_value=[
                {"content": "The answer is 42", "memory_type": "semantic"},
            ]
        )
        set_memory_manager(mgr)
        result = await _recall_memory("answer")
        assert "Found 1" in result
        assert "42" in result
        mgr.search.assert_called_once_with("answer", memory_type="all", limit=10)

    @pytest.mark.asyncio()
    async def test_recall_memory_backend_empty(self) -> None:
        mgr = AsyncMock()
        mgr.search = AsyncMock(return_value=[])
        set_memory_manager(mgr)
        result = await _recall_memory("nothing")
        assert "No memories found" in result

    @pytest.mark.asyncio()
    async def test_store_memory_backend_failure_falls_back(self) -> None:
        backend = AsyncMock()
        backend.store = AsyncMock(side_effect=RuntimeError("db error"))
        mgr = AsyncMock()
        mgr._backend = backend
        set_memory_manager(mgr)
        result = await _store_memory("fallback test", "semantic")
        assert "Stored" in result
        # Should be in the fallback list
        assert len(get_stored_memories()) == 1

    @pytest.mark.asyncio()
    async def test_memory_tools_category(self) -> None:
        tools = get_memory_tools()
        assert all(t.category == "memory" for t in tools)

    def test_recall_memory_schema(self) -> None:
        tools = get_memory_tools()
        recall = next(t for t in tools if t.name == "recall_memory")
        assert "query" in recall.parameters["properties"]


# ======================================================================
# TestBuiltinTools — get_all_builtin_tools
# ======================================================================


class TestGetAllBuiltinTools:
    def test_returns_list_of_tool_definitions(self) -> None:
        tools = get_all_builtin_tools()
        assert isinstance(tools, list)
        assert all(isinstance(t, ToolDefinition) for t in tools)

    def test_includes_all_expected_tools(self) -> None:
        tools = get_all_builtin_tools()
        names = {t.name for t in tools}
        expected = {
            "web_search",
            "web_fetch",
            "file_read",
            "file_write",
            "shell_exec",
            "send_message",
            "store_memory",
            "recall_memory",
            "set_reminder",
        }
        assert expected.issubset(names)

    def test_no_duplicate_names(self) -> None:
        tools = get_all_builtin_tools()
        names = [t.name for t in tools]
        assert len(names) == len(set(names))

    def test_all_have_handlers(self) -> None:
        tools = get_all_builtin_tools()
        assert all(callable(t.handler) for t in tools)

    def test_all_have_descriptions(self) -> None:
        tools = get_all_builtin_tools()
        assert all(t.description for t in tools)

    def test_all_have_parameter_schemas(self) -> None:
        tools = get_all_builtin_tools()
        assert all(isinstance(t.parameters, dict) for t in tools)


# ======================================================================
# TestMCPToolBridge
# ======================================================================


class TestMCPToolBridge:
    @pytest.mark.asyncio()
    async def test_discover_servers_with_valid_config(self, tmp_path: Path) -> None:
        config = tmp_path / "mcp.json"
        config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "memory": {"command": "memory-server", "args": []},
                        "fetch": {"command": "fetch-server", "args": ["--port", "8080"]},
                    }
                }
            ),
            encoding="utf-8",
        )
        bridge = MCPToolBridge(config_path=config)
        servers = await bridge.discover_servers()
        assert set(servers) == {"memory", "fetch"}

    @pytest.mark.asyncio()
    async def test_discover_servers_no_config(self) -> None:
        bridge = MCPToolBridge(config_path=None)
        servers = await bridge.discover_servers()
        assert servers == []

    @pytest.mark.asyncio()
    async def test_discover_servers_missing_file(self, tmp_path: Path) -> None:
        bridge = MCPToolBridge(config_path=tmp_path / "nonexistent.json")
        servers = await bridge.discover_servers()
        assert servers == []

    @pytest.mark.asyncio()
    async def test_discover_servers_invalid_json(self, tmp_path: Path) -> None:
        config = tmp_path / "bad.json"
        config.write_text("not json!", encoding="utf-8")
        bridge = MCPToolBridge(config_path=config)
        servers = await bridge.discover_servers()
        assert servers == []

    @pytest.mark.asyncio()
    async def test_discover_servers_no_mcp_key(self, tmp_path: Path) -> None:
        config = tmp_path / "empty.json"
        config.write_text("{}", encoding="utf-8")
        bridge = MCPToolBridge(config_path=config)
        servers = await bridge.discover_servers()
        assert servers == []

    @pytest.mark.asyncio()
    async def test_import_tools_fails_on_missing_command(self, tmp_path: Path) -> None:
        config = tmp_path / "mcp.json"
        config.write_text(
            json.dumps({"mcpServers": {"test": {"command": "nonexistent_mcp_server_xyz"}}}),
            encoding="utf-8",
        )
        bridge = MCPToolBridge(config_path=config)
        await bridge.discover_servers()
        tools = await bridge.import_tools("test")
        assert tools == []

    @pytest.mark.asyncio()
    async def test_import_tools_unknown_server(self) -> None:
        bridge = MCPToolBridge()
        tools = await bridge.import_tools("unknown")
        assert tools == []

    @pytest.mark.asyncio()
    async def test_call_tool_not_connected(self) -> None:
        bridge = MCPToolBridge()
        result = await bridge.call_tool("server", "tool", {})
        assert "not connected" in result

    @pytest.mark.asyncio()
    async def test_import_tools_with_mock_process(self) -> None:
        bridge = MCPToolBridge()
        bridge._servers = {"test": {"command": "echo"}}

        # Mock the subprocess to return proper JSON-RPC responses
        mock_proc = AsyncMock()
        mock_proc.pid = 999
        mock_proc.returncode = None
        mock_proc.stdin = AsyncMock()
        mock_proc.stdin.write = lambda data: None
        mock_proc.stdin.drain = AsyncMock()

        # Response queue: initialize, then tools/list
        init_response = (
            json.dumps(
                {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}
            ).encode()
            + b"\n"
        )
        tools_response = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {
                        "tools": [
                            {
                                "name": "greet",
                                "description": "Say hello",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"name": {"type": "string"}},
                                },
                            }
                        ]
                    },
                }
            ).encode()
            + b"\n"
        )

        responses = [init_response, tools_response]
        response_idx = 0

        async def mock_readline():
            nonlocal response_idx
            if response_idx < len(responses):
                data = responses[response_idx]
                response_idx += 1
                return data
            return b""

        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = mock_readline
        mock_proc.terminate = lambda: None
        mock_proc.kill = lambda: None
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            tools = await bridge.import_tools("test")

        assert len(tools) == 1
        assert tools[0].name == "mcp_test_greet"
        assert tools[0].description == "Say hello"
        assert tools[0].category == "mcp:test"
        assert callable(tools[0].handler)

    @pytest.mark.asyncio()
    async def test_close_terminates_connections(self) -> None:
        bridge = MCPToolBridge()
        mock_conn = AsyncMock()
        bridge._connections = {"test": mock_conn}
        await bridge.close()
        mock_conn.close.assert_called_once()
        assert bridge._connections == {}

    @pytest.mark.asyncio()
    async def test_discover_populates_servers_dict(self, tmp_path: Path) -> None:
        config = tmp_path / "mcp.json"
        server_config = {"command": "srv", "args": []}
        config.write_text(
            json.dumps({"mcpServers": {"myserver": server_config}}),
            encoding="utf-8",
        )
        bridge = MCPToolBridge(config_path=config)
        await bridge.discover_servers()
        assert "myserver" in bridge._servers
        assert bridge._servers["myserver"] == server_config

    @pytest.mark.asyncio()
    async def test_discover_servers_bad_mcp_type(self, tmp_path: Path) -> None:
        config = tmp_path / "mcp.json"
        config.write_text(json.dumps({"mcpServers": "not-a-dict"}), encoding="utf-8")
        bridge = MCPToolBridge(config_path=config)
        servers = await bridge.discover_servers()
        assert servers == []


# ------------------------------------------------------------------
# MCP auto-discovery in runtime
# ------------------------------------------------------------------


class TestMCPAutoDiscovery:
    """Tests for MCP auto-discovery wiring in AnimusRuntime."""

    @pytest.mark.asyncio()
    async def test_discover_no_config_file(self, tmp_path: Path) -> None:
        """Runtime skips MCP when config file doesn't exist."""
        from animus_bootstrap.config.schema import AnimusConfig

        cfg = AnimusConfig()
        cfg.intelligence.mcp.config_path = str(tmp_path / "nonexistent.json")
        cfg.intelligence.mcp.auto_discover = True

        from animus_bootstrap.runtime import AnimusRuntime

        runtime = AnimusRuntime(config=cfg)
        # _discover_mcp_tools should not crash
        runtime.tool_executor = runtime._create_tool_executor()
        await runtime._discover_mcp_tools()
        assert runtime._mcp_bridge is None

    @pytest.mark.asyncio()
    async def test_discover_empty_config(self, tmp_path: Path) -> None:
        """Runtime handles empty MCP config gracefully."""
        config = tmp_path / "mcp.json"
        config.write_text('{"mcpServers": {}}', encoding="utf-8")

        from animus_bootstrap.config.schema import AnimusConfig

        cfg = AnimusConfig()
        cfg.intelligence.mcp.config_path = str(config)
        cfg.intelligence.mcp.auto_discover = True

        from animus_bootstrap.runtime import AnimusRuntime

        runtime = AnimusRuntime(config=cfg)
        runtime.tool_executor = runtime._create_tool_executor()
        await runtime._discover_mcp_tools()
        assert runtime._mcp_bridge is None

    @pytest.mark.asyncio()
    async def test_discover_with_servers(self, tmp_path: Path) -> None:
        """Runtime discovers servers and attempts import."""
        config = tmp_path / "mcp.json"
        config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "test_srv": {"command": "nonexistent_binary", "args": []},
                    }
                }
            ),
            encoding="utf-8",
        )

        from animus_bootstrap.config.schema import AnimusConfig

        cfg = AnimusConfig()
        cfg.intelligence.mcp.config_path = str(config)
        cfg.intelligence.mcp.auto_discover = True

        from animus_bootstrap.runtime import AnimusRuntime

        runtime = AnimusRuntime(config=cfg)
        runtime.tool_executor = runtime._create_tool_executor()
        # import_tools will fail because the binary doesn't exist,
        # but it should not crash
        await runtime._discover_mcp_tools()
        # Bridge was created (discover found servers) but no tools imported
        assert runtime._mcp_bridge is not None


# ======================================================================
# ToolHistoryStore — persistent SQLite storage
# ======================================================================


class TestToolHistoryStore:
    """Tests for the persistent tool history store."""

    def test_save_and_list(self, tmp_path: Path) -> None:
        from animus_bootstrap.intelligence.tools.history_store import ToolHistoryStore

        store = ToolHistoryStore(tmp_path / "history.db")
        result = ToolResult(
            id="abc-123",
            tool_name="web_search",
            success=True,
            output="found it",
            duration_ms=150.5,
            timestamp=datetime(2026, 2, 20, 12, 0, 0, tzinfo=UTC),
            arguments={"query": "test"},
        )
        store.save(result)
        items = store.list_recent(limit=10)
        assert len(items) == 1
        assert items[0]["tool_name"] == "web_search"
        assert items[0]["success"] is True
        assert items[0]["arguments"] == {"query": "test"}
        store.close()

    def test_count(self, tmp_path: Path) -> None:
        from animus_bootstrap.intelligence.tools.history_store import ToolHistoryStore

        store = ToolHistoryStore(tmp_path / "history.db")
        for i in range(5):
            store.save(
                ToolResult(
                    id=f"id-{i}",
                    tool_name="tool",
                    success=True,
                    output="ok",
                    duration_ms=10.0,
                    timestamp=datetime(2026, 1, 1, i, 0, 0, tzinfo=UTC),
                )
            )
        assert store.count() == 5
        store.close()

    def test_list_recent_respects_limit(self, tmp_path: Path) -> None:
        from animus_bootstrap.intelligence.tools.history_store import ToolHistoryStore

        store = ToolHistoryStore(tmp_path / "history.db")
        for i in range(10):
            store.save(
                ToolResult(
                    id=f"id-{i}",
                    tool_name="tool",
                    success=True,
                    output="ok",
                    duration_ms=10.0,
                    timestamp=datetime(2026, 1, 1, i, 0, 0, tzinfo=UTC),
                )
            )
        items = store.list_recent(limit=3)
        assert len(items) == 3
        store.close()

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        from animus_bootstrap.intelligence.tools.history_store import ToolHistoryStore

        db_path = tmp_path / "history.db"
        store1 = ToolHistoryStore(db_path)
        store1.save(
            ToolResult(
                id="persist-1",
                tool_name="shell_exec",
                success=False,
                output="error",
                duration_ms=500.0,
                timestamp=datetime(2026, 2, 20, tzinfo=UTC),
            )
        )
        store1.close()

        store2 = ToolHistoryStore(db_path)
        items = store2.list_recent()
        assert len(items) == 1
        assert items[0]["tool_name"] == "shell_exec"
        assert items[0]["success"] is False
        store2.close()

    def test_duplicate_id_ignored(self, tmp_path: Path) -> None:
        from animus_bootstrap.intelligence.tools.history_store import ToolHistoryStore

        store = ToolHistoryStore(tmp_path / "history.db")
        result = ToolResult(
            id="dup-1",
            tool_name="tool",
            success=True,
            output="ok",
            duration_ms=10.0,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        store.save(result)
        store.save(result)  # duplicate — should be ignored
        assert store.count() == 1
        store.close()


# ======================================================================
# ToolExecutor + HistoryStore integration
# ======================================================================


class TestExecutorHistoryPersistence:
    """Tests for ToolExecutor persisting to ToolHistoryStore."""

    @pytest.mark.asyncio()
    async def test_execute_persists_to_store(self, tmp_path: Path) -> None:
        from animus_bootstrap.intelligence.tools.history_store import ToolHistoryStore

        store = ToolHistoryStore(tmp_path / "history.db")
        executor = ToolExecutor(history_store=store)

        async def _echo(text: str = "hello") -> str:
            return f"echo: {text}"

        executor.register(
            ToolDefinition(
                name="echo",
                description="Echo tool",
                parameters={"type": "object", "properties": {}},
                handler=_echo,
            )
        )
        await executor.execute("echo", {"text": "world"})
        saved = store.list_recent()
        assert len(saved) == 1
        assert saved[0]["tool_name"] == "echo"
        assert saved[0]["success"] is True
        store.close()

    @pytest.mark.asyncio()
    async def test_set_history_store(self, tmp_path: Path) -> None:
        from animus_bootstrap.intelligence.tools.history_store import ToolHistoryStore

        store = ToolHistoryStore(tmp_path / "history.db")
        executor = ToolExecutor()

        async def _noop() -> str:
            return "ok"

        executor.register(
            ToolDefinition(
                name="noop",
                description="No-op",
                parameters={"type": "object", "properties": {}},
                handler=_noop,
            )
        )

        # Execute without store — nothing persisted
        await executor.execute("noop", {})
        assert store.count() == 0

        # Set store and execute again
        executor.set_history_store(store)
        await executor.execute("noop", {})
        assert store.count() == 1
        store.close()

    @pytest.mark.asyncio()
    async def test_store_failure_does_not_break_execution(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        mock_store = MagicMock()
        mock_store.save.side_effect = RuntimeError("db error")
        executor = ToolExecutor(history_store=mock_store)

        async def _ok() -> str:
            return "ok"

        executor.register(
            ToolDefinition(
                name="ok",
                description="OK",
                parameters={"type": "object", "properties": {}},
                handler=_ok,
            )
        )

        # Should succeed despite store failure
        result = await executor.execute("ok", {})
        assert result.success is True
        assert result.output == "ok"
