"""Tests for self-modification capabilities: code_edit, forge_ctl, timer_ctl, self_improve."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_bootstrap.intelligence.tools.builtin import get_all_builtin_tools
from animus_bootstrap.intelligence.tools.builtin.code_edit import (
    _ANIMUS_ROOT,
    _code_list,
    _code_patch,
    _code_read,
    _code_write,
    _resolve_animus_path,
    get_code_edit_tools,
)
from animus_bootstrap.intelligence.tools.builtin.forge_ctl import (
    _forge_invoke,
    _forge_start,
    _forge_status,
    _forge_stop,
    get_forge_tools,
)
from animus_bootstrap.intelligence.tools.builtin.self_improve import (
    _analyze_behavior,
    _apply_improvement,
    _list_improvements,
    _propose_improvement,
    clear_improvement_log,
    get_improvement_log,
    get_self_improve_tools,
    set_self_improve_deps,
)
from animus_bootstrap.intelligence.tools.builtin.timer_ctl import (
    _timer_cancel,
    _timer_create,
    _timer_fire,
    _timer_list,
    clear_dynamic_timers,
    get_dynamic_timers,
    get_timer_tools,
    set_proactive_engine,
)
from animus_bootstrap.intelligence.tools.executor import ToolResult

# ======================================================================
# code_edit tools
# ======================================================================


class TestResolveAnimusPath:
    def test_valid_relative_path(self) -> None:
        resolved = _resolve_animus_path("packages/bootstrap/pyproject.toml")
        assert resolved.is_relative_to(_ANIMUS_ROOT)

    def test_rejects_escape(self) -> None:
        with pytest.raises(ValueError, match="escapes"):
            _resolve_animus_path("../../etc/passwd")

    def test_root_is_correct(self) -> None:
        # _ANIMUS_ROOT should be the animus monorepo root
        assert (_ANIMUS_ROOT / "packages").is_dir()


class TestCodeRead:
    @pytest.mark.asyncio()
    async def test_reads_existing_file(self) -> None:
        result = await _code_read("packages/bootstrap/pyproject.toml")
        assert "[project]" in result or "[build-system]" in result

    @pytest.mark.asyncio()
    async def test_file_not_found(self) -> None:
        result = await _code_read("nonexistent_file_xyz.py")
        assert "not found" in result.lower()

    @pytest.mark.asyncio()
    async def test_rejects_escape(self) -> None:
        result = await _code_read("../../../../etc/passwd")
        assert "Permission denied" in result

    @pytest.mark.asyncio()
    async def test_directory_error(self) -> None:
        result = await _code_read("packages")
        assert "directory" in result.lower()


class TestCodeWrite:
    @pytest.mark.asyncio()
    async def test_writes_file(self, tmp_path: Path) -> None:
        target = tmp_path / "test_write.txt"
        with patch(
            "animus_bootstrap.intelligence.tools.builtin.code_edit._ANIMUS_ROOT",
            tmp_path,
        ):
            result = await _code_write("test_write.txt", "hello self-mod")
        assert "Wrote" in result
        assert target.read_text() == "hello self-mod"

    @pytest.mark.asyncio()
    async def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        with patch(
            "animus_bootstrap.intelligence.tools.builtin.code_edit._ANIMUS_ROOT",
            tmp_path,
        ):
            result = await _code_write("sub/dir/file.py", "content")
        assert "Wrote" in result
        assert (tmp_path / "sub" / "dir" / "file.py").exists()

    @pytest.mark.asyncio()
    async def test_rejects_escape(self) -> None:
        result = await _code_write("../../../../tmp/evil.py", "bad")
        assert "Permission denied" in result


class TestCodePatch:
    @pytest.mark.asyncio()
    async def test_patches_file(self, tmp_path: Path) -> None:
        f = tmp_path / "patch_me.py"
        f.write_text("def old_func():\n    pass\n", encoding="utf-8")
        with patch(
            "animus_bootstrap.intelligence.tools.builtin.code_edit._ANIMUS_ROOT",
            tmp_path,
        ):
            result = await _code_patch("patch_me.py", "old_func", "new_func")
        assert "Patched" in result
        assert "new_func" in f.read_text()
        assert "old_func" not in f.read_text()

    @pytest.mark.asyncio()
    async def test_returns_diff(self, tmp_path: Path) -> None:
        f = tmp_path / "diff_me.py"
        f.write_text("x = 1\n", encoding="utf-8")
        with patch(
            "animus_bootstrap.intelligence.tools.builtin.code_edit._ANIMUS_ROOT",
            tmp_path,
        ):
            result = await _code_patch("diff_me.py", "x = 1", "x = 2")
        assert "-x = 1" in result
        assert "+x = 2" in result

    @pytest.mark.asyncio()
    async def test_search_not_found(self, tmp_path: Path) -> None:
        f = tmp_path / "no_match.py"
        f.write_text("hello\n", encoding="utf-8")
        with patch(
            "animus_bootstrap.intelligence.tools.builtin.code_edit._ANIMUS_ROOT",
            tmp_path,
        ):
            result = await _code_patch("no_match.py", "MISSING", "replacement")
        assert "not found" in result

    @pytest.mark.asyncio()
    async def test_file_not_found(self) -> None:
        result = await _code_patch("nonexistent.py", "a", "b")
        assert "not found" in result.lower()

    @pytest.mark.asyncio()
    async def test_rejects_escape(self) -> None:
        result = await _code_patch("../../../../etc/hosts", "a", "b")
        assert "Permission denied" in result


class TestCodeList:
    @pytest.mark.asyncio()
    async def test_lists_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").touch()
        (tmp_path / "b.py").touch()
        with patch(
            "animus_bootstrap.intelligence.tools.builtin.code_edit._ANIMUS_ROOT",
            tmp_path,
        ):
            result = await _code_list("", "*.py")
        assert "a.py" in result
        assert "b.py" in result

    @pytest.mark.asyncio()
    async def test_no_files_found(self, tmp_path: Path) -> None:
        with patch(
            "animus_bootstrap.intelligence.tools.builtin.code_edit._ANIMUS_ROOT",
            tmp_path,
        ):
            result = await _code_list("", "*.xyz")
        assert "No files" in result

    @pytest.mark.asyncio()
    async def test_not_a_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.touch()
        with patch(
            "animus_bootstrap.intelligence.tools.builtin.code_edit._ANIMUS_ROOT",
            tmp_path,
        ):
            result = await _code_list("file.txt")
        assert "Not a directory" in result

    @pytest.mark.asyncio()
    async def test_caps_at_100(self, tmp_path: Path) -> None:
        for i in range(110):
            (tmp_path / f"f{i:03d}.py").touch()
        with patch(
            "animus_bootstrap.intelligence.tools.builtin.code_edit._ANIMUS_ROOT",
            tmp_path,
        ):
            result = await _code_list("", "*.py")
        assert "and 10 more" in result


class TestCodeEditToolDefinitions:
    def test_tool_count(self) -> None:
        tools = get_code_edit_tools()
        assert len(tools) == 4

    def test_tool_names(self) -> None:
        names = {t.name for t in get_code_edit_tools()}
        assert names == {"code_read", "code_write", "code_patch", "code_list"}

    def test_all_require_approval(self) -> None:
        tools = get_code_edit_tools()
        assert all(t.permission == "approve" for t in tools)

    def test_category(self) -> None:
        tools = get_code_edit_tools()
        assert all(t.category == "self_modification" for t in tools)


# ======================================================================
# forge_ctl tools
# ======================================================================


class TestForgeStatus:
    @pytest.mark.asyncio()
    async def test_status_running(self) -> None:
        import httpx

        _req = httpx.Request("GET", "http://127.0.0.1:8000/health")
        mock_resp = httpx.Response(200, json={"status": "ok"}, request=_req)
        with patch("httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.get.return_value = mock_resp
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _forge_status()
        assert "running" in result.lower()

    @pytest.mark.asyncio()
    async def test_status_unreachable(self) -> None:
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.get.side_effect = httpx.ConnectError("refused")
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _forge_status()
        assert "not reachable" in result

    @pytest.mark.asyncio()
    async def test_status_no_httpx(self) -> None:
        with patch.dict("sys.modules", {"httpx": None}):
            # Need to reimport to trigger the ImportError
            import importlib

            import animus_bootstrap.intelligence.tools.builtin.forge_ctl as mod

            importlib.reload(mod)
            result = await mod._forge_status()
            assert "httpx not installed" in result
            importlib.reload(mod)  # Restore


class TestForgeStart:
    @pytest.mark.asyncio()
    async def test_start_already_running(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"active\n", b"")
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await _forge_start()
        assert "already running" in result

    @pytest.mark.asyncio()
    async def test_start_via_systemd(self) -> None:
        # First call: is-active → inactive
        mock_check = AsyncMock()
        mock_check.communicate.return_value = (b"inactive\n", b"")
        mock_check.returncode = 3  # inactive

        # Second call: start → success
        mock_start = AsyncMock()
        mock_start.communicate.return_value = (b"", b"")
        mock_start.returncode = 0

        calls = [mock_check, mock_start]
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=calls,
        ):
            result = await _forge_start()
        assert "systemd" in result.lower()

    @pytest.mark.asyncio()
    async def test_start_no_systemctl(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = None
        mock_proc.pid = 12345
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=[
                FileNotFoundError,  # is-active
                FileNotFoundError,  # start
                mock_proc,  # uvicorn
            ],
        ):
            result = await _forge_start()
        assert "uvicorn" in result.lower() or "12345" in result


class TestForgeStop:
    @pytest.mark.asyncio()
    async def test_stop_via_systemd(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await _forge_stop()
        assert "systemd" in result.lower()

    @pytest.mark.asyncio()
    async def test_stop_fallback_kill(self) -> None:
        mock_systemd = AsyncMock()
        mock_systemd.communicate.return_value = (b"", b"not found")
        mock_systemd.returncode = 5

        mock_pkill = AsyncMock()
        mock_pkill.communicate.return_value = (b"", b"")
        mock_pkill.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_systemd):
            with patch("asyncio.create_subprocess_shell", return_value=mock_pkill):
                result = await _forge_stop()
        assert "killed" in result.lower()


class TestForgeInvoke:
    @pytest.mark.asyncio()
    async def test_invoke_get(self) -> None:
        import httpx

        _req = httpx.Request("GET", "http://127.0.0.1:8000/api/v1/workflows")
        mock_resp = httpx.Response(200, text='[{"id": 1}]', request=_req)
        with patch("httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.get.return_value = mock_resp
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _forge_invoke("/api/v1/workflows")
        assert "HTTP 200" in result

    @pytest.mark.asyncio()
    async def test_invoke_post(self) -> None:
        import httpx

        _req = httpx.Request("POST", "http://127.0.0.1:8000/api/v1/run")
        mock_resp = httpx.Response(201, text='{"ok": true}', request=_req)
        with patch("httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.post.return_value = mock_resp
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _forge_invoke("/api/v1/run", method="POST", body='{"wf": "test"}')
        assert "HTTP 201" in result

    @pytest.mark.asyncio()
    async def test_invoke_unsupported_method(self) -> None:
        result = await _forge_invoke("/test", method="PATCH")
        assert "Unsupported" in result

    @pytest.mark.asyncio()
    async def test_invoke_invalid_json_body(self) -> None:

        with patch("httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock()
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _forge_invoke("/test", method="POST", body="not json")
        assert "Invalid JSON" in result

    @pytest.mark.asyncio()
    async def test_invoke_connection_error(self) -> None:
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.get.side_effect = httpx.ConnectError("refused")
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _forge_invoke("/api")
        assert "not reachable" in result


class TestForgeToolDefinitions:
    def test_tool_count(self) -> None:
        assert len(get_forge_tools()) == 4

    def test_tool_names(self) -> None:
        names = {t.name for t in get_forge_tools()}
        assert names == {"forge_status", "forge_start", "forge_stop", "forge_invoke"}

    def test_status_is_auto(self) -> None:
        status = next(t for t in get_forge_tools() if t.name == "forge_status")
        assert status.permission == "auto"

    def test_start_requires_approval(self) -> None:
        start = next(t for t in get_forge_tools() if t.name == "forge_start")
        assert start.permission == "approve"

    def test_category(self) -> None:
        assert all(t.category == "forge" for t in get_forge_tools())


# ======================================================================
# timer_ctl tools
# ======================================================================


class TestTimerCreate:
    def setup_method(self) -> None:
        clear_dynamic_timers()
        set_proactive_engine(None)

    @pytest.mark.asyncio()
    async def test_create_interval_timer(self) -> None:
        result = await _timer_create("test", "every 30m", "check stuff")
        assert "created" in result.lower()
        timers = get_dynamic_timers()
        assert len(timers) == 1
        assert timers[0]["name"] == "test"

    @pytest.mark.asyncio()
    async def test_create_cron_timer(self) -> None:
        result = await _timer_create("daily", "0 9 * * *", "morning check")
        assert "created" in result.lower()

    @pytest.mark.asyncio()
    async def test_create_invalid_interval(self) -> None:
        result = await _timer_create("bad", "every xyz", "test")
        assert "Invalid" in result

    @pytest.mark.asyncio()
    async def test_create_with_proactive_engine(self) -> None:
        engine = MagicMock()
        engine.register_check = MagicMock()
        set_proactive_engine(engine)
        result = await _timer_create("live", "every 5m", "ping")
        assert "registered" in result.lower()
        engine.register_check.assert_called_once()
        check_arg = engine.register_check.call_args[0][0]
        assert check_arg.name == "timer:live"

    @pytest.mark.asyncio()
    async def test_create_with_channels(self) -> None:
        await _timer_create("ch", "every 1h", "test", channels="telegram,discord")
        timers = get_dynamic_timers()
        assert timers[0]["channels"] == ["telegram", "discord"]


class TestTimerList:
    def setup_method(self) -> None:
        clear_dynamic_timers()
        set_proactive_engine(None)

    @pytest.mark.asyncio()
    async def test_empty_list(self) -> None:
        result = await _timer_list()
        assert "No dynamic timers" in result

    @pytest.mark.asyncio()
    async def test_list_with_timers(self) -> None:
        await _timer_create("t1", "every 10m", "action1")
        await _timer_create("t2", "every 20m", "action2")
        result = await _timer_list()
        assert "t1" in result
        assert "t2" in result
        assert "Dynamic timers (2)" in result


class TestTimerCancel:
    def setup_method(self) -> None:
        clear_dynamic_timers()
        set_proactive_engine(None)

    @pytest.mark.asyncio()
    async def test_cancel_existing(self) -> None:
        await _timer_create("rm_me", "every 5m", "test")
        result = await _timer_cancel("rm_me")
        assert "cancelled" in result.lower()
        assert get_dynamic_timers() == []

    @pytest.mark.asyncio()
    async def test_cancel_not_found(self) -> None:
        result = await _timer_cancel("ghost")
        assert "not found" in result

    @pytest.mark.asyncio()
    async def test_cancel_unregisters_from_engine(self) -> None:
        engine = MagicMock()
        engine.register_check = MagicMock()
        engine.unregister_check = MagicMock()
        set_proactive_engine(engine)
        await _timer_create("rm", "every 5m", "test")
        await _timer_cancel("rm")
        engine.unregister_check.assert_called_once_with("timer:rm")


class TestTimerFire:
    def setup_method(self) -> None:
        clear_dynamic_timers()
        set_proactive_engine(None)

    @pytest.mark.asyncio()
    async def test_fire_no_engine(self) -> None:
        result = await _timer_fire("test")
        assert "No ProactiveEngine" in result

    @pytest.mark.asyncio()
    async def test_fire_success(self) -> None:
        engine = AsyncMock()
        nudge = MagicMock()
        nudge.text = "hello timer"
        engine.run_check.return_value = nudge
        set_proactive_engine(engine)
        result = await _timer_fire("test")
        assert "hello timer" in result
        engine.run_check.assert_called_once_with("timer:test")

    @pytest.mark.asyncio()
    async def test_fire_not_found(self) -> None:
        engine = AsyncMock()
        engine.run_check.side_effect = KeyError("not found")
        set_proactive_engine(engine)
        result = await _timer_fire("missing")
        assert "not found" in result


class TestTimerToolDefinitions:
    def test_tool_count(self) -> None:
        assert len(get_timer_tools()) == 4

    def test_tool_names(self) -> None:
        names = {t.name for t in get_timer_tools()}
        assert names == {"timer_create", "timer_list", "timer_cancel", "timer_fire"}

    def test_category(self) -> None:
        assert all(t.category == "timer" for t in get_timer_tools())


# ======================================================================
# self_improve tools
# ======================================================================


class TestAnalyzeBehavior:
    def setup_method(self) -> None:
        set_self_improve_deps(None, None)

    @pytest.mark.asyncio()
    async def test_no_executor(self) -> None:
        result = await _analyze_behavior()
        assert "No tool executor" in result

    @pytest.mark.asyncio()
    async def test_no_history(self) -> None:
        executor = MagicMock()
        executor.get_history.return_value = []
        set_self_improve_deps(executor, None)
        result = await _analyze_behavior()
        assert "No tool execution history" in result

    @pytest.mark.asyncio()
    async def test_with_history(self) -> None:
        now = datetime.now(UTC)
        history = [
            ToolResult(
                id="1",
                tool_name="web_search",
                success=True,
                output="ok",
                duration_ms=100.0,
                timestamp=now,
            ),
            ToolResult(
                id="2",
                tool_name="web_search",
                success=False,
                output="error occurred",
                duration_ms=6000.0,
                timestamp=now,
            ),
            ToolResult(
                id="3",
                tool_name="file_read",
                success=True,
                output="data",
                duration_ms=50.0,
                timestamp=now,
            ),
        ]
        executor = MagicMock()
        executor.get_history.return_value = history
        set_self_improve_deps(executor, None)

        result = await _analyze_behavior()
        assert "Behavior Analysis" in result
        assert "Errors: 1/3" in result
        assert "Slow executions" in result
        assert "web_search" in result

    @pytest.mark.asyncio()
    async def test_focus_errors(self) -> None:
        now = datetime.now(UTC)
        history = [
            ToolResult(
                id="1", tool_name="t", success=False, output="err", duration_ms=10, timestamp=now
            ),
        ]
        executor = MagicMock()
        executor.get_history.return_value = history
        set_self_improve_deps(executor, None)
        result = await _analyze_behavior(focus="errors")
        assert "Errors:" in result


class TestProposeImprovement:
    def setup_method(self) -> None:
        clear_improvement_log()
        set_self_improve_deps(None, None)

    @pytest.mark.asyncio()
    async def test_propose_without_backend(self) -> None:
        result = await _propose_improvement("tool:web_search", "Too slow")
        assert "Proposal #1" in result
        assert "Manual review" in result
        log = get_improvement_log()
        assert len(log) == 1
        assert log[0]["area"] == "tool:web_search"

    @pytest.mark.asyncio()
    async def test_propose_with_backend(self) -> None:
        backend = AsyncMock()
        backend.generate_response.return_value = "Optimize the caching layer"
        set_self_improve_deps(None, backend)
        result = await _propose_improvement("prompt", "Improve response quality")
        assert "Optimize the caching layer" in result
        assert get_improvement_log()[0]["analysis"] == "Optimize the caching layer"

    @pytest.mark.asyncio()
    async def test_propose_increments_id(self) -> None:
        await _propose_improvement("a", "desc1")
        await _propose_improvement("b", "desc2")
        log = get_improvement_log()
        assert log[0]["id"] == 1
        assert log[1]["id"] == 2


class TestApplyImprovement:
    def setup_method(self) -> None:
        clear_improvement_log()
        set_self_improve_deps(None, None)

    @pytest.mark.asyncio()
    async def test_apply_not_found(self) -> None:
        result = await _apply_improvement(999)
        assert "not found" in result

    @pytest.mark.asyncio()
    async def test_apply_preview(self) -> None:
        await _propose_improvement("test", "test desc")
        result = await _apply_improvement(1, confirm=False)
        assert "ready for application" in result

    @pytest.mark.asyncio()
    async def test_apply_confirm(self) -> None:
        await _propose_improvement("test", "test desc")
        result = await _apply_improvement(1, confirm=True)
        assert "applied" in result.lower()
        assert get_improvement_log()[0]["status"] == "applied"

    @pytest.mark.asyncio()
    async def test_apply_already_applied(self) -> None:
        await _propose_improvement("test", "test desc")
        await _apply_improvement(1, confirm=True)
        result = await _apply_improvement(1, confirm=True)
        assert "already been applied" in result


class TestListImprovements:
    def setup_method(self) -> None:
        clear_improvement_log()
        set_self_improve_deps(None, None)

    @pytest.mark.asyncio()
    async def test_empty_list(self) -> None:
        result = await _list_improvements()
        assert "No improvement proposals" in result

    @pytest.mark.asyncio()
    async def test_list_all(self) -> None:
        await _propose_improvement("a", "desc1")
        await _propose_improvement("b", "desc2")
        result = await _list_improvements()
        assert "Improvement Proposals (2)" in result

    @pytest.mark.asyncio()
    async def test_filter_by_status(self) -> None:
        await _propose_improvement("a", "desc1")
        await _propose_improvement("b", "desc2")
        await _apply_improvement(1, confirm=True)
        result = await _list_improvements(status="applied")
        assert "#1" in result
        assert "#2" not in result


class TestSelfImproveToolDefinitions:
    def test_tool_count(self) -> None:
        assert len(get_self_improve_tools()) == 4

    def test_tool_names(self) -> None:
        names = {t.name for t in get_self_improve_tools()}
        assert names == {
            "analyze_behavior",
            "propose_improvement",
            "apply_improvement",
            "list_improvements",
        }

    def test_propose_requires_approval(self) -> None:
        propose = next(t for t in get_self_improve_tools() if t.name == "propose_improvement")
        assert propose.permission == "approve"

    def test_analyze_is_auto(self) -> None:
        analyze = next(t for t in get_self_improve_tools() if t.name == "analyze_behavior")
        assert analyze.permission == "auto"

    def test_category(self) -> None:
        assert all(t.category == "self_improvement" for t in get_self_improve_tools())


# ======================================================================
# Integration: all new tools in get_all_builtin_tools
# ======================================================================


class TestNewToolsInBuiltinRegistry:
    def test_all_new_tools_present(self) -> None:
        tools = get_all_builtin_tools()
        names = {t.name for t in tools}
        new_tools = {
            "code_read",
            "code_write",
            "code_patch",
            "code_list",
            "forge_status",
            "forge_start",
            "forge_stop",
            "forge_invoke",
            "timer_create",
            "timer_list",
            "timer_cancel",
            "timer_fire",
            "analyze_behavior",
            "propose_improvement",
            "apply_improvement",
            "list_improvements",
        }
        assert new_tools.issubset(names)

    def test_no_duplicate_names(self) -> None:
        tools = get_all_builtin_tools()
        names = [t.name for t in tools]
        assert len(names) == len(set(names))

    def test_total_tool_count(self) -> None:
        tools = get_all_builtin_tools()
        # 8 original + 16 new = 24
        assert len(tools) == 24

    def test_all_have_handlers(self) -> None:
        tools = get_all_builtin_tools()
        assert all(callable(t.handler) for t in tools)
