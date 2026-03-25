"""Tests for the Animus MCP server."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from animus.memory import Memory, MemoryType

# Skip all tests if mcp not installed
mcp = pytest.importorskip("mcp")


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _sync_run_coro(coro):
    """Helper to run a coroutine in a new event loop (for nested asyncio.run patches)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _patch_nested_asyncio_run():
    """Patch asyncio.run to handle nested calls from within MCP tool handlers.

    When the test calls _run() (asyncio.run) → MCP tool calls asyncio.run() internally,
    this patches the inner asyncio.run to just await the coroutine in the running loop.
    """
    _original_run = asyncio.run

    def _smart_run(coro, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # We're inside an event loop already — create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        return _original_run(coro, **kwargs)

    return patch("asyncio.run", side_effect=_smart_run)


def _make_memory(content: str, tags: list[str] | None = None) -> Memory:
    from datetime import datetime

    return Memory(
        id="mem-test-001",
        content=content,
        memory_type=MemoryType.SEMANTIC,
        created_at=datetime(2026, 3, 5),
        updated_at=datetime(2026, 3, 5),
        metadata={},
        tags=tags or [],
    )


@pytest.fixture
def mock_config():
    with patch("animus.mcp_server.AnimusConfig") as mock_cls:
        cfg = MagicMock()
        cfg.data_dir = "/tmp/animus-test"
        cfg.memory.backend = "json"
        mock_cls.load.return_value = cfg
        yield cfg


@pytest.fixture
def mock_memory():
    with patch("animus.mcp_server.MemoryLayer") as mock_cls:
        mem = MagicMock()
        mock_cls.return_value = mem
        yield mem


@pytest.fixture
def mock_tasks():
    with patch("animus.mcp_server.TaskTracker") as mock_cls:
        tracker = MagicMock()
        mock_cls.return_value = tracker
        yield tracker


@pytest.fixture
def server(mock_config, mock_memory, mock_tasks):
    from animus.mcp_server import create_mcp_server

    return create_mcp_server()


class TestMcpServerCreation:
    def test_server_created(self, server):
        assert server is not None
        assert server.name == "animus"

    def test_server_has_tools(self, server):
        tools = server._tool_manager.list_tools()
        tool_names = {t.name for t in tools}
        assert "animus_remember" in tool_names
        assert "animus_recall" in tool_names
        assert "animus_search_tags" in tool_names
        assert "animus_memory_stats" in tool_names
        assert "animus_list_tasks" in tool_names
        assert "animus_create_task" in tool_names
        assert "animus_complete_task" in tool_names
        assert "animus_brief" in tool_names
        assert "animus_run_workflow" in tool_names
        assert "animus_self_improve" in tool_names
        assert "animus_harvest" in tool_names

    def test_tool_count(self, server):
        tools = server._tool_manager.list_tools()
        # 4 memory + 3 task + 1 brief + 1 workflow + 1 harvest + 4 watchlist + 1 self-improve = 15
        assert len(tools) == 15


class TestMemoryTools:
    def test_remember(self, server, mock_memory):
        mock_memory.remember.return_value = _make_memory("test content")
        result = _run(server.call_tool("animus_remember", {"content": "test fact", "tags": "a,b"}))
        assert "Stored memory" in result[0][0].text
        mock_memory.remember.assert_called_once()
        call_kwargs = mock_memory.remember.call_args.kwargs
        assert call_kwargs["content"] == "test fact"
        assert call_kwargs["tags"] == ["a", "b"]

    def test_remember_no_tags(self, server, mock_memory):
        mock_memory.remember.return_value = _make_memory("test")
        result = _run(server.call_tool("animus_remember", {"content": "test"}))
        assert "Stored memory" in result[0][0].text
        call_kwargs = mock_memory.remember.call_args.kwargs
        assert call_kwargs["tags"] == []

    def test_remember_invalid_type(self, server, mock_memory):
        mock_memory.remember.return_value = _make_memory("test")
        result = _run(
            server.call_tool("animus_remember", {"content": "test", "memory_type": "invalid"})
        )
        assert "Stored memory" in result[0][0].text
        call_kwargs = mock_memory.remember.call_args.kwargs
        assert call_kwargs["memory_type"] == MemoryType.SEMANTIC

    def test_recall_with_results(self, server, mock_memory):
        mock_memory.recall.return_value = [
            _make_memory("Python is great", ["python"]),
            _make_memory("Rust is fast"),
        ]
        result = _run(server.call_tool("animus_recall", {"query": "languages"}))
        text = result[0][0].text
        assert "Python is great" in text
        assert "Rust is fast" in text
        assert "[python]" in text

    def test_recall_empty(self, server, mock_memory):
        mock_memory.recall.return_value = []
        result = _run(server.call_tool("animus_recall", {"query": "nothing"}))
        assert "No matching" in result[0][0].text

    def test_search_tags(self, server, mock_memory):
        mock_memory.recall_by_tags.return_value = [_make_memory("tagged item")]
        result = _run(server.call_tool("animus_search_tags", {"tags": "python,code"}))
        assert "tagged item" in result[0][0].text
        mock_memory.recall_by_tags.assert_called_once_with(tags=["python", "code"], limit=10)

    def test_search_tags_empty(self, server, mock_memory):
        result = _run(server.call_tool("animus_search_tags", {"tags": ""}))
        assert "No tags provided" in result[0][0].text

    def test_memory_stats(self, server, mock_memory):
        mock_memory.get_statistics.return_value = {"total": 42, "unique_tags": 10}
        result = _run(server.call_tool("animus_memory_stats", {}))
        data = json.loads(result[0][0].text)
        assert data["total"] == 42


class TestTaskTools:
    def test_list_tasks(self, server, mock_tasks):
        task = MagicMock()
        task.id = "task-001"
        task.status = MagicMock(value="pending")
        task.description = "Fix the bug"
        mock_tasks.list.return_value = [task]
        result = _run(server.call_tool("animus_list_tasks", {"status": "pending"}))
        assert "Fix the bug" in result[0][0].text

    def test_list_tasks_empty(self, server, mock_tasks):
        mock_tasks.list.return_value = []
        result = _run(server.call_tool("animus_list_tasks", {"status": "pending"}))
        assert "No pending" in result[0][0].text

    def test_create_task(self, server, mock_tasks):
        task = MagicMock()
        task.id = "task-002"
        mock_tasks.add_task.return_value = task
        result = _run(server.call_tool("animus_create_task", {"description": "Write tests"}))
        assert "Created task" in result[0][0].text

    def test_complete_task(self, server, mock_tasks):
        mock_tasks.complete.return_value = True
        result = _run(server.call_tool("animus_complete_task", {"task_id": "task-001"}))
        assert "complete" in result[0][0].text

    def test_complete_task_not_found(self, server, mock_tasks):
        mock_tasks.complete.return_value = False
        result = _run(server.call_tool("animus_complete_task", {"task_id": "bad-id"}))
        assert "not found" in result[0][0].text


class TestBriefTool:
    def test_brief_with_results(self, server, mock_memory):
        mock_memory.recall.return_value = [
            _make_memory("Sprint 3 is in progress"),
            _make_memory("Deploy deadline is Friday"),
        ]
        result = _run(server.call_tool("animus_brief", {"topic": "sprint"}))
        text = result[0][0].text
        assert "Animus Briefing" in text
        assert "Sprint 3" in text
        assert "Friday" in text

    def test_brief_empty(self, server, mock_memory):
        mock_memory.recall.return_value = []
        result = _run(server.call_tool("animus_brief", {}))
        assert "No relevant context" in result[0][0].text


class TestRunWorkflow:
    """Test animus_run_workflow MCP tool."""

    @pytest.fixture
    def server(self, tmp_path):
        with patch("animus.mcp_server.AnimusConfig") as mock_config_cls:
            mock_config = MagicMock()
            mock_config.data_dir = tmp_path
            mock_config.memory.backend = "dict"
            mock_config_cls.load.return_value = mock_config
            from animus.mcp_server import create_mcp_server

            return create_mcp_server()

    def test_workflow_not_found(self, server):
        result = _run(
            server.call_tool("animus_run_workflow", {"workflow_path": "/nonexistent/wf.yaml"})
        )
        assert "not found" in result[0][0].text.lower()

    def test_workflow_load_error(self, server, tmp_path):
        """Test ForgeError when loading invalid workflow YAML."""
        bad_yaml = tmp_path / "bad_wf.yaml"
        bad_yaml.write_text("not: valid: workflow")
        from animus.forge.models import ForgeError

        with patch("animus.forge.loader.load_workflow", side_effect=ForgeError("bad schema")):
            result = _run(
                server.call_tool("animus_run_workflow", {"workflow_path": str(bad_yaml)})
            )
            assert "Failed to load workflow" in result[0][0].text

    def test_workflow_success_with_results(self, server, tmp_path):
        """Test successful workflow execution with result formatting."""
        wf_yaml = tmp_path / "ok_wf.yaml"
        wf_yaml.write_text("placeholder")

        mock_wf_config = MagicMock()
        mock_wf_config.name = "test_pipeline"
        mock_wf_config.agents = []

        mock_agent_result = MagicMock()
        mock_agent_result.success = True
        mock_agent_result.agent_name = "analyzer"
        mock_agent_result.tokens_used = 500
        mock_agent_result.error = None

        mock_fail_result = MagicMock()
        mock_fail_result.success = False
        mock_fail_result.agent_name = "broken_step"
        mock_fail_result.tokens_used = 100
        mock_fail_result.error = "timeout"

        mock_state = MagicMock()
        mock_state.status = "completed"
        mock_state.results = [mock_agent_result, mock_fail_result]
        mock_state.total_tokens = 600
        mock_state.total_cost = 0.0042

        with patch("animus.forge.loader.load_workflow", return_value=mock_wf_config), \
             patch("animus.cognitive.CognitiveLayer"), \
             patch("animus.cognitive.ModelConfig") as mock_mc, \
             patch("animus.tools.create_default_registry"), \
             patch("animus.forge.ForgeEngine") as mock_engine_cls:
            mock_mc.ollama.return_value = MagicMock()
            mock_engine_cls.return_value.run.return_value = mock_state
            result = _run(
                server.call_tool(
                    "animus_run_workflow", {"workflow_path": str(wf_yaml)}
                )
            )
            text = result[0][0].text
            assert "test_pipeline" in text
            assert "completed" in text
            assert "[OK] analyzer" in text
            assert "[FAIL] broken_step" in text
            assert "timeout" in text
            assert "$0.0042" in text

    def test_workflow_runs(self, server, tmp_path):
        # Create a minimal workflow YAML
        wf_yaml = tmp_path / "test_wf.yaml"
        wf_yaml.write_text(
            "name: test_wf\n"
            "description: Test\n"
            "provider: mock\n"
            "model: mock\n"
            "max_cost_usd: 1.0\n"
            "agents:\n"
            "  - name: step1\n"
            "    archetype: writer\n"
            "    budget_tokens: 100\n"
            "    outputs: [result]\n"
            "gates: []\n"
        )
        result = _run(
            server.call_tool(
                "animus_run_workflow",
                {"workflow_path": str(wf_yaml), "task_description": "test task"},
            )
        )
        text = result[0][0].text
        # Should complete or fail gracefully
        assert "test_wf" in text or "failed" in text.lower()


class TestMCPAuth:
    """Test MCP server API key authentication."""

    def test_no_auth_configured(self):
        """Without ANIMUS_MCP_API_KEY, all calls pass."""
        from animus.mcp_server import _check_auth

        with patch.dict("os.environ", {}, clear=False):
            with patch("animus.mcp_server._MCP_API_KEY", None):
                assert _check_auth() is None
                assert _check_auth("anything") is None

    def test_auth_required_no_key(self):
        """With ANIMUS_MCP_API_KEY set, missing key is rejected."""
        from animus.mcp_server import _check_auth

        with patch("animus.mcp_server._MCP_API_KEY", "secret123"):
            result = _check_auth("")
            assert result is not None
            assert "Authentication required" in result

    def test_auth_required_wrong_key(self):
        """Wrong key is rejected."""
        from animus.mcp_server import _check_auth

        with patch("animus.mcp_server._MCP_API_KEY", "secret123"):
            result = _check_auth("wrong")
            assert result is not None

    def test_auth_required_correct_key(self):
        """Correct key passes."""
        from animus.mcp_server import _check_auth

        with patch("animus.mcp_server._MCP_API_KEY", "secret123"):
            assert _check_auth("secret123") is None

    def test_remember_with_auth(self, tmp_path):
        """animus_remember blocks without valid key when auth is configured."""
        with patch("animus.mcp_server._MCP_API_KEY", "testkey"):
            with patch("animus.mcp_server.AnimusConfig") as mock_config_cls:
                mock_config = MagicMock()
                mock_config.data_dir = tmp_path
                mock_config.memory.backend = "sqlite"
                mock_config_cls.load.return_value = mock_config

                from animus.mcp_server import create_mcp_server

                server = create_mcp_server()

                # No key → blocked
                result = _run(server.call_tool("animus_remember", {"content": "test"}))
                assert "Authentication required" in result[0][0].text

                # Correct key → passes
                result = _run(
                    server.call_tool(
                        "animus_remember",
                        {"content": "test", "api_key": "testkey"},
                    )
                )
                assert "Authentication required" not in result[0][0].text


class TestHarvestTool:
    """Test animus_harvest MCP tool."""

    def test_harvest_success(self, server, mock_memory):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"repo": "test/repo", "patterns": 5}
        with patch("animus.harvest.harvest_repo", return_value=mock_result):
            result = _run(
                server.call_tool("animus_harvest", {"target": "test/repo"})
            )
            data = json.loads(result[0][0].text)
            assert data["repo"] == "test/repo"

    def test_harvest_value_error(self, server, mock_memory):
        with patch("animus.harvest.harvest_repo", side_effect=ValueError("bad target")):
            result = _run(
                server.call_tool("animus_harvest", {"target": "bad"})
            )
            assert "Harvest failed" in result[0][0].text

    def test_harvest_runtime_error(self, server, mock_memory):
        with patch("animus.harvest.harvest_repo", side_effect=RuntimeError("clone failed")):
            result = _run(
                server.call_tool("animus_harvest", {"target": "bad/repo"})
            )
            assert "Harvest failed" in result[0][0].text

    def test_harvest_unexpected_error(self, server, mock_memory):
        with patch("animus.harvest.harvest_repo", side_effect=OSError("disk full")):
            result = _run(
                server.call_tool("animus_harvest", {"target": "test/repo"})
            )
            assert "Harvest error" in result[0][0].text

    def test_harvest_auth_blocked(self, server, mock_memory):
        with patch("animus.mcp_server._MCP_API_KEY", "secret"):
            result = _run(
                server.call_tool("animus_harvest", {"target": "test/repo"})
            )
            assert "Authentication required" in result[0][0].text

    def test_harvest_auth_passes(self, server, mock_memory):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"repo": "ok"}
        with patch("animus.mcp_server._MCP_API_KEY", "key123"):
            with patch("animus.harvest.harvest_repo", return_value=mock_result):
                result = _run(
                    server.call_tool(
                        "animus_harvest", {"target": "test/repo", "api_key": "key123"}
                    )
                )
                assert "Authentication required" not in result[0][0].text


class TestWatchlistTools:
    """Test watchlist MCP tools."""

    def test_watchlist_add(self, server):
        entry = {"target": "test/repo", "tags": ["ai"], "added": "2026-03-25"}
        with patch("animus.harvest_watchlist.add_to_watchlist", return_value=entry):
            result = _run(
                server.call_tool(
                    "animus_watchlist_add",
                    {"target": "test/repo", "tags": "ai,ml", "notes": "competitor"},
                )
            )
            data = json.loads(result[0][0].text)
            assert data["target"] == "test/repo"

    def test_watchlist_add_no_tags(self, server):
        entry = {"target": "test/repo", "tags": [], "added": "2026-03-25"}
        with patch("animus.harvest_watchlist.add_to_watchlist", return_value=entry):
            result = _run(
                server.call_tool("animus_watchlist_add", {"target": "test/repo"})
            )
            data = json.loads(result[0][0].text)
            assert data["target"] == "test/repo"

    def test_watchlist_add_value_error(self, server):
        with patch(
            "animus.harvest_watchlist.add_to_watchlist",
            side_effect=ValueError("duplicate"),
        ):
            result = _run(
                server.call_tool("animus_watchlist_add", {"target": "test/repo"})
            )
            assert "Watchlist add failed" in result[0][0].text

    def test_watchlist_add_unexpected_error(self, server):
        with patch(
            "animus.harvest_watchlist.add_to_watchlist",
            side_effect=OSError("disk"),
        ):
            result = _run(
                server.call_tool("animus_watchlist_add", {"target": "test/repo"})
            )
            assert "Watchlist error" in result[0][0].text

    def test_watchlist_add_auth_blocked(self, server):
        with patch("animus.mcp_server._MCP_API_KEY", "secret"):
            result = _run(
                server.call_tool("animus_watchlist_add", {"target": "test/repo"})
            )
            assert "Authentication required" in result[0][0].text

    def test_watchlist_remove_success(self, server):
        with patch("animus.harvest_watchlist.remove_from_watchlist", return_value=True):
            result = _run(
                server.call_tool("animus_watchlist_remove", {"target": "test/repo"})
            )
            assert "Removed" in result[0][0].text

    def test_watchlist_remove_not_found(self, server):
        with patch("animus.harvest_watchlist.remove_from_watchlist", return_value=False):
            result = _run(
                server.call_tool("animus_watchlist_remove", {"target": "test/repo"})
            )
            assert "not found" in result[0][0].text

    def test_watchlist_remove_auth_blocked(self, server):
        with patch("animus.mcp_server._MCP_API_KEY", "secret"):
            result = _run(
                server.call_tool("animus_watchlist_remove", {"target": "test/repo"})
            )
            assert "Authentication required" in result[0][0].text

    def test_watchlist_list_with_repos(self, server):
        repos = [{"target": "a/b", "last_scan": "2026-03-20"}]
        with patch("animus.harvest_watchlist.get_watchlist", return_value=repos):
            result = _run(server.call_tool("animus_watchlist_list", {}))
            data = json.loads(result[0][0].text)
            assert len(data) == 1

    def test_watchlist_list_empty(self, server):
        with patch("animus.harvest_watchlist.get_watchlist", return_value=[]):
            result = _run(server.call_tool("animus_watchlist_list", {}))
            assert "empty" in result[0][0].text.lower()

    def test_watchlist_scan_success(self, server):
        report = {"scanned": 2, "changes": 1}

        async def fake_scan(**kwargs):
            return report

        with patch(
            "animus.harvest_watchlist.run_watchlist_scan", side_effect=fake_scan
        ), _patch_nested_asyncio_run():
            result = _run(server.call_tool("animus_watchlist_scan", {}))
            data = json.loads(result[0][0].text)
            assert data["scanned"] == 2

    def test_watchlist_scan_with_interval(self, server):
        report = {"scanned": 1}
        call_log = {}

        async def fake_scan(**kwargs):
            call_log.update(kwargs)
            return report

        with patch(
            "animus.harvest_watchlist.run_watchlist_scan", side_effect=fake_scan
        ), _patch_nested_asyncio_run():
            result = _run(
                server.call_tool("animus_watchlist_scan", {"interval_hours": 24})
            )
            data = json.loads(result[0][0].text)
            assert data["scanned"] == 1
            assert call_log["interval_hours"] == 24

    def test_watchlist_scan_failure(self, server):
        with patch(
            "animus.harvest_watchlist.run_watchlist_scan",
            side_effect=RuntimeError("network error"),
        ):
            result = _run(server.call_tool("animus_watchlist_scan", {}))
            assert "Watchlist scan failed" in result[0][0].text

    def test_watchlist_scan_auth_blocked(self, server):
        with patch("animus.mcp_server._MCP_API_KEY", "secret"):
            result = _run(server.call_tool("animus_watchlist_scan", {}))
            assert "Authentication required" in result[0][0].text


class TestSelfImproveTool:
    """Test animus_self_improve MCP tool."""

    def test_self_improve_path_not_found(self, server):
        result = _run(
            server.call_tool(
                "animus_self_improve", {"codebase_path": "/nonexistent/path"}
            )
        )
        assert "Path not found" in result[0][0].text

    def test_self_improve_forge_not_installed(self, server, tmp_path):
        # Patch the specific imports that animus_self_improve tries
        with patch.dict("sys.modules", {"animus_forge.agents.provider_wrapper": None}):
            result = _run(
                server.call_tool(
                    "animus_self_improve", {"codebase_path": str(tmp_path)}
                )
            )
            assert "Forge not installed" in result[0][0].text

    def test_self_improve_provider_error(self, server, tmp_path):
        with patch(
            "animus_forge.agents.provider_wrapper.create_agent_provider",
            side_effect=ValueError("bad provider"),
        ):
            result = _run(
                server.call_tool(
                    "animus_self_improve",
                    {"codebase_path": str(tmp_path), "provider": "bad"},
                )
            )
            assert "Failed to create" in result[0][0].text

    def test_self_improve_success(self, server, tmp_path):
        mock_result = MagicMock()
        mock_result.stage_reached.value = "completed"
        mock_result.success = True
        mock_result.plan = MagicMock()
        mock_result.plan.title = "Fix bare excepts"
        mock_result.plan.suggestions = [MagicMock(description="Replace bare except in foo.py")]
        mock_result.error = None
        mock_result.sandbox_result = MagicMock()
        mock_result.sandbox_result.tests_passed = True
        mock_result.pull_request = MagicMock()
        mock_result.pull_request.url = "https://github.com/test/pr/1"
        mock_result.pull_request.branch = "self-improve-1"

        mock_orch = MagicMock()

        async def mock_run(**kwargs):
            return mock_result

        mock_orch.run = mock_run

        with patch(
            "animus_forge.agents.provider_wrapper.create_agent_provider",
            return_value=MagicMock(),
        ), patch(
            "animus_forge.self_improve.orchestrator.SelfImproveOrchestrator",
            return_value=mock_orch,
        ), _patch_nested_asyncio_run():
            result = _run(
                server.call_tool(
                    "animus_self_improve",
                    {"codebase_path": str(tmp_path)},
                )
            )
            text = result[0][0].text
            assert "completed" in text
            assert "Fix bare excepts" in text
            assert "passed" in text

    def test_self_improve_failure(self, server, tmp_path):
        mock_result = MagicMock()
        mock_result.stage_reached.value = "analysis"
        mock_result.success = False
        mock_result.plan = None
        mock_result.error = "No issues found"
        mock_result.sandbox_result = None
        mock_result.pull_request = None

        mock_orch = MagicMock()

        async def mock_run(**kwargs):
            return mock_result

        mock_orch.run = mock_run

        with patch(
            "animus_forge.agents.provider_wrapper.create_agent_provider",
            return_value=MagicMock(),
        ), patch(
            "animus_forge.self_improve.orchestrator.SelfImproveOrchestrator",
            return_value=mock_orch,
        ), _patch_nested_asyncio_run():
            result = _run(
                server.call_tool(
                    "animus_self_improve",
                    {"codebase_path": str(tmp_path)},
                )
            )
            text = result[0][0].text
            assert "analysis" in text
            assert "No issues found" in text

    def test_self_improve_exception(self, server, tmp_path):
        mock_orch = MagicMock()

        async def mock_run(**kwargs):
            raise RuntimeError("sandbox crashed")

        mock_orch.run = mock_run

        with patch(
            "animus_forge.agents.provider_wrapper.create_agent_provider",
            return_value=MagicMock(),
        ), patch(
            "animus_forge.self_improve.orchestrator.SelfImproveOrchestrator",
            return_value=mock_orch,
        ), _patch_nested_asyncio_run():
            result = _run(
                server.call_tool(
                    "animus_self_improve",
                    {"codebase_path": str(tmp_path)},
                )
            )
            assert "Self-improve failed" in result[0][0].text

    def test_self_improve_auth_blocked(self, server, tmp_path):
        with patch("animus.mcp_server._MCP_API_KEY", "secret"):
            result = _run(
                server.call_tool(
                    "animus_self_improve",
                    {"codebase_path": str(tmp_path)},
                )
            )
            assert "Authentication required" in result[0][0].text


class TestWriteToolsAuth:
    """Test auth blocks on all write tools."""

    def test_create_task_auth(self, server):
        with patch("animus.mcp_server._MCP_API_KEY", "authkey"):
            result = _run(
                server.call_tool("animus_create_task", {"description": "test"})
            )
            assert "Authentication required" in result[0][0].text

    def test_complete_task_auth(self, server):
        with patch("animus.mcp_server._MCP_API_KEY", "authkey"):
            result = _run(
                server.call_tool("animus_complete_task", {"task_id": "t1"})
            )
            assert "Authentication required" in result[0][0].text

    def test_run_workflow_auth(self, server):
        with patch("animus.mcp_server._MCP_API_KEY", "authkey"):
            result = _run(
                server.call_tool("animus_run_workflow", {"workflow_path": "/test.yaml"})
            )
            assert "Authentication required" in result[0][0].text


class TestListTasksFilter:
    """Test task list filtering edge cases."""

    def test_list_all_tasks(self, server, mock_tasks):
        task = MagicMock()
        task.id = "task-001"
        task.status = MagicMock(value="completed")
        task.description = "Done task"
        mock_tasks.list.return_value = [task]
        result = _run(server.call_tool("animus_list_tasks", {"status": "all"}))
        assert "Done task" in result[0][0].text

    def test_list_tasks_filters_by_status(self, server, mock_tasks):
        pending = MagicMock()
        pending.id = "t1"
        pending.status = MagicMock(value="pending")
        pending.description = "Pending one"
        done = MagicMock()
        done.id = "t2"
        done.status = MagicMock(value="completed")
        done.description = "Done one"
        mock_tasks.list.return_value = [pending, done]
        result = _run(server.call_tool("animus_list_tasks", {"status": "pending"}))
        text = result[0][0].text
        assert "Pending one" in text
        assert "Done one" not in text


class TestSearchTagsNoResults:
    """Test search_tags with no matching results."""

    def test_search_tags_no_results(self, server, mock_memory):
        mock_memory.recall_by_tags.return_value = []
        result = _run(server.call_tool("animus_search_tags", {"tags": "nonexistent"}))
        assert "No memories found" in result[0][0].text


class TestBriefMemoryType:
    """Test brief tool memory_type attribute handling."""

    def test_brief_with_memory_type(self, server, mock_memory):
        mem = _make_memory("Important context")
        mock_memory.recall.return_value = [mem]
        result = _run(server.call_tool("animus_brief", {"topic": "test"}))
        text = result[0][0].text
        assert "semantic" in text
        assert "Important context" in text

    def test_brief_default_topic(self, server, mock_memory):
        mock_memory.recall.return_value = [_make_memory("data")]
        _run(server.call_tool("animus_brief", {}))
        mock_memory.recall.assert_called_with(query="recent important context", limit=10)


class TestMcpImportError:
    """Test create_mcp_server when mcp is not installed."""

    def test_import_error_raised(self):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mcp.server.fastmcp":
                raise ImportError("No module named 'mcp'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            from animus.mcp_server import create_mcp_server

            with pytest.raises(ImportError, match="MCP server requires"):
                create_mcp_server()


class TestMainEntrypoint:
    """Test the main() entrypoint."""

    def test_main_calls_run(self, mock_config, mock_memory, mock_tasks):
        from animus.mcp_server import main

        with patch("animus.mcp_server.create_mcp_server") as mock_create:
            mock_server = MagicMock()
            mock_create.return_value = mock_server
            main()
            mock_server.run.assert_called_once()
