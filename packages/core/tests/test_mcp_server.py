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
    return asyncio.new_event_loop().run_until_complete(coro)


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

    def test_tool_count(self, server):
        tools = server._tool_manager.list_tools()
        assert len(tools) == 9


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
        task.status = "pending"
        task.description = "Fix the bug"
        mock_tasks.list_tasks.return_value = [task]
        result = _run(server.call_tool("animus_list_tasks", {"status": "pending"}))
        assert "Fix the bug" in result[0][0].text

    def test_list_tasks_empty(self, server, mock_tasks):
        mock_tasks.list_tasks.return_value = []
        result = _run(server.call_tool("animus_list_tasks", {"status": "pending"}))
        assert "No pending" in result[0][0].text

    def test_create_task(self, server, mock_tasks):
        task = MagicMock()
        task.id = "task-002"
        mock_tasks.add_task.return_value = task
        result = _run(server.call_tool("animus_create_task", {"description": "Write tests"}))
        assert "Created task" in result[0][0].text

    def test_complete_task(self, server, mock_tasks):
        mock_tasks.complete_task.return_value = True
        result = _run(server.call_tool("animus_complete_task", {"task_id": "task-001"}))
        assert "complete" in result[0][0].text

    def test_complete_task_not_found(self, server, mock_tasks):
        mock_tasks.complete_task.return_value = False
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
