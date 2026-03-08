"""Tests for AgentTaskRunner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from animus_forge.agents.task_runner import AgentTaskRunner, TaskResult

# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------


class TestTaskResult:
    """Test TaskResult dataclass."""

    def test_defaults(self):
        r = TaskResult(task_id="t1", agent="builder", task="build it")
        assert r.status == "completed"
        assert r.error is None
        assert r.tool_calls == 0
        assert r.tokens_used == 0
        assert r.duration_ms == 0

    def test_to_dict(self):
        r = TaskResult(
            task_id="t1",
            agent="builder",
            task="build it",
            output="done",
            duration_ms=500,
            tool_calls=3,
        )
        d = r.to_dict()
        assert d["task_id"] == "t1"
        assert d["agent"] == "builder"
        assert d["output"] == "done"
        assert d["duration_ms"] == 500
        assert d["tool_calls"] == 3

    def test_to_dict_truncates_output(self):
        r = TaskResult(task_id="t1", agent="a", task="t", output="x" * 5000)
        d = r.to_dict()
        assert len(d["output"]) == 2000

    def test_to_dict_truncates_task(self):
        r = TaskResult(task_id="t1", agent="a", task="y" * 500)
        d = r.to_dict()
        assert len(d["task"]) == 200


# ---------------------------------------------------------------------------
# AgentTaskRunner.run()
# ---------------------------------------------------------------------------


class TestAgentTaskRunnerRun:
    """Test the run() method."""

    @pytest.fixture
    def provider(self):
        p = MagicMock()
        p.complete = AsyncMock(return_value="Task completed successfully")
        p.complete_with_tools = AsyncMock(return_value="Built the feature")
        return p

    @pytest.mark.asyncio
    async def test_run_without_tools(self, provider):
        runner = AgentTaskRunner(provider=provider)
        result = await runner.run("builder", "build it", use_tools=False)
        assert result.status == "completed"
        assert result.output == "Task completed successfully"
        provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_tools(self, provider):
        registry = MagicMock()
        runner = AgentTaskRunner(provider=provider, tool_registry=registry)
        result = await runner.run("builder", "build it", use_tools=True)
        assert result.status == "completed"
        assert result.output == "Built the feature"
        provider.complete_with_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_no_registry_falls_back_to_no_tools(self, provider):
        runner = AgentTaskRunner(provider=provider, tool_registry=None)
        result = await runner.run("builder", "build it", use_tools=True)
        assert result.status == "completed"
        provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_failure(self, provider):
        provider.complete = AsyncMock(side_effect=RuntimeError("model error"))
        runner = AgentTaskRunner(provider=provider)
        result = await runner.run("builder", "build it", use_tools=False)
        assert result.status == "failed"
        assert "model error" in result.error

    @pytest.mark.asyncio
    async def test_run_with_context(self, provider):
        runner = AgentTaskRunner(provider=provider)
        result = await runner.run("builder", "build it", use_tools=False, context="Use Python")
        assert result.status == "completed"
        # Check context was included in messages
        call_args = provider.complete.call_args[0][0]
        assert any("Use Python" in m.get("content", "") for m in call_args)

    @pytest.mark.asyncio
    async def test_run_with_memory(self, provider):
        mem = MagicMock()
        mem.recall_context.return_value = {"facts": [MagicMock(content="uses python")]}
        mem.format_context.return_value = "Known Facts:\n- uses python"
        runner = AgentTaskRunner(provider=provider, agent_memory=mem)
        result = await runner.run("builder", "build it", use_tools=False)
        assert result.status == "completed"
        mem.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_memory_recall_failure_swallowed(self, provider):
        mem = MagicMock()
        mem.recall_context.side_effect = RuntimeError("db error")
        runner = AgentTaskRunner(provider=provider, agent_memory=mem)
        result = await runner.run("builder", "build it", use_tools=False)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_run_memory_store_failure_swallowed(self, provider):
        mem = MagicMock()
        mem.recall_context.return_value = {}
        mem.store.side_effect = RuntimeError("db error")
        runner = AgentTaskRunner(provider=provider, agent_memory=mem)
        result = await runner.run("builder", "build it", use_tools=False)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_run_emits_broadcaster_status(self, provider):
        bc = MagicMock()
        runner = AgentTaskRunner(provider=provider, broadcaster=bc)
        await runner.run("builder", "build it", use_tools=False)
        assert bc.on_status_change.call_count >= 2  # running + completed

    @pytest.mark.asyncio
    async def test_run_broadcaster_error_swallowed(self, provider):
        bc = MagicMock()
        bc.on_status_change.side_effect = RuntimeError("ws error")
        runner = AgentTaskRunner(provider=provider, broadcaster=bc)
        result = await runner.run("builder", "build it", use_tools=False)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_get_result(self, provider):
        runner = AgentTaskRunner(provider=provider)
        result = await runner.run("builder", "build it", use_tools=False)
        fetched = runner.get_result(result.task_id)
        assert fetched is result

    @pytest.mark.asyncio
    async def test_list_results(self, provider):
        runner = AgentTaskRunner(provider=provider)
        await runner.run("builder", "task1", use_tools=False)
        provider.complete = AsyncMock(side_effect=RuntimeError("fail"))
        await runner.run("tester", "task2", use_tools=False)
        all_results = runner.list_results()
        assert len(all_results) == 2
        completed = runner.list_results(status="completed")
        assert len(completed) == 1
        failed = runner.list_results(status="failed")
        assert len(failed) == 1

    @pytest.mark.asyncio
    async def test_duration_tracked(self, provider):
        runner = AgentTaskRunner(provider=provider)
        result = await runner.run("builder", "build it", use_tools=False)
        assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# AgentTaskRunner.spawn()
# ---------------------------------------------------------------------------


class TestAgentTaskRunnerSpawn:
    """Test background spawning."""

    @pytest.mark.asyncio
    async def test_spawn_requires_subagent_manager(self):
        provider = MagicMock()
        runner = AgentTaskRunner(provider=provider)
        with pytest.raises(RuntimeError, match="SubAgentManager not configured"):
            await runner.spawn("builder", "build it")

    @pytest.mark.asyncio
    async def test_spawn_delegates_to_sam(self):
        provider = MagicMock()
        sam = MagicMock()
        run_mock = MagicMock()
        run_mock.run_id = "run-123"
        sam.spawn = AsyncMock(return_value=run_mock)
        runner = AgentTaskRunner(provider=provider, subagent_manager=sam)
        run_id = await runner.spawn("builder", "build it")
        assert run_id == "run-123"
        sam.spawn.assert_called_once()


# ---------------------------------------------------------------------------
# Build messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_basic_messages(self):
        provider = MagicMock()
        runner = AgentTaskRunner(provider=provider)
        msgs = runner._build_messages("builder", "do stuff")
        assert msgs[0]["role"] == "system"
        assert "builder" in msgs[0]["content"]
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "do stuff"

    def test_messages_with_context(self):
        provider = MagicMock()
        runner = AgentTaskRunner(provider=provider)
        msgs = runner._build_messages("builder", "do stuff", context="Use Rust")
        assert len(msgs) == 3
        assert "Use Rust" in msgs[1]["content"]
