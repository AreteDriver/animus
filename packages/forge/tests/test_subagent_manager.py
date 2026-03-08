"""Tests for SubAgentManager and AgentConfig."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from animus_forge.agents.agent_config import (
    DEFAULT_AGENT_CONFIGS,
    AgentConfig,
    get_agent_config,
)
from animus_forge.agents.subagent_manager import (
    AgentRun,
    RunStatus,
    SubAgentManager,
)

# ---------------------------------------------------------------------------
# AgentConfig tests
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_default_config(self):
        config = AgentConfig(role="test")
        assert config.role == "test"
        assert config.timeout_seconds == 300.0
        assert config.max_tool_iterations == 8
        assert config.workspace is None
        assert config.allowed_tools is None
        assert config.denied_tools is None

    def test_get_effective_tools_no_filters(self):
        config = AgentConfig(role="builder")
        tools = config.get_effective_tools(["read_file", "write_file", "run_command"])
        assert tools == ["read_file", "write_file", "run_command"]

    def test_get_effective_tools_allow_list(self):
        config = AgentConfig(role="reviewer", allowed_tools=["read_file", "search_code"])
        tools = config.get_effective_tools(["read_file", "write_file", "search_code"])
        assert tools == ["read_file", "search_code"]

    def test_get_effective_tools_deny_list(self):
        config = AgentConfig(role="analyst", denied_tools=["write_file", "run_command"])
        tools = config.get_effective_tools(["read_file", "write_file", "run_command"])
        assert tools == ["read_file"]

    def test_get_effective_tools_allow_and_deny(self):
        """Deny wins over allow."""
        config = AgentConfig(
            role="test",
            allowed_tools=["read_file", "write_file"],
            denied_tools=["write_file"],
        )
        tools = config.get_effective_tools(["read_file", "write_file", "run_command"])
        assert tools == ["read_file"]

    def test_default_configs_exist(self):
        assert "builder" in DEFAULT_AGENT_CONFIGS
        assert "tester" in DEFAULT_AGENT_CONFIGS
        assert "reviewer" in DEFAULT_AGENT_CONFIGS
        assert "planner" in DEFAULT_AGENT_CONFIGS

    def test_builder_has_shell(self):
        assert DEFAULT_AGENT_CONFIGS["builder"].enable_shell is True

    def test_reviewer_denies_write(self):
        config = DEFAULT_AGENT_CONFIGS["reviewer"]
        assert "write_file" in config.denied_tools
        assert "run_command" in config.denied_tools

    def test_planner_no_tools(self):
        config = DEFAULT_AGENT_CONFIGS["planner"]
        assert config.allowed_tools == []

    def test_get_agent_config_default(self):
        config = get_agent_config("builder")
        assert config.role == "builder"
        assert config.enable_shell is True

    def test_get_agent_config_override(self):
        override = AgentConfig(role="builder", enable_shell=False, timeout_seconds=60.0)
        config = get_agent_config("builder", overrides={"builder": override})
        assert config.enable_shell is False
        assert config.timeout_seconds == 60.0

    def test_get_agent_config_unknown_role(self):
        config = get_agent_config("custom_agent")
        assert config.role == "custom_agent"
        assert config.timeout_seconds == 300.0  # Default

    def test_metadata_field(self):
        config = AgentConfig(role="test", metadata={"project": "forge"})
        assert config.metadata["project"] == "forge"


# ---------------------------------------------------------------------------
# AgentRun tests
# ---------------------------------------------------------------------------


class TestAgentRun:
    def test_duration_zero_when_not_started(self):
        run = AgentRun(
            run_id="test-1",
            agent="builder",
            task="build it",
            config=AgentConfig(role="builder"),
        )
        assert run.duration_ms == 0

    def test_duration_calculated(self):
        import time

        run = AgentRun(
            run_id="test-1",
            agent="builder",
            task="build it",
            config=AgentConfig(role="builder"),
            started_at=time.time() - 1.0,
            completed_at=time.time(),
        )
        assert 900 < run.duration_ms < 1200

    def test_to_dict(self):
        run = AgentRun(
            run_id="test-1",
            agent="builder",
            task="build a feature",
            config=AgentConfig(role="builder"),
            status=RunStatus.COMPLETED,
            result="done",
        )
        d = run.to_dict()
        assert d["run_id"] == "test-1"
        assert d["agent"] == "builder"
        assert d["status"] == "completed"
        assert d["result"] == "done"
        assert "task_handle" not in d

    def test_to_dict_truncates(self):
        run = AgentRun(
            run_id="test-1",
            agent="builder",
            task="x" * 300,
            config=AgentConfig(role="builder"),
            result="y" * 1000,
        )
        d = run.to_dict()
        assert len(d["task"]) == 200
        assert len(d["result"]) == 500


# ---------------------------------------------------------------------------
# SubAgentManager tests
# ---------------------------------------------------------------------------


class TestSubAgentManager:
    async def test_spawn_returns_run(self):
        mgr = SubAgentManager()

        async def executor(agent, task, config):
            return "done"

        run = await mgr.spawn("builder", "build it", executor)
        assert run.run_id.startswith("run-")
        assert run.agent == "builder"
        assert run.task == "build it"

        # Wait for completion
        await run.task_handle
        assert run.status == RunStatus.COMPLETED
        assert run.result == "done"

    async def test_spawn_batch_parallel(self):
        mgr = SubAgentManager(max_concurrent=4)
        call_order = []

        async def executor(agent, task, config):
            call_order.append(agent)
            await asyncio.sleep(0.01)
            return f"{agent} done"

        delegations = [
            {"agent": "builder", "task": "build"},
            {"agent": "tester", "task": "test"},
            {"agent": "reviewer", "task": "review"},
        ]
        runs = await mgr.spawn_batch(delegations, executor)
        assert len(runs) == 3
        assert all(r.status == RunStatus.COMPLETED for r in runs)
        assert all(r.result is not None for r in runs)

    async def test_spawn_timeout(self):
        mgr = SubAgentManager()
        config = AgentConfig(role="slow", timeout_seconds=0.05)

        async def slow_executor(agent, task, cfg):
            await asyncio.sleep(10)
            return "never"

        run = await mgr.spawn("slow", "do thing", slow_executor, config_override=config)
        await run.task_handle
        assert run.status == RunStatus.TIMED_OUT
        assert "timed out" in run.error

    async def test_spawn_failure(self):
        mgr = SubAgentManager()

        async def failing_executor(agent, task, config):
            raise ValueError("bad input")

        run = await mgr.spawn("builder", "fail", failing_executor)
        await run.task_handle
        assert run.status == RunStatus.FAILED
        assert "bad input" in run.error

    async def test_cancel_run(self):
        mgr = SubAgentManager()

        async def slow_executor(agent, task, config):
            await asyncio.sleep(10)
            return "never"

        run = await mgr.spawn("builder", "build", slow_executor)
        await asyncio.sleep(0.01)
        result = await mgr.cancel(run.run_id)
        assert result is True
        # Wait for task to actually finish
        try:
            await run.task_handle
        except asyncio.CancelledError:
            pass
        assert run.status == RunStatus.CANCELLED

    async def test_cancel_nonexistent(self):
        mgr = SubAgentManager()
        assert await mgr.cancel("nonexistent") is False

    async def test_cancel_completed(self):
        mgr = SubAgentManager()

        async def executor(agent, task, config):
            return "done"

        run = await mgr.spawn("builder", "build", executor)
        await run.task_handle
        assert await mgr.cancel(run.run_id) is False

    async def test_cascade_cancel(self):
        mgr = SubAgentManager()

        async def slow_executor(agent, task, config):
            await asyncio.sleep(10)
            return "never"

        parent = await mgr.spawn("supervisor", "oversee", slow_executor)
        child = await mgr.spawn("builder", "build", slow_executor, parent_id=parent.run_id)
        await asyncio.sleep(0.01)

        await mgr.cancel(parent.run_id, cascade=True)

        assert parent.status == RunStatus.CANCELLED
        assert child.status == RunStatus.CANCELLED

    async def test_max_depth_exceeded(self):
        mgr = SubAgentManager(max_depth=2)

        async def executor(agent, task, config):
            return "done"

        r1 = await mgr.spawn("a", "t1", executor)
        r2 = await mgr.spawn("b", "t2", executor, parent_id=r1.run_id)

        with pytest.raises(RuntimeError, match="Max sub-agent depth"):
            await mgr.spawn("c", "t3", executor, parent_id=r2.run_id)

    async def test_concurrency_limit(self):
        mgr = SubAgentManager(max_concurrent=2)
        max_concurrent_seen = 0
        current_concurrent = 0

        async def counting_executor(agent, task, config):
            nonlocal max_concurrent_seen, current_concurrent
            current_concurrent += 1
            max_concurrent_seen = max(max_concurrent_seen, current_concurrent)
            await asyncio.sleep(0.05)
            current_concurrent -= 1
            return "done"

        delegations = [{"agent": f"a{i}", "task": "t"} for i in range(5)]
        runs = await mgr.spawn_batch(delegations, counting_executor)
        assert all(r.status == RunStatus.COMPLETED for r in runs)
        assert max_concurrent_seen <= 2

    async def test_cancel_all(self):
        mgr = SubAgentManager()

        async def slow_executor(agent, task, config):
            await asyncio.sleep(10)
            return "never"

        runs = []
        for i in range(3):
            runs.append(await mgr.spawn(f"a{i}", "task", slow_executor))

        await asyncio.sleep(0.01)
        count = await mgr.cancel_all()
        assert count == 3

    async def test_cleanup_old_runs(self):
        mgr = SubAgentManager()

        async def executor(agent, task, config):
            return "done"

        run = await mgr.spawn("builder", "build", executor)
        await run.task_handle

        # Fake old timestamp
        run.completed_at = 0.0
        removed = mgr.cleanup(max_age_seconds=1)
        assert removed == 1
        assert run.run_id not in mgr.runs

    def test_list_runs(self):
        mgr = SubAgentManager()
        config = AgentConfig(role="test")
        run = AgentRun(
            run_id="test-1",
            agent="builder",
            task="build",
            config=config,
            status=RunStatus.COMPLETED,
        )
        mgr._runs["test-1"] = run
        assert len(mgr.list_runs()) == 1
        assert len(mgr.list_runs(status=RunStatus.COMPLETED)) == 1
        assert len(mgr.list_runs(status=RunStatus.RUNNING)) == 0

    def test_get_run(self):
        mgr = SubAgentManager()
        config = AgentConfig(role="test")
        run = AgentRun(
            run_id="test-1",
            agent="builder",
            task="build",
            config=config,
        )
        mgr._runs["test-1"] = run
        assert mgr.get_run("test-1") is run
        assert mgr.get_run("nonexistent") is None

    def test_active_count(self):
        mgr = SubAgentManager()
        assert mgr.active_count == 0

    async def test_output_truncation(self):
        mgr = SubAgentManager()
        config = AgentConfig(role="builder", max_output_chars=100)

        async def verbose_executor(agent, task, cfg):
            return "x" * 500

        run = await mgr.spawn("builder", "build", verbose_executor, config_override=config)
        await run.task_handle
        assert run.status == RunStatus.COMPLETED
        assert len(run.result) <= 112  # 100 + len("\n[truncated]")
        assert run.result.endswith("[truncated]")


# ---------------------------------------------------------------------------
# SupervisorAgent parallel integration
# ---------------------------------------------------------------------------


class TestSupervisorParallelDelegation:
    """Test SupervisorAgent with SubAgentManager."""

    def _make_supervisor(self, with_manager: bool = True):
        from animus_forge.agents.supervisor import SupervisorAgent

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value="direct response")
        mock_provider.complete_with_tools = AsyncMock(return_value="tools response")

        mgr = SubAgentManager(max_concurrent=4) if with_manager else None

        sup = SupervisorAgent(
            provider=mock_provider,
            subagent_manager=mgr,
        )
        return sup

    async def test_parallel_delegation_with_manager(self):
        sup = self._make_supervisor(with_manager=True)

        delegations = [
            {"agent": "planner", "task": "plan"},
            {"agent": "builder", "task": "build"},
        ]

        results = await sup._execute_delegations(
            delegations=delegations,
            context=[],
            progress_callback=None,
        )

        assert "planner" in results
        assert "builder" in results

    async def test_sequential_delegation_without_manager(self):
        sup = self._make_supervisor(with_manager=False)

        delegations = [
            {"agent": "planner", "task": "plan"},
        ]

        results = await sup._execute_delegations(
            delegations=delegations,
            context=[],
            progress_callback=None,
        )

        assert "planner" in results


# ---------------------------------------------------------------------------
# _FilteredToolRegistry
# ---------------------------------------------------------------------------


class TestFilteredToolRegistry:
    def test_filters_tools(self):
        from animus_forge.agents.supervisor import _FilteredToolRegistry

        mock_tool_read = MagicMock(name="read_file")
        mock_tool_read.name = "read_file"
        mock_tool_write = MagicMock(name="write_file")
        mock_tool_write.name = "write_file"

        mock_registry = MagicMock()
        mock_registry.tools = [mock_tool_read, mock_tool_write]
        mock_registry.execute.return_value = "read result"

        filtered = _FilteredToolRegistry(mock_registry, ["read_file"])
        assert len(filtered.tools) == 1
        assert filtered.tools[0].name == "read_file"

    def test_execute_allowed(self):
        from animus_forge.agents.supervisor import _FilteredToolRegistry

        mock_registry = MagicMock()
        mock_registry.execute.return_value = "ok"

        filtered = _FilteredToolRegistry(mock_registry, ["read_file"])
        result = filtered.execute("read_file", {"path": "."})
        assert result == "ok"
        mock_registry.execute.assert_called_once()

    def test_execute_denied(self):
        from animus_forge.agents.supervisor import _FilteredToolRegistry

        mock_registry = MagicMock()
        filtered = _FilteredToolRegistry(mock_registry, ["read_file"])
        result = filtered.execute("write_file", {"path": ".", "content": "x"})
        assert "not available" in result
        mock_registry.execute.assert_not_called()
