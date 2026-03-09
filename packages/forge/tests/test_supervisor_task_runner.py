"""Tests for SupervisorAgent <-> TaskRunner bridge.

Covers set_task_runner(), _run_agent() delegation through TaskRunner,
fallback to direct provider on TaskRunner failure, and direct provider
path when no TaskRunner is set.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from animus_forge.agents.supervisor import SupervisorAgent
from animus_forge.agents.task_runner import TaskResult


@pytest.fixture
def mock_provider():
    """Provider mock with both complete and complete_with_tools."""
    provider = MagicMock()
    provider.complete = AsyncMock(return_value="direct-provider-response")
    provider.complete_with_tools = AsyncMock(return_value="direct-tools-response")
    return provider


@pytest.fixture
def supervisor(mock_provider):
    """Bare supervisor with no optional components."""
    return SupervisorAgent(provider=mock_provider)


class TestSetTaskRunner:
    """Test set_task_runner() sets the internal attribute."""

    def test_task_runner_none_by_default(self, supervisor):
        assert supervisor._task_runner is None

    def test_set_task_runner_stores_runner(self, supervisor):
        runner = MagicMock()
        supervisor.set_task_runner(runner)
        assert supervisor._task_runner is runner

    def test_set_task_runner_replaces_previous(self, supervisor):
        runner_a = MagicMock(name="runner_a")
        runner_b = MagicMock(name="runner_b")
        supervisor.set_task_runner(runner_a)
        supervisor.set_task_runner(runner_b)
        assert supervisor._task_runner is runner_b

    def test_set_task_runner_to_none_clears(self, supervisor):
        runner = MagicMock()
        supervisor.set_task_runner(runner)
        supervisor.set_task_runner(None)
        assert supervisor._task_runner is None


class TestRunAgentWithTaskRunner:
    """Test _run_agent() uses TaskRunner when set."""

    @pytest.mark.asyncio
    async def test_uses_task_runner_when_set(self, supervisor):
        """_run_agent() delegates to TaskRunner.run() and returns output."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(
            return_value=TaskResult(
                task_id="task-abc123",
                agent="builder",
                task="build login",
                output="TaskRunner output",
                status="completed",
            )
        )
        supervisor.set_task_runner(mock_runner)

        result = await supervisor._run_agent("builder", "build login", [])
        assert result == "TaskRunner output"
        mock_runner.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_task_runner_receives_correct_params(self, supervisor):
        """TaskRunner.run() receives agent, task, use_tools, context, max_iterations."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(
            return_value=TaskResult(
                task_id="task-xyz",
                agent="tester",
                task="test login",
                output="ok",
                status="completed",
            )
        )
        supervisor.set_task_runner(mock_runner)

        await supervisor._run_agent("tester", "test login", [])

        call_kwargs = mock_runner.run.call_args.kwargs
        assert call_kwargs["agent"] == "tester"
        assert call_kwargs["task"] == "test login"
        assert "use_tools" in call_kwargs
        assert "context" in call_kwargs
        assert "max_iterations" in call_kwargs

    @pytest.mark.asyncio
    async def test_task_runner_uses_agent_config_max_iterations(self, supervisor):
        """When agent_config is provided, max_iterations comes from it."""
        from animus_forge.agents.agent_config import AgentConfig

        config = AgentConfig(role="builder", max_tool_iterations=20)

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(
            return_value=TaskResult(
                task_id="t1",
                agent="builder",
                task="x",
                output="ok",
                status="completed",
            )
        )
        supervisor.set_task_runner(mock_runner)

        await supervisor._run_agent("builder", "x", [], agent_config=config)

        call_kwargs = mock_runner.run.call_args.kwargs
        assert call_kwargs["max_iterations"] == 20

    @pytest.mark.asyncio
    async def test_task_runner_failed_result_returns_error_message(self, supervisor):
        """When TaskRunner returns non-completed status, returns error string."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(
            return_value=TaskResult(
                task_id="t2",
                agent="builder",
                task="x",
                output="",
                status="failed",
                error="timeout exceeded",
            )
        )
        supervisor.set_task_runner(mock_runner)

        result = await supervisor._run_agent("builder", "x", [])
        assert "failed" in result.lower()
        assert "builder" in result.lower() or "timeout" in result.lower()


class TestRunAgentTaskRunnerFallback:
    """Test _run_agent() falls back to direct provider when TaskRunner raises."""

    @pytest.mark.asyncio
    async def test_falls_back_on_task_runner_exception(self, supervisor, mock_provider):
        """When TaskRunner.run() raises, falls back to direct provider."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=RuntimeError("runner crashed"))
        supervisor.set_task_runner(mock_runner)

        result = await supervisor._run_agent("planner", "plan something", [])

        # Should have used the direct provider fallback
        assert (
            mock_provider.complete.await_count > 0
            or mock_provider.complete_with_tools.await_count > 0
        )
        # Should return something (not crash)
        assert isinstance(result, str)


class TestRunAgentDirectProvider:
    """Test _run_agent() uses direct provider when no TaskRunner is set."""

    @pytest.mark.asyncio
    async def test_no_task_runner_uses_provider_complete(self, supervisor, mock_provider):
        """Without TaskRunner, non-tool agents use provider.complete()."""
        result = await supervisor._run_agent("planner", "plan something", [])

        mock_provider.complete.assert_awaited_once()
        assert result == "direct-provider-response"

    @pytest.mark.asyncio
    async def test_no_task_runner_tool_equipped_without_registry(self, supervisor, mock_provider):
        """Tool-equipped agent without tool_registry falls back to complete()."""
        # supervisor has no tool_registry, so even "builder" uses text-only
        result = await supervisor._run_agent("builder", "build it", [])

        mock_provider.complete.assert_awaited_once()
        assert result == "direct-provider-response"

    @pytest.mark.asyncio
    async def test_no_task_runner_tool_equipped_with_registry(self, mock_provider):
        """Tool-equipped agent with tool_registry uses complete_with_tools()."""
        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_registry = MagicMock()
        mock_registry.tools = [mock_tool]

        sup = SupervisorAgent(provider=mock_provider, tool_registry=mock_registry)
        result = await sup._run_agent("builder", "build it", [])

        mock_provider.complete_with_tools.assert_awaited_once()
        assert result == "direct-tools-response"
