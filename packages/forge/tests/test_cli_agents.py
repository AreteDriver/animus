"""Tests for CLI agent commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from animus_forge.agents.task_runner import TaskResult
from animus_forge.cli.commands.agents import agent_app

cli = CliRunner()


def _make_result(**kwargs) -> TaskResult:
    defaults = {
        "task_id": "t1",
        "agent": "builder",
        "task": "build it",
        "output": "Built successfully",
        "status": "completed",
        "duration_ms": 500,
    }
    defaults.update(kwargs)
    return TaskResult(**defaults)


# ---------------------------------------------------------------------------
# agent run
# ---------------------------------------------------------------------------

# The agent_run CLI command imports create_agent_provider and AgentTaskRunner
# inside its body, then calls asyncio.run(runner.run(...)). We patch asyncio.run
# to return fake results, and patch the real source modules.

_PROVIDER_PATCH = "animus_forge.agents.provider_wrapper.create_agent_provider"
_ASYNCIO_PATCH = "animus_forge.cli.commands.agents.asyncio"


class TestAgentRun:
    """Test gorgon agent run command."""

    @patch(_ASYNCIO_PATCH)
    @patch(_PROVIDER_PATCH)
    def test_run_success(self, mock_create, mock_asyncio):
        mock_create.return_value = MagicMock()
        mock_asyncio.run.return_value = _make_result()
        result = cli.invoke(agent_app, ["run", "builder", "build it", "--no-tools"])
        assert result.exit_code == 0
        assert "completed" in result.output

    @patch(_ASYNCIO_PATCH)
    @patch(_PROVIDER_PATCH)
    def test_run_json(self, mock_create, mock_asyncio):
        mock_create.return_value = MagicMock()
        mock_asyncio.run.return_value = _make_result()
        result = cli.invoke(agent_app, ["run", "builder", "build it", "--no-tools", "--json"])
        assert result.exit_code == 0
        assert '"task_id"' in result.output

    @patch(_PROVIDER_PATCH)
    def test_run_provider_failure(self, mock_create):
        mock_create.side_effect = RuntimeError("no API key")
        result = cli.invoke(agent_app, ["run", "builder", "build it"])
        assert result.exit_code == 1

    @patch(_ASYNCIO_PATCH)
    @patch(_PROVIDER_PATCH)
    def test_run_failed_task(self, mock_create, mock_asyncio):
        mock_create.return_value = MagicMock()
        mock_asyncio.run.return_value = _make_result(status="failed", error="model crashed")
        result = cli.invoke(agent_app, ["run", "builder", "fail", "--no-tools"])
        assert result.exit_code == 1
        assert "failed" in result.output


# ---------------------------------------------------------------------------
# agent list / status / cancel
# ---------------------------------------------------------------------------

# These commands do `from animus_forge.agents.subagent_manager import SubAgentManager`
# inside the function body. We patch at the source.

_SAM_PATCH = "animus_forge.agents.subagent_manager.SubAgentManager"


class TestAgentList:
    """Test gorgon agent list command."""

    @patch(_SAM_PATCH + ".list_runs")
    def test_list_empty(self, mock_list):
        mock_list.return_value = []
        result = cli.invoke(agent_app, ["list"])
        assert result.exit_code == 0
        assert "No agent runs" in result.output

    @patch(_SAM_PATCH + ".list_runs")
    def test_list_with_runs(self, mock_list):
        run = MagicMock()
        run.run_id = "run-abc123456789"
        run.agent = "builder"
        run.status.value = "completed"
        run.duration_ms = 1200
        run.task = "Build a widget"
        run.to_dict.return_value = {"run_id": "run-abc123"}
        mock_list.return_value = [run]

        result = cli.invoke(agent_app, ["list"])
        assert result.exit_code == 0
        assert "builder" in result.output

    @patch(_SAM_PATCH + ".list_runs")
    def test_list_json(self, mock_list):
        run = MagicMock()
        run.run_id = "run-abc123"
        run.status.value = "completed"
        run.to_dict.return_value = {"run_id": "run-abc123", "agent": "builder"}
        mock_list.return_value = [run]

        result = cli.invoke(agent_app, ["list", "--json"])
        assert result.exit_code == 0
        assert "run-abc123" in result.output


class TestAgentStatus:
    """Test gorgon agent status command."""

    @patch(_SAM_PATCH + ".get_run")
    def test_status_found(self, mock_get):
        run = MagicMock()
        run.run_id = "run-xyz"
        run.agent = "tester"
        run.status.value = "completed"
        run.task = "Run tests"
        run.duration_ms = 800
        run.result = "All tests passed"
        run.error = None
        run.children = []
        mock_get.return_value = run

        result = cli.invoke(agent_app, ["status", "run-xyz"])
        assert result.exit_code == 0
        assert "tester" in result.output

    @patch(_SAM_PATCH + ".get_run")
    def test_status_not_found(self, mock_get):
        mock_get.return_value = None
        result = cli.invoke(agent_app, ["status", "run-missing"])
        assert result.exit_code == 1

    @patch(_SAM_PATCH + ".get_run")
    def test_status_json(self, mock_get):
        run = MagicMock()
        run.run_id = "run-xyz"
        run.to_dict.return_value = {"run_id": "run-xyz"}
        mock_get.return_value = run

        result = cli.invoke(agent_app, ["status", "run-xyz", "--json"])
        assert result.exit_code == 0
        assert "run-xyz" in result.output


class TestAgentCancel:
    """Test gorgon agent cancel command."""

    @patch(_ASYNCIO_PATCH)
    @patch(_SAM_PATCH + ".cancel")
    def test_cancel_success(self, mock_cancel, mock_asyncio):
        mock_asyncio.run.return_value = True
        result = cli.invoke(agent_app, ["cancel", "run-abc"])
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    @patch(_ASYNCIO_PATCH)
    @patch(_SAM_PATCH + ".cancel")
    def test_cancel_not_running(self, mock_cancel, mock_asyncio):
        mock_asyncio.run.return_value = False
        result = cli.invoke(agent_app, ["cancel", "run-abc"])
        assert result.exit_code == 0
        assert "not running" in result.output


# ---------------------------------------------------------------------------
# agent memory
# ---------------------------------------------------------------------------

_MEM_PATCH = "animus_forge.state.agent_memory.AgentMemory"


class TestAgentMemory:
    """Test gorgon agent memory command."""

    @patch(_MEM_PATCH + ".recall")
    @patch(_MEM_PATCH + "._init_schema")
    def test_memory_empty(self, mock_schema, mock_recall):
        mock_recall.return_value = []
        result = cli.invoke(agent_app, ["memory", "builder"])
        assert result.exit_code == 0
        assert "No memories" in result.output

    @patch(_MEM_PATCH + ".recall")
    @patch(_MEM_PATCH + "._init_schema")
    def test_memory_with_entries(self, mock_schema, mock_recall):
        entry = MagicMock()
        entry.id = 1
        entry.memory_type = "fact"
        entry.importance = 0.8
        entry.content = "Uses Python 3.12"
        mock_recall.return_value = [entry]

        result = cli.invoke(agent_app, ["memory", "builder"])
        assert result.exit_code == 0
        assert "Python" in result.output

    @patch(_MEM_PATCH + ".recall")
    @patch(_MEM_PATCH + "._init_schema")
    def test_memory_json(self, mock_schema, mock_recall):
        entry = MagicMock()
        entry.id = 1
        entry.memory_type = "fact"
        entry.importance = 0.8
        entry.content = "Uses Python 3.12"
        mock_recall.return_value = [entry]

        result = cli.invoke(agent_app, ["memory", "builder", "--json"])
        assert result.exit_code == 0
        assert "fact" in result.output
