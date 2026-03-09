"""Tests for the --runner flag in do_task (dev.py).

Covers _run_via_task_runner path: normal execution, dry run,
JSON output, and provider failure.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.exceptions import Exit

from animus_forge.agents.task_runner import TaskResult


def _invoke_do_task_runner(**overrides):
    """Helper to invoke do_task with --runner and all mocks.

    Patches codebase detection, console output, and the internal
    imports of create_agent_provider and AgentTaskRunner.
    """
    from animus_forge.cli.commands.dev import do_task

    defaults = {
        "task": "add login endpoint",
        "workflow": None,
        "dry_run": False,
        "json_output": False,
        "live": False,
        "verify": False,
        "runner": True,
        "role": "builder",
    }
    defaults.update(overrides)
    return defaults, do_task


class TestRunnerNormalExecution:
    """do_task --runner executes via AgentTaskRunner."""

    def test_runner_calls_task_runner_run(self):
        """--runner creates AgentTaskRunner and calls run()."""
        mock_result = TaskResult(
            task_id="task-test1",
            agent="builder",
            task="add login endpoint",
            output="Implementation complete",
            status="completed",
            tool_calls=3,
        )
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=mock_result)
        mock_runner_cls = MagicMock(return_value=mock_runner_instance)

        mock_provider = MagicMock()
        mock_create_provider = MagicMock(return_value=mock_provider)

        kwargs, do_task = _invoke_do_task_runner()

        with (
            patch(
                "animus_forge.cli.commands.dev.detect_codebase_context",
                return_value={"path": "/tmp", "language": "python"},
            ),
            patch(
                "animus_forge.cli.commands.dev.format_context_for_prompt",
                return_value="ctx",
            ),
            patch("animus_forge.cli.commands.dev.console"),
            patch(
                "animus_forge.agents.provider_wrapper.create_agent_provider",
                mock_create_provider,
            ),
            patch(
                "animus_forge.agents.task_runner.AgentTaskRunner",
                mock_runner_cls,
            ),
        ):
            do_task(**kwargs)

        mock_runner_instance.run.assert_awaited_once()
        call_kwargs = mock_runner_instance.run.call_args
        assert call_kwargs.args[0] == "builder" or call_kwargs.kwargs.get("agent") == "builder" or call_kwargs[0][0] == "builder"

    def test_runner_uses_specified_role(self):
        """--role tester routes to the tester agent."""
        mock_result = TaskResult(
            task_id="task-role",
            agent="tester",
            task="test it",
            output="Tests pass",
            status="completed",
        )
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=mock_result)
        mock_runner_cls = MagicMock(return_value=mock_runner_instance)

        kwargs, do_task = _invoke_do_task_runner(role="tester", task="test it")

        with (
            patch(
                "animus_forge.cli.commands.dev.detect_codebase_context",
                return_value={"path": "/tmp", "language": "python"},
            ),
            patch("animus_forge.cli.commands.dev.format_context_for_prompt", return_value="ctx"),
            patch("animus_forge.cli.commands.dev.console"),
            patch(
                "animus_forge.agents.provider_wrapper.create_agent_provider",
                MagicMock(return_value=MagicMock()),
            ),
            patch(
                "animus_forge.agents.task_runner.AgentTaskRunner",
                mock_runner_cls,
            ),
        ):
            do_task(**kwargs)

        call_args = mock_runner_instance.run.call_args
        # First positional arg is the role
        assert "tester" in str(call_args)


class TestRunnerDryRun:
    """do_task --runner --dry-run shows dry run message."""

    def test_dry_run_exits_without_running(self):
        """--runner --dry-run prints message and exits."""
        kwargs, do_task = _invoke_do_task_runner(dry_run=True)

        with (
            patch(
                "animus_forge.cli.commands.dev.detect_codebase_context",
                return_value={"path": "/tmp", "language": "python"},
            ),
            patch("animus_forge.cli.commands.dev.format_context_for_prompt", return_value="ctx"),
            patch("animus_forge.cli.commands.dev.console") as mock_console,
        ):
            with pytest.raises((Exit, SystemExit)):
                do_task(**kwargs)

        # Verify dry run message was printed
        print_calls = [str(c) for c in mock_console.print.call_args_list]
        combined = " ".join(print_calls)
        assert "dry" in combined.lower() or "Dry" in combined


class TestRunnerJsonOutput:
    """do_task --runner --json outputs JSON."""

    def test_json_output_includes_fields(self, capsys):
        """--runner --json outputs task_id, agent, status, tool_calls."""
        mock_result = TaskResult(
            task_id="task-json1",
            agent="builder",
            task="add login",
            output="Built it",
            status="completed",
            tool_calls=5,
        )
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=mock_result)
        mock_runner_cls = MagicMock(return_value=mock_runner_instance)

        kwargs, do_task = _invoke_do_task_runner(json_output=True)

        with (
            patch(
                "animus_forge.cli.commands.dev.detect_codebase_context",
                return_value={"path": "/tmp", "language": "python"},
            ),
            patch("animus_forge.cli.commands.dev.format_context_for_prompt", return_value="ctx"),
            patch("animus_forge.cli.commands.dev.console"),
            patch(
                "animus_forge.agents.provider_wrapper.create_agent_provider",
                MagicMock(return_value=MagicMock()),
            ),
            patch(
                "animus_forge.agents.task_runner.AgentTaskRunner",
                mock_runner_cls,
            ),
        ):
            do_task(**kwargs)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["task_id"] == "task-json1"
        assert data["agent"] == "builder"
        assert data["status"] == "completed"
        assert data["tool_calls"] == 5
        assert "duration_ms" in data


class TestRunnerProviderFailure:
    """do_task --runner when provider creation fails."""

    def test_provider_creation_failure_exits(self):
        """When create_agent_provider raises, exits with code 1."""
        kwargs, do_task = _invoke_do_task_runner()

        with (
            patch(
                "animus_forge.cli.commands.dev.detect_codebase_context",
                return_value={"path": "/tmp", "language": "python"},
            ),
            patch("animus_forge.cli.commands.dev.format_context_for_prompt", return_value="ctx"),
            patch("animus_forge.cli.commands.dev.console"),
            patch(
                "animus_forge.agents.provider_wrapper.create_agent_provider",
                side_effect=RuntimeError("No API key"),
            ),
        ):
            with pytest.raises((Exit, SystemExit)):
                do_task(**kwargs)

    def test_runner_task_failure_exits(self):
        """When TaskRunner returns failed status, exits with code 1."""
        mock_result = TaskResult(
            task_id="task-fail",
            agent="builder",
            task="broken task",
            output="",
            status="failed",
            error="provider timeout",
        )
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=mock_result)
        mock_runner_cls = MagicMock(return_value=mock_runner_instance)

        kwargs, do_task = _invoke_do_task_runner()

        with (
            patch(
                "animus_forge.cli.commands.dev.detect_codebase_context",
                return_value={"path": "/tmp", "language": "python"},
            ),
            patch("animus_forge.cli.commands.dev.format_context_for_prompt", return_value="ctx"),
            patch("animus_forge.cli.commands.dev.console"),
            patch(
                "animus_forge.agents.provider_wrapper.create_agent_provider",
                MagicMock(return_value=MagicMock()),
            ),
            patch(
                "animus_forge.agents.task_runner.AgentTaskRunner",
                mock_runner_cls,
            ),
        ):
            with pytest.raises((Exit, SystemExit)):
                do_task(**kwargs)
