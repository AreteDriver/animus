"""Tests for SupervisorAgent.process_message() — the public entry point."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_forge.agents.supervisor import SupervisorAgent


@pytest.fixture
def mock_provider():
    """Create a mock provider with async complete."""
    provider = MagicMock()
    provider.complete = AsyncMock(return_value="Direct response — no delegation needed.")
    return provider


class TestProcessMessageDirect:
    """Tests for direct responses (no delegation)."""

    def test_direct_response_no_delegation(self, mock_provider):
        """When LLM doesn't produce a delegation plan, return response directly."""
        result = asyncio.run(
            mock_provider_sup(mock_provider).process_message("What is Python?")
        )
        assert result == "Direct response — no delegation needed."

    def test_passes_context_to_llm(self, mock_provider):
        """Context messages are included in the LLM call."""
        sup = SupervisorAgent(provider=mock_provider)
        context = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        asyncio.run(sup.process_message("Follow-up", context=context))
        call_args = mock_provider.complete.call_args[0][0]
        # Should have system + context + user message
        assert any("[Context]" in m.get("content", "") for m in call_args)

    def test_none_context_defaults_to_empty(self, mock_provider):
        """None context doesn't crash."""
        sup = SupervisorAgent(provider=mock_provider)
        result = asyncio.run(sup.process_message("Hello"))
        assert result is not None

    def test_progress_callback_called(self, mock_provider):
        """Progress callback is invoked during processing."""
        sup = SupervisorAgent(provider=mock_provider)
        stages = []

        def track(stage, detail=""):
            stages.append(stage)

        asyncio.run(sup.process_message("Do something", progress_callback=track))
        assert "analyzing" in stages

    def test_llm_error_returns_error_message(self, mock_provider):
        """LLM failure returns a clean error message."""
        mock_provider.complete = AsyncMock(side_effect=Exception("API timeout"))
        sup = SupervisorAgent(provider=mock_provider)
        result = asyncio.run(sup.process_message("Build something"))
        assert "Error" in result
        assert "API timeout" in result


class TestProcessMessageDelegation:
    """Tests for delegation-based responses."""

    def test_delegates_and_synthesizes(self, mock_provider):
        """When LLM returns a delegation plan, executes agents and synthesizes."""
        delegation_response = json.dumps({
            "analysis": "This requires planning and building.",
            "delegations": [
                {"agent": "planner", "task": "Create implementation plan"},
                {"agent": "builder", "task": "Implement the feature"},
            ],
            "synthesis_approach": "Combine plan with implementation",
        })
        # First call: supervisor analysis returns delegation plan
        # Second call: planner agent response
        # Third call: builder agent response
        # Fourth call: synthesis
        mock_provider.complete = AsyncMock(
            side_effect=[
                f"```json\n{delegation_response}\n```",
                "Plan: Step 1, Step 2, Step 3",
                "Code: def feature(): pass",
                "Combined result: plan + code delivered.",
            ]
        )
        sup = SupervisorAgent(provider=mock_provider)
        result = asyncio.run(sup.process_message("Add user auth"))
        assert "Combined result" in result

    def test_delegation_progress_stages(self, mock_provider):
        """Progress callback hits all stages during delegation."""
        delegation_response = json.dumps({
            "analysis": "Needs analysis.",
            "delegations": [{"agent": "analyst", "task": "Analyze data"}],
            "synthesis_approach": "Report findings",
        })
        mock_provider.complete = AsyncMock(
            side_effect=[
                f"```json\n{delegation_response}\n```",
                "Analysis complete.",
                "Synthesized report.",
            ]
        )
        sup = SupervisorAgent(provider=mock_provider)
        stages = []

        def track(stage, detail=""):
            stages.append(stage)

        asyncio.run(sup.process_message("Analyze the data", progress_callback=track))
        assert "analyzing" in stages
        assert "delegating" in stages
        assert "synthesizing" in stages

    def test_single_agent_delegation(self, mock_provider):
        """Single agent delegation works."""
        delegation_response = json.dumps({
            "analysis": "Just needs testing.",
            "delegations": [{"agent": "tester", "task": "Write tests"}],
            "synthesis_approach": "Return test results",
        })
        mock_provider.complete = AsyncMock(
            side_effect=[
                f"```json\n{delegation_response}\n```",
                "Tests: test_foo passes",
                "Final: tests written.",
            ]
        )
        sup = SupervisorAgent(provider=mock_provider)
        result = asyncio.run(sup.process_message("Write tests for auth"))
        assert result is not None
        assert mock_provider.complete.call_count == 3  # supervisor + agent + synthesis


class TestProcessMessageWithBudget:
    """Tests for budget integration in process_message."""

    def test_budget_manager_passed_to_delegation(self, mock_provider):
        """Budget manager is available during delegation."""
        bm = MagicMock()
        bm.can_allocate.return_value = True
        bm.remaining = 50000
        bm.get_budget_context.return_value = "Budget: 50K tokens remaining"

        delegation_response = json.dumps({
            "analysis": "Quick task.",
            "delegations": [{"agent": "builder", "task": "Build it"}],
            "synthesis_approach": "Return code",
        })
        mock_provider.complete = AsyncMock(
            side_effect=[
                f"```json\n{delegation_response}\n```",
                "def foo(): pass",
                "Code delivered.",
            ]
        )
        sup = SupervisorAgent(provider=mock_provider, budget_manager=bm)
        result = asyncio.run(sup.process_message("Build a function"))
        assert result is not None

    def test_budget_exhausted_skips_delegation(self, mock_provider):
        """When budget is critical, delegations are skipped."""
        bm = MagicMock()
        bm.can_allocate.return_value = False
        bm.remaining = 100

        delegation_response = json.dumps({
            "analysis": "Complex task.",
            "delegations": [{"agent": "builder", "task": "Build everything"}],
            "synthesis_approach": "Return code",
        })
        mock_provider.complete = AsyncMock(
            side_effect=[
                f"```json\n{delegation_response}\n```",
                "Budget skip synthesized.",
            ]
        )
        sup = SupervisorAgent(provider=mock_provider, budget_manager=bm)
        result = asyncio.run(sup.process_message("Build everything"))
        # Should still return something (synthesis of skipped results)
        assert result is not None


class TestGetSupervisor:
    """Tests for the get_supervisor() CLI helper."""

    @patch("animus_forge.agents.create_agent_provider")
    def test_creates_supervisor_with_provider(self, mock_create):
        """get_supervisor creates a SupervisorAgent with provider."""
        mock_provider = MagicMock()
        mock_create.return_value = mock_provider

        from animus_forge.cli.helpers import get_supervisor

        with patch.dict("os.environ", {"DEFAULT_PROVIDER": "ollama"}):
            sup = get_supervisor()

        assert sup is not None
        assert sup.provider == mock_provider

    @patch("animus_forge.agents.create_agent_provider")
    def test_budget_manager_optional(self, mock_create):
        """get_supervisor works even if BudgetManager import fails."""
        mock_create.return_value = MagicMock()

        from animus_forge.cli.helpers import get_supervisor

        with patch("animus_forge.budget.BudgetManager", side_effect=ImportError):
            sup = get_supervisor()

        assert sup is not None

    @patch(
        "animus_forge.agents.create_agent_provider",
        side_effect=ValueError("No provider"),
    )
    def test_exits_on_provider_failure(self, mock_create):
        """get_supervisor exits cleanly if provider creation fails."""
        from click.exceptions import Exit

        from animus_forge.cli.helpers import get_supervisor

        with pytest.raises((SystemExit, Exit)):
            get_supervisor()


class TestDoTaskAgentMode:
    """Tests for do_task() with SupervisorAgent (default mode)."""

    @patch("animus_forge.cli.commands.dev.get_supervisor")
    @patch("animus_forge.cli.commands.dev.format_context_for_prompt", return_value="ctx")
    @patch("animus_forge.cli.commands.dev.detect_codebase_context")
    def test_default_uses_supervisor(self, mock_ctx, mock_fmt, mock_sup):
        """do_task without --workflow uses SupervisorAgent."""
        from typer.testing import CliRunner

        from animus_forge.cli.main import app

        mock_ctx.return_value = {"path": "/tmp", "language": "python"}
        mock_supervisor = MagicMock()
        mock_supervisor.process_message = AsyncMock(return_value="Task completed successfully.")
        mock_sup.return_value = mock_supervisor

        runner = CliRunner()
        result = runner.invoke(app, ["do", "fix the login bug"])

        assert result.exit_code == 0
        assert "Task completed" in result.output
        mock_supervisor.process_message.assert_called_once()

    @patch("animus_forge.cli.commands.dev.get_supervisor")
    @patch("animus_forge.cli.commands.dev.format_context_for_prompt", return_value="ctx")
    @patch("animus_forge.cli.commands.dev.detect_codebase_context")
    def test_dry_run_agent_mode(self, mock_ctx, mock_fmt, mock_sup):
        """--dry-run in agent mode shows available agents."""
        from typer.testing import CliRunner

        from animus_forge.cli.main import app

        mock_ctx.return_value = {"path": "/tmp", "language": "python"}

        runner = CliRunner()
        result = runner.invoke(app, ["do", "test task", "--dry-run"])

        assert result.exit_code == 0
        assert "Dry run" in result.output
        # Supervisor should NOT be called in dry-run
        mock_sup.assert_not_called()

    @patch("animus_forge.cli.commands.dev.get_supervisor")
    @patch("animus_forge.cli.commands.dev.format_context_for_prompt", return_value="ctx")
    @patch("animus_forge.cli.commands.dev.detect_codebase_context")
    def test_json_output_agent_mode(self, mock_ctx, mock_fmt, mock_sup):
        """--json outputs structured JSON in agent mode."""
        from typer.testing import CliRunner

        from animus_forge.cli.main import app

        mock_ctx.return_value = {"path": "/tmp", "language": "python"}
        mock_supervisor = MagicMock()
        mock_supervisor.process_message = AsyncMock(return_value="Done.")
        mock_sup.return_value = mock_supervisor

        runner = CliRunner()
        result = runner.invoke(app, ["do", "build it", "--json"])

        assert result.exit_code == 0
        # Output contains Rich panel + JSON; extract the JSON block
        output = result.output
        # Find the JSON object (pretty-printed, so { is on its own line)
        import re as _re

        json_match = _re.search(r"\{[^╭]*\}", output, _re.DOTALL)
        assert json_match is not None, f"No JSON found in output: {output!r}"
        data = json.loads(json_match.group(0))
        assert data["task"] == "build it"
        assert data["result"] == "Done."


def mock_provider_sup(provider):
    """Helper to create SupervisorAgent from mock provider."""
    return SupervisorAgent(provider=provider)
