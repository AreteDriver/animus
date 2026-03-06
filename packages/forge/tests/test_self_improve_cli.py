"""Tests for the self-improve CLI commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from animus_forge.cli.main import app

runner = CliRunner()


def _mock_provider_and_orchestrator(mock_result):
    """Patch both create_agent_provider and SelfImproveOrchestrator via sys.modules."""
    mock_orch = MagicMock()
    mock_orch.run = AsyncMock(return_value=mock_result)

    mock_create = MagicMock(return_value=MagicMock())
    mock_orch_cls = MagicMock(return_value=mock_orch)

    return mock_create, mock_orch_cls


class TestSelfImproveRun:
    """Tests for 'gorgon self-improve run' command."""

    def test_help(self):
        result = runner.invoke(app, ["self-improve", "run", "--help"])
        assert result.exit_code == 0
        assert "self-improvement workflow" in result.output.lower()

    def test_run_success(self):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.stage_reached.value = "complete"
        mock_result.plan = MagicMock(title="Test Plan", changes=["a"])
        mock_result.sandbox_result = MagicMock(tests_passed=True)
        mock_result.pull_request = None
        mock_result.error = None
        mock_result.violations = []

        mock_create, mock_orch_cls = _mock_provider_and_orchestrator(mock_result)

        with patch.dict(
            "sys.modules",
            {
                "animus_forge.agents.provider_wrapper": MagicMock(
                    create_agent_provider=mock_create,
                ),
                "animus_forge.self_improve.orchestrator": MagicMock(
                    SelfImproveOrchestrator=mock_orch_cls,
                ),
            },
        ):
            # Re-import so the lazy imports pick up the mocks
            import importlib

            import animus_forge.cli.commands.self_improve as mod

            importlib.reload(mod)

            result = runner.invoke(app, ["self-improve", "run", "--provider", "ollama"])
            assert result.exit_code == 0

    def test_run_provider_error(self):
        with patch.dict(
            "sys.modules",
            {
                "animus_forge.agents.provider_wrapper": MagicMock(
                    create_agent_provider=MagicMock(side_effect=ValueError("No API key")),
                ),
            },
        ):
            result = runner.invoke(app, ["self-improve", "run", "--provider", "anthropic"])
            assert result.exit_code == 1
            assert "Failed" in result.output

    def test_run_bad_path(self):
        result = runner.invoke(
            app, ["self-improve", "run", "--path", "/nonexistent/path/xyz"]
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_run_with_focus(self):
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.stage_reached.value = "analyzing"
        mock_result.plan = None
        mock_result.sandbox_result = None
        mock_result.pull_request = None
        mock_result.error = "No suggestions found"
        mock_result.violations = []

        mock_create, mock_orch_cls = _mock_provider_and_orchestrator(mock_result)

        with patch.dict(
            "sys.modules",
            {
                "animus_forge.agents.provider_wrapper": MagicMock(
                    create_agent_provider=mock_create,
                ),
                "animus_forge.self_improve.orchestrator": MagicMock(
                    SelfImproveOrchestrator=mock_orch_cls,
                ),
            },
        ):
            result = runner.invoke(
                app, ["self-improve", "run", "--focus", "security"]
            )
            assert "No suggestions found" in result.output or result.exit_code == 0


class TestSelfImproveAnalyze:
    """Tests for 'gorgon self-improve analyze' command."""

    def test_help(self):
        result = runner.invoke(app, ["self-improve", "analyze", "--help"])
        assert result.exit_code == 0
        assert "analyze" in result.output.lower()

    def test_analyze_no_suggestions(self):
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []

        with patch.dict(
            "sys.modules",
            {
                "animus_forge.self_improve.analyzer": MagicMock(
                    CodebaseAnalyzer=MagicMock(return_value=mock_analyzer),
                ),
            },
        ):
            result = runner.invoke(app, ["self-improve", "analyze"])
            assert result.exit_code == 0
            assert "No improvement" in result.output

    def test_analyze_with_suggestions(self):
        suggestion = MagicMock()
        suggestion.category = "performance"
        suggestion.file_path = "foo.py"
        suggestion.description = "Optimize loop"
        suggestion.priority = "high"

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [suggestion]

        with patch.dict(
            "sys.modules",
            {
                "animus_forge.self_improve.analyzer": MagicMock(
                    CodebaseAnalyzer=MagicMock(return_value=mock_analyzer),
                ),
            },
        ):
            result = runner.invoke(app, ["self-improve", "analyze"])
            assert result.exit_code == 0
            assert "performance" in result.output.lower()

    def test_analyze_focus_filter(self):
        s1 = MagicMock(category="security", file_path="a.py", description="x", priority="high")
        s2 = MagicMock(category="performance", file_path="b.py", description="y", priority="low")

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [s1, s2]

        with patch.dict(
            "sys.modules",
            {
                "animus_forge.self_improve.analyzer": MagicMock(
                    CodebaseAnalyzer=MagicMock(return_value=mock_analyzer),
                ),
            },
        ):
            result = runner.invoke(app, ["self-improve", "analyze", "--focus", "security"])
            assert result.exit_code == 0
            assert "security" in result.output.lower()
