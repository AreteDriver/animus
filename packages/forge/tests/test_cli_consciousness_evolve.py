"""Tests for consciousness and evolve CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.exceptions import Exit

# ---------------------------------------------------------------------------
# Consciousness CLI tests
# ---------------------------------------------------------------------------


class TestConsciousnessStatus:
    """Tests for 'consciousness status' command."""

    def test_status_with_bridge(self):
        """Shows bridge status when api_state has a bridge."""
        from animus_forge.cli.commands.consciousness import status

        mock_bridge = MagicMock()
        mock_bridge.status.return_value = {
            "running": True,
            "enabled": True,
            "reflection_count": 5,
        }

        with (
            patch("animus_forge.cli.commands.consciousness.console"),
            patch("animus_forge.api_state.consciousness_bridge", mock_bridge),
        ):
            status()

    def test_status_without_bridge(self):
        """Shows default status when no bridge in api_state."""
        from animus_forge.cli.commands.consciousness import status

        with (
            patch("animus_forge.cli.commands.consciousness.console"),
            patch("animus_forge.api_state.consciousness_bridge", None),
        ):
            status()

    def test_status_exception_fallback(self):
        """Shows default status when bridge.status() raises."""
        from animus_forge.cli.commands.consciousness import status

        mock_bridge = MagicMock()
        mock_bridge.status.side_effect = RuntimeError("bridge broken")

        with (
            patch("animus_forge.cli.commands.consciousness.console"),
            patch("animus_forge.api_state.consciousness_bridge", mock_bridge),
        ):
            # The except block catches any exception and shows defaults
            status()


class TestConsciousnessReflect:
    """Tests for 'consciousness reflect' command."""

    def test_reflect_success(self):
        """Successful reflection shows output."""
        from animus_forge.cli.commands.consciousness import reflect

        mock_output = MagicMock()
        mock_output.summary = "All is well"
        mock_output.insights = ["insight1"]
        mock_output.principle_tensions = ["tension1"]
        mock_output.workflow_patch_ids = ["wf1"]
        mock_output.next_reflection_in = 300

        mock_bridge = MagicMock()
        mock_bridge.reflect_once.return_value = mock_output

        with (
            patch("animus_forge.cli.commands.consciousness._get_bridge", return_value=mock_bridge),
            patch("animus_forge.cli.commands.consciousness.console") as mock_console,
        ):
            reflect()

        # Verify all output sections were printed
        printed = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "All is well" in printed

    def test_reflect_no_insights(self):
        """Reflection with empty insights/tensions."""
        from animus_forge.cli.commands.consciousness import reflect

        mock_output = MagicMock()
        mock_output.summary = "Nothing notable"
        mock_output.insights = []
        mock_output.principle_tensions = []
        mock_output.workflow_patch_ids = []
        mock_output.next_reflection_in = 600

        mock_bridge = MagicMock()
        mock_bridge.reflect_once.return_value = mock_output

        with (
            patch("animus_forge.cli.commands.consciousness._get_bridge", return_value=mock_bridge),
            patch("animus_forge.cli.commands.consciousness.console"),
        ):
            reflect()

    def test_reflect_failure(self):
        """Reflection error exits with code 1."""
        from animus_forge.cli.commands.consciousness import reflect

        mock_bridge = MagicMock()
        mock_bridge.reflect_once.side_effect = RuntimeError("LLM down")

        with (
            patch("animus_forge.cli.commands.consciousness._get_bridge", return_value=mock_bridge),
            patch("animus_forge.cli.commands.consciousness.console"),
            pytest.raises(Exit),
        ):
            reflect()


class TestConsciousnessHistory:
    """Tests for 'consciousness history' command."""

    def test_history_no_file(self):
        """No log file exits cleanly."""
        from animus_forge.cli.commands.consciousness import history

        mock_settings = MagicMock()
        mock_settings.base_dir = Path("/nonexistent")

        with (
            patch(
                "animus_forge.cli.commands.consciousness.get_settings",
                return_value=mock_settings,
                create=True,
            ),
            patch("animus_forge.config.get_settings", return_value=mock_settings),
            patch("animus_forge.cli.commands.consciousness.console"),
            pytest.raises(Exit),
        ):
            history(last=10)

    def test_history_with_records(self, tmp_path):
        """Shows records from JSONL log."""
        from animus_forge.cli.commands.consciousness import history

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "reflections.jsonl"
        records = [
            {
                "timestamp": "2026-03-07T12:00:00Z",
                "model": "ollama",
                "tokens_used": 100,
                "output": {"summary": "All good"},
            },
            {
                "timestamp": "2026-03-07T13:00:00Z",
                "model": "ollama",
                "tokens_used": 200,
                "output": {"summary": "A" * 100},
            },
        ]
        log_file.write_text("\n".join(json.dumps(r) for r in records))

        mock_settings = MagicMock()
        mock_settings.base_dir = tmp_path

        with (
            patch("animus_forge.config.get_settings", return_value=mock_settings),
            patch("animus_forge.cli.commands.consciousness.console"),
        ):
            history(last=10)

    def test_history_with_invalid_json(self, tmp_path):
        """Skips invalid JSON lines."""
        from animus_forge.cli.commands.consciousness import history

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "reflections.jsonl"
        log_file.write_text('not json\n{"timestamp": "now"}\n')

        mock_settings = MagicMock()
        mock_settings.base_dir = tmp_path

        with (
            patch("animus_forge.config.get_settings", return_value=mock_settings),
            patch("animus_forge.cli.commands.consciousness.console"),
        ):
            history(last=10)


class TestConsciousnessReviews:
    """Tests for 'consciousness reviews' command."""

    def test_reviews_no_file(self):
        """No queue file exits cleanly."""
        from animus_forge.cli.commands.consciousness import reviews

        mock_settings = MagicMock()
        mock_settings.base_dir = Path("/nonexistent")

        with (
            patch("animus_forge.config.get_settings", return_value=mock_settings),
            patch("animus_forge.cli.commands.consciousness.console"),
            pytest.raises(Exit),
        ):
            reviews()

    def test_reviews_empty_file(self, tmp_path):
        """Empty queue file exits cleanly."""
        from animus_forge.cli.commands.consciousness import reviews

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "workflow_review_queue.jsonl").write_text("")

        mock_settings = MagicMock()
        mock_settings.base_dir = tmp_path

        with (
            patch("animus_forge.config.get_settings", return_value=mock_settings),
            patch("animus_forge.cli.commands.consciousness.console"),
            pytest.raises(Exit),
        ):
            reviews()

    def test_reviews_with_records(self, tmp_path):
        """Shows pending reviews."""
        from animus_forge.cli.commands.consciousness import reviews

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        records = [
            {
                "workflow_id": "wf1",
                "flagged_by": "reflection",
                "timestamp": "2026-03-07T12:00:00Z",
                "cycle": 3,
            },
        ]
        (log_dir / "workflow_review_queue.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records)
        )

        mock_settings = MagicMock()
        mock_settings.base_dir = tmp_path

        with (
            patch("animus_forge.config.get_settings", return_value=mock_settings),
            patch("animus_forge.cli.commands.consciousness.console"),
        ):
            reviews()

    def test_reviews_invalid_json(self, tmp_path):
        """Skips invalid JSON in review queue."""
        from animus_forge.cli.commands.consciousness import reviews

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "workflow_review_queue.jsonl").write_text(
            'bad\n{"workflow_id": "wf1", "flagged_by": "r", "timestamp": "2026-01-01T00:00:00Z", "cycle": 1}\n'
        )

        mock_settings = MagicMock()
        mock_settings.base_dir = tmp_path

        with (
            patch("animus_forge.config.get_settings", return_value=mock_settings),
            patch("animus_forge.cli.commands.consciousness.console"),
        ):
            reviews()


class TestGetBridge:
    """Tests for _get_bridge helper."""

    def test_get_bridge_ollama(self):
        """_get_bridge creates bridge with ollama provider."""
        from animus_forge.cli.commands.consciousness import _get_bridge

        mock_provider = MagicMock()
        mock_settings = MagicMock()
        mock_settings.base_dir = Path("/tmp")

        with (
            patch("animus_forge.agents.create_agent_provider", return_value=mock_provider),
            patch("animus_forge.config.get_settings", return_value=mock_settings),
            patch("animus_forge.budget.manager.BudgetManager"),
            patch("animus_forge.coordination.consciousness_bridge.ConsciousnessBridge"),
            patch("animus_forge.coordination.consciousness_bridge.ConsciousnessConfig"),
        ):
            result = _get_bridge()
            assert result is not None

    def test_get_bridge_fallback_anthropic(self):
        """_get_bridge falls back to anthropic when ollama fails."""
        from animus_forge.cli.commands.consciousness import _get_bridge

        mock_provider = MagicMock()
        mock_settings = MagicMock()
        mock_settings.base_dir = Path("/tmp")
        call_count = 0

        def mock_create(name):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("ollama not available")
            return mock_provider

        with (
            patch("animus_forge.agents.create_agent_provider", side_effect=mock_create),
            patch("animus_forge.config.get_settings", return_value=mock_settings),
            patch("animus_forge.budget.manager.BudgetManager"),
            patch("animus_forge.coordination.consciousness_bridge.ConsciousnessBridge"),
            patch("animus_forge.coordination.consciousness_bridge.ConsciousnessConfig"),
        ):
            result = _get_bridge()
            assert result is not None

    def test_get_bridge_no_provider(self):
        """_get_bridge exits when no provider available."""
        from animus_forge.cli.commands.consciousness import _get_bridge

        with (
            patch("animus_forge.agents.create_agent_provider", side_effect=RuntimeError("none")),
            patch("animus_forge.cli.commands.consciousness.console"),
            pytest.raises(Exit),
        ):
            _get_bridge()


# ---------------------------------------------------------------------------
# Evolve CLI tests
# ---------------------------------------------------------------------------


class TestEvolveStatus:
    """Tests for 'evolve status' command."""

    def test_status_no_pending(self):
        """Shows message when no pending patches."""
        from animus_forge.cli.commands.evolve import status

        mock_evo = MagicMock()
        mock_evo.list_pending.return_value = []

        with (
            patch("animus_forge.cli.commands.evolve._get_evolution", return_value=mock_evo),
            patch("animus_forge.cli.commands.evolve.console") as mock_console,
        ):
            status()

        mock_console.print.assert_called_once()

    def test_status_with_pending(self):
        """Lists pending patches."""
        from animus_forge.cli.commands.evolve import status

        mock_evo = MagicMock()
        mock_evo.list_pending.return_value = ["wf1", "wf2"]

        with (
            patch("animus_forge.cli.commands.evolve._get_evolution", return_value=mock_evo),
            patch("animus_forge.cli.commands.evolve.console") as mock_console,
        ):
            status()

        printed = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "2 pending" in printed


class TestEvolveList:
    """Tests for 'evolve list' command."""

    def test_list_no_workflows(self):
        """Shows message when no workflows."""
        from animus_forge.cli.commands.evolve import list_workflows

        mock_evo = MagicMock()
        mock_evo.list_workflows.return_value = []

        with (
            patch("animus_forge.cli.commands.evolve._get_evolution", return_value=mock_evo),
            patch("animus_forge.cli.commands.evolve.console"),
        ):
            list_workflows()

    def test_list_with_workflows(self):
        """Lists workflows in a table."""
        from animus_forge.cli.commands.evolve import list_workflows

        mock_evo = MagicMock()
        mock_evo.list_workflows.return_value = [
            {
                "id": "wf1",
                "name": "Build",
                "version": "1.0",
                "steps": 3,
                "last_evolved": "2026-03-07",
                "has_pending": True,
            },
            {
                "id": "wf2",
                "name": "Test",
                "version": "2.0",
                "steps": 5,
                "last_evolved": None,
                "has_pending": False,
            },
        ]

        with (
            patch("animus_forge.cli.commands.evolve._get_evolution", return_value=mock_evo),
            patch("animus_forge.cli.commands.evolve.console"),
        ):
            list_workflows()


class TestEvolveApprove:
    """Tests for 'evolve approve' command."""

    def test_approve_success(self):
        """Successful approval."""
        from animus_forge.cli.commands.evolve import approve

        mock_result = MagicMock()
        mock_result.workflow_id = "1.1"
        mock_evo = MagicMock()
        mock_evo.approve.return_value = mock_result

        with (
            patch("animus_forge.cli.commands.evolve._get_evolution", return_value=mock_evo),
            patch("animus_forge.cli.commands.evolve.console"),
        ):
            approve(workflow_id="wf1")

    def test_approve_error(self):
        """Approval error exits with code 1."""
        from animus_forge.cli.commands.evolve import approve

        mock_evo = MagicMock()
        mock_evo.approve.side_effect = RuntimeError("No pending patch")

        with (
            patch("animus_forge.cli.commands.evolve._get_evolution", return_value=mock_evo),
            patch("animus_forge.cli.commands.evolve.console"),
            pytest.raises(Exit),
        ):
            approve(workflow_id="wf1")


class TestEvolveReject:
    """Tests for 'evolve reject' command."""

    def test_reject_with_reason(self):
        """Rejection with reason."""
        from animus_forge.cli.commands.evolve import reject

        mock_evo = MagicMock()

        with (
            patch("animus_forge.cli.commands.evolve._get_evolution", return_value=mock_evo),
            patch("animus_forge.cli.commands.evolve.console") as mock_console,
        ):
            reject(workflow_id="wf1", reason="Not ready")

        mock_evo.reject.assert_called_once_with("wf1", "Not ready")
        printed = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "Not ready" in printed

    def test_reject_no_reason(self):
        """Rejection without reason."""
        from animus_forge.cli.commands.evolve import reject

        mock_evo = MagicMock()

        with (
            patch("animus_forge.cli.commands.evolve._get_evolution", return_value=mock_evo),
            patch("animus_forge.cli.commands.evolve.console"),
        ):
            reject(workflow_id="wf1", reason="")


class TestEvolveHistory:
    """Tests for 'evolve history' command."""

    def test_history_no_notes(self):
        """No history shows message."""
        from animus_forge.cli.commands.evolve import history

        mock_evo = MagicMock()
        mock_evo.history.return_value = []

        with (
            patch("animus_forge.cli.commands.evolve._get_evolution", return_value=mock_evo),
            patch("animus_forge.cli.commands.evolve.console"),
        ):
            history(workflow_id="wf1")

    def test_history_with_notes(self):
        """Shows evolution history."""
        from animus_forge.cli.commands.evolve import history

        mock_evo = MagicMock()
        mock_evo.history.return_value = [
            {"version": "1.0", "date": "2026-03-01", "change": "Initial", "proposed_by": "system"},
            {
                "version": "1.1",
                "date": "2026-03-07",
                "change": "Added step",
                "proposed_by": "reflection",
            },
        ]

        with (
            patch("animus_forge.cli.commands.evolve._get_evolution", return_value=mock_evo),
            patch("animus_forge.cli.commands.evolve.console"),
        ):
            history(workflow_id="wf1")


class TestGetEvolution:
    """Tests for _get_evolution helper."""

    def test_get_evolution_creates_instance(self):
        """_get_evolution creates WorkflowEvolution with settings."""
        from animus_forge.cli.commands.evolve import _get_evolution

        mock_settings = MagicMock()
        mock_settings.base_dir = Path("/tmp")

        with (
            patch("animus_forge.config.get_settings", return_value=mock_settings),
            patch("animus_forge.coordination.workflow_evolution.WorkflowEvolution") as MockEvo,
        ):
            _get_evolution()
            MockEvo.assert_called_once()
