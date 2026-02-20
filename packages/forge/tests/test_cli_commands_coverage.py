"""Coverage tests for CLI command functions and validation helpers."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from typer.testing import CliRunner

from animus_forge.cli.main import (
    _display_workflow_preview,
    _load_workflow_from_source,
    _output_run_results,
    _output_validation_results,
    _validate_cli_next_step_refs,
    _validate_cli_required_fields,
    _validate_cli_steps,
    app,
)

runner = CliRunner()


class TestValidateCliRequiredFields:
    def test_valid(self):
        errors, warnings = _validate_cli_required_fields({"id": "x", "steps": [{"id": "s1"}]})
        assert not errors
        assert not warnings

    def test_missing_id(self):
        errors, warnings = _validate_cli_required_fields({"steps": []})
        assert any("id" in e for e in errors)

    def test_missing_steps(self):
        errors, warnings = _validate_cli_required_fields({"id": "x"})
        assert any("steps" in e for e in errors)

    def test_steps_not_list(self):
        errors, warnings = _validate_cli_required_fields({"id": "x", "steps": "bad"})
        assert any("list" in e for e in errors)

    def test_empty_steps(self):
        errors, warnings = _validate_cli_required_fields({"id": "x", "steps": []})
        assert not errors
        assert any("no steps" in w.lower() for w in warnings)


class TestValidateCliSteps:
    def test_valid_steps(self):
        steps = [{"id": "s1", "type": "claude_code", "action": "do"}]
        errors, warnings, ids = _validate_cli_steps(steps)
        assert not errors
        assert not warnings
        assert "s1" in ids

    def test_missing_id(self):
        steps = [{"type": "claude_code", "action": "do"}]
        errors, warnings, ids = _validate_cli_steps(steps)
        assert any("id" in e.lower() for e in errors)

    def test_duplicate_id(self):
        steps = [
            {"id": "s1", "type": "claude_code", "action": "a"},
            {"id": "s1", "type": "claude_code", "action": "b"},
        ]
        errors, warnings, ids = _validate_cli_steps(steps)
        assert any("Duplicate" in e for e in errors)

    def test_missing_type(self):
        steps = [{"id": "s1", "action": "do"}]
        errors, warnings, ids = _validate_cli_steps(steps)
        assert any("type" in e.lower() for e in errors)

    def test_unknown_type(self):
        steps = [{"id": "s1", "type": "unknown_type", "action": "do"}]
        errors, warnings, ids = _validate_cli_steps(steps)
        assert any("Unknown" in w for w in warnings)

    def test_missing_action(self):
        steps = [{"id": "s1", "type": "claude_code"}]
        errors, warnings, ids = _validate_cli_steps(steps)
        assert any("action" in e.lower() for e in errors)


class TestValidateCliNextStepRefs:
    def test_valid_refs(self):
        steps = [{"id": "s1", "next_step": "s2"}, {"id": "s2"}]
        errors = _validate_cli_next_step_refs(steps, {"s1", "s2"})
        assert not errors

    def test_invalid_ref(self):
        steps = [{"id": "s1", "next_step": "s999"}]
        errors = _validate_cli_next_step_refs(steps, {"s1"})
        assert any("s999" in e for e in errors)

    def test_no_next_step(self):
        steps = [{"id": "s1"}]
        errors = _validate_cli_next_step_refs(steps, {"s1"})
        assert not errors

    def test_null_next_step(self):
        steps = [{"id": "s1", "next_step": None}]
        errors = _validate_cli_next_step_refs(steps, {"s1"})
        assert not errors


class TestOutputValidationResults:
    def test_errors(self):
        import typer

        with pytest.raises(typer.Exit):
            _output_validation_results(["err1"], [], Path("test.json"))

    def test_warnings_only(self):
        _output_validation_results([], ["warn1"], Path("test.json"))

    def test_clean(self):
        _output_validation_results([], [], Path("test.json"))


class TestLoadWorkflowFromSource:
    def test_json_file(self, tmp_path):
        wf_file = tmp_path / "test.json"
        wf_file.write_text(json.dumps({"id": "test_wf", "steps": []}))
        engine = MagicMock()
        wf_id, data, path = _load_workflow_from_source(str(wf_file), engine)
        assert wf_id == "test_wf"
        assert path == wf_file

    def test_json_file_no_id(self, tmp_path):
        wf_file = tmp_path / "my_workflow.json"
        wf_file.write_text(json.dumps({"steps": []}))
        engine = MagicMock()
        wf_id, data, path = _load_workflow_from_source(str(wf_file), engine)
        assert wf_id == "my_workflow"

    def test_invalid_json(self, tmp_path):
        wf_file = tmp_path / "bad.json"
        wf_file.write_text("not json{")
        engine = MagicMock()
        import typer

        with pytest.raises(typer.Exit):
            _load_workflow_from_source(str(wf_file), engine)

    def test_by_id_found(self):
        engine = MagicMock()
        mock_wf = MagicMock()
        mock_wf.model_dump.return_value = {"id": "wf1", "steps": []}
        engine.load_workflow.return_value = mock_wf
        wf_id, data, path = _load_workflow_from_source("wf1", engine)
        assert wf_id == "wf1"
        assert path is None

    def test_by_id_not_found(self):
        engine = MagicMock()
        engine.load_workflow.return_value = None
        import typer

        with pytest.raises(typer.Exit):
            _load_workflow_from_source("missing", engine)


class TestDisplayWorkflowPreview:
    def test_with_steps_and_vars(self):
        data = {
            "name": "Test WF",
            "description": "A test",
            "steps": [{"id": "s1", "type": "shell", "action": "run"}],
        }
        _display_workflow_preview("test", data, {"key": "val"})

    def test_empty(self):
        _display_workflow_preview("test", {}, {})


class TestOutputRunResults:
    def test_json_output(self, capsys):
        result = MagicMock()
        result.model_dump.return_value = {"status": "completed"}
        _output_run_results(result, json_output=True)
        out = capsys.readouterr().out
        assert "completed" in out

    def test_completed(self):
        result = MagicMock()
        result.status = "completed"
        result.step_results = {"s1": {"status": "success"}}
        result.error = None
        _output_run_results(result, json_output=False)

    def test_failed_with_error(self):
        result = MagicMock()
        result.status = "failed"
        result.step_results = {"s1": {"status": "failed"}}
        result.error = "boom"
        _output_run_results(result, json_output=False)

    def test_no_step_results(self):
        result = MagicMock()
        result.status = "completed"
        result.step_results = {}
        result.error = None
        _output_run_results(result, json_output=False)


class TestRunCommand:
    def test_dry_run(self, tmp_path):
        wf_file = tmp_path / "test.json"
        wf_file.write_text(
            json.dumps(
                {
                    "id": "test",
                    "name": "Test",
                    "steps": [{"id": "s1", "type": "shell", "action": "run"}],
                }
            )
        )
        result = runner.invoke(app, ["run", str(wf_file), "--dry-run"])
        assert result.exit_code == 0

    def test_run_json_file(self, tmp_path):
        wf_file = tmp_path / "test.json"
        wf_file.write_text(
            json.dumps(
                {
                    "id": "test",
                    "name": "Test",
                    "steps": [],
                }
            )
        )
        with patch("animus_forge.cli.commands.workflow.get_workflow_engine") as mock_engine:
            mock_result = MagicMock()
            mock_result.status = "completed"
            mock_result.step_results = {}
            mock_result.error = None
            mock_engine.return_value.execute_workflow.return_value = mock_result
            runner.invoke(app, ["run", str(wf_file)])  # Called for side effects


class TestValidateCommand:
    def test_valid_workflow(self, tmp_path):
        wf_file = tmp_path / "test.json"
        wf_file.write_text(
            json.dumps(
                {
                    "id": "test",
                    "steps": [{"id": "s1", "type": "claude_code", "action": "run"}],
                }
            )
        )
        result = runner.invoke(app, ["validate", str(wf_file)])
        assert result.exit_code == 0

    def test_invalid_json(self, tmp_path):
        wf_file = tmp_path / "bad.json"
        wf_file.write_text("not json")
        result = runner.invoke(app, ["validate", str(wf_file)])
        assert result.exit_code != 0

    def test_missing_fields(self, tmp_path):
        wf_file = tmp_path / "test.json"
        wf_file.write_text(json.dumps({"name": "only name"}))
        result = runner.invoke(app, ["validate", str(wf_file)])
        assert result.exit_code != 0


class TestListCommand:
    def test_list_json(self):
        with patch("animus_forge.cli.commands.workflow.get_workflow_engine") as mock_engine:
            mock_engine.return_value.list_workflows.return_value = []
            result = runner.invoke(app, ["list", "--json"])
            assert result.exit_code == 0

    def test_list_empty(self):
        with patch("animus_forge.cli.commands.workflow.get_workflow_engine") as mock_engine:
            mock_engine.return_value.list_workflows.return_value = []
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0

    def test_list_with_workflows(self):
        with patch("animus_forge.cli.commands.workflow.get_workflow_engine") as mock_engine:
            mock_wf = MagicMock()
            mock_wf.steps = [MagicMock()]
            mock_engine.return_value.list_workflows.return_value = [
                {"id": "wf1", "name": "Test", "description": "A test workflow"},
            ]
            mock_engine.return_value.load_workflow.return_value = mock_wf
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0


class TestStatusCommand:
    def test_status_json(self):
        with patch("animus_forge.cli.commands.workflow.get_tracker") as mock_tracker:
            mock_tracker.return_value.get_dashboard_data.return_value = {
                "summary": {},
                "active_workflows": [],
                "recent_executions": [],
            }
            result = runner.invoke(app, ["status", "--json"])
            assert result.exit_code == 0

    def test_status_no_tracker(self):
        with patch(
            "animus_forge.cli.commands.workflow.get_tracker",
            side_effect=Exception("no tracker"),
        ):
            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0

    def test_status_with_data(self):
        with patch("animus_forge.cli.commands.workflow.get_tracker") as mock_tracker:
            mock_tracker.return_value.get_dashboard_data.return_value = {
                "summary": {
                    "active_workflows": 2,
                    "total_executions": 10,
                    "success_rate": 85.0,
                    "avg_duration_ms": 1500,
                },
                "active_workflows": [
                    {
                        "workflow_name": "test",
                        "execution_id": "abc123def456",
                        "completed_steps": 3,
                        "total_steps": 5,
                    }
                ],
                "recent_executions": [
                    {
                        "workflow_name": "test",
                        "status": "completed",
                        "duration_ms": 1200,
                        "completed_steps": 5,
                        "total_steps": 5,
                    }
                ],
            }
            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0


class TestInitCommand:
    def test_init_basic(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init", "my-workflow", "-o", str(tmp_path / "out.json")])
        assert result.exit_code == 0
        assert (tmp_path / "out.json").exists()
        data = json.loads((tmp_path / "out.json").read_text())
        assert data["id"] == "my_workflow"

    def test_init_overwrite_decline(self, tmp_path):
        out = tmp_path / "out.json"
        out.write_text("{}")
        runner.invoke(app, ["init", "test", "-o", str(out)], input="n\n")
        # Should abort - called for side effects only


class TestVersionCommand:
    def test_version_hidden(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
