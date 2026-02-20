"""Tests for interactive workflow runner."""

from unittest.mock import MagicMock, patch

import pytest

from animus_forge.cli.interactive_runner import (
    InteractiveRunner,
    WorkflowInput,
    WorkflowTemplate,
)


class TestWorkflowInput:
    """Tests for WorkflowInput dataclass."""

    def test_defaults(self):
        wi = WorkflowInput(name="test", description="A test input")
        assert wi.name == "test"
        assert wi.input_type == "string"
        assert wi.required is True
        assert wi.default is None
        assert wi.choices is None
        assert wi.validation is None

    def test_select_type(self):
        wi = WorkflowInput(
            name="opt",
            description="Pick one",
            input_type="select",
            choices=["a", "b"],
            default="a",
        )
        assert wi.input_type == "select"
        assert wi.choices == ["a", "b"]


class TestWorkflowTemplate:
    """Tests for WorkflowTemplate dataclass."""

    def test_defaults(self):
        wt = WorkflowTemplate(id="test", name="Test", description="desc", category="Dev")
        assert wt.inputs == []
        assert wt.tags == []

    def test_with_inputs(self):
        inp = WorkflowInput(name="x", description="X")
        wt = WorkflowTemplate(
            id="t", name="T", description="d", category="C", inputs=[inp], tags=["a"]
        )
        assert len(wt.inputs) == 1
        assert wt.tags == ["a"]


class TestInteractiveRunner:
    """Tests for InteractiveRunner."""

    @pytest.fixture
    def runner(self):
        """Create runner with mocked output."""
        with patch("animus_forge.cli.interactive_runner.get_output") as mock_get:
            mock_output = MagicMock()
            mock_get.return_value = mock_output
            r = InteractiveRunner()
            r.output = mock_output
            return r

    def test_templates_exist(self, runner):
        assert len(InteractiveRunner.TEMPLATES) >= 4

    def test_templates_have_categories(self, runner):
        categories = {t.category for t in InteractiveRunner.TEMPLATES}
        assert "Development" in categories

    def test_check_dependencies_no_inquirer(self):
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("animus_forge.cli.interactive_runner.get_output") as mock_get:
                mock_output = MagicMock()
                mock_get.return_value = mock_output
                InteractiveRunner()
                mock_output.warning.assert_called_once()

    def test_select_category_fallback(self, runner):
        """Test category selection with basic input fallback."""
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="0"):
                result = runner._select_category()
                assert result is None

    def test_select_category_fallback_valid(self, runner):
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="1"):
                result = runner._select_category()
                assert result is not None

    def test_select_category_fallback_invalid(self, runner):
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="999"):
                result = runner._select_category()
                assert result is None

    def test_select_category_fallback_non_numeric(self, runner):
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="abc"):
                result = runner._select_category()
                assert result is None

    def test_select_workflow_fallback_cancel(self, runner):
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="0"):
                result = runner._select_workflow("Development")
                assert result is None

    def test_select_workflow_fallback_valid(self, runner):
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="1"):
                result = runner._select_workflow("Development")
                assert result is not None

    def test_select_workflow_fallback_invalid(self, runner):
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="bad"):
                result = runner._select_workflow("Development")
                assert result is None

    def test_gather_inputs_all_provided(self, runner):
        template = WorkflowTemplate(
            id="t",
            name="T",
            description="d",
            category="C",
            inputs=[WorkflowInput(name="x", description="X", required=False, default="val")],
        )
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value=""):
                result = runner._gather_inputs(template)
                assert result == {"x": "val"}

    def test_gather_inputs_required_missing(self, runner):
        template = WorkflowTemplate(
            id="t",
            name="T",
            description="d",
            category="C",
            inputs=[WorkflowInput(name="x", description="X", required=True)],
        )
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value=""):
                result = runner._gather_inputs(template)
                assert result is None

    def test_prompt_input_string_fallback(self, runner):
        inp = WorkflowInput(name="x", description="Enter X", default="def")
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value=""):
                result = runner._prompt_input(inp)
                assert result == "def"

    def test_prompt_input_string_value(self, runner):
        inp = WorkflowInput(name="x", description="Enter X", required=False)
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="hello"):
                result = runner._prompt_input(inp)
                assert result == "hello"

    def test_prompt_input_boolean_fallback(self, runner):
        inp = WorkflowInput(name="x", description="Yes?", input_type="boolean")
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="yes"):
                assert runner._prompt_input(inp) is True
            with patch("builtins.input", return_value="no"):
                assert runner._prompt_input(inp) is False

    def test_prompt_input_number_fallback(self, runner):
        inp = WorkflowInput(name="x", description="Count", input_type="number")
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="42"):
                assert runner._prompt_input(inp) == 42

    def test_prompt_input_select_fallback(self, runner):
        inp = WorkflowInput(
            name="x",
            description="Pick",
            input_type="select",
            choices=["a", "b"],
            required=False,
        )
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="a"):
                assert runner._prompt_input(inp) == "a"

    def test_prompt_input_multiselect_fallback(self, runner):
        inp = WorkflowInput(
            name="x",
            description="Pick",
            input_type="multiselect",
            choices=["a", "b"],
            required=False,
        )
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="a,b"):
                assert runner._prompt_input(inp) == ["a", "b"]

    def test_confirm_execution_yes(self, runner):
        template = WorkflowTemplate(id="t", name="T", description="d", category="C")
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="y"):
                assert runner._confirm_execution(template, {"k": "v"}) is True

    def test_confirm_execution_no(self, runner):
        template = WorkflowTemplate(id="t", name="T", description="d", category="C")
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="n"):
                assert runner._confirm_execution(template, {"k": "v"}) is False

    def test_confirm_execution_long_value(self, runner):
        """Long input values get truncated in display."""
        template = WorkflowTemplate(id="t", name="T", description="d", category="C")
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="y"):
                result = runner._confirm_execution(template, {"k": "x" * 100})
                assert result is True

    @patch("time.sleep")
    def test_execute_workflow(self, mock_sleep, runner):
        template = WorkflowTemplate(id="feat", name="Feature", description="d", category="C")
        result = runner._execute_workflow(template, {"input": "val"})
        assert result["workflow_id"] == "feat"
        assert result["status"] == "completed"
        assert result["inputs"] == {"input": "val"}

    def test_run_cancelled_at_category(self, runner):
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="0"):
                result = runner.run()
                assert result is None

    def test_list_workflows(self, runner):
        """list_workflows should not raise."""
        runner.list_workflows()

    def test_quick_run_not_found(self, runner):
        result = runner.quick_run("nonexistent")
        assert result is None
        runner.output.error.assert_called()

    @patch("time.sleep")
    def test_quick_run_with_inputs(self, mock_sleep, runner):
        result = runner.quick_run(
            "feature-build",
            inputs={
                "feature_request": "add login",
                "codebase_path": ".",
                "test_command": "pytest",
            },
        )
        assert result is not None
        assert result["status"] == "completed"

    @patch("time.sleep")
    def test_quick_run_missing_inputs(self, mock_sleep, runner):
        """Quick run with missing required inputs prompts for them."""
        with patch("animus_forge.cli.interactive_runner.INQUIRER_AVAILABLE", False):
            with patch("builtins.input", return_value="my feature"):
                result = runner.quick_run("feature-build", inputs={})
                assert result is not None


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    @patch("animus_forge.cli.interactive_runner.InteractiveRunner")
    def test_run_interactive(self, mock_cls):
        from animus_forge.cli.interactive_runner import run_interactive

        run_interactive()
        mock_cls.return_value.run.assert_called_once()

    @patch("animus_forge.cli.interactive_runner.InteractiveRunner")
    def test_list_workflows(self, mock_cls):
        from animus_forge.cli.interactive_runner import list_workflows

        list_workflows()
        mock_cls.return_value.list_workflows.assert_called_once()
