"""Tests for the workflow module."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, "src")

from animus_forge.workflow import (
    ConditionConfig,
    StepConfig,
    WorkflowConfig,
    WorkflowExecutor,
    list_workflows,
    load_workflow,
    validate_workflow,
)
from animus_forge.workflow.executor import StepStatus


class TestConditionConfig:
    """Tests for ConditionConfig class."""

    def test_equals(self):
        """Equals operator works."""
        cond = ConditionConfig("status", "equals", "success")
        assert cond.evaluate({"status": "success"}) is True
        assert cond.evaluate({"status": "failed"}) is False

    def test_not_equals(self):
        """Not equals operator works."""
        cond = ConditionConfig("status", "not_equals", "failed")
        assert cond.evaluate({"status": "success"}) is True
        assert cond.evaluate({"status": "failed"}) is False

    def test_contains(self):
        """Contains operator works."""
        cond = ConditionConfig("tags", "contains", "urgent")
        assert cond.evaluate({"tags": ["urgent", "bug"]}) is True
        assert cond.evaluate({"tags": ["normal"]}) is False
        assert cond.evaluate({"tags": "urgent-fix"}) is True

    def test_greater_than(self):
        """Greater than operator works."""
        cond = ConditionConfig("count", "greater_than", 5)
        assert cond.evaluate({"count": 10}) is True
        assert cond.evaluate({"count": 3}) is False

    def test_less_than(self):
        """Less than operator works."""
        cond = ConditionConfig("count", "less_than", 5)
        assert cond.evaluate({"count": 3}) is True
        assert cond.evaluate({"count": 10}) is False

    def test_missing_field(self):
        """Missing field returns False."""
        cond = ConditionConfig("missing", "equals", "value")
        assert cond.evaluate({}) is False


class TestStepConfig:
    """Tests for StepConfig class."""

    def test_from_dict(self):
        """Can create from dictionary."""
        data = {
            "id": "step1",
            "type": "shell",
            "params": {"command": "echo hello"},
            "on_failure": "skip",
            "max_retries": 2,
        }
        step = StepConfig.from_dict(data)
        assert step.id == "step1"
        assert step.type == "shell"
        assert step.on_failure == "skip"
        assert step.max_retries == 2

    def test_with_condition(self):
        """Can create with condition."""
        data = {
            "id": "step1",
            "type": "shell",
            "condition": {
                "field": "run",
                "operator": "equals",
                "value": True,
            },
        }
        step = StepConfig.from_dict(data)
        assert step.condition is not None
        assert step.condition.field == "run"


class TestWorkflowConfig:
    """Tests for WorkflowConfig class."""

    def test_from_dict(self):
        """Can create from dictionary."""
        data = {
            "name": "Test Workflow",
            "version": "1.0",
            "description": "A test workflow",
            "token_budget": 50000,
            "steps": [
                {"id": "step1", "type": "shell", "params": {"command": "echo 1"}},
                {"id": "step2", "type": "shell", "params": {"command": "echo 2"}},
            ],
            "inputs": {"input1": {"type": "string", "required": True}},
            "outputs": ["result"],
        }
        config = WorkflowConfig.from_dict(data)
        assert config.name == "Test Workflow"
        assert len(config.steps) == 2
        assert config.token_budget == 50000

    def test_get_step(self):
        """Can get step by ID."""
        config = WorkflowConfig(
            name="Test",
            version="1.0",
            description="",
            steps=[
                StepConfig(id="step1", type="shell"),
                StepConfig(id="step2", type="shell"),
            ],
        )
        step = config.get_step("step1")
        assert step is not None
        assert step.id == "step1"

        missing = config.get_step("nonexistent")
        assert missing is None


class TestValidateWorkflow:
    """Tests for validate_workflow function."""

    def test_valid_workflow(self):
        """Valid workflow has no errors."""
        data = {
            "name": "Valid Workflow",
            "steps": [
                {"id": "step1", "type": "shell"},
            ],
        }
        errors = validate_workflow(data)
        assert len(errors) == 0

    def test_missing_name(self):
        """Missing name is an error."""
        data = {"steps": [{"id": "s1", "type": "shell"}]}
        errors = validate_workflow(data)
        assert any("name" in e.lower() for e in errors)

    def test_missing_steps(self):
        """Missing steps is an error."""
        data = {"name": "Test"}
        errors = validate_workflow(data)
        assert any("steps" in e.lower() for e in errors)

    def test_empty_steps(self):
        """Empty steps list is an error."""
        data = {"name": "Test", "steps": []}
        errors = validate_workflow(data)
        assert any("at least one step" in e.lower() for e in errors)

    def test_duplicate_step_ids(self):
        """Duplicate step IDs are errors."""
        data = {
            "name": "Test",
            "steps": [
                {"id": "same", "type": "shell"},
                {"id": "same", "type": "shell"},
            ],
        }
        errors = validate_workflow(data)
        assert any("duplicate" in e.lower() for e in errors)

    def test_invalid_step_type(self):
        """Invalid step type is an error."""
        data = {
            "name": "Test",
            "steps": [{"id": "s1", "type": "invalid_type"}],
        }
        errors = validate_workflow(data)
        assert any("invalid type" in e.lower() for e in errors)


class TestLoadWorkflow:
    """Tests for load_workflow function."""

    def test_load_existing_workflow(self):
        """Can load an existing YAML workflow."""
        # Use one of the example workflows - disable path validation for test
        config = load_workflow("workflows/feature-build.yaml", validate_path=False)
        assert config.name == "Feature Build"
        assert len(config.steps) > 0

    def test_load_nonexistent_raises(self):
        """Loading nonexistent file raises."""
        with pytest.raises(FileNotFoundError):
            load_workflow("nonexistent.yaml", validate_path=False)

    def test_load_invalid_yaml_raises(self):
        """Loading invalid YAML raises."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"invalid: yaml: content: [")
            f.flush()
            try:
                with pytest.raises(ValueError):
                    load_workflow(f.name, validate_path=False)
            finally:
                os.unlink(f.name)


class TestListWorkflows:
    """Tests for list_workflows function."""

    def test_list_workflows(self):
        """Can list workflows in directory."""
        workflows = list_workflows("workflows")
        assert len(workflows) >= 3  # feature-build, bug-fix, refactor
        names = [w["name"] for w in workflows]
        assert "Feature Build" in names

    def test_list_empty_directory(self):
        """Empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflows = list_workflows(tmpdir)
            assert workflows == []


class TestWorkflowExecutor:
    """Tests for WorkflowExecutor class."""

    def test_execute_shell_step(self):
        """Can execute shell steps."""
        workflow = WorkflowConfig(
            name="Shell Test",
            version="1.0",
            description="",
            steps=[
                StepConfig(id="echo", type="shell", params={"command": "echo hello"}),
            ],
        )
        executor = WorkflowExecutor()
        result = executor.execute(workflow)

        assert result.status == "success"
        assert len(result.steps) == 1
        assert result.steps[0].status == StepStatus.SUCCESS
        assert "hello" in result.steps[0].output["stdout"]

    def test_execute_with_condition_skip(self):
        """Conditional step is skipped when condition is false."""
        workflow = WorkflowConfig(
            name="Conditional Test",
            version="1.0",
            description="",
            steps=[
                StepConfig(
                    id="skipped",
                    type="shell",
                    params={"command": "echo should not run"},
                    condition=ConditionConfig("run", "equals", True),
                ),
            ],
        )
        executor = WorkflowExecutor()
        result = executor.execute(workflow, inputs={"run": False})

        assert result.status == "success"
        assert result.steps[0].status == StepStatus.SKIPPED

    def test_execute_with_condition_run(self):
        """Conditional step runs when condition is true."""
        workflow = WorkflowConfig(
            name="Conditional Test",
            version="1.0",
            description="",
            steps=[
                StepConfig(
                    id="runs",
                    type="shell",
                    params={"command": "echo runs"},
                    condition=ConditionConfig("run", "equals", True),
                ),
            ],
        )
        executor = WorkflowExecutor()
        result = executor.execute(workflow, inputs={"run": True})

        assert result.steps[0].status == StepStatus.SUCCESS

    def test_execute_failure_abort(self):
        """Step failure with on_failure=abort stops workflow."""
        workflow = WorkflowConfig(
            name="Failure Test",
            version="1.0",
            description="",
            steps=[
                StepConfig(
                    id="fail",
                    type="shell",
                    params={"command": "exit 1"},
                    on_failure="abort",
                ),
                StepConfig(
                    id="after",
                    type="shell",
                    params={"command": "echo after"},
                ),
            ],
        )
        executor = WorkflowExecutor()
        result = executor.execute(workflow)

        assert result.status == "failed"
        assert len(result.steps) == 1  # Second step never ran

    def test_execute_failure_skip(self):
        """Step failure with on_failure=skip continues workflow."""
        workflow = WorkflowConfig(
            name="Skip Test",
            version="1.0",
            description="",
            steps=[
                StepConfig(
                    id="fail",
                    type="shell",
                    params={"command": "exit 1"},
                    on_failure="skip",
                ),
                StepConfig(
                    id="after",
                    type="shell",
                    params={"command": "echo after"},
                ),
            ],
        )
        executor = WorkflowExecutor()
        result = executor.execute(workflow)

        assert result.status == "success"
        assert len(result.steps) == 2
        assert result.steps[1].status == StepStatus.SUCCESS

    def test_dry_run_mode(self):
        """Dry run mode returns mock responses."""
        workflow = WorkflowConfig(
            name="Dry Run Test",
            version="1.0",
            description="",
            steps=[
                StepConfig(
                    id="claude",
                    type="claude_code",
                    params={"role": "planner", "prompt": "test"},
                ),
                StepConfig(
                    id="openai",
                    type="openai",
                    params={"prompt": "test"},
                ),
            ],
        )
        executor = WorkflowExecutor(dry_run=True)
        result = executor.execute(workflow)

        assert result.status == "success"
        assert result.steps[0].output.get("dry_run") is True
        assert result.steps[1].output.get("dry_run") is True

    def test_context_variable_substitution(self):
        """Context variables are substituted in prompts."""
        workflow = WorkflowConfig(
            name="Substitution Test",
            version="1.0",
            description="",
            steps=[
                StepConfig(
                    id="echo",
                    type="shell",
                    params={"command": "echo ${greeting}"},
                    outputs=["stdout"],
                ),
            ],
        )
        executor = WorkflowExecutor()
        result = executor.execute(workflow, inputs={"greeting": "hello world"})

        assert "hello world" in result.steps[0].output["stdout"]

    def test_required_input_missing(self):
        """Missing required input fails workflow."""
        workflow = WorkflowConfig(
            name="Input Test",
            version="1.0",
            description="",
            steps=[StepConfig(id="s1", type="shell", params={"command": "echo"})],
            inputs={"required_field": {"type": "string", "required": True}},
        )
        executor = WorkflowExecutor()
        result = executor.execute(workflow, inputs={})

        assert result.status == "failed"
        assert "required_field" in result.error

    def test_default_input_value(self):
        """Default input values are used."""
        workflow = WorkflowConfig(
            name="Default Test",
            version="1.0",
            description="",
            steps=[
                StepConfig(
                    id="echo",
                    type="shell",
                    params={"command": "echo ${value}"},
                ),
            ],
            inputs={
                "value": {
                    "type": "string",
                    "required": True,
                    "default": "default_value",
                }
            },
        )
        executor = WorkflowExecutor()
        result = executor.execute(workflow, inputs={})

        assert "default_value" in result.steps[0].output["stdout"]

    def test_register_custom_handler(self):
        """Can register custom step handlers."""

        def custom_handler(step, context):
            return {"custom": True, "step_id": step.id}

        workflow = WorkflowConfig(
            name="Custom Handler Test",
            version="1.0",
            description="",
            steps=[StepConfig(id="custom", type="custom_type")],
        )

        executor = WorkflowExecutor()
        executor.register_handler("custom_type", custom_handler)
        result = executor.execute(workflow)

        assert result.status == "success"
        assert result.steps[0].output["custom"] is True

    def test_execution_result_to_dict(self):
        """ExecutionResult can be converted to dict."""
        workflow = WorkflowConfig(
            name="Dict Test",
            version="1.0",
            description="",
            steps=[StepConfig(id="s1", type="shell", params={"command": "echo test"})],
        )
        executor = WorkflowExecutor()
        result = executor.execute(workflow)

        result_dict = result.to_dict()
        assert result_dict["workflow_name"] == "Dict Test"
        assert result_dict["status"] == "success"
        assert len(result_dict["steps"]) == 1
