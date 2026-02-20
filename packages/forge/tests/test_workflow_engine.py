"""Tests for workflow models and adapter."""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_forge.orchestrator import WorkflowEngineAdapter
from animus_forge.orchestrator.workflow_engine import (
    StepType,
    Workflow,
    WorkflowResult,
    WorkflowStep,
)
from animus_forge.orchestrator.workflow_engine_adapter import (
    convert_execution_result,
    convert_step_type,
    convert_workflow,
    convert_workflow_step,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for workflows and logs."""
    workflows_dir = tmp_path / "workflows"
    logs_dir = tmp_path / "logs"
    workflows_dir.mkdir()
    logs_dir.mkdir()
    return {"workflows_dir": workflows_dir, "logs_dir": logs_dir}


@pytest.fixture
def mock_settings(temp_dirs):
    """Create mock settings."""
    settings = MagicMock()
    settings.workflows_dir = temp_dirs["workflows_dir"]
    settings.logs_dir = temp_dirs["logs_dir"]
    return settings


@pytest.fixture
def workflow_adapter(mock_settings):
    """Create workflow adapter with mocked settings."""
    with patch(
        "animus_forge.orchestrator.workflow_engine_adapter.get_settings"
    ) as mock_get_settings:
        mock_get_settings.return_value = mock_settings
        with patch("animus_forge.workflow.executor.WorkflowExecutor"):
            adapter = WorkflowEngineAdapter(dry_run=True)
            # Manually set settings for testing
            adapter._test_settings = mock_settings
            yield adapter


@pytest.fixture
def simple_workflow():
    """Create a simple workflow."""
    return Workflow(
        id="test-workflow",
        name="Test Workflow",
        description="A test workflow",
        steps=[
            WorkflowStep(
                id="step1",
                type=StepType.TRANSFORM,
                action="format",
                params={"template": "Hello, {name}!"},
                next_step="step2",
            ),
            WorkflowStep(
                id="step2",
                type=StepType.TRANSFORM,
                action="extract",
                params={"source": "data", "key": "value"},
            ),
        ],
        variables={"name": "World", "data": {"value": 42}},
    )


# =============================================================================
# Test StepType Enum
# =============================================================================


class TestStepType:
    """Tests for StepType enum."""

    def test_openai_value(self):
        """Test OPENAI type value."""
        assert StepType.OPENAI.value == "openai"

    def test_github_value(self):
        """Test GITHUB type value."""
        assert StepType.GITHUB.value == "github"

    def test_notion_value(self):
        """Test NOTION type value."""
        assert StepType.NOTION.value == "notion"

    def test_gmail_value(self):
        """Test GMAIL type value."""
        assert StepType.GMAIL.value == "gmail"

    def test_transform_value(self):
        """Test TRANSFORM type value."""
        assert StepType.TRANSFORM.value == "transform"

    def test_claude_code_value(self):
        """Test CLAUDE_CODE type value."""
        assert StepType.CLAUDE_CODE.value == "claude_code"

    def test_from_string(self):
        """Test creating from string value."""
        assert StepType("openai") == StepType.OPENAI
        assert StepType("github") == StepType.GITHUB


# =============================================================================
# Test WorkflowStep Model
# =============================================================================


class TestWorkflowStep:
    """Tests for WorkflowStep model."""

    def test_create_step(self):
        """Test creating a workflow step."""
        step = WorkflowStep(
            id="test-step",
            type=StepType.OPENAI,
            action="generate_completion",
        )
        assert step.id == "test-step"
        assert step.type == StepType.OPENAI
        assert step.action == "generate_completion"

    def test_step_with_params(self):
        """Test step with parameters."""
        step = WorkflowStep(
            id="test",
            type=StepType.OPENAI,
            action="generate_completion",
            params={"prompt": "Hello", "max_tokens": 100},
        )
        assert step.params["prompt"] == "Hello"
        assert step.params["max_tokens"] == 100

    def test_step_with_next(self):
        """Test step with next_step."""
        step = WorkflowStep(
            id="step1",
            type=StepType.TRANSFORM,
            action="format",
            next_step="step2",
        )
        assert step.next_step == "step2"

    def test_default_params(self):
        """Test default params is empty dict."""
        step = WorkflowStep(
            id="test",
            type=StepType.TRANSFORM,
            action="extract",
        )
        assert step.params == {}

    def test_default_next_step(self):
        """Test default next_step is None."""
        step = WorkflowStep(
            id="test",
            type=StepType.TRANSFORM,
            action="extract",
        )
        assert step.next_step is None

    def test_step_serialization(self):
        """Test step can be serialized."""
        step = WorkflowStep(
            id="test",
            type=StepType.OPENAI,
            action="generate_completion",
            params={"prompt": "test"},
        )
        data = step.model_dump()
        assert data["id"] == "test"
        assert data["type"] == "openai"


# =============================================================================
# Test Workflow Model
# =============================================================================


class TestWorkflow:
    """Tests for Workflow model."""

    def test_create_workflow(self):
        """Test creating a workflow."""
        workflow = Workflow(
            id="test",
            name="Test Workflow",
            description="A test",
        )
        assert workflow.id == "test"
        assert workflow.name == "Test Workflow"
        assert workflow.steps == []
        assert workflow.variables == {}

    def test_workflow_with_steps(self, simple_workflow):
        """Test workflow with steps."""
        assert len(simple_workflow.steps) == 2
        assert simple_workflow.steps[0].id == "step1"
        assert simple_workflow.steps[1].id == "step2"

    def test_workflow_with_variables(self, simple_workflow):
        """Test workflow with variables."""
        assert simple_workflow.variables["name"] == "World"
        assert simple_workflow.variables["data"]["value"] == 42

    def test_workflow_serialization(self, simple_workflow):
        """Test workflow can be serialized."""
        data = simple_workflow.model_dump()
        assert data["id"] == "test-workflow"
        assert data["name"] == "Test Workflow"
        assert len(data["steps"]) == 2


# =============================================================================
# Test WorkflowResult Model
# =============================================================================


class TestWorkflowResult:
    """Tests for WorkflowResult model."""

    def test_create_result(self):
        """Test creating a workflow result."""
        result = WorkflowResult(
            workflow_id="test",
            status="running",
            started_at=datetime.now(),
        )
        assert result.workflow_id == "test"
        assert result.status == "running"
        assert result.completed_at is None

    def test_result_with_outputs(self):
        """Test result with outputs."""
        result = WorkflowResult(
            workflow_id="test",
            status="completed",
            started_at=datetime.now(),
            steps_executed=["step1", "step2"],
            outputs={"step1": "result1", "step2": "result2"},
        )
        assert len(result.steps_executed) == 2
        assert result.outputs["step1"] == "result1"

    def test_result_with_errors(self):
        """Test result with errors."""
        result = WorkflowResult(
            workflow_id="test",
            status="failed",
            started_at=datetime.now(),
            errors=["Error 1", "Error 2"],
        )
        assert result.status == "failed"
        assert len(result.errors) == 2

    def test_result_serialization(self):
        """Test result can be serialized."""
        result = WorkflowResult(
            workflow_id="test",
            status="completed",
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
        data = result.model_dump(mode="json")
        assert data["workflow_id"] == "test"
        assert "started_at" in data


# =============================================================================
# Test WorkflowEngineAdapter Initialization
# =============================================================================


class TestWorkflowEngineAdapterInit:
    """Tests for WorkflowEngineAdapter initialization."""

    def test_adapter_init(self):
        """Test adapter initializes correctly."""
        with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
            adapter = WorkflowEngineAdapter(dry_run=True)
            assert adapter is not None

    def test_adapter_has_executor(self):
        """Test adapter has internal executor."""
        with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
            adapter = WorkflowEngineAdapter()
            assert adapter._executor is not None


# =============================================================================
# Test Workflow Persistence
# =============================================================================


class TestWorkflowPersistence:
    """Tests for workflow save/load/list operations."""

    def test_save_workflow(self, mock_settings, simple_workflow):
        """Test saving a workflow."""
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
                adapter = WorkflowEngineAdapter()
                result = adapter.save_workflow(simple_workflow)
                assert result is True
                file_path = mock_settings.workflows_dir / f"{simple_workflow.id}.json"
                assert file_path.exists()

    def test_save_workflow_creates_valid_json(self, mock_settings, simple_workflow):
        """Test saved workflow is valid JSON."""
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
                adapter = WorkflowEngineAdapter()
                adapter.save_workflow(simple_workflow)
                file_path = mock_settings.workflows_dir / f"{simple_workflow.id}.json"
                with open(file_path) as f:
                    data = json.load(f)
                assert data["id"] == "test-workflow"
                assert len(data["steps"]) == 2

    def test_load_workflow(self, mock_settings, simple_workflow):
        """Test loading a workflow."""
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
                adapter = WorkflowEngineAdapter()
                adapter.save_workflow(simple_workflow)
                loaded = adapter.load_workflow(simple_workflow.id)
                assert loaded is not None
                assert loaded.id == simple_workflow.id
                assert len(loaded.steps) == 2

    def test_load_nonexistent_workflow(self, mock_settings):
        """Test loading nonexistent workflow returns None."""
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
                adapter = WorkflowEngineAdapter()
                result = adapter.load_workflow("nonexistent")
                assert result is None

    def test_list_workflows_empty(self, mock_settings):
        """Test listing workflows when none exist."""
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
                adapter = WorkflowEngineAdapter()
                result = adapter.list_workflows()
                assert result == []

    def test_list_workflows(self, mock_settings):
        """Test listing workflows."""
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
                adapter = WorkflowEngineAdapter()
                # Save some workflows
                for i in range(3):
                    workflow = Workflow(
                        id=f"workflow-{i}",
                        name=f"Workflow {i}",
                        description=f"Description {i}",
                    )
                    adapter.save_workflow(workflow)

                result = adapter.list_workflows()
                assert len(result) == 3
                ids = [w["id"] for w in result]
                assert "workflow-0" in ids
                assert "workflow-1" in ids
                assert "workflow-2" in ids

    def test_list_workflows_handles_invalid_files(self, mock_settings):
        """Test list_workflows handles invalid JSON files."""
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
                adapter = WorkflowEngineAdapter()
                # Create valid workflow
                workflow = Workflow(id="valid", name="Valid", description="Valid workflow")
                adapter.save_workflow(workflow)

                # Create invalid JSON file
                invalid_path = mock_settings.workflows_dir / "invalid.json"
                with open(invalid_path, "w") as f:
                    f.write("not valid json")

                result = adapter.list_workflows()
                assert len(result) == 1
                assert result[0]["id"] == "valid"

    def test_settings_property(self, mock_settings):
        """Test settings property returns settings."""
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
                adapter = WorkflowEngineAdapter()
                assert adapter.settings is not None


# =============================================================================
# Test Workflow Execution
# =============================================================================


class TestWorkflowExecution:
    """Tests for workflow execution through adapter."""

    def test_execute_workflow_calls_executor(self, mock_settings, simple_workflow):
        """Test execute_workflow uses internal executor."""
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch(
                "animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"
            ) as mock_executor_class:
                mock_executor = MagicMock()
                mock_result = MagicMock()
                mock_result.status = "completed"
                mock_result.started_at = datetime.now()
                mock_result.completed_at = datetime.now()
                mock_result.steps = []
                mock_result.outputs = {}
                mock_result.error = None
                mock_executor.execute.return_value = mock_result
                mock_executor_class.return_value = mock_executor

                adapter = WorkflowEngineAdapter()
                result = adapter.execute_workflow(simple_workflow)

                assert result.status == "completed"
                mock_executor.execute.assert_called_once()

    def test_execute_workflow_returns_workflow_result(self, mock_settings, simple_workflow):
        """Test execute_workflow returns WorkflowResult."""
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch(
                "animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"
            ) as mock_executor_class:
                mock_executor = MagicMock()
                mock_result = MagicMock()
                mock_result.status = "completed"
                mock_result.started_at = datetime.now()
                mock_result.completed_at = datetime.now()
                mock_result.steps = []
                mock_result.outputs = {"key": "value"}
                mock_result.error = None
                mock_executor.execute.return_value = mock_result
                mock_executor_class.return_value = mock_executor

                adapter = WorkflowEngineAdapter()
                result = adapter.execute_workflow(simple_workflow)

                assert isinstance(result, WorkflowResult)
                assert result.workflow_id == simple_workflow.id


# =============================================================================
# Test convert_step_type
# =============================================================================


class TestConvertStepType:
    """Tests for convert_step_type function."""

    def test_openai_maps_to_openai(self):
        assert convert_step_type(StepType.OPENAI) == "openai"

    def test_claude_code_maps_to_claude_code(self):
        assert convert_step_type(StepType.CLAUDE_CODE) == "claude_code"

    def test_github_maps_to_shell(self):
        assert convert_step_type(StepType.GITHUB) == "shell"

    def test_notion_maps_to_shell(self):
        assert convert_step_type(StepType.NOTION) == "shell"

    def test_gmail_maps_to_shell(self):
        assert convert_step_type(StepType.GMAIL) == "shell"

    def test_transform_maps_to_shell(self):
        assert convert_step_type(StepType.TRANSFORM) == "shell"

    def test_unknown_type_defaults_to_shell(self):
        """If an unknown step type were added, it defaults to 'shell'."""
        # Use a MagicMock to simulate an unknown enum member
        unknown = MagicMock()
        assert convert_step_type(unknown) == "shell"


# =============================================================================
# Test convert_workflow_step
# =============================================================================


class TestConvertWorkflowStep:
    """Tests for convert_workflow_step function."""

    def test_openai_step_adds_prompt(self):
        step = WorkflowStep(id="s1", type=StepType.OPENAI, action="Generate code")
        result = convert_workflow_step(step)
        assert result["id"] == "s1"
        assert result["type"] == "openai"
        assert result["params"]["prompt"] == "Generate code"

    def test_openai_step_preserves_existing_prompt(self):
        step = WorkflowStep(
            id="s1",
            type=StepType.OPENAI,
            action="Generate code",
            params={"prompt": "Existing prompt"},
        )
        result = convert_workflow_step(step)
        assert result["params"]["prompt"] == "Existing prompt"

    def test_claude_code_step_adds_prompt_and_role(self):
        step = WorkflowStep(id="s1", type=StepType.CLAUDE_CODE, action="Build feature")
        result = convert_workflow_step(step)
        assert result["type"] == "claude_code"
        assert result["params"]["prompt"] == "Build feature"
        assert result["params"]["role"] == "builder"

    def test_claude_code_step_preserves_existing_prompt(self):
        step = WorkflowStep(
            id="s1",
            type=StepType.CLAUDE_CODE,
            action="Build feature",
            params={"prompt": "Custom prompt"},
        )
        result = convert_workflow_step(step)
        assert result["params"]["prompt"] == "Custom prompt"

    def test_claude_code_step_preserves_existing_role(self):
        step = WorkflowStep(
            id="s1",
            type=StepType.CLAUDE_CODE,
            action="Review",
            params={"role": "reviewer"},
        )
        result = convert_workflow_step(step)
        assert result["params"]["role"] == "reviewer"

    def test_shell_step_adds_command(self):
        step = WorkflowStep(id="s1", type=StepType.TRANSFORM, action="format data")
        result = convert_workflow_step(step)
        assert result["type"] == "shell"
        assert "format data" in result["params"]["command"]

    def test_shell_step_preserves_existing_command(self):
        step = WorkflowStep(
            id="s1",
            type=StepType.GITHUB,
            action="push",
            params={"command": "git push origin main"},
        )
        result = convert_workflow_step(step)
        assert result["params"]["command"] == "git push origin main"

    def test_github_step_maps_to_shell(self):
        step = WorkflowStep(id="s1", type=StepType.GITHUB, action="create_pr")
        result = convert_workflow_step(step)
        assert result["type"] == "shell"

    def test_notion_step_maps_to_shell(self):
        step = WorkflowStep(id="s1", type=StepType.NOTION, action="update_page")
        result = convert_workflow_step(step)
        assert result["type"] == "shell"

    def test_gmail_step_maps_to_shell(self):
        step = WorkflowStep(id="s1", type=StepType.GMAIL, action="send_email")
        result = convert_workflow_step(step)
        assert result["type"] == "shell"

    def test_step_params_copied_not_mutated(self):
        original_params = {"prompt": "test", "extra": "data"}
        step = WorkflowStep(
            id="s1",
            type=StepType.OPENAI,
            action="generate",
            params=original_params,
        )
        result = convert_workflow_step(step)
        # Original should not be mutated
        assert result["params"] is not original_params

    def test_openai_step_empty_action_no_prompt(self):
        step = WorkflowStep(id="s1", type=StepType.OPENAI, action="")
        result = convert_workflow_step(step)
        # Empty action is falsy, so no prompt added
        assert "prompt" not in result["params"]

    def test_additional_params_preserved(self):
        step = WorkflowStep(
            id="s1",
            type=StepType.OPENAI,
            action="gen",
            params={"max_tokens": 100, "temperature": 0.7},
        )
        result = convert_workflow_step(step)
        assert result["params"]["max_tokens"] == 100
        assert result["params"]["temperature"] == 0.7
        assert result["params"]["prompt"] == "gen"


# =============================================================================
# Test convert_workflow
# =============================================================================


class TestConvertWorkflow:
    """Tests for convert_workflow function."""

    def test_converts_name(self, simple_workflow):
        config = convert_workflow(simple_workflow)
        assert config.name == "Test Workflow"

    def test_converts_description(self, simple_workflow):
        config = convert_workflow(simple_workflow)
        assert config.description == "A test workflow"

    def test_default_version(self, simple_workflow):
        config = convert_workflow(simple_workflow)
        assert config.version == "1.0.0"

    def test_converts_steps(self, simple_workflow):
        config = convert_workflow(simple_workflow)
        assert len(config.steps) == 2

    def test_converts_variables_to_inputs(self, simple_workflow):
        config = convert_workflow(simple_workflow)
        assert "name" in config.inputs
        assert config.inputs["name"]["type"] == "string"
        assert config.inputs["name"]["default"] == "World"

    def test_empty_workflow(self):
        workflow = Workflow(id="empty", name="Empty", description="No steps")
        config = convert_workflow(workflow)
        assert config.steps == []
        assert config.inputs == {}

    def test_complex_variable_values(self):
        workflow = Workflow(
            id="test",
            name="Test",
            description="Test",
            variables={"count": 42, "flag": True, "items": [1, 2, 3]},
        )
        config = convert_workflow(workflow)
        assert config.inputs["count"]["default"] == 42
        assert config.inputs["flag"]["default"] is True
        assert config.inputs["items"]["default"] == [1, 2, 3]


# =============================================================================
# Test convert_execution_result
# =============================================================================


class TestConvertExecutionResult:
    """Tests for convert_execution_result function."""

    def test_maps_workflow_id(self):
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.started_at = datetime.now()
        mock_result.completed_at = datetime.now()
        mock_result.steps = []
        mock_result.outputs = {}
        mock_result.error = None

        result = convert_execution_result(mock_result, "wf-123")
        assert result.workflow_id == "wf-123"

    def test_maps_status(self):
        mock_result = MagicMock()
        mock_result.status = "failed"
        mock_result.started_at = datetime.now()
        mock_result.completed_at = None
        mock_result.steps = []
        mock_result.outputs = {}
        mock_result.error = "Something broke"

        result = convert_execution_result(mock_result, "wf-1")
        assert result.status == "failed"

    def test_maps_timestamps(self):
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 5, 0)
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.started_at = start
        mock_result.completed_at = end
        mock_result.steps = []
        mock_result.outputs = {}
        mock_result.error = None

        result = convert_execution_result(mock_result, "wf-1")
        assert result.started_at == start
        assert result.completed_at == end

    def test_maps_step_ids(self):
        mock_step1 = MagicMock()
        mock_step1.step_id = "step-a"
        mock_step2 = MagicMock()
        mock_step2.step_id = "step-b"
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.started_at = datetime.now()
        mock_result.completed_at = datetime.now()
        mock_result.steps = [mock_step1, mock_step2]
        mock_result.outputs = {}
        mock_result.error = None

        result = convert_execution_result(mock_result, "wf-1")
        assert result.steps_executed == ["step-a", "step-b"]

    def test_maps_outputs(self):
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.started_at = datetime.now()
        mock_result.completed_at = datetime.now()
        mock_result.steps = []
        mock_result.outputs = {"key": "value", "count": 42}
        mock_result.error = None

        result = convert_execution_result(mock_result, "wf-1")
        assert result.outputs == {"key": "value", "count": 42}

    def test_maps_error_to_errors_list(self):
        mock_result = MagicMock()
        mock_result.status = "failed"
        mock_result.started_at = datetime.now()
        mock_result.completed_at = None
        mock_result.steps = []
        mock_result.outputs = {}
        mock_result.error = "Timeout exceeded"

        result = convert_execution_result(mock_result, "wf-1")
        assert result.errors == ["Timeout exceeded"]

    def test_no_error_produces_empty_errors_list(self):
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.started_at = datetime.now()
        mock_result.completed_at = datetime.now()
        mock_result.steps = []
        mock_result.outputs = {}
        mock_result.error = None

        result = convert_execution_result(mock_result, "wf-1")
        assert result.errors == []

    def test_returns_workflow_result_type(self):
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.started_at = datetime.now()
        mock_result.completed_at = datetime.now()
        mock_result.steps = []
        mock_result.outputs = {}
        mock_result.error = None

        result = convert_execution_result(mock_result, "wf-1")
        assert isinstance(result, WorkflowResult)


# =============================================================================
# Test WorkflowEngineAdapter — Additional Coverage
# =============================================================================


class TestWorkflowEngineAdapterAdditional:
    """Additional tests for WorkflowEngineAdapter."""

    def test_adapter_with_checkpoint_manager(self, mock_settings):
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch(
                "animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"
            ) as mock_exec_cls:
                checkpoint_mgr = MagicMock()
                _ = WorkflowEngineAdapter(checkpoint_manager=checkpoint_mgr)
                mock_exec_cls.assert_called_once_with(
                    checkpoint_manager=checkpoint_mgr,
                    contract_validator=None,
                    budget_manager=None,
                    dry_run=False,
                    execution_manager=None,
                )

    def test_adapter_with_contract_validator(self, mock_settings):
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch(
                "animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"
            ) as mock_exec_cls:
                validator = MagicMock()
                _ = WorkflowEngineAdapter(contract_validator=validator)
                call_kwargs = mock_exec_cls.call_args[1]
                assert call_kwargs["contract_validator"] is validator

    def test_adapter_with_budget_manager(self, mock_settings):
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch(
                "animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"
            ) as mock_exec_cls:
                budget_mgr = MagicMock()
                _ = WorkflowEngineAdapter(budget_manager=budget_mgr)
                call_kwargs = mock_exec_cls.call_args[1]
                assert call_kwargs["budget_manager"] is budget_mgr

    def test_adapter_dry_run_flag(self, mock_settings):
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch(
                "animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"
            ) as mock_exec_cls:
                _ = WorkflowEngineAdapter(dry_run=True)
                call_kwargs = mock_exec_cls.call_args[1]
                assert call_kwargs["dry_run"] is True

    def test_execute_workflow_passes_variables_as_inputs(self, mock_settings, simple_workflow):
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch(
                "animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"
            ) as mock_exec_cls:
                mock_executor = MagicMock()
                mock_result = MagicMock()
                mock_result.status = "completed"
                mock_result.started_at = datetime.now()
                mock_result.completed_at = datetime.now()
                mock_result.steps = []
                mock_result.outputs = {}
                mock_result.error = None
                mock_executor.execute.return_value = mock_result
                mock_exec_cls.return_value = mock_executor

                adapter = WorkflowEngineAdapter()
                adapter.execute_workflow(simple_workflow)

                # Verify inputs= was passed
                call_kwargs = mock_executor.execute.call_args[1]
                assert call_kwargs["inputs"] == simple_workflow.variables

    def test_execute_workflow_async(self, mock_settings, simple_workflow):
        """Test async workflow execution."""

        async def _test():
            with patch(
                "animus_forge.orchestrator.workflow_engine_adapter.get_settings"
            ) as mock_get:
                mock_get.return_value = mock_settings
                with patch(
                    "animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"
                ) as mock_exec_cls:
                    mock_executor = MagicMock()
                    mock_result = MagicMock()
                    mock_result.status = "completed"
                    mock_result.started_at = datetime.now()
                    mock_result.completed_at = datetime.now()
                    mock_result.steps = []
                    mock_result.outputs = {}
                    mock_result.error = None
                    mock_executor.execute_async = AsyncMock(return_value=mock_result)
                    mock_exec_cls.return_value = mock_executor

                    adapter = WorkflowEngineAdapter()
                    result = await adapter.execute_workflow_async(simple_workflow)

                    assert isinstance(result, WorkflowResult)
                    assert result.workflow_id == simple_workflow.id
                    assert result.status == "completed"

        asyncio.run(_test())

    def test_execute_workflow_async_passes_variables(self, mock_settings, simple_workflow):
        """Test async execution passes variables as inputs."""

        async def _test():
            with patch(
                "animus_forge.orchestrator.workflow_engine_adapter.get_settings"
            ) as mock_get:
                mock_get.return_value = mock_settings
                with patch(
                    "animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"
                ) as mock_exec_cls:
                    mock_executor = MagicMock()
                    mock_result = MagicMock()
                    mock_result.status = "completed"
                    mock_result.started_at = datetime.now()
                    mock_result.completed_at = datetime.now()
                    mock_result.steps = []
                    mock_result.outputs = {}
                    mock_result.error = None
                    mock_executor.execute_async = AsyncMock(return_value=mock_result)
                    mock_exec_cls.return_value = mock_executor

                    adapter = WorkflowEngineAdapter()
                    await adapter.execute_workflow_async(simple_workflow)

                    call_kwargs = mock_executor.execute_async.call_args[1]
                    assert call_kwargs["inputs"] == simple_workflow.variables

        asyncio.run(_test())

    def test_save_workflow_error_returns_false(self, mock_settings):
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
                adapter = WorkflowEngineAdapter()
                workflow = Workflow(id="fail", name="Fail", description="Should fail")
                # Patch builtins.open to force an error during save
                with patch("builtins.open", side_effect=OSError("Permission denied")):
                    result = adapter.save_workflow(workflow)
                assert result is False

    def test_load_workflow_error_returns_none(self, mock_settings):
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
                adapter = WorkflowEngineAdapter()
                # Create a file with invalid JSON that can be parsed but
                # won't make a valid Workflow
                bad_file = mock_settings.workflows_dir / "bad.json"
                bad_file.write_text('{"id": 123}')  # id should be str
                result = adapter.load_workflow("bad")
                assert result is None

    def test_list_workflows_nonexistent_directory(self, mock_settings, tmp_path):
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_settings.workflows_dir = tmp_path / "nonexistent"
            mock_get.return_value = mock_settings
            with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
                adapter = WorkflowEngineAdapter()
                result = adapter.list_workflows()
                assert result == []

    def test_list_workflows_metadata_fields(self, mock_settings):
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            mock_get.return_value = mock_settings
            with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
                adapter = WorkflowEngineAdapter()
                workflow = Workflow(
                    id="meta-test",
                    name="Metadata Test",
                    description="Check metadata",
                )
                adapter.save_workflow(workflow)
                result = adapter.list_workflows()
                assert len(result) == 1
                assert result[0]["id"] == "meta-test"
                assert result[0]["name"] == "Metadata Test"
                assert result[0]["description"] == "Check metadata"

    def test_settings_property_returns_get_settings(self, mock_settings):
        with patch("animus_forge.orchestrator.workflow_engine_adapter.get_settings") as mock_get:
            sentinel = MagicMock()
            mock_get.return_value = sentinel
            with patch("animus_forge.orchestrator.workflow_engine_adapter.WorkflowExecutor"):
                adapter = WorkflowEngineAdapter()
                assert adapter.settings is sentinel


# =============================================================================
# Test WorkflowStep — Additional Edge Cases
# =============================================================================


class TestWorkflowStepEdgeCases:
    """Additional edge case tests for WorkflowStep."""

    def test_step_from_dict(self):
        data = {
            "id": "s1",
            "type": "openai",
            "action": "generate",
            "params": {"prompt": "Hello"},
            "next_step": "s2",
        }
        step = WorkflowStep(**data)
        assert step.id == "s1"
        assert step.type == StepType.OPENAI
        assert step.next_step == "s2"

    def test_step_model_dump_mode_json(self):
        step = WorkflowStep(id="s1", type=StepType.CLAUDE_CODE, action="build")
        data = step.model_dump(mode="json")
        assert data["type"] == "claude_code"
        assert isinstance(data["type"], str)


# =============================================================================
# Test Workflow — Additional Edge Cases
# =============================================================================


class TestWorkflowEdgeCases:
    """Additional edge case tests for Workflow."""

    def test_workflow_model_dump_mode_json(self, simple_workflow):
        data = simple_workflow.model_dump(mode="json")
        assert isinstance(data["steps"], list)
        assert data["steps"][0]["type"] == "transform"

    def test_workflow_from_json_roundtrip(self, simple_workflow):
        json_str = simple_workflow.model_dump_json()
        loaded = Workflow.model_validate_json(json_str)
        assert loaded.id == simple_workflow.id
        assert len(loaded.steps) == len(simple_workflow.steps)

    def test_workflow_deep_copy(self, simple_workflow):
        data = simple_workflow.model_dump()
        copy = Workflow(**data)
        assert copy.id == simple_workflow.id
        assert copy is not simple_workflow


# =============================================================================
# Test WorkflowResult — Additional Edge Cases
# =============================================================================


class TestWorkflowResultEdgeCases:
    """Additional edge case tests for WorkflowResult."""

    def test_result_no_errors_default(self):
        result = WorkflowResult(
            workflow_id="test",
            status="completed",
            started_at=datetime.now(),
        )
        assert result.errors == []

    def test_result_no_steps_executed_default(self):
        result = WorkflowResult(
            workflow_id="test",
            status="completed",
            started_at=datetime.now(),
        )
        assert result.steps_executed == []

    def test_result_no_outputs_default(self):
        result = WorkflowResult(
            workflow_id="test",
            status="completed",
            started_at=datetime.now(),
        )
        assert result.outputs == {}

    def test_result_json_roundtrip(self):
        now = datetime.now()
        result = WorkflowResult(
            workflow_id="test",
            status="completed",
            started_at=now,
            completed_at=now,
            steps_executed=["s1"],
            outputs={"key": "val"},
            errors=[],
        )
        json_str = result.model_dump_json()
        loaded = WorkflowResult.model_validate_json(json_str)
        assert loaded.workflow_id == "test"
        assert loaded.status == "completed"
        assert loaded.steps_executed == ["s1"]
