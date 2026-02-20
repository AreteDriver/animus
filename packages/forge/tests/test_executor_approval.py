"""Tests for approval gate step type in WorkflowExecutor."""

import os
import shutil
import sys
import tempfile
from unittest.mock import patch

import pytest

sys.path.insert(0, "src")

from animus_forge.state.backends import SQLiteBackend
from animus_forge.workflow.approval_store import ResumeTokenStore, reset_approval_store
from animus_forge.workflow.executor_core import WorkflowExecutor
from animus_forge.workflow.executor_results import StepStatus
from animus_forge.workflow.loader import StepConfig, WorkflowConfig


@pytest.fixture
def backend():
    """Create a temp SQLite backend with approval migration applied."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "test.db")
        backend = SQLiteBackend(db_path=db_path)

        migration_path = os.path.join(
            os.path.dirname(__file__), "..", "migrations", "014_approval_tokens.sql"
        )
        with open(migration_path) as f:
            sql = f.read()
        backend.executescript(sql)

        yield backend
        backend.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def store(backend):
    """Create a ResumeTokenStore with the test backend."""
    return ResumeTokenStore(backend)


@pytest.fixture
def executor():
    """Create a bare WorkflowExecutor with no optional managers."""
    return WorkflowExecutor()


@pytest.fixture(autouse=True)
def cleanup_singleton():
    """Reset the approval store singleton before and after each test."""
    reset_approval_store()
    yield
    reset_approval_store()


# =============================================================================
# TestApprovalHandler
# =============================================================================


class TestApprovalHandler:
    """Tests for the _execute_approval handler."""

    def test_handler_registered(self, executor):
        """Approval handler is registered in _handlers."""
        assert "approval" in executor._handlers

    def test_handler_returns_awaiting_approval(self, executor, store):
        """Handler returns dict with status=awaiting_approval."""
        step = StepConfig(
            id="gate",
            type="approval",
            params={"prompt": "Deploy to prod?"},
        )

        with patch(
            "animus_forge.workflow.approval_store.get_approval_store",
            return_value=store,
        ):
            executor._execution_id = "exec-1"
            executor._current_workflow_id = "wf-1"
            result = executor._execute_approval(step, {"step1": "output"})

        assert result["status"] == "awaiting_approval"
        assert len(result["token"]) == 16
        assert result["prompt"] == "Deploy to prod?"

    def test_handler_gathers_preview(self, executor, store):
        """Handler collects preview data from referenced steps."""
        step = StepConfig(
            id="gate",
            type="approval",
            params={
                "prompt": "Apply changes?",
                "preview_from": ["analyze", "review"],
            },
        )
        context = {
            "analyze": {"findings": ["issue1"]},
            "review": {"suggestions": ["fix1"]},
            "unrelated": "data",
        }

        with patch(
            "animus_forge.workflow.approval_store.get_approval_store",
            return_value=store,
        ):
            executor._execution_id = "exec-1"
            executor._current_workflow_id = "wf-1"
            result = executor._execute_approval(step, context)

        assert result["preview"] == {
            "analyze": {"findings": ["issue1"]},
            "review": {"suggestions": ["fix1"]},
        }
        # unrelated step should not be in preview
        assert "unrelated" not in result["preview"]

    def test_handler_stores_context(self, executor, store):
        """Handler serializes full context to the token."""
        step = StepConfig(
            id="gate",
            type="approval",
            params={"prompt": "Continue?"},
        )
        context = {"step1": "result1", "step2": "result2"}

        with patch(
            "animus_forge.workflow.approval_store.get_approval_store",
            return_value=store,
        ):
            executor._execution_id = "exec-1"
            executor._current_workflow_id = "wf-1"
            result = executor._execute_approval(step, context)

        token_data = store.get_by_token(result["token"])
        assert token_data["context"] == context

    def test_handler_custom_timeout(self, executor, store):
        """Handler respects timeout_hours param."""
        step = StepConfig(
            id="gate",
            type="approval",
            params={"prompt": "Continue?", "timeout_hours": 72},
        )

        with patch(
            "animus_forge.workflow.approval_store.get_approval_store",
            return_value=store,
        ):
            executor._execution_id = "exec-1"
            executor._current_workflow_id = "wf-1"
            result = executor._execute_approval(step, {})

        assert result["timeout_hours"] == 72

    def test_handler_missing_preview_steps_skipped(self, executor, store):
        """Preview references to non-existent steps are silently skipped."""
        step = StepConfig(
            id="gate",
            type="approval",
            params={
                "prompt": "Continue?",
                "preview_from": ["exists", "missing"],
            },
        )
        context = {"exists": "data"}

        with patch(
            "animus_forge.workflow.approval_store.get_approval_store",
            return_value=store,
        ):
            executor._execution_id = "exec-1"
            executor._current_workflow_id = "wf-1"
            result = executor._execute_approval(step, context)

        assert result["preview"] == {"exists": "data"}


# =============================================================================
# TestSequentialHalt
# =============================================================================


class TestSequentialHalt:
    """Tests for approval gate halting sequential execution."""

    def _make_workflow(self, steps):
        return WorkflowConfig(
            name="test-workflow",
            version="1.0",
            description="Test",
            steps=steps,
        )

    def test_execution_halts_at_approval(self, store):
        """Workflow execution stops at approval step."""
        workflow = self._make_workflow(
            [
                StepConfig(id="step1", type="shell", params={"command": "echo hello"}),
                StepConfig(
                    id="gate",
                    type="approval",
                    params={"prompt": "Continue?"},
                ),
                StepConfig(id="step3", type="shell", params={"command": "echo done"}),
            ]
        )

        executor = WorkflowExecutor()

        with patch(
            "animus_forge.workflow.approval_store.get_approval_store",
            return_value=store,
        ):
            result = executor.execute(workflow, inputs={})

        assert result.status == "awaiting_approval"
        assert "__approval_token" in result.outputs
        assert len(result.outputs["__approval_token"]) == 16
        assert result.outputs["__approval_prompt"] == "Continue?"

        # Only 2 steps should have been executed (step1 + gate)
        assert len(result.steps) == 2

    def test_approval_sets_next_step_id(self, store):
        """Token is updated with the correct next step ID."""
        workflow = self._make_workflow(
            [
                StepConfig(id="step1", type="shell", params={"command": "echo a"}),
                StepConfig(
                    id="gate",
                    type="approval",
                    params={"prompt": "Continue?"},
                ),
                StepConfig(id="step3", type="shell", params={"command": "echo b"}),
            ]
        )

        executor = WorkflowExecutor()

        with patch(
            "animus_forge.workflow.approval_store.get_approval_store",
            return_value=store,
        ):
            result = executor.execute(workflow, inputs={})

        token = result.outputs["__approval_token"]
        token_data = store.get_by_token(token)
        assert token_data["next_step_id"] == "step3"

    def test_approval_at_end_of_workflow(self, store):
        """Approval at the last step sets empty next_step_id."""
        workflow = self._make_workflow(
            [
                StepConfig(id="step1", type="shell", params={"command": "echo a"}),
                StepConfig(
                    id="gate",
                    type="approval",
                    params={"prompt": "Finalize?"},
                ),
            ]
        )

        executor = WorkflowExecutor()

        with patch(
            "animus_forge.workflow.approval_store.get_approval_store",
            return_value=store,
        ):
            result = executor.execute(workflow, inputs={})

        token = result.outputs["__approval_token"]
        token_data = store.get_by_token(token)
        assert token_data["next_step_id"] == ""


# =============================================================================
# TestResumeFlow
# =============================================================================


class TestResumeFlow:
    """Tests for resuming execution after approval."""

    def _make_workflow(self, steps):
        return WorkflowConfig(
            name="test-workflow",
            version="1.0",
            description="Test",
            steps=steps,
        )

    def test_resume_from_next_step(self, store):
        """After approval, execution can resume from the next step."""
        workflow = self._make_workflow(
            [
                StepConfig(id="step1", type="shell", params={"command": "echo first"}),
                StepConfig(
                    id="gate",
                    type="approval",
                    params={"prompt": "Continue?"},
                ),
                StepConfig(
                    id="step3",
                    type="shell",
                    params={"command": "echo resumed"},
                    outputs=["final_output"],
                ),
            ]
        )

        executor = WorkflowExecutor()

        # Phase 1: Execute until approval
        with patch(
            "animus_forge.workflow.approval_store.get_approval_store",
            return_value=store,
        ):
            result1 = executor.execute(workflow, inputs={})

        assert result1.status == "awaiting_approval"
        token = result1.outputs["__approval_token"]
        token_data = store.get_by_token(token)
        next_step = token_data["next_step_id"]

        # Phase 2: Approve and resume
        store.approve(token)
        context = token_data["context"]

        executor2 = WorkflowExecutor()
        result2 = executor2.execute(
            workflow,
            inputs=context if isinstance(context, dict) else {},
            resume_from=next_step,
        )

        assert result2.status == "success"
        assert len(result2.steps) == 1  # Only step3


# =============================================================================
# TestLoaderValidation
# =============================================================================


class TestLoaderValidation:
    """Tests for approval step validation in the YAML loader."""

    def test_approval_type_valid(self):
        """Approval is a valid step type."""
        from animus_forge.workflow.loader import VALID_STEP_TYPES

        assert "approval" in VALID_STEP_TYPES

    def test_approval_step_from_dict(self):
        """StepConfig can be created from dict with type=approval."""
        step = StepConfig.from_dict(
            {
                "id": "gate",
                "type": "approval",
                "params": {"prompt": "Deploy?"},
            }
        )
        assert step.type == "approval"
        assert step.params["prompt"] == "Deploy?"

    def test_validation_requires_prompt(self):
        """Approval step without prompt in params fails validation."""
        from animus_forge.workflow.loader import validate_workflow

        data = {
            "name": "test",
            "steps": [
                {"id": "gate", "type": "approval", "params": {}},
            ],
        }
        errors = validate_workflow(data)
        assert any("prompt" in e for e in errors)

    def test_validation_prompt_present_passes(self):
        """Approval step with prompt passes validation."""
        from animus_forge.workflow.loader import validate_workflow

        data = {
            "name": "test",
            "steps": [
                {
                    "id": "gate",
                    "type": "approval",
                    "params": {"prompt": "Continue?"},
                },
            ],
        }
        errors = validate_workflow(data)
        assert not any("prompt" in e for e in errors)

    def test_validation_preview_from_must_be_list(self):
        """preview_from must be a list of step IDs."""
        from animus_forge.workflow.loader import validate_workflow

        data = {
            "name": "test",
            "steps": [
                {
                    "id": "gate",
                    "type": "approval",
                    "params": {"prompt": "Continue?", "preview_from": "step1"},
                },
            ],
        }
        errors = validate_workflow(data)
        assert any("preview_from" in e for e in errors)


# =============================================================================
# TestStatusEnums
# =============================================================================


class TestStatusEnums:
    """Tests for AWAITING_APPROVAL status in all enums."""

    def test_step_status_has_awaiting_approval(self):
        """StepStatus has AWAITING_APPROVAL."""
        assert StepStatus.AWAITING_APPROVAL.value == "awaiting_approval"

    def test_execution_status_has_awaiting_approval(self):
        """ExecutionStatus has AWAITING_APPROVAL."""
        from animus_forge.executions.models import ExecutionStatus

        assert ExecutionStatus.AWAITING_APPROVAL.value == "awaiting_approval"

    def test_workflow_status_has_awaiting_approval(self):
        """WorkflowStatus has AWAITING_APPROVAL."""
        from animus_forge.state.persistence import WorkflowStatus

        assert WorkflowStatus.AWAITING_APPROVAL.value == "awaiting_approval"
