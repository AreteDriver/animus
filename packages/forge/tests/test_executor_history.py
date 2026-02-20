"""Tests for task history recording in WorkflowExecutor."""

from unittest.mock import MagicMock, patch

from animus_forge.workflow.executor_core import WorkflowExecutor
from animus_forge.workflow.loader import StepConfig, WorkflowConfig


class TestExecutorHistoryRecording:
    """Verify _record_step_completion writes to task history."""

    def test_record_task_called_on_step_completion(self):
        """record_task is called when a shell step completes."""
        mock_store = MagicMock()

        with patch(
            "animus_forge.db.get_task_store",
            return_value=mock_store,
        ):
            workflow = WorkflowConfig(
                name="History Test",
                version="1.0",
                description="",
                steps=[
                    StepConfig(
                        id="echo-step",
                        type="shell",
                        params={"command": "echo hello"},
                    ),
                ],
            )
            executor = WorkflowExecutor()
            result = executor.execute(workflow)

        assert result.status == "success"
        mock_store.record_task.assert_called_once()
        call_kwargs = mock_store.record_task.call_args
        assert call_kwargs[1]["job_id"] == "echo-step"

    def test_record_task_captures_failure(self):
        """record_task records failed step status."""
        mock_store = MagicMock()

        with patch(
            "animus_forge.db.get_task_store",
            return_value=mock_store,
        ):
            workflow = WorkflowConfig(
                name="Fail Test",
                version="1.0",
                description="",
                steps=[
                    StepConfig(
                        id="fail-step",
                        type="shell",
                        params={"command": "exit 1"},
                    ),
                ],
            )
            executor = WorkflowExecutor()
            executor.execute(workflow)

        # The step was recorded (may have been called with error info)
        assert mock_store.record_task.called

    def test_record_task_failure_does_not_break_execution(self):
        """If task history recording fails, workflow still completes."""
        with patch(
            "animus_forge.db.get_task_store",
            side_effect=Exception("DB unavailable"),
        ):
            workflow = WorkflowConfig(
                name="Resilient Test",
                version="1.0",
                description="",
                steps=[
                    StepConfig(
                        id="resilient-step",
                        type="shell",
                        params={"command": "echo still works"},
                    ),
                ],
            )
            executor = WorkflowExecutor()
            result = executor.execute(workflow)

        # Workflow should succeed despite history recording failure
        assert result.status == "success"
        assert len(result.steps) == 1

    def test_record_task_uses_agent_role_from_params(self):
        """agent_role is extracted from step params."""
        mock_store = MagicMock()

        with patch(
            "animus_forge.db.get_task_store",
            return_value=mock_store,
        ):
            workflow = WorkflowConfig(
                name="Role Test",
                version="1.0",
                description="",
                steps=[
                    StepConfig(
                        id="role-step",
                        type="shell",
                        params={"command": "echo hi", "role": "reviewer"},
                    ),
                ],
            )
            executor = WorkflowExecutor()
            executor.execute(workflow)

        mock_store.record_task.assert_called_once()
        call_kwargs = mock_store.record_task.call_args[1]
        assert call_kwargs["agent_role"] == "reviewer"
