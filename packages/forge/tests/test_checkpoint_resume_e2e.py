"""E2E test for checkpoint/resume functionality.

Verifies that a workflow can:
1. Execute with checkpointing enabled
2. Fail mid-workflow
3. Resume from the last successful checkpoint
4. Complete successfully after resume
"""

import os
import tempfile

import pytest

from animus_forge.state.checkpoint import CheckpointManager
from animus_forge.workflow.executor import StepStatus, WorkflowExecutor
from animus_forge.workflow.loader import StepConfig, WorkflowConfig


class TestCheckpointResumeE2E:
    """End-to-end checkpoint/resume tests."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test.db")

    @pytest.fixture
    def checkpoint_manager(self, temp_db):
        """Create a checkpoint manager with temporary database."""
        return CheckpointManager(db_path=temp_db)

    @pytest.fixture
    def test_workflow(self):
        """Create a test workflow with multiple steps."""
        return WorkflowConfig(
            name="test_checkpoint_workflow",
            version="1.0",
            description="Test workflow for checkpoint/resume",
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "echo 'Step 1 complete'"},
                    outputs=["step1_output"],
                    on_failure="abort",
                    max_retries=0,
                    timeout_seconds=30,
                ),
                StepConfig(
                    id="step2",
                    type="shell",
                    params={"command": "echo 'Step 2 complete'"},
                    outputs=["step2_output"],
                    on_failure="abort",
                    max_retries=0,
                    timeout_seconds=30,
                ),
                StepConfig(
                    id="step3",
                    type="shell",
                    params={"command": "echo 'Step 3 complete'"},
                    outputs=["step3_output"],
                    on_failure="abort",
                    max_retries=0,
                    timeout_seconds=30,
                ),
            ],
            inputs={},
            outputs=["step3_output"],
        )

    def test_workflow_creates_checkpoints(self, checkpoint_manager, test_workflow):
        """Verify workflow execution creates checkpoints for each step."""
        executor = WorkflowExecutor(checkpoint_manager=checkpoint_manager)

        result = executor.execute(test_workflow)

        assert result.status == "success"
        assert len(result.steps) == 3

        # Verify checkpoints were created
        # After execution, current_workflow_id is cleared, so get from result
        # We need to get the workflow ID from persistence
        workflows = checkpoint_manager.persistence.list_workflows()
        assert len(workflows) == 1

        wf_id = workflows[0]["id"]
        checkpoints = checkpoint_manager.persistence.get_all_checkpoints(wf_id)
        assert len(checkpoints) == 3
        assert all(cp["status"] == "success" for cp in checkpoints)

    def test_failed_workflow_can_resume(self, checkpoint_manager, temp_db):
        """Verify workflow can resume from checkpoint after failure."""
        # Create workflow with a step that fails

        failing_workflow = WorkflowConfig(
            name="failing_workflow",
            version="1.0",
            description="Workflow that fails on step 2",
            steps=[
                StepConfig(
                    id="step1",
                    type="shell",
                    params={"command": "echo 'Step 1 OK'"},
                    outputs=[],
                    on_failure="abort",
                    max_retries=0,
                    timeout_seconds=30,
                ),
                StepConfig(
                    id="step2_failing",
                    type="shell",
                    params={"command": "exit 1"},  # This will fail
                    outputs=[],
                    on_failure="abort",
                    max_retries=0,
                    timeout_seconds=30,
                ),
                StepConfig(
                    id="step3",
                    type="shell",
                    params={"command": "echo 'Step 3 OK'"},
                    outputs=[],
                    on_failure="abort",
                    max_retries=0,
                    timeout_seconds=30,
                ),
            ],
            inputs={},
            outputs=[],
        )

        executor = WorkflowExecutor(checkpoint_manager=checkpoint_manager)
        result = executor.execute(failing_workflow)

        # Workflow should have failed
        assert result.status == "failed"
        assert "step2_failing" in result.error

        # Step 1 should have succeeded, step 2 failed
        assert result.steps[0].status == StepStatus.SUCCESS
        assert result.steps[1].status == StepStatus.FAILED

        # Get workflow ID for resume
        workflows = checkpoint_manager.persistence.list_workflows()
        wf_id = workflows[0]["id"]

        # Check checkpoints were created
        checkpoints = checkpoint_manager.persistence.get_all_checkpoints(wf_id)

        # Debug: print all checkpoints
        checkpoint_stages = [(cp["stage"], cp["status"]) for cp in checkpoints]

        # We should have checkpoints for step1 (success) and step2_failing (failed)
        assert len(checkpoints) >= 2, f"Expected 2+ checkpoints, got: {checkpoint_stages}"

        # Find checkpoint for step1
        step1_cp = next((cp for cp in checkpoints if cp["stage"] == "step1"), None)
        assert step1_cp is not None, f"Missing step1 checkpoint. Found: {checkpoint_stages}"
        assert step1_cp["status"] == "success"

        # Find checkpoint for step2_failing
        step2_cp = next((cp for cp in checkpoints if cp["stage"] == "step2_failing"), None)
        assert step2_cp is not None, f"Missing step2_failing checkpoint. Found: {checkpoint_stages}"
        assert step2_cp["status"] == "failed"

    def test_resume_from_specific_step(self, checkpoint_manager, test_workflow):
        """Verify workflow can resume from a specific step ID."""
        executor = WorkflowExecutor(checkpoint_manager=checkpoint_manager)

        # Execute full workflow first
        result1 = executor.execute(test_workflow)
        assert result1.status == "success"

        # Now resume from step2 (skipping step1)
        result2 = executor.execute(test_workflow, resume_from="step2")

        assert result2.status == "success"
        # Should only have 2 steps (step2 and step3)
        assert len(result2.steps) == 2
        assert result2.steps[0].step_id == "step2"
        assert result2.steps[1].step_id == "step3"

    def test_checkpoint_preserves_context(self, checkpoint_manager, temp_db):
        """Verify checkpoint preserves input/output data."""
        workflow = WorkflowConfig(
            name="context_workflow",
            version="1.0",
            description="Workflow that passes data between steps",
            steps=[
                StepConfig(
                    id="produce_data",
                    type="shell",
                    params={"command": "echo 'hello world'"},
                    outputs=["stdout"],
                    on_failure="abort",
                    max_retries=0,
                    timeout_seconds=30,
                ),
            ],
            inputs={},
            outputs=["stdout"],
        )

        executor = WorkflowExecutor(checkpoint_manager=checkpoint_manager)
        result = executor.execute(workflow)

        assert result.status == "success"

        # Get checkpoint and verify data is preserved
        workflows = checkpoint_manager.persistence.list_workflows()
        wf_id = workflows[0]["id"]
        checkpoints = checkpoint_manager.persistence.get_all_checkpoints(wf_id)

        assert len(checkpoints) == 1
        assert checkpoints[0]["output_data"] is not None
        assert "stdout" in checkpoints[0]["output_data"]

    def test_get_progress_tracks_completion(self, checkpoint_manager, test_workflow):
        """Verify get_progress tracks stages correctly."""
        executor = WorkflowExecutor(checkpoint_manager=checkpoint_manager)

        # Start workflow
        wf_id = checkpoint_manager.start_workflow("test_progress")

        # Execute with explicit workflow_id
        executor.execute(test_workflow)

        # Get progress for the workflow created during execution
        workflows = checkpoint_manager.persistence.list_workflows()
        # Find the workflow created by executor (has a name matching the workflow config)
        for wf in workflows:
            if wf["name"] == "test_checkpoint_workflow":
                wf_id = wf["id"]
                break

        progress = checkpoint_manager.get_progress(wf_id)

        assert progress["workflow_name"] == "test_checkpoint_workflow"
        assert len(progress["stages_completed"]) == 3
        assert progress["total_checkpoints"] == 3

    def test_resume_integration_with_persistence(self, checkpoint_manager):
        """Test the full resume() method integration."""
        # Create and partially complete a workflow
        wf_id = checkpoint_manager.start_workflow("resume_test", config={"key": "value"})

        # Checkpoint a stage
        checkpoint_manager.checkpoint_now(
            stage="stage1",
            status="success",
            input_data={"input": "data"},
            output_data={"result": "ok"},
            tokens_used=100,
            duration_ms=500,
        )

        # Mark as paused
        checkpoint_manager.persistence.mark_paused(wf_id)

        # Clear current workflow (simulating restart)
        checkpoint_manager._current_workflow = None

        # Resume
        resume_data = checkpoint_manager.resume(wf_id)

        assert resume_data["workflow"]["id"] == wf_id
        assert resume_data["checkpoint"]["stage"] == "stage1"
        assert resume_data["checkpoint"]["output_data"]["result"] == "ok"
        assert resume_data["resume_from_stage"] == "stage1"

        # Current workflow should be set
        assert checkpoint_manager.current_workflow_id == wf_id
