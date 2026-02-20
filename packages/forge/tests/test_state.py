"""Tests for the state persistence module."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, "src")

from animus_forge.state import CheckpointManager, StatePersistence, WorkflowStatus


class TestStatePersistence:
    """Tests for StatePersistence class."""

    @pytest.fixture
    def persistence(self):
        """Create a temporary persistence instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            yield StatePersistence(db_path)

    def test_create_workflow(self, persistence):
        """Can create a workflow."""
        persistence.create_workflow("wf-1", "Test Workflow", {"key": "value"})
        workflow = persistence.get_workflow("wf-1")
        assert workflow is not None
        assert workflow["name"] == "Test Workflow"
        assert workflow["status"] == "pending"
        assert workflow["config"]["key"] == "value"

    def test_update_status(self, persistence):
        """Can update workflow status."""
        persistence.create_workflow("wf-1", "Test")
        persistence.update_status("wf-1", WorkflowStatus.RUNNING)
        workflow = persistence.get_workflow("wf-1")
        assert workflow["status"] == "running"

    def test_checkpoint(self, persistence):
        """Can create checkpoints."""
        persistence.create_workflow("wf-1", "Test")
        cp_id = persistence.checkpoint(
            workflow_id="wf-1",
            stage="planning",
            status="success",
            input_data={"task": "test"},
            output_data={"plan": "..."},
            tokens_used=100,
            duration_ms=500,
        )
        assert cp_id > 0

        checkpoint = persistence.get_last_checkpoint("wf-1")
        assert checkpoint["stage"] == "planning"
        assert checkpoint["tokens_used"] == 100

    def test_get_all_checkpoints(self, persistence):
        """Can get all checkpoints for a workflow."""
        persistence.create_workflow("wf-1", "Test")
        persistence.checkpoint("wf-1", "step1", "success")
        persistence.checkpoint("wf-1", "step2", "success")
        persistence.checkpoint("wf-1", "step3", "failed")

        checkpoints = persistence.get_all_checkpoints("wf-1")
        assert len(checkpoints) == 3
        stages = [cp["stage"] for cp in checkpoints]
        assert "step1" in stages
        assert "step2" in stages
        assert "step3" in stages

    def test_resumable_workflows(self, persistence):
        """Can find resumable workflows."""
        persistence.create_workflow("wf-1", "Running")
        persistence.update_status("wf-1", WorkflowStatus.RUNNING)

        persistence.create_workflow("wf-2", "Completed")
        persistence.update_status("wf-2", WorkflowStatus.COMPLETED)

        persistence.create_workflow("wf-3", "Failed")
        persistence.update_status("wf-3", WorkflowStatus.FAILED)

        resumable = persistence.get_resumable_workflows()
        ids = [w["id"] for w in resumable]
        assert "wf-1" in ids
        assert "wf-3" in ids
        assert "wf-2" not in ids

    def test_resume_from_checkpoint(self, persistence):
        """Can resume from checkpoint."""
        persistence.create_workflow("wf-1", "Test")
        persistence.checkpoint("wf-1", "step1", "success", output_data={"result": "ok"})
        persistence.update_status("wf-1", WorkflowStatus.PAUSED)

        checkpoint = persistence.resume_from_checkpoint("wf-1")
        assert checkpoint["stage"] == "step1"

        workflow = persistence.get_workflow("wf-1")
        assert workflow["status"] == "running"

    def test_mark_complete(self, persistence):
        """Can mark workflow complete."""
        persistence.create_workflow("wf-1", "Test")
        persistence.mark_complete("wf-1")
        workflow = persistence.get_workflow("wf-1")
        assert workflow["status"] == "completed"

    def test_mark_failed(self, persistence):
        """Can mark workflow failed with error."""
        persistence.create_workflow("wf-1", "Test")
        persistence.mark_failed("wf-1", "Something went wrong")
        workflow = persistence.get_workflow("wf-1")
        assert workflow["status"] == "failed"
        assert workflow["error"] == "Something went wrong"

    def test_delete_workflow(self, persistence):
        """Can delete workflow and checkpoints."""
        persistence.create_workflow("wf-1", "Test")
        persistence.checkpoint("wf-1", "step1", "success")

        result = persistence.delete_workflow("wf-1")
        assert result is True

        workflow = persistence.get_workflow("wf-1")
        assert workflow is None

    def test_get_stats(self, persistence):
        """Can get persistence statistics."""
        persistence.create_workflow("wf-1", "Test")
        persistence.checkpoint("wf-1", "step1", "success")
        # Set status after checkpoint since checkpoint sets to RUNNING
        persistence.update_status("wf-1", WorkflowStatus.COMPLETED)

        stats = persistence.get_stats()
        assert stats["total_workflows"] == 1
        assert stats["by_status"]["completed"] == 1
        assert stats["total_checkpoints"] == 1

    # -- additional coverage --

    def test_get_workflow_not_found(self, persistence):
        """get_workflow returns None for nonexistent workflow."""
        assert persistence.get_workflow("nonexistent") is None

    def test_get_last_checkpoint_no_checkpoints(self, persistence):
        """get_last_checkpoint returns None when no checkpoints exist."""
        persistence.create_workflow("wf-1", "Test")
        assert persistence.get_last_checkpoint("wf-1") is None

    def test_resume_from_checkpoint_no_checkpoint_raises(self, persistence):
        """resume_from_checkpoint raises ValueError when no checkpoint exists."""
        persistence.create_workflow("wf-1", "Test")
        with pytest.raises(ValueError, match="No checkpoint found"):
            persistence.resume_from_checkpoint("wf-1")

    def test_mark_paused(self, persistence):
        """Can mark workflow as paused."""
        persistence.create_workflow("wf-1", "Test")
        persistence.update_status("wf-1", WorkflowStatus.RUNNING)
        persistence.mark_paused("wf-1")
        workflow = persistence.get_workflow("wf-1")
        assert workflow["status"] == "paused"

    def test_delete_workflow_not_found(self, persistence):
        """Deleting nonexistent workflow returns False."""
        result = persistence.delete_workflow("nonexistent")
        assert result is False

    def test_create_workflow_no_config(self, persistence):
        """Can create workflow without config."""
        persistence.create_workflow("wf-1", "Test")
        workflow = persistence.get_workflow("wf-1")
        assert workflow["config"] is None

    def test_checkpoint_without_data(self, persistence):
        """Checkpoint works with no input/output data."""
        persistence.create_workflow("wf-1", "Test")
        cp_id = persistence.checkpoint("wf-1", "step1", "success")
        assert cp_id > 0
        checkpoint = persistence.get_last_checkpoint("wf-1")
        assert checkpoint["input_data"] is None
        assert checkpoint["output_data"] is None

    def test_checkpoint_updates_current_stage(self, persistence):
        """Checkpoint updates the workflow's current_stage."""
        persistence.create_workflow("wf-1", "Test")
        persistence.checkpoint("wf-1", "planning", "success")
        workflow = persistence.get_workflow("wf-1")
        assert workflow["current_stage"] == "planning"
        assert workflow["status"] == "running"

    def test_list_workflows_all(self, persistence):
        """list_workflows returns all workflows."""
        persistence.create_workflow("wf-1", "Alpha")
        persistence.create_workflow("wf-2", "Beta")
        persistence.create_workflow("wf-3", "Gamma")
        workflows = persistence.list_workflows()
        assert len(workflows) == 3

    def test_list_workflows_with_status_filter(self, persistence):
        """list_workflows filters by status."""
        persistence.create_workflow("wf-1", "Running")
        persistence.update_status("wf-1", WorkflowStatus.RUNNING)
        persistence.create_workflow("wf-2", "Done")
        persistence.update_status("wf-2", WorkflowStatus.COMPLETED)
        persistence.create_workflow("wf-3", "Also Running")
        persistence.update_status("wf-3", WorkflowStatus.RUNNING)

        running = persistence.list_workflows(status=WorkflowStatus.RUNNING)
        assert len(running) == 2
        completed = persistence.list_workflows(status=WorkflowStatus.COMPLETED)
        assert len(completed) == 1

    def test_list_workflows_with_limit(self, persistence):
        """list_workflows respects limit."""
        for i in range(5):
            persistence.create_workflow(f"wf-{i}", f"Workflow {i}")
        workflows = persistence.list_workflows(limit=2)
        assert len(workflows) == 2

    def test_get_all_checkpoints_empty(self, persistence):
        """get_all_checkpoints returns empty for workflow with no checkpoints."""
        persistence.create_workflow("wf-1", "Test")
        assert persistence.get_all_checkpoints("wf-1") == []

    def test_update_status_with_error(self, persistence):
        """update_status stores error message."""
        persistence.create_workflow("wf-1", "Test")
        persistence.update_status("wf-1", WorkflowStatus.FAILED, error="Timeout")
        workflow = persistence.get_workflow("wf-1")
        assert workflow["error"] == "Timeout"

    def test_get_stats_empty_db(self, persistence):
        """get_stats works on empty database."""
        stats = persistence.get_stats()
        assert stats["total_workflows"] == 0
        assert stats["total_checkpoints"] == 0
        for status in WorkflowStatus:
            assert stats["by_status"][status.value] == 0

    def test_get_stats_multiple_statuses(self, persistence):
        """get_stats counts multiple statuses correctly."""
        persistence.create_workflow("wf-1", "A")
        persistence.update_status("wf-1", WorkflowStatus.RUNNING)
        persistence.create_workflow("wf-2", "B")
        persistence.update_status("wf-2", WorkflowStatus.RUNNING)
        persistence.create_workflow("wf-3", "C")
        persistence.update_status("wf-3", WorkflowStatus.COMPLETED)

        stats = persistence.get_stats()
        assert stats["total_workflows"] == 3
        assert stats["by_status"]["running"] == 2
        assert stats["by_status"]["completed"] == 1

    def test_resumable_workflows_includes_paused(self, persistence):
        """Paused workflows are included in resumable list."""
        persistence.create_workflow("wf-1", "Paused")
        persistence.update_status("wf-1", WorkflowStatus.PAUSED)
        resumable = persistence.get_resumable_workflows()
        ids = [w["id"] for w in resumable]
        assert "wf-1" in ids

    def test_init_with_backend(self, tmp_path):
        """StatePersistence can be initialized with explicit backend."""
        from animus_forge.state.backends import SQLiteBackend

        db_path = str(tmp_path / "test.db")
        backend = SQLiteBackend(db_path=db_path)
        persistence = StatePersistence(backend=backend)
        persistence.create_workflow("wf-1", "Test")
        assert persistence.get_workflow("wf-1") is not None


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def manager(self):
        """Create a temporary checkpoint manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            yield CheckpointManager(db_path=db_path)

    def test_start_workflow(self, manager):
        """Can start a workflow."""
        wf_id = manager.start_workflow("Test Workflow", {"config": "value"})
        assert wf_id.startswith("wf-")
        assert manager.current_workflow_id == wf_id

    def test_complete_workflow(self, manager):
        """Can complete a workflow."""
        wf_id = manager.start_workflow("Test")
        manager.complete_workflow()
        assert manager.current_workflow_id is None

        workflow = manager.persistence.get_workflow(wf_id)
        assert workflow["status"] == "completed"

    def test_fail_workflow(self, manager):
        """Can fail a workflow."""
        wf_id = manager.start_workflow("Test")
        manager.fail_workflow("Error occurred")
        assert manager.current_workflow_id is None

        workflow = manager.persistence.get_workflow(wf_id)
        assert workflow["status"] == "failed"
        assert workflow["error"] == "Error occurred"

    def test_workflow_context_manager(self, manager):
        """Workflow context manager handles success."""
        with manager.workflow("Test Workflow") as wf_id:
            assert wf_id is not None
            assert manager.current_workflow_id == wf_id

        workflow = manager.persistence.get_workflow(wf_id)
        assert workflow["status"] == "completed"

    def test_workflow_context_manager_failure(self, manager):
        """Workflow context manager handles failure."""
        with pytest.raises(ValueError):
            with manager.workflow("Test") as wf_id:
                raise ValueError("Test error")

        workflow = manager.persistence.get_workflow(wf_id)
        assert workflow["status"] == "failed"

    def test_stage_context_manager(self, manager):
        """Stage context manager creates checkpoints."""
        manager.start_workflow("Test")

        with manager.stage("planning", {"task": "test"}) as ctx:
            ctx.output_data = {"plan": "..."}
            ctx.tokens_used = 100

        checkpoints = manager.persistence.get_all_checkpoints(manager.current_workflow_id)
        assert len(checkpoints) == 1
        assert checkpoints[0]["stage"] == "planning"
        assert checkpoints[0]["tokens_used"] == 100

    def test_stage_without_workflow_raises(self, manager):
        """Stage without active workflow raises."""
        with pytest.raises(ValueError):
            with manager.stage("test"):
                pass

    def test_checkpoint_now(self, manager):
        """Can create immediate checkpoint."""
        manager.start_workflow("Test")
        cp_id = manager.checkpoint_now(
            stage="manual",
            status="success",
            tokens_used=50,
        )
        assert cp_id > 0

    def test_resume(self, manager):
        """Can resume a workflow."""
        wf_id = manager.start_workflow("Test")
        with manager.stage("step1") as ctx:
            ctx.output_data = {"result": "step1"}

        manager.persistence.update_status(wf_id, WorkflowStatus.PAUSED)
        manager._current_workflow = None

        resume_data = manager.resume(wf_id)
        assert resume_data["resume_from_stage"] == "step1"
        assert manager.current_workflow_id == wf_id

    def test_get_progress(self, manager):
        """Can get workflow progress."""
        manager.start_workflow("Test")

        with manager.stage("step1") as ctx:
            ctx.tokens_used = 50

        with manager.stage("step2") as ctx:
            ctx.tokens_used = 75

        progress = manager.get_progress()
        assert len(progress["stages_completed"]) == 2
        assert progress["total_tokens"] == 125
        assert progress["total_checkpoints"] == 2

    # -- additional coverage --

    def test_start_workflow_custom_id(self, manager):
        """Can start workflow with custom id."""
        wf_id = manager.start_workflow("Test", workflow_id="custom-id")
        assert wf_id == "custom-id"
        assert manager.current_workflow_id == "custom-id"

    def test_complete_workflow_explicit_id(self, manager):
        """Can complete workflow with explicit id."""
        wf_id = manager.start_workflow("Test")
        manager.complete_workflow(workflow_id=wf_id)
        assert manager.current_workflow_id is None
        workflow = manager.persistence.get_workflow(wf_id)
        assert workflow["status"] == "completed"

    def test_complete_workflow_different_id(self, manager):
        """Completing different workflow doesn't clear current."""
        _ = manager.start_workflow("WF1", workflow_id="wf1")
        _ = manager.start_workflow("WF2", workflow_id="wf2")
        manager.complete_workflow(workflow_id="wf1")
        # Current should still be wf2
        assert manager.current_workflow_id == "wf2"

    def test_complete_workflow_no_workflow(self, manager):
        """Completing without any workflow is a no-op."""
        manager.complete_workflow()  # Should not raise

    def test_fail_workflow_explicit_id(self, manager):
        """Can fail workflow with explicit id."""
        wf_id = manager.start_workflow("Test")
        manager.fail_workflow("Error", workflow_id=wf_id)
        assert manager.current_workflow_id is None

    def test_fail_workflow_no_workflow(self, manager):
        """Failing without any workflow is a no-op."""
        manager.fail_workflow("Error")  # Should not raise

    def test_checkpoint_now_without_workflow_raises(self, manager):
        """checkpoint_now without active workflow raises ValueError."""
        with pytest.raises(ValueError, match="No active workflow"):
            manager.checkpoint_now(stage="test", status="success")

    def test_checkpoint_now_with_explicit_workflow_id(self, manager):
        """checkpoint_now works with explicit workflow_id."""
        wf_id = manager.start_workflow("Test")
        manager._current_workflow = None  # Clear current
        cp_id = manager.checkpoint_now(
            stage="manual",
            status="success",
            workflow_id=wf_id,
        )
        assert cp_id > 0

    def test_get_progress_without_workflow_raises(self, manager):
        """get_progress without workflow raises ValueError."""
        with pytest.raises(ValueError, match="No workflow specified"):
            manager.get_progress()

    def test_get_progress_explicit_workflow_id(self, manager):
        """get_progress works with explicit workflow_id."""
        wf_id = manager.start_workflow("Test")
        with manager.stage("step1") as ctx:
            ctx.tokens_used = 42
        manager._current_workflow = None

        progress = manager.get_progress(workflow_id=wf_id)
        assert progress["workflow_id"] == wf_id
        assert progress["total_tokens"] == 42

    def test_resume_nonexistent_workflow_raises(self, manager):
        """Resuming nonexistent workflow raises ValueError."""
        with pytest.raises(ValueError, match="Workflow not found"):
            manager.resume("nonexistent")

    def test_stage_failure_records_checkpoint(self, manager):
        """Stage failure still creates a checkpoint with 'failed' status."""
        wf_id = manager.start_workflow("Test")

        with pytest.raises(RuntimeError):
            with manager.stage("failing_step") as ctx:
                ctx.tokens_used = 10
                raise RuntimeError("stage exploded")

        # Checkpoint should be recorded with failed status
        checkpoints = manager.persistence.get_all_checkpoints(wf_id)
        assert len(checkpoints) == 1
        assert checkpoints[0]["status"] == "failed"
        assert checkpoints[0]["tokens_used"] == 10

    def test_stage_with_explicit_workflow_id(self, manager):
        """Stage can use explicit workflow_id."""
        wf_id = manager.start_workflow("Test")
        manager._current_workflow = None

        with manager.stage("step1", workflow_id=wf_id) as ctx:
            ctx.output_data = {"result": "ok"}

        checkpoints = manager.persistence.get_all_checkpoints(wf_id)
        assert len(checkpoints) == 1

    def test_current_stage_property(self, manager):
        """current_stage returns active stage context."""
        assert manager.current_stage is None
        manager.start_workflow("Test")
        with manager.stage("step1") as ctx:
            assert manager.current_stage is ctx
        assert manager.current_stage is None

    def test_get_progress_with_failed_stages(self, manager):
        """get_progress separates completed and failed stages."""
        _ = manager.start_workflow("Test")

        with manager.stage("step1") as ctx:
            ctx.tokens_used = 10

        with pytest.raises(RuntimeError):
            with manager.stage("step2") as ctx:
                raise RuntimeError("fail")

        progress = manager.get_progress()
        assert "step1" in progress["stages_completed"]
        assert "step2" in progress["stages_failed"]

    def test_workflow_context_with_config(self, manager):
        """Workflow context manager passes config."""
        config = {"model": "gpt-4", "max_tokens": 1000}
        with manager.workflow("Configured Workflow", config=config) as wf_id:
            workflow = manager.persistence.get_workflow(wf_id)
            assert workflow["config"] == config

    def test_init_with_persistence(self, tmp_path):
        """CheckpointManager can use pre-built persistence."""
        db_path = str(tmp_path / "test.db")
        persistence = StatePersistence(db_path)
        manager = CheckpointManager(persistence=persistence)
        wf_id = manager.start_workflow("Test")
        assert persistence.get_workflow(wf_id) is not None

    def test_init_with_backend(self, tmp_path):
        """CheckpointManager can use a database backend directly."""
        from animus_forge.state.backends import SQLiteBackend

        db_path = str(tmp_path / "test.db")
        backend = SQLiteBackend(db_path=db_path)
        manager = CheckpointManager(backend=backend)
        wf_id = manager.start_workflow("Test")
        assert manager.persistence.get_workflow(wf_id) is not None


class TestStageContext:
    """Tests for StageContext dataclass."""

    def test_default_values(self):
        """StageContext has sensible defaults."""
        from animus_forge.state.checkpoint import StageContext

        ctx = StageContext(
            workflow_id="wf-1",
            stage="test",
            started_at=1000.0,
        )
        assert ctx.input_data == {}
        assert ctx.output_data == {}
        assert ctx.tokens_used == 0
        assert ctx.status == "running"
        assert ctx.error is None


class TestWorkflowStatus:
    """Tests for WorkflowStatus enum."""

    def test_values(self):
        """All expected workflow statuses exist."""
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.PAUSED.value == "paused"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"

    def test_from_value(self):
        """Can create WorkflowStatus from string."""
        assert WorkflowStatus("running") == WorkflowStatus.RUNNING
