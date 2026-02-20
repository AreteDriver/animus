"""Tests for JobManager database operations."""

import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from animus_forge.jobs.job_manager import Job, JobManager, JobStatus
from animus_forge.state.backends import SQLiteBackend


class TestJobManager:
    """Tests for JobManager class."""

    @pytest.fixture
    def backend(self):
        """Create a temporary SQLite backend."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            backend = SQLiteBackend(db_path=db_path)
            yield backend
            backend.close()
        finally:
            # Use ignore_errors to handle SQLite WAL/SHM files that may still be locked
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def manager(self, backend):
        """Create a JobManager with mocked workflow engine."""
        with patch("animus_forge.jobs.job_manager.get_database", return_value=backend):
            with patch("animus_forge.jobs.job_manager.WorkflowEngineAdapter") as mock_engine:
                # Mock workflow
                mock_workflow = MagicMock()
                mock_workflow.variables = {}
                mock_engine.return_value.load_workflow.return_value = mock_workflow

                # Mock execute_workflow to return a result with model_dump
                mock_result = MagicMock()
                mock_result.status = "completed"
                mock_result.errors = []
                mock_result.model_dump.return_value = {
                    "status": "completed",
                    "output": "test",
                }
                mock_engine.return_value.execute_workflow.return_value = mock_result

                manager = JobManager(backend=backend, max_workers=2)
                yield manager
                manager.shutdown(wait=True)

    def test_init_creates_schema(self, backend):
        """JobManager creates jobs table on init."""
        with patch("animus_forge.jobs.job_manager.WorkflowEngineAdapter"):
            JobManager(backend=backend, max_workers=1).shutdown(wait=False)

        # Verify table exists
        row = backend.fetchone("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
        assert row is not None

    def test_submit_creates_job(self, manager, backend):
        """submit() creates a job and persists to database."""
        job = manager.submit("test-workflow", {"key": "value"})

        assert job.id is not None
        assert job.workflow_id == "test-workflow"
        # Job may be pending, running, or completed depending on timing
        assert job.status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.COMPLETED)
        assert job.variables == {"key": "value"}

        # Verify in database
        row = backend.fetchone("SELECT * FROM jobs WHERE id = ?", (job.id,))
        assert row is not None
        assert row["workflow_id"] == "test-workflow"
        # Status may have changed by the time we query
        assert row["status"] in ("pending", "running", "completed")

    def test_submit_validates_workflow(self, backend):
        """submit() raises if workflow doesn't exist."""
        with patch("animus_forge.jobs.job_manager.WorkflowEngineAdapter") as mock_engine:
            mock_engine.return_value.load_workflow.return_value = None
            manager = JobManager(backend=backend, max_workers=1)

            with pytest.raises(ValueError) as exc:
                manager.submit("nonexistent-workflow")

            assert "not found" in str(exc.value)
            manager.shutdown(wait=False)

    def test_get_job_returns_job(self, manager):
        """get_job() returns the job by ID."""
        job = manager.submit("test-workflow")

        retrieved = manager.get_job(job.id)
        assert retrieved is not None
        assert retrieved.id == job.id
        assert retrieved.workflow_id == job.workflow_id

    def test_get_job_returns_none_for_missing(self, manager):
        """get_job() returns None for nonexistent ID."""
        result = manager.get_job("nonexistent-id")
        assert result is None

    def test_list_jobs_returns_all(self, manager):
        """list_jobs() returns all jobs."""
        manager.submit("workflow-1")
        manager.submit("workflow-2")
        manager.submit("workflow-3")

        jobs = manager.list_jobs()
        assert len(jobs) == 3

    def test_list_jobs_filters_by_status(self, manager):
        """list_jobs() filters by status."""
        job1 = manager.submit("workflow-1")
        job2 = manager.submit("workflow-2")

        # Wait for jobs to complete
        import time

        time.sleep(0.2)

        # Manually update status for testing
        job1.status = JobStatus.COMPLETED
        manager._save_job(job1)
        job2.status = JobStatus.FAILED
        manager._save_job(job2)

        completed_jobs = manager.list_jobs(status=JobStatus.COMPLETED)
        assert len(completed_jobs) == 1
        assert completed_jobs[0].id == job1.id

        failed_jobs = manager.list_jobs(status=JobStatus.FAILED)
        assert len(failed_jobs) == 1
        assert failed_jobs[0].id == job2.id

    def test_list_jobs_filters_by_workflow(self, manager):
        """list_jobs() filters by workflow_id."""
        manager.submit("workflow-a")
        manager.submit("workflow-a")
        manager.submit("workflow-b")

        jobs = manager.list_jobs(workflow_id="workflow-a")
        assert len(jobs) == 2
        assert all(j.workflow_id == "workflow-a" for j in jobs)

    def test_list_jobs_respects_limit(self, manager):
        """list_jobs() respects limit parameter."""
        for i in range(10):
            manager.submit(f"workflow-{i}")

        jobs = manager.list_jobs(limit=5)
        assert len(jobs) == 5

    def test_cancel_pending_job(self, manager):
        """cancel() cancels a pending job."""
        job = manager.submit("test-workflow")

        result = manager.cancel(job.id)
        assert result is True

        updated = manager.get_job(job.id)
        assert updated.status == JobStatus.CANCELLED
        assert updated.error == "Cancelled by user"

    def test_cancel_nonexistent_job(self, manager):
        """cancel() returns False for nonexistent job."""
        result = manager.cancel("nonexistent-id")
        assert result is False

    def test_cancel_completed_job_fails(self, manager):
        """cancel() returns False for completed job."""
        job = manager.submit("test-workflow")
        job.status = JobStatus.COMPLETED
        manager._save_job(job)

        result = manager.cancel(job.id)
        assert result is False

    def test_delete_job(self, manager, backend):
        """delete_job() removes job from memory and database."""
        job = manager.submit("test-workflow")
        job.status = JobStatus.COMPLETED
        manager._save_job(job)

        result = manager.delete_job(job.id)
        assert result is True

        # Verify removed from memory
        assert manager.get_job(job.id) is None

        # Verify removed from database
        row = backend.fetchone("SELECT * FROM jobs WHERE id = ?", (job.id,))
        assert row is None

    def test_delete_running_job_fails(self, manager):
        """delete_job() returns False for running job."""
        # Create job directly with RUNNING status to avoid race condition
        # with the async executor that would complete it immediately
        job = Job(workflow_id="test-workflow", status=JobStatus.RUNNING)
        manager._jobs[job.id] = job
        manager._save_job(job)

        result = manager.delete_job(job.id)
        assert result is False

    def test_cleanup_old_jobs(self, manager, backend):
        """cleanup_old_jobs() removes old completed jobs."""
        import time

        # Create old completed job
        old_job = manager.submit("workflow-1")
        # Create recent completed job
        recent_job = manager.submit("workflow-2")

        # Wait for jobs to complete
        time.sleep(0.3)

        # Now set the completed_at times
        old_job.status = JobStatus.COMPLETED
        old_job.completed_at = datetime.now() - timedelta(hours=48)
        manager._save_job(old_job)

        recent_job.status = JobStatus.COMPLETED
        recent_job.completed_at = datetime.now()
        manager._save_job(recent_job)

        deleted = manager.cleanup_old_jobs(max_age_hours=24)
        assert deleted == 1

        # Old job should be gone
        assert manager.get_job(old_job.id) is None

        # Recent job should remain
        assert manager.get_job(recent_job.id) is not None

    def test_get_stats(self, manager):
        """get_stats() returns correct counts."""
        job1 = manager.submit("workflow-1")
        job2 = manager.submit("workflow-2")
        job3 = manager.submit("workflow-3")

        # Wait for jobs to complete
        import time

        time.sleep(0.2)

        # Set specific statuses for testing
        job1.status = JobStatus.COMPLETED
        manager._save_job(job1)
        job2.status = JobStatus.FAILED
        manager._save_job(job2)
        job3.status = JobStatus.PENDING
        manager._save_job(job3)

        stats = manager.get_stats()
        assert stats["total"] == 3
        assert stats["pending"] == 1
        assert stats["completed"] == 1
        assert stats["failed"] == 1

    def test_job_persists_across_restart(self, backend):
        """Jobs persist across manager restart."""
        with patch("animus_forge.jobs.job_manager.WorkflowEngineAdapter") as mock_engine:
            mock_engine.return_value.load_workflow.return_value = MagicMock()

            # Create job with first manager
            manager1 = JobManager(backend=backend, max_workers=1)
            job = manager1.submit("test-workflow", {"foo": "bar"})
            job_id = job.id
            manager1.shutdown(wait=False)

            # Verify with second manager
            manager2 = JobManager(backend=backend, max_workers=1)
            loaded_job = manager2.get_job(job_id)

            assert loaded_job is not None
            assert loaded_job.workflow_id == "test-workflow"
            assert loaded_job.variables == {"foo": "bar"}
            manager2.shutdown(wait=False)

    def test_running_jobs_marked_failed_on_restart(self, backend):
        """Running jobs are marked failed when manager restarts."""
        with patch("animus_forge.jobs.job_manager.WorkflowEngineAdapter") as mock_engine:
            mock_engine.return_value.load_workflow.return_value = MagicMock()

            # Create running job with first manager
            manager1 = JobManager(backend=backend, max_workers=1)
            job = manager1.submit("test-workflow")
            job.status = JobStatus.RUNNING
            manager1._save_job(job)
            job_id = job.id
            manager1.shutdown(wait=False)

            # Restart manager - running job should be marked failed
            manager2 = JobManager(backend=backend, max_workers=1)
            loaded_job = manager2.get_job(job_id)

            assert loaded_job.status == JobStatus.FAILED
            assert "Server restarted" in loaded_job.error
            manager2.shutdown(wait=False)
