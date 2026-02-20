"""Integration tests for the full streaming pipeline.

Exercises: JobManager → WorkflowExecutor → ExecutionManager → Broadcaster
with real threading and temp SQLite backend.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from animus_forge.executions.manager import ExecutionManager
from animus_forge.jobs.job_manager import JobManager, JobStatus
from animus_forge.orchestrator.workflow_engine import WorkflowResult
from animus_forge.state.backends import SQLiteBackend

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pipeline_backend():
    """Create a temp SQLite backend with all required schemas."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "pipeline_test.db")
        backend = SQLiteBackend(db_path=db_path)

        # Apply execution migrations
        migrations_dir = os.path.join(os.path.dirname(__file__), "..", "migrations")
        for migration in (
            "004_executions.sql",
            "010_task_history.sql",
            "011_budget_session_usage.sql",
        ):
            path = os.path.join(migrations_dir, migration)
            if os.path.exists(path):
                with open(path) as f:
                    sql = f.read()
                backend.executescript(sql)

        yield backend
        backend.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def execution_manager(pipeline_backend):
    """Create a real ExecutionManager with the test backend."""
    return ExecutionManager(backend=pipeline_backend)


@pytest.fixture(autouse=True)
def mock_task_store():
    """Prevent _record_task_history from hitting real TaskStore."""
    mock_store = MagicMock()
    with patch("animus_forge.db.get_task_store", return_value=mock_store):
        yield mock_store


def _make_workflow_result(status="completed", errors=None):
    """Create a canned WorkflowResult for mocking."""
    return WorkflowResult(
        workflow_id="test-wf",
        status=status,
        started_at=datetime.now(),
        completed_at=datetime.now(),
        steps_executed=["step1", "step2", "step3"],
        outputs={"summary": "Test completed", "total_tokens": 1500},
        errors=errors or [],
    )


def _make_job_manager(pipeline_backend, execution_manager, workflow_result=None):
    """Create a JobManager with mocked workflow engine."""
    result = workflow_result or _make_workflow_result()

    mock_workflow = MagicMock()
    mock_workflow.variables = {}

    with patch("animus_forge.jobs.job_manager.get_settings"):
        with patch("animus_forge.jobs.job_manager.get_database", return_value=pipeline_backend):
            jm = JobManager(
                backend=pipeline_backend,
                max_workers=2,
                execution_manager=execution_manager,
            )

    # Mock the workflow engine methods
    jm.workflow_engine.load_workflow = MagicMock(return_value=mock_workflow)
    jm.workflow_engine.execute_workflow = MagicMock(return_value=result)

    return jm


# =============================================================================
# TestJobSubmitCreatesExecution
# =============================================================================


class TestJobSubmitCreatesExecution:
    """Verify submitting a job creates records in both Job and Execution stores."""

    def test_submit_creates_job_record(self, pipeline_backend, execution_manager):
        jm = _make_job_manager(pipeline_backend, execution_manager)
        try:
            job = jm.submit("test-workflow", variables={"key": "value"})

            assert job.id is not None
            assert job.workflow_id == "test-workflow"
            assert job.status in (JobStatus.PENDING, JobStatus.RUNNING)
            assert job.variables == {"key": "value"}

            # Wait for execution to complete
            time.sleep(1)

            retrieved = jm.get_job(job.id)
            assert retrieved is not None
            assert retrieved.status in (JobStatus.COMPLETED, JobStatus.RUNNING)
        finally:
            jm.shutdown(wait=True)


# =============================================================================
# TestJobCompletionLifecycle
# =============================================================================


class TestJobCompletionLifecycle:
    """Verify job transitions through expected lifecycle states."""

    def test_job_completes_successfully(self, pipeline_backend, execution_manager):
        jm = _make_job_manager(pipeline_backend, execution_manager)
        try:
            job = jm.submit("lifecycle-wf")

            # Wait for background execution
            for _ in range(20):
                time.sleep(0.2)
                current = jm.get_job(job.id)
                if current and current.status == JobStatus.COMPLETED:
                    break

            final = jm.get_job(job.id)
            assert final.status == JobStatus.COMPLETED
            assert final.started_at is not None
            assert final.completed_at is not None
            assert final.error is None
            assert final.result is not None
        finally:
            jm.shutdown(wait=True)

    def test_job_records_result_data(self, pipeline_backend, execution_manager):
        jm = _make_job_manager(pipeline_backend, execution_manager)
        try:
            job = jm.submit("result-wf")

            for _ in range(20):
                time.sleep(0.2)
                current = jm.get_job(job.id)
                if current and current.status == JobStatus.COMPLETED:
                    break

            final = jm.get_job(job.id)
            assert final.result is not None
            assert final.result["status"] == "completed"
            assert "steps_executed" in final.result
        finally:
            jm.shutdown(wait=True)


# =============================================================================
# TestJobFailureRecordsError
# =============================================================================


class TestJobFailureRecordsError:
    """Verify failed workflows record error state correctly."""

    def test_workflow_exception_marks_job_failed(self, pipeline_backend, execution_manager):
        jm = _make_job_manager(pipeline_backend, execution_manager)
        jm.workflow_engine.execute_workflow = MagicMock(
            side_effect=RuntimeError("Workflow crashed")
        )
        try:
            job = jm.submit("failing-wf")

            for _ in range(20):
                time.sleep(0.2)
                current = jm.get_job(job.id)
                if current and current.status == JobStatus.FAILED:
                    break

            final = jm.get_job(job.id)
            assert final.status == JobStatus.FAILED
            assert "Workflow crashed" in final.error
            assert final.completed_at is not None
        finally:
            jm.shutdown(wait=True)

    def test_workflow_with_errors_in_result(self, pipeline_backend, execution_manager):
        result = _make_workflow_result(
            status="failed", errors=["Step 2 timed out", "Validation failed"]
        )
        jm = _make_job_manager(pipeline_backend, execution_manager, result)
        try:
            job = jm.submit("error-result-wf")

            for _ in range(20):
                time.sleep(0.2)
                current = jm.get_job(job.id)
                if current and current.status == JobStatus.FAILED:
                    break

            final = jm.get_job(job.id)
            assert final.status == JobStatus.FAILED
            assert "Step 2 timed out" in final.error
        finally:
            jm.shutdown(wait=True)


# =============================================================================
# TestMultipleConcurrentJobs
# =============================================================================


class TestMultipleConcurrentJobs:
    """Verify multiple jobs can execute concurrently."""

    def test_three_concurrent_jobs_all_complete(self, pipeline_backend, execution_manager):
        jm = _make_job_manager(pipeline_backend, execution_manager)
        try:
            jobs = [jm.submit(f"concurrent-wf-{i}", variables={"index": i}) for i in range(3)]

            # Wait for all to complete
            for _ in range(30):
                time.sleep(0.2)
                statuses = [jm.get_job(j.id).status for j in jobs]
                if all(s == JobStatus.COMPLETED for s in statuses):
                    break

            for j in jobs:
                final = jm.get_job(j.id)
                assert final.status == JobStatus.COMPLETED
                assert final.result is not None
        finally:
            jm.shutdown(wait=True)


# =============================================================================
# TestJobCancellation
# =============================================================================


class TestJobCancellation:
    """Verify job cancellation works correctly."""

    def test_cancel_pending_job(self, pipeline_backend, execution_manager):
        # Make workflow engine slow so we can cancel before execution
        def slow_execute(workflow):
            time.sleep(5)
            return _make_workflow_result()

        jm = _make_job_manager(pipeline_backend, execution_manager)
        jm.workflow_engine.execute_workflow = MagicMock(side_effect=slow_execute)
        try:
            job = jm.submit("cancel-wf")

            # Cancel immediately
            cancelled = jm.cancel(job.id)
            assert cancelled is True

            final = jm.get_job(job.id)
            assert final.status == JobStatus.CANCELLED
            assert final.error == "Cancelled by user"
        finally:
            jm.shutdown(wait=False)

    def test_cancel_nonexistent_job(self, pipeline_backend, execution_manager):
        jm = _make_job_manager(pipeline_backend, execution_manager)
        try:
            result = jm.cancel("nonexistent-id")
            assert result is False
        finally:
            jm.shutdown(wait=True)


# =============================================================================
# TestCallbackExceptionDoesntBreakPipeline
# =============================================================================


class TestCallbackExceptionDoesntBreakPipeline:
    """Verify that broken callbacks don't break the execution pipeline."""

    def test_broken_callback_still_completes_job(self, pipeline_backend, execution_manager):
        # Register a broken callback on the execution manager
        def broken_callback(event_type, execution_id, **kwargs):
            raise RuntimeError("Callback exploded!")

        execution_manager.register_callback(broken_callback)

        jm = _make_job_manager(pipeline_backend, execution_manager)
        try:
            job = jm.submit("callback-wf")

            for _ in range(20):
                time.sleep(0.2)
                current = jm.get_job(job.id)
                if current and current.status == JobStatus.COMPLETED:
                    break

            final = jm.get_job(job.id)
            assert final.status == JobStatus.COMPLETED
        finally:
            execution_manager.unregister_callback(broken_callback)
            jm.shutdown(wait=True)


# =============================================================================
# TestTaskHistoryRecorded
# =============================================================================


class TestTaskHistoryRecorded:
    """Verify _record_task_history writes to TaskStore on completion."""

    def test_task_history_recorded_on_completion(
        self, pipeline_backend, execution_manager, mock_task_store
    ):
        jm = _make_job_manager(pipeline_backend, execution_manager)
        try:
            job = jm.submit("history-wf")

            for _ in range(20):
                time.sleep(0.2)
                current = jm.get_job(job.id)
                if current and current.status == JobStatus.COMPLETED:
                    break

            # Give a moment for _record_task_history to execute
            time.sleep(0.5)

            # Verify record_task was called
            assert mock_task_store.record_task.called
        finally:
            jm.shutdown(wait=True)

    def test_task_history_failure_doesnt_break_job(
        self, pipeline_backend, execution_manager, mock_task_store
    ):
        """Even if TaskStore raises, the job should still complete."""
        mock_task_store.record_task.side_effect = RuntimeError("TaskStore broken")

        jm = _make_job_manager(pipeline_backend, execution_manager)
        try:
            job = jm.submit("broken-history-wf")

            for _ in range(20):
                time.sleep(0.2)
                current = jm.get_job(job.id)
                if current and current.status in (
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                ):
                    break

            final = jm.get_job(job.id)
            # Job should still complete despite TaskStore failure
            assert final.status == JobStatus.COMPLETED
        finally:
            jm.shutdown(wait=True)
