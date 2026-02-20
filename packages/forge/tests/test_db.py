"""Tests for TaskStore — task history, agent scores, and budget log."""

import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

sys.path.insert(0, "src")

from animus_forge.db import TaskStore, get_task_store, reset_task_store
from animus_forge.state.backends import SQLiteBackend


@pytest.fixture
def backend():
    """Create a temp SQLite backend with migration 010 applied."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "test.db")
        backend = SQLiteBackend(db_path=db_path)

        # Apply migration 010 schema
        migration_path = os.path.join(
            os.path.dirname(__file__), "..", "migrations", "010_task_history.sql"
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
    """Create a TaskStore with the test backend."""
    return TaskStore(backend)


# =============================================================================
# TestRecordTask
# =============================================================================


class TestRecordTask:
    """Tests for recording tasks to history."""

    def test_record_completed_task(self, store):
        """Completed task is inserted into task_history."""
        task_id = store.record_task(
            job_id="job-1",
            workflow_id="wf-1",
            status="completed",
            agent_role="builder",
            model="claude-3",
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            cost_usd=0.01,
            duration_ms=5000,
        )
        assert task_id >= 1

        task = store.get_task(task_id)
        assert task is not None
        assert task["job_id"] == "job-1"
        assert task["workflow_id"] == "wf-1"
        assert task["status"] == "completed"
        assert task["agent_role"] == "builder"
        assert task["model"] == "claude-3"
        assert task["total_tokens"] == 300
        assert task["cost_usd"] == 0.01
        assert task["duration_ms"] == 5000

    def test_record_failed_task(self, store):
        """Failed task records error message."""
        task_id = store.record_task(
            job_id="job-2",
            workflow_id="wf-1",
            status="failed",
            error="timeout exceeded",
        )
        task = store.get_task(task_id)
        assert task["status"] == "failed"
        assert task["error"] == "timeout exceeded"

    def test_record_cancelled_task(self, store):
        """Cancelled task is recorded normally."""
        task_id = store.record_task(
            job_id="job-3",
            workflow_id="wf-2",
            status="cancelled",
        )
        task = store.get_task(task_id)
        assert task["status"] == "cancelled"

    def test_record_without_agent_role(self, store):
        """Task without agent_role defaults to 'unknown' in aggregates."""
        store.record_task(
            job_id="job-4",
            workflow_id="wf-1",
            status="completed",
        )
        stats = store.get_agent_stats(agent_role="unknown")
        assert len(stats) == 1
        assert stats[0]["total_tasks"] == 1

    def test_record_with_metadata(self, store):
        """Metadata dict round-trips through JSON serialization."""
        meta = {"step": "build", "retries": 2, "tags": ["fast"]}
        task_id = store.record_task(
            job_id="job-5",
            workflow_id="wf-1",
            status="completed",
            metadata=meta,
        )
        task = store.get_task(task_id)
        assert task["metadata"] == meta

    def test_record_with_explicit_timestamps(self, store):
        """Explicit created_at/completed_at are persisted."""
        created = datetime(2025, 1, 1, 10, 0, 0)
        completed = datetime(2025, 1, 1, 10, 5, 0)
        task_id = store.record_task(
            job_id="job-6",
            workflow_id="wf-1",
            status="completed",
            created_at=created,
            completed_at=completed,
        )
        task = store.get_task(task_id)
        assert "2025-01-01" in task["created_at"]
        assert "2025-01-01" in task["completed_at"]

    def test_record_calculates_duration(self, store):
        """Duration is stored as provided by caller."""
        task_id = store.record_task(
            job_id="job-7",
            workflow_id="wf-1",
            status="completed",
            duration_ms=12345,
        )
        task = store.get_task(task_id)
        assert task["duration_ms"] == 12345

    def test_record_increments_ids(self, store):
        """Each record gets a unique incrementing id."""
        id1 = store.record_task(job_id="j1", workflow_id="w1", status="completed")
        id2 = store.record_task(job_id="j2", workflow_id="w1", status="completed")
        assert id2 > id1


# =============================================================================
# TestQueryTasks
# =============================================================================


class TestQueryTasks:
    """Tests for querying task history."""

    def _seed(self, store):
        """Insert a mix of tasks for query tests."""
        store.record_task(
            job_id="j1",
            workflow_id="wf-a",
            status="completed",
            agent_role="builder",
        )
        store.record_task(
            job_id="j2",
            workflow_id="wf-a",
            status="failed",
            agent_role="tester",
        )
        store.record_task(
            job_id="j3",
            workflow_id="wf-b",
            status="completed",
            agent_role="builder",
        )
        store.record_task(
            job_id="j4",
            workflow_id="wf-b",
            status="completed",
            agent_role="reviewer",
        )

    def test_query_all(self, store):
        """Query with no filters returns all tasks."""
        self._seed(store)
        tasks = store.query_tasks()
        assert len(tasks) == 4

    def test_query_filter_by_status(self, store):
        """Filter by status returns matching tasks."""
        self._seed(store)
        tasks = store.query_tasks(status="failed")
        assert len(tasks) == 1
        assert tasks[0]["status"] == "failed"

    def test_query_filter_by_agent_role(self, store):
        """Filter by agent_role returns matching tasks."""
        self._seed(store)
        tasks = store.query_tasks(agent_role="builder")
        assert len(tasks) == 2
        assert all(t["agent_role"] == "builder" for t in tasks)

    def test_query_filter_by_workflow_id(self, store):
        """Filter by workflow_id returns matching tasks."""
        self._seed(store)
        tasks = store.query_tasks(workflow_id="wf-b")
        assert len(tasks) == 2

    def test_query_limit_offset(self, store):
        """Limit and offset pagination works."""
        self._seed(store)
        page1 = store.query_tasks(limit=2, offset=0)
        page2 = store.query_tasks(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        # No overlap
        ids1 = {t["id"] for t in page1}
        ids2 = {t["id"] for t in page2}
        assert ids1.isdisjoint(ids2)

    def test_query_empty_results(self, store):
        """Query on empty DB returns empty list."""
        tasks = store.query_tasks(status="completed")
        assert tasks == []

    def test_query_ordering(self, store):
        """Results are ordered by completed_at descending."""
        self._seed(store)
        tasks = store.query_tasks()
        # Most recent first
        assert tasks[0]["job_id"] == "j4"


# =============================================================================
# TestAgentScores
# =============================================================================


class TestAgentScores:
    """Tests for agent performance aggregates."""

    def test_scores_created_on_first_record(self, store):
        """First task for an agent creates an agent_scores row."""
        store.record_task(
            job_id="j1",
            workflow_id="wf-1",
            status="completed",
            agent_role="builder",
            total_tokens=500,
            cost_usd=0.01,
            duration_ms=3000,
        )
        stats = store.get_agent_stats(agent_role="builder")
        assert len(stats) == 1
        assert stats[0]["total_tasks"] == 1
        assert stats[0]["successful_tasks"] == 1
        assert stats[0]["failed_tasks"] == 0
        assert stats[0]["success_rate"] == 100.0

    def test_multiple_tasks_aggregate(self, store):
        """Multiple tasks update running aggregates."""
        store.record_task(
            job_id="j1",
            workflow_id="wf-1",
            status="completed",
            agent_role="builder",
            total_tokens=500,
            cost_usd=0.01,
            duration_ms=2000,
        )
        store.record_task(
            job_id="j2",
            workflow_id="wf-1",
            status="completed",
            agent_role="builder",
            total_tokens=300,
            cost_usd=0.005,
            duration_ms=4000,
        )
        stats = store.get_agent_stats(agent_role="builder")
        assert stats[0]["total_tasks"] == 2
        assert stats[0]["total_tokens"] == 800
        assert stats[0]["avg_duration_ms"] == 3000.0

    def test_success_rate_calculation(self, store):
        """Success rate is calculated from completed vs total."""
        store.record_task(
            job_id="j1",
            workflow_id="wf-1",
            status="completed",
            agent_role="tester",
        )
        store.record_task(
            job_id="j2",
            workflow_id="wf-1",
            status="failed",
            agent_role="tester",
        )
        stats = store.get_agent_stats(agent_role="tester")
        assert stats[0]["success_rate"] == 50.0

    def test_get_all_agents(self, store):
        """get_agent_stats() without filter returns all agents."""
        store.record_task(
            job_id="j1",
            workflow_id="wf-1",
            status="completed",
            agent_role="builder",
        )
        store.record_task(
            job_id="j2",
            workflow_id="wf-1",
            status="completed",
            agent_role="tester",
        )
        stats = store.get_agent_stats()
        assert len(stats) == 2

    def test_get_nonexistent_agent(self, store):
        """Querying a non-existent agent returns empty list."""
        stats = store.get_agent_stats(agent_role="nonexistent")
        assert stats == []

    def test_failed_task_increments_failed(self, store):
        """Failed task increments failed_tasks counter."""
        store.record_task(
            job_id="j1",
            workflow_id="wf-1",
            status="failed",
            agent_role="reviewer",
        )
        stats = store.get_agent_stats(agent_role="reviewer")
        assert stats[0]["failed_tasks"] == 1
        assert stats[0]["successful_tasks"] == 0
        assert stats[0]["success_rate"] == 0.0


# =============================================================================
# TestBudgetLog
# =============================================================================


class TestBudgetLog:
    """Tests for daily budget rollups."""

    def test_log_created_on_record(self, store):
        """Recording a task creates a budget_log entry."""
        store.record_task(
            job_id="j1",
            workflow_id="wf-1",
            status="completed",
            agent_role="builder",
            total_tokens=500,
            cost_usd=0.01,
        )
        logs = store.get_daily_budget(days=1)
        assert len(logs) == 1
        assert logs[0]["total_tokens"] == 500
        assert logs[0]["task_count"] == 1

    def test_same_day_aggregates(self, store):
        """Multiple tasks on the same day aggregate in one row."""
        store.record_task(
            job_id="j1",
            workflow_id="wf-1",
            status="completed",
            agent_role="builder",
            total_tokens=500,
            cost_usd=0.01,
        )
        store.record_task(
            job_id="j2",
            workflow_id="wf-1",
            status="completed",
            agent_role="builder",
            total_tokens=300,
            cost_usd=0.005,
        )
        logs = store.get_daily_budget(days=1, agent_role="builder")
        assert len(logs) == 1
        assert logs[0]["total_tokens"] == 800
        assert logs[0]["task_count"] == 2

    def test_multi_day_query(self, store):
        """get_daily_budget returns only entries within the day window."""
        today = datetime.now()
        old = today - timedelta(days=30)
        store.record_task(
            job_id="j1",
            workflow_id="wf-1",
            status="completed",
            agent_role="builder",
            total_tokens=100,
            completed_at=old,
        )
        store.record_task(
            job_id="j2",
            workflow_id="wf-1",
            status="completed",
            agent_role="builder",
            total_tokens=200,
            completed_at=today,
        )
        logs = store.get_daily_budget(days=7)
        assert len(logs) == 1
        assert logs[0]["total_tokens"] == 200

    def test_filter_by_agent(self, store):
        """get_daily_budget filters by agent_role."""
        store.record_task(
            job_id="j1",
            workflow_id="wf-1",
            status="completed",
            agent_role="builder",
            total_tokens=100,
        )
        store.record_task(
            job_id="j2",
            workflow_id="wf-1",
            status="completed",
            agent_role="tester",
            total_tokens=200,
        )
        logs = store.get_daily_budget(days=1, agent_role="tester")
        assert len(logs) == 1
        assert logs[0]["agent_role"] == "tester"

    def test_empty_budget_log(self, store):
        """get_daily_budget on empty DB returns empty list."""
        logs = store.get_daily_budget(days=7)
        assert logs == []


# =============================================================================
# TestGetSummary
# =============================================================================


class TestGetSummary:
    """Tests for the summary aggregate."""

    def test_empty_db_returns_zeros(self, store):
        """Summary on empty DB returns zero values."""
        summary = store.get_summary()
        assert summary["total_tasks"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["total_cost_usd"] == 0.0
        assert summary["top_agents"] == []

    def test_summary_with_data(self, store):
        """Summary calculates totals from recorded tasks."""
        store.record_task(
            job_id="j1",
            workflow_id="wf-1",
            status="completed",
            agent_role="builder",
            total_tokens=500,
            cost_usd=0.01,
        )
        store.record_task(
            job_id="j2",
            workflow_id="wf-1",
            status="failed",
            agent_role="tester",
            total_tokens=300,
            cost_usd=0.005,
        )
        summary = store.get_summary()
        assert summary["total_tasks"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert summary["success_rate"] == 50.0
        assert summary["total_tokens"] == 800
        assert summary["total_cost_usd"] == 0.015
        assert len(summary["top_agents"]) == 2

    def test_summary_success_rate(self, store):
        """Success rate calculated correctly for all-success case."""
        for i in range(3):
            store.record_task(
                job_id=f"j{i}",
                workflow_id="wf-1",
                status="completed",
                agent_role="builder",
            )
        summary = store.get_summary()
        assert summary["success_rate"] == 100.0


# =============================================================================
# TestGetTaskStore
# =============================================================================


class TestGetTaskStore:
    """Tests for the singleton factory."""

    def test_singleton_behavior(self, backend):
        """get_task_store returns the same instance."""
        reset_task_store()
        with patch("animus_forge.state.database.get_database", return_value=backend):
            s1 = get_task_store()
            s2 = get_task_store()
            assert s1 is s2
        reset_task_store()

    def test_reset_clears_singleton(self, backend):
        """reset_task_store clears the cached instance."""
        reset_task_store()
        with patch("animus_forge.state.database.get_database", return_value=backend):
            s1 = get_task_store()
            reset_task_store()
            s2 = get_task_store()
            assert s1 is not s2
        reset_task_store()

    def test_uses_get_database(self, backend):
        """get_task_store calls get_database for the backend."""
        reset_task_store()
        with patch("animus_forge.state.database.get_database", return_value=backend) as mock_db:
            get_task_store()
            mock_db.assert_called_once()
        reset_task_store()
