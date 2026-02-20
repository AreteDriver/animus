"""Tests for monitoring/metrics.py module."""

import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from animus_forge.monitoring.metrics import MetricsStore, StepMetrics, WorkflowMetrics

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reset_metrics_store():
    """Reset MetricsStore singleton before and after each test."""
    MetricsStore._instance = None
    yield
    MetricsStore._instance = None


@pytest.fixture
def step_metrics():
    """Create a sample StepMetrics."""
    return StepMetrics(
        step_id="step_1",
        step_type="openai",
        action="generate",
        started_at=datetime.now(UTC),
    )


@pytest.fixture
def workflow_metrics():
    """Create a sample WorkflowMetrics."""
    return WorkflowMetrics(
        workflow_id="wf_123",
        execution_id="exec_456",
        workflow_name="Test Workflow",
        started_at=datetime.now(UTC),
    )


@pytest.fixture
def metrics_store(reset_metrics_store):
    """Create a MetricsStore without persistence."""
    return MetricsStore()


@pytest.fixture
def metrics_store_with_db(reset_metrics_store, tmp_path):
    """Create a MetricsStore with SQLite persistence."""
    db_path = str(tmp_path / "test_metrics.db")
    return MetricsStore(db_path=db_path)


# =============================================================================
# StepMetrics Tests
# =============================================================================


class TestStepMetricsCreation:
    """Tests for StepMetrics creation."""

    def test_create_step_metrics(self):
        """Test creating StepMetrics with required fields."""
        now = datetime.now(UTC)
        step = StepMetrics(
            step_id="step_1",
            step_type="openai",
            action="generate",
            started_at=now,
        )
        assert step.step_id == "step_1"
        assert step.step_type == "openai"
        assert step.action == "generate"
        assert step.started_at == now

    def test_default_values(self):
        """Test default values for StepMetrics."""
        step = StepMetrics(
            step_id="step_1",
            step_type="openai",
            action="generate",
            started_at=datetime.now(UTC),
        )
        assert step.completed_at is None
        assert step.duration_ms == 0
        assert step.status == "running"
        assert step.error is None
        assert step.tokens_used == 0
        assert step.metadata == {}

    def test_create_with_all_fields(self):
        """Test creating StepMetrics with all fields."""
        now = datetime.now(UTC)
        step = StepMetrics(
            step_id="step_1",
            step_type="claude",
            action="analyze",
            started_at=now,
            completed_at=now + timedelta(seconds=5),
            duration_ms=5000,
            status="success",
            error=None,
            tokens_used=1500,
            metadata={"model": "claude-3"},
        )
        assert step.duration_ms == 5000
        assert step.status == "success"
        assert step.tokens_used == 1500
        assert step.metadata == {"model": "claude-3"}


class TestStepMetricsComplete:
    """Tests for StepMetrics.complete() method."""

    def test_complete_success(self, step_metrics):
        """Test completing step with success."""
        time.sleep(0.01)  # Small delay for duration calculation
        step_metrics.complete(status="success")

        assert step_metrics.completed_at is not None
        assert step_metrics.status == "success"
        assert step_metrics.duration_ms > 0
        assert step_metrics.error is None

    def test_complete_failed_with_error(self, step_metrics):
        """Test completing step with failure and error."""
        step_metrics.complete(status="failed", error="Connection timeout")

        assert step_metrics.completed_at is not None
        assert step_metrics.status == "failed"
        assert step_metrics.error == "Connection timeout"

    def test_complete_skipped(self, step_metrics):
        """Test completing step as skipped."""
        step_metrics.complete(status="skipped")

        assert step_metrics.status == "skipped"
        assert step_metrics.error is None

    def test_duration_calculation(self):
        """Test that duration is calculated correctly."""
        now = datetime.now(UTC)
        step = StepMetrics(
            step_id="step_1",
            step_type="openai",
            action="generate",
            started_at=now - timedelta(seconds=2),
        )
        step.complete()

        # Duration should be approximately 2000ms
        assert step.duration_ms >= 2000
        assert step.duration_ms < 3000  # Allow some tolerance


class TestStepMetricsToDict:
    """Tests for StepMetrics.to_dict() method."""

    def test_to_dict_running(self, step_metrics):
        """Test to_dict for running step."""
        result = step_metrics.to_dict()

        assert result["step_id"] == "step_1"
        assert result["step_type"] == "openai"
        assert result["action"] == "generate"
        assert result["status"] == "running"
        assert result["completed_at"] is None
        assert "started_at" in result

    def test_to_dict_completed(self, step_metrics):
        """Test to_dict for completed step."""
        step_metrics.complete(status="success")
        result = step_metrics.to_dict()

        assert result["status"] == "success"
        assert result["completed_at"] is not None
        assert result["duration_ms"] > 0

    def test_to_dict_with_error(self, step_metrics):
        """Test to_dict includes error when present."""
        step_metrics.complete(status="failed", error="API error")
        result = step_metrics.to_dict()

        assert result["error"] == "API error"


# =============================================================================
# WorkflowMetrics Tests
# =============================================================================


class TestWorkflowMetricsCreation:
    """Tests for WorkflowMetrics creation."""

    def test_create_workflow_metrics(self):
        """Test creating WorkflowMetrics with required fields."""
        now = datetime.now(UTC)
        wf = WorkflowMetrics(
            workflow_id="wf_123",
            execution_id="exec_456",
            workflow_name="Test Workflow",
            started_at=now,
        )
        assert wf.workflow_id == "wf_123"
        assert wf.execution_id == "exec_456"
        assert wf.workflow_name == "Test Workflow"
        assert wf.started_at == now

    def test_default_values(self):
        """Test default values for WorkflowMetrics."""
        wf = WorkflowMetrics(
            workflow_id="wf_123",
            execution_id="exec_456",
            workflow_name="Test",
            started_at=datetime.now(UTC),
        )
        assert wf.completed_at is None
        assert wf.duration_ms == 0
        assert wf.status == "running"
        assert wf.total_steps == 0
        assert wf.completed_steps == 0
        assert wf.failed_steps == 0
        assert wf.total_tokens == 0
        assert wf.steps == []
        assert wf.error is None


class TestWorkflowMetricsAddStep:
    """Tests for WorkflowMetrics.add_step() method."""

    def test_add_single_step(self, workflow_metrics, step_metrics):
        """Test adding a single step."""
        workflow_metrics.add_step(step_metrics)

        assert len(workflow_metrics.steps) == 1
        assert workflow_metrics.total_steps == 1
        assert workflow_metrics.steps[0] == step_metrics

    def test_add_multiple_steps(self, workflow_metrics):
        """Test adding multiple steps."""
        for i in range(5):
            step = StepMetrics(
                step_id=f"step_{i}",
                step_type="openai",
                action="generate",
                started_at=datetime.now(UTC),
            )
            workflow_metrics.add_step(step)

        assert len(workflow_metrics.steps) == 5
        assert workflow_metrics.total_steps == 5


class TestWorkflowMetricsUpdateStep:
    """Tests for WorkflowMetrics.update_step() method."""

    def test_update_step_success(self, workflow_metrics, step_metrics):
        """Test updating step with success."""
        step_metrics.tokens_used = 100
        workflow_metrics.add_step(step_metrics)
        workflow_metrics.update_step("step_1", "success")

        assert workflow_metrics.completed_steps == 1
        assert workflow_metrics.failed_steps == 0
        assert workflow_metrics.total_tokens == 100
        assert workflow_metrics.steps[0].status == "success"

    def test_update_step_failed(self, workflow_metrics, step_metrics):
        """Test updating step with failure."""
        workflow_metrics.add_step(step_metrics)
        workflow_metrics.update_step("step_1", "failed", error="Connection error")

        assert workflow_metrics.completed_steps == 0
        assert workflow_metrics.failed_steps == 1
        assert workflow_metrics.steps[0].status == "failed"
        assert workflow_metrics.steps[0].error == "Connection error"

    def test_update_nonexistent_step(self, workflow_metrics):
        """Test updating non-existent step does nothing."""
        workflow_metrics.update_step("nonexistent", "success")

        assert workflow_metrics.completed_steps == 0
        assert workflow_metrics.failed_steps == 0

    def test_update_accumulates_tokens(self, workflow_metrics):
        """Test that tokens accumulate across steps."""
        for i in range(3):
            step = StepMetrics(
                step_id=f"step_{i}",
                step_type="openai",
                action="generate",
                started_at=datetime.now(UTC),
            )
            step.tokens_used = 100 * (i + 1)
            workflow_metrics.add_step(step)
            workflow_metrics.update_step(f"step_{i}", "success")

        assert workflow_metrics.total_tokens == 600  # 100 + 200 + 300


class TestWorkflowMetricsComplete:
    """Tests for WorkflowMetrics.complete() method."""

    def test_complete_success(self, workflow_metrics):
        """Test completing workflow successfully."""
        time.sleep(0.01)
        workflow_metrics.complete(status="completed")

        assert workflow_metrics.completed_at is not None
        assert workflow_metrics.status == "completed"
        assert workflow_metrics.duration_ms > 0
        assert workflow_metrics.error is None

    def test_complete_failed(self, workflow_metrics):
        """Test completing workflow with failure."""
        workflow_metrics.complete(status="failed", error="Critical error")

        assert workflow_metrics.status == "failed"
        assert workflow_metrics.error == "Critical error"


class TestWorkflowMetricsToDict:
    """Tests for WorkflowMetrics.to_dict() method."""

    def test_to_dict_basic(self, workflow_metrics):
        """Test basic to_dict output."""
        result = workflow_metrics.to_dict()

        assert result["workflow_id"] == "wf_123"
        assert result["execution_id"] == "exec_456"
        assert result["workflow_name"] == "Test Workflow"
        assert result["status"] == "running"
        assert "started_at" in result

    def test_to_dict_with_steps(self, workflow_metrics, step_metrics):
        """Test to_dict includes steps."""
        workflow_metrics.add_step(step_metrics)
        result = workflow_metrics.to_dict()

        assert len(result["steps"]) == 1
        assert result["steps"][0]["step_id"] == "step_1"

    def test_to_dict_completed(self, workflow_metrics):
        """Test to_dict for completed workflow."""
        workflow_metrics.complete()
        result = workflow_metrics.to_dict()

        assert result["completed_at"] is not None
        assert result["duration_ms"] > 0


# =============================================================================
# MetricsStore Tests
# =============================================================================


class TestMetricsStoreSingleton:
    """Tests for MetricsStore singleton pattern."""

    def test_singleton_returns_same_instance(self, reset_metrics_store):
        """Test that MetricsStore returns same instance."""
        store1 = MetricsStore()
        store2 = MetricsStore()
        assert store1 is store2

    def test_singleton_with_db_path(self, reset_metrics_store, tmp_path):
        """Test singleton with db_path parameter."""
        db_path = str(tmp_path / "test.db")
        store1 = MetricsStore(db_path=db_path)
        store2 = MetricsStore()  # Should return same instance
        assert store1 is store2

    def test_reset_singleton(self, reset_metrics_store):
        """Test that resetting singleton creates new instance."""
        store1 = MetricsStore()
        MetricsStore._instance = None
        store2 = MetricsStore()
        # These should be different objects since we reset
        assert store1 is not store2


class TestMetricsStoreInit:
    """Tests for MetricsStore initialization."""

    def test_init_without_db(self, metrics_store):
        """Test initialization without database."""
        assert metrics_store._db_path is None
        assert metrics_store._workflows == {}
        assert metrics_store._recent_executions == []

    def test_init_with_db(self, metrics_store_with_db, tmp_path):
        """Test initialization with database."""
        assert metrics_store_with_db._db_path is not None
        assert Path(metrics_store_with_db._db_path).exists()

    def test_db_tables_created(self, metrics_store_with_db):
        """Test that database tables are created."""
        conn = sqlite3.connect(metrics_store_with_db._db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "workflow_executions" in tables
        assert "step_executions" in tables


class TestMetricsStoreStartWorkflow:
    """Tests for MetricsStore.start_workflow() method."""

    def test_start_workflow(self, metrics_store, workflow_metrics):
        """Test starting a workflow."""
        metrics_store.start_workflow(workflow_metrics)

        assert workflow_metrics.execution_id in metrics_store._workflows
        assert metrics_store._counters["workflows_started"] == 1

    def test_start_multiple_workflows(self, metrics_store):
        """Test starting multiple workflows."""
        for i in range(3):
            wf = WorkflowMetrics(
                workflow_id=f"wf_{i}",
                execution_id=f"exec_{i}",
                workflow_name=f"Workflow {i}",
                started_at=datetime.now(UTC),
            )
            metrics_store.start_workflow(wf)

        assert len(metrics_store._workflows) == 3
        assert metrics_store._counters["workflows_started"] == 3


class TestMetricsStoreCompleteWorkflow:
    """Tests for MetricsStore.complete_workflow() method."""

    def test_complete_workflow_success(self, metrics_store, workflow_metrics):
        """Test completing a workflow successfully."""
        metrics_store.start_workflow(workflow_metrics)
        metrics_store.complete_workflow(workflow_metrics.execution_id, "completed")

        assert workflow_metrics.execution_id not in metrics_store._workflows
        assert len(metrics_store._recent_executions) == 1
        assert metrics_store._counters["workflows_completed"] == 1
        assert metrics_store._counters["workflows_failed"] == 0

    def test_complete_workflow_failed(self, metrics_store, workflow_metrics):
        """Test completing a workflow with failure."""
        metrics_store.start_workflow(workflow_metrics)
        metrics_store.complete_workflow(
            workflow_metrics.execution_id, "failed", error="Error occurred"
        )

        assert metrics_store._counters["workflows_completed"] == 1
        assert metrics_store._counters["workflows_failed"] == 1

    def test_complete_nonexistent_workflow(self, metrics_store):
        """Test completing non-existent workflow does nothing."""
        metrics_store.complete_workflow("nonexistent", "completed")

        assert metrics_store._counters["workflows_completed"] == 0

    def test_recent_executions_limit(self, metrics_store):
        """Test that recent executions are limited."""
        metrics_store._max_recent = 5

        for i in range(10):
            wf = WorkflowMetrics(
                workflow_id=f"wf_{i}",
                execution_id=f"exec_{i}",
                workflow_name=f"Workflow {i}",
                started_at=datetime.now(UTC),
            )
            metrics_store.start_workflow(wf)
            metrics_store.complete_workflow(f"exec_{i}", "completed")

        assert len(metrics_store._recent_executions) == 5

    def test_complete_persists_to_db(self, metrics_store_with_db):
        """Test that completed workflows are persisted."""
        wf = WorkflowMetrics(
            workflow_id="wf_1",
            execution_id="exec_1",
            workflow_name="Test",
            started_at=datetime.now(UTC),
        )
        metrics_store_with_db.start_workflow(wf)
        metrics_store_with_db.complete_workflow("exec_1", "completed")

        conn = sqlite3.connect(metrics_store_with_db._db_path)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM workflow_executions WHERE execution_id = ?",
            ("exec_1",),
        )
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1


class TestMetricsStoreStartStep:
    """Tests for MetricsStore.start_step() method."""

    def test_start_step(self, metrics_store, workflow_metrics, step_metrics):
        """Test starting a step."""
        metrics_store.start_workflow(workflow_metrics)
        metrics_store.start_step(workflow_metrics.execution_id, step_metrics)

        assert len(metrics_store._workflows[workflow_metrics.execution_id].steps) == 1
        assert metrics_store._counters["steps_started"] == 1

    def test_start_step_nonexistent_workflow(self, metrics_store, step_metrics):
        """Test starting step for non-existent workflow does nothing."""
        metrics_store.start_step("nonexistent", step_metrics)

        assert metrics_store._counters["steps_started"] == 0


class TestMetricsStoreCompleteStep:
    """Tests for MetricsStore.complete_step() method."""

    def test_complete_step_success(self, metrics_store, workflow_metrics, step_metrics):
        """Test completing a step successfully."""
        metrics_store.start_workflow(workflow_metrics)
        metrics_store.start_step(workflow_metrics.execution_id, step_metrics)
        metrics_store.complete_step(workflow_metrics.execution_id, "step_1", "success", tokens=100)

        assert metrics_store._counters["steps_completed"] == 1
        wf = metrics_store._workflows[workflow_metrics.execution_id]
        assert wf.steps[0].tokens_used == 100

    def test_complete_step_failed(self, metrics_store, workflow_metrics, step_metrics):
        """Test completing a step with failure."""
        metrics_store.start_workflow(workflow_metrics)
        metrics_store.start_step(workflow_metrics.execution_id, step_metrics)
        metrics_store.complete_step(
            workflow_metrics.execution_id, "step_1", "failed", error="Step error"
        )

        assert metrics_store._counters["steps_completed"] == 1
        assert metrics_store._counters["steps_failed"] == 1


class TestMetricsStoreGetActiveWorkflows:
    """Tests for MetricsStore.get_active_workflows() method."""

    def test_get_active_workflows_empty(self, metrics_store):
        """Test getting active workflows when none running."""
        result = metrics_store.get_active_workflows()
        assert result == []

    def test_get_active_workflows(self, metrics_store, workflow_metrics):
        """Test getting active workflows."""
        metrics_store.start_workflow(workflow_metrics)
        result = metrics_store.get_active_workflows()

        assert len(result) == 1
        assert result[0]["execution_id"] == "exec_456"

    def test_get_active_workflows_excludes_completed(self, metrics_store, workflow_metrics):
        """Test that completed workflows are excluded."""
        metrics_store.start_workflow(workflow_metrics)
        metrics_store.complete_workflow(workflow_metrics.execution_id, "completed")

        result = metrics_store.get_active_workflows()
        assert result == []


class TestMetricsStoreGetRecentExecutions:
    """Tests for MetricsStore.get_recent_executions() method."""

    def test_get_recent_executions_empty(self, metrics_store):
        """Test getting recent executions when none completed."""
        result = metrics_store.get_recent_executions()
        assert result == []

    def test_get_recent_executions(self, metrics_store, workflow_metrics):
        """Test getting recent executions."""
        metrics_store.start_workflow(workflow_metrics)
        metrics_store.complete_workflow(workflow_metrics.execution_id, "completed")

        result = metrics_store.get_recent_executions()
        assert len(result) == 1

    def test_get_recent_executions_limit(self, metrics_store):
        """Test that limit is respected."""
        for i in range(10):
            wf = WorkflowMetrics(
                workflow_id=f"wf_{i}",
                execution_id=f"exec_{i}",
                workflow_name=f"Workflow {i}",
                started_at=datetime.now(UTC),
            )
            metrics_store.start_workflow(wf)
            metrics_store.complete_workflow(f"exec_{i}", "completed")

        result = metrics_store.get_recent_executions(limit=5)
        assert len(result) == 5


class TestMetricsStoreGetSummary:
    """Tests for MetricsStore.get_summary() method."""

    def test_get_summary_empty(self, metrics_store):
        """Test getting summary with no data."""
        result = metrics_store.get_summary()

        assert result["active_workflows"] == 0
        assert result["total_executions"] == 0
        assert result["failed_executions"] == 0
        assert result["success_rate"] == 0
        assert result["avg_duration_ms"] == 0

    def test_get_summary_with_data(self, metrics_store):
        """Test getting summary with workflow data."""
        for i in range(5):
            wf = WorkflowMetrics(
                workflow_id=f"wf_{i}",
                execution_id=f"exec_{i}",
                workflow_name=f"Workflow {i}",
                started_at=datetime.now(UTC),
            )
            metrics_store.start_workflow(wf)
            status = "failed" if i == 0 else "completed"
            metrics_store.complete_workflow(f"exec_{i}", status)

        result = metrics_store.get_summary()

        assert result["total_executions"] == 5
        assert result["failed_executions"] == 1
        assert result["success_rate"] == 80.0

    def test_get_summary_includes_tokens(self, metrics_store):
        """Test that summary includes token usage."""
        wf = WorkflowMetrics(
            workflow_id="wf_1",
            execution_id="exec_1",
            workflow_name="Test",
            started_at=datetime.now(UTC),
        )
        wf.total_tokens = 500
        metrics_store.start_workflow(wf)
        metrics_store.complete_workflow("exec_1", "completed")

        result = metrics_store.get_summary()
        assert result["total_tokens_used"] == 500


class TestMetricsStoreGetStepPerformance:
    """Tests for MetricsStore.get_step_performance() method."""

    def test_get_step_performance_empty(self, metrics_store):
        """Test getting step performance with no data."""
        result = metrics_store.get_step_performance()
        assert result == {}

    def test_get_step_performance(self, metrics_store):
        """Test getting step performance metrics."""
        wf = WorkflowMetrics(
            workflow_id="wf_1",
            execution_id="exec_1",
            workflow_name="Test",
            started_at=datetime.now(UTC),
        )
        step = StepMetrics(
            step_id="step_1",
            step_type="openai",
            action="generate",
            started_at=datetime.now(UTC) - timedelta(seconds=1),
        )
        step.complete(status="success")
        wf.steps.append(step)
        metrics_store.start_workflow(wf)
        metrics_store.complete_workflow("exec_1", "completed")

        result = metrics_store.get_step_performance()

        assert "openai:generate" in result
        assert result["openai:generate"]["count"] == 1
        assert result["openai:generate"]["avg_ms"] > 0

    def test_get_step_performance_with_failures(self, metrics_store):
        """Test step performance includes failure rate."""
        wf = WorkflowMetrics(
            workflow_id="wf_1",
            execution_id="exec_1",
            workflow_name="Test",
            started_at=datetime.now(UTC),
        )
        for i in range(4):
            step = StepMetrics(
                step_id=f"step_{i}",
                step_type="openai",
                action="generate",
                started_at=datetime.now(UTC),
            )
            status = "failed" if i == 0 else "success"
            step.complete(status=status)
            wf.steps.append(step)

        metrics_store.start_workflow(wf)
        metrics_store.complete_workflow("exec_1", "completed")

        result = metrics_store.get_step_performance()
        assert result["openai:generate"]["failure_rate"] == 25.0


class TestMetricsStoreGetHistoricalData:
    """Tests for MetricsStore.get_historical_data() method."""

    def test_get_historical_data_no_db(self, metrics_store, workflow_metrics):
        """Test getting historical data without database falls back to recent."""
        metrics_store.start_workflow(workflow_metrics)
        metrics_store.complete_workflow(workflow_metrics.execution_id, "completed")

        result = metrics_store.get_historical_data()
        assert len(result) == 1

    def test_get_historical_data_with_db(self, metrics_store_with_db):
        """Test getting historical data from database."""
        wf = WorkflowMetrics(
            workflow_id="wf_1",
            execution_id="exec_1",
            workflow_name="Test",
            started_at=datetime.now(UTC),
        )
        metrics_store_with_db.start_workflow(wf)
        metrics_store_with_db.complete_workflow("exec_1", "completed")

        result = metrics_store_with_db.get_historical_data(hours=1)
        assert len(result) == 1
        assert result[0]["execution_id"] == "exec_1"


class TestMetricsStorePersistence:
    """Tests for MetricsStore database persistence."""

    def test_persist_workflow_with_steps(self, metrics_store_with_db):
        """Test persisting workflow with steps."""
        wf = WorkflowMetrics(
            workflow_id="wf_1",
            execution_id="exec_1",
            workflow_name="Test",
            started_at=datetime.now(UTC),
        )
        for i in range(3):
            step = StepMetrics(
                step_id=f"step_{i}",
                step_type="openai",
                action="generate",
                started_at=datetime.now(UTC),
            )
            step.complete(status="success")
            wf.steps.append(step)

        metrics_store_with_db.start_workflow(wf)
        metrics_store_with_db.complete_workflow("exec_1", "completed")

        conn = sqlite3.connect(metrics_store_with_db._db_path)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM step_executions WHERE execution_id = ?",
            ("exec_1",),
        )
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 3

    def test_persisted_data_survives_reset(self, tmp_path, reset_metrics_store):
        """Test that persisted data survives store reset."""
        db_path = str(tmp_path / "test.db")

        # Create store and add data
        store1 = MetricsStore(db_path=db_path)
        wf = WorkflowMetrics(
            workflow_id="wf_1",
            execution_id="exec_1",
            workflow_name="Test",
            started_at=datetime.now(UTC),
        )
        store1.start_workflow(wf)
        store1.complete_workflow("exec_1", "completed")

        # Reset singleton
        MetricsStore._instance = None

        # Create new store with same db
        store2 = MetricsStore(db_path=db_path)
        result = store2.get_historical_data(hours=1)

        assert len(result) == 1


class TestMetricsStoreThreadSafety:
    """Tests for MetricsStore thread safety."""

    def test_concurrent_workflow_operations(self, metrics_store):
        """Test concurrent workflow start/complete operations."""
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    wf = WorkflowMetrics(
                        workflow_id=f"wf_{worker_id}_{i}",
                        execution_id=f"exec_{worker_id}_{i}",
                        workflow_name=f"Workflow {worker_id}-{i}",
                        started_at=datetime.now(UTC),
                    )
                    metrics_store.start_workflow(wf)
                    metrics_store.complete_workflow(f"exec_{worker_id}_{i}", "completed")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert metrics_store._counters["workflows_started"] == 50
        assert metrics_store._counters["workflows_completed"] == 50


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetricsIntegration:
    """Integration tests for complete metrics workflow."""

    def test_full_workflow_lifecycle(self, metrics_store):
        """Test complete workflow lifecycle with steps."""
        # Start workflow
        wf = WorkflowMetrics(
            workflow_id="wf_integration",
            execution_id="exec_integration",
            workflow_name="Integration Test",
            started_at=datetime.now(UTC),
        )
        metrics_store.start_workflow(wf)

        # Add and complete steps
        for i in range(3):
            step = StepMetrics(
                step_id=f"step_{i}",
                step_type="openai",
                action="generate",
                started_at=datetime.now(UTC),
            )
            metrics_store.start_step("exec_integration", step)
            metrics_store.complete_step("exec_integration", f"step_{i}", "success", tokens=100)

        # Complete workflow
        metrics_store.complete_workflow("exec_integration", "completed")

        # Verify results
        summary = metrics_store.get_summary()
        assert summary["total_executions"] == 1
        assert summary["total_steps_executed"] == 3

        recent = metrics_store.get_recent_executions()
        assert len(recent) == 1
        assert recent[0]["completed_steps"] == 3

    def test_mixed_success_failure_workflow(self, metrics_store):
        """Test workflow with mixed step results."""
        wf = WorkflowMetrics(
            workflow_id="wf_mixed",
            execution_id="exec_mixed",
            workflow_name="Mixed Test",
            started_at=datetime.now(UTC),
        )
        metrics_store.start_workflow(wf)

        # Success step
        step1 = StepMetrics(
            step_id="step_1",
            step_type="openai",
            action="generate",
            started_at=datetime.now(UTC),
        )
        metrics_store.start_step("exec_mixed", step1)
        metrics_store.complete_step("exec_mixed", "step_1", "success")

        # Failed step
        step2 = StepMetrics(
            step_id="step_2",
            step_type="github",
            action="create_pr",
            started_at=datetime.now(UTC),
        )
        metrics_store.start_step("exec_mixed", step2)
        metrics_store.complete_step("exec_mixed", "step_2", "failed", error="API error")

        metrics_store.complete_workflow("exec_mixed", "failed", error="Step failed")

        summary = metrics_store.get_summary()
        assert summary["failed_executions"] == 1
