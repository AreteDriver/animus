"""Tests for Monitoring and Metrics Collection.

Tests for:
- StepMetrics and WorkflowMetrics dataclasses
- MetricsStore with in-memory and SQLite persistence
- ExecutionTracker context managers and decorators
- AgentTracker for AI agent activity tracking
"""

import os
import sys
import tempfile
import time
from datetime import UTC, datetime

sys.path.insert(0, "src")

from animus_forge.monitoring.metrics import (
    MetricsStore,
    StepMetrics,
    WorkflowMetrics,
)
from animus_forge.monitoring.tracker import (
    AgentTracker,
    ExecutionTracker,
    get_tracker,
)


# Reset MetricsStore singleton between tests
def reset_metrics_store():
    """Reset the MetricsStore singleton for test isolation."""
    MetricsStore._instance = None


class TestStepMetrics:
    """Tests for StepMetrics dataclass."""

    def test_step_metrics_creation(self):
        """StepMetrics can be created with required fields."""
        step = StepMetrics(
            step_id="step_1",
            step_type="claude_code",
            action="execute_agent",
            started_at=datetime.now(UTC),
        )

        assert step.step_id == "step_1"
        assert step.step_type == "claude_code"
        assert step.action == "execute_agent"
        assert step.status == "running"
        assert step.tokens_used == 0
        assert step.completed_at is None

    def test_step_metrics_complete_success(self):
        """StepMetrics.complete() marks step as successful."""
        step = StepMetrics(
            step_id="step_1",
            step_type="transform",
            action="format",
            started_at=datetime.now(UTC),
        )

        time.sleep(0.01)  # Small delay for measurable duration
        step.complete("success")

        assert step.status == "success"
        assert step.completed_at is not None
        assert step.duration_ms > 0
        assert step.error is None

    def test_step_metrics_complete_failed(self):
        """StepMetrics.complete() marks step as failed with error."""
        step = StepMetrics(
            step_id="step_1",
            step_type="api_call",
            action="fetch",
            started_at=datetime.now(UTC),
        )

        step.complete("failed", "Connection timeout")

        assert step.status == "failed"
        assert step.error == "Connection timeout"
        assert step.completed_at is not None

    def test_step_metrics_to_dict(self):
        """StepMetrics.to_dict() returns serializable dict."""
        step = StepMetrics(
            step_id="step_1",
            step_type="transform",
            action="parse",
            started_at=datetime.now(UTC),
            tokens_used=150,
        )
        step.complete("success")

        data = step.to_dict()

        assert data["step_id"] == "step_1"
        assert data["step_type"] == "transform"
        assert data["action"] == "parse"
        assert data["status"] == "success"
        assert data["tokens_used"] == 150
        assert "started_at" in data
        assert "completed_at" in data
        assert "duration_ms" in data


class TestWorkflowMetrics:
    """Tests for WorkflowMetrics dataclass."""

    def test_workflow_metrics_creation(self):
        """WorkflowMetrics can be created with required fields."""
        wf = WorkflowMetrics(
            workflow_id="test_workflow",
            execution_id="exec_123",
            workflow_name="Test Workflow",
            started_at=datetime.now(UTC),
        )

        assert wf.workflow_id == "test_workflow"
        assert wf.execution_id == "exec_123"
        assert wf.workflow_name == "Test Workflow"
        assert wf.status == "running"
        assert wf.total_steps == 0
        assert wf.completed_steps == 0
        assert wf.total_tokens == 0

    def test_workflow_metrics_add_step(self):
        """WorkflowMetrics.add_step() adds step and updates count."""
        wf = WorkflowMetrics(
            workflow_id="test_workflow",
            execution_id="exec_123",
            workflow_name="Test Workflow",
            started_at=datetime.now(UTC),
        )

        step = StepMetrics(
            step_id="step_1",
            step_type="transform",
            action="parse",
            started_at=datetime.now(UTC),
        )

        wf.add_step(step)

        assert wf.total_steps == 1
        assert len(wf.steps) == 1
        assert wf.steps[0].step_id == "step_1"

    def test_workflow_metrics_update_step_success(self):
        """WorkflowMetrics.update_step() updates step status and counters."""
        wf = WorkflowMetrics(
            workflow_id="test_workflow",
            execution_id="exec_123",
            workflow_name="Test Workflow",
            started_at=datetime.now(UTC),
        )

        step = StepMetrics(
            step_id="step_1",
            step_type="transform",
            action="parse",
            started_at=datetime.now(UTC),
            tokens_used=100,
        )
        wf.add_step(step)

        wf.update_step("step_1", "success")

        assert wf.completed_steps == 1
        assert wf.failed_steps == 0
        assert wf.total_tokens == 100
        assert wf.steps[0].status == "success"

    def test_workflow_metrics_update_step_failed(self):
        """WorkflowMetrics.update_step() handles failed steps."""
        wf = WorkflowMetrics(
            workflow_id="test_workflow",
            execution_id="exec_123",
            workflow_name="Test Workflow",
            started_at=datetime.now(UTC),
        )

        step = StepMetrics(
            step_id="step_1",
            step_type="api_call",
            action="fetch",
            started_at=datetime.now(UTC),
        )
        wf.add_step(step)

        wf.update_step("step_1", "failed", "API error")

        assert wf.completed_steps == 0
        assert wf.failed_steps == 1
        assert wf.steps[0].status == "failed"
        assert wf.steps[0].error == "API error"

    def test_workflow_metrics_complete(self):
        """WorkflowMetrics.complete() finalizes workflow."""
        wf = WorkflowMetrics(
            workflow_id="test_workflow",
            execution_id="exec_123",
            workflow_name="Test Workflow",
            started_at=datetime.now(UTC),
        )

        time.sleep(0.01)
        wf.complete("completed")

        assert wf.status == "completed"
        assert wf.completed_at is not None
        assert wf.duration_ms > 0

    def test_workflow_metrics_to_dict(self):
        """WorkflowMetrics.to_dict() returns serializable dict."""
        wf = WorkflowMetrics(
            workflow_id="test_workflow",
            execution_id="exec_123",
            workflow_name="Test Workflow",
            started_at=datetime.now(UTC),
        )

        step = StepMetrics(
            step_id="step_1",
            step_type="transform",
            action="parse",
            started_at=datetime.now(UTC),
        )
        wf.add_step(step)
        wf.update_step("step_1", "success")
        wf.complete("completed")

        data = wf.to_dict()

        assert data["workflow_id"] == "test_workflow"
        assert data["execution_id"] == "exec_123"
        assert data["status"] == "completed"
        assert data["total_steps"] == 1
        assert data["completed_steps"] == 1
        assert len(data["steps"]) == 1


class TestMetricsStore:
    """Tests for MetricsStore."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_metrics_store()

    def test_metrics_store_singleton(self):
        """MetricsStore is a singleton."""
        store1 = MetricsStore()
        store2 = MetricsStore()

        assert store1 is store2

    def test_metrics_store_start_workflow(self):
        """MetricsStore.start_workflow() registers workflow."""
        store = MetricsStore()

        wf = WorkflowMetrics(
            workflow_id="test_wf",
            execution_id="exec_1",
            workflow_name="Test",
            started_at=datetime.now(UTC),
        )
        store.start_workflow(wf)

        active = store.get_active_workflows()
        assert len(active) == 1
        assert active[0]["execution_id"] == "exec_1"

    def test_metrics_store_complete_workflow(self):
        """MetricsStore.complete_workflow() finalizes and archives workflow."""
        store = MetricsStore()

        wf = WorkflowMetrics(
            workflow_id="test_wf",
            execution_id="exec_1",
            workflow_name="Test",
            started_at=datetime.now(UTC),
        )
        store.start_workflow(wf)
        store.complete_workflow("exec_1", "completed")

        # Should no longer be active
        active = store.get_active_workflows()
        assert len(active) == 0

        # Should be in recent executions
        recent = store.get_recent_executions()
        assert len(recent) == 1
        assert recent[0]["status"] == "completed"

    def test_metrics_store_complete_workflow_failed(self):
        """MetricsStore tracks failed workflows."""
        store = MetricsStore()

        wf = WorkflowMetrics(
            workflow_id="test_wf",
            execution_id="exec_1",
            workflow_name="Test",
            started_at=datetime.now(UTC),
        )
        store.start_workflow(wf)
        store.complete_workflow("exec_1", "failed", "Something went wrong")

        recent = store.get_recent_executions()
        assert recent[0]["status"] == "failed"
        assert recent[0]["error"] == "Something went wrong"

    def test_metrics_store_start_step(self):
        """MetricsStore.start_step() adds step to workflow."""
        store = MetricsStore()

        wf = WorkflowMetrics(
            workflow_id="test_wf",
            execution_id="exec_1",
            workflow_name="Test",
            started_at=datetime.now(UTC),
        )
        store.start_workflow(wf)

        step = StepMetrics(
            step_id="step_1",
            step_type="transform",
            action="parse",
            started_at=datetime.now(UTC),
        )
        store.start_step("exec_1", step)

        active = store.get_active_workflows()
        assert active[0]["total_steps"] == 1

    def test_metrics_store_complete_step(self):
        """MetricsStore.complete_step() finalizes step."""
        store = MetricsStore()

        wf = WorkflowMetrics(
            workflow_id="test_wf",
            execution_id="exec_1",
            workflow_name="Test",
            started_at=datetime.now(UTC),
        )
        store.start_workflow(wf)

        step = StepMetrics(
            step_id="step_1",
            step_type="transform",
            action="parse",
            started_at=datetime.now(UTC),
        )
        store.start_step("exec_1", step)
        store.complete_step("exec_1", "step_1", "success", tokens=150)

        active = store.get_active_workflows()
        assert active[0]["completed_steps"] == 1
        assert active[0]["total_tokens"] == 150

    def test_metrics_store_get_summary(self):
        """MetricsStore.get_summary() returns aggregate metrics."""
        store = MetricsStore()

        # Create and complete multiple workflows
        for i in range(3):
            wf = WorkflowMetrics(
                workflow_id=f"test_wf_{i}",
                execution_id=f"exec_{i}",
                workflow_name=f"Test {i}",
                started_at=datetime.now(UTC),
            )
            store.start_workflow(wf)
            status = "completed" if i < 2 else "failed"
            store.complete_workflow(f"exec_{i}", status)

        summary = store.get_summary()

        assert summary["total_executions"] == 3
        assert summary["failed_executions"] == 1
        assert summary["success_rate"] == 66.7  # 2/3 * 100
        assert summary["active_workflows"] == 0

    def test_metrics_store_get_step_performance(self):
        """MetricsStore.get_step_performance() returns per-type stats."""
        store = MetricsStore()

        wf = WorkflowMetrics(
            workflow_id="test_wf",
            execution_id="exec_1",
            workflow_name="Test",
            started_at=datetime.now(UTC),
        )
        store.start_workflow(wf)

        # Add multiple steps of same type
        for i in range(3):
            step = StepMetrics(
                step_id=f"step_{i}",
                step_type="transform",
                action="parse",
                started_at=datetime.now(UTC),
            )
            store.start_step("exec_1", step)
            store.complete_step("exec_1", f"step_{i}", "success")

        store.complete_workflow("exec_1", "completed")

        perf = store.get_step_performance()

        assert "transform:parse" in perf
        assert perf["transform:parse"]["count"] == 3

    def test_metrics_store_sqlite_persistence(self):
        """MetricsStore persists to SQLite when db_path provided."""
        reset_metrics_store()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "metrics.db")
            store = MetricsStore(db_path)

            wf = WorkflowMetrics(
                workflow_id="test_wf",
                execution_id="exec_persist",
                workflow_name="Persisted Test",
                started_at=datetime.now(UTC),
            )
            store.start_workflow(wf)

            step = StepMetrics(
                step_id="step_1",
                step_type="transform",
                action="parse",
                started_at=datetime.now(UTC),
            )
            store.start_step("exec_persist", step)
            store.complete_step("exec_persist", "step_1", "success", tokens=100)
            store.complete_workflow("exec_persist", "completed")

            # Verify database file exists
            assert os.path.exists(db_path)

            # Query historical data
            historical = store.get_historical_data(hours=1)
            assert len(historical) == 1
            assert historical[0]["execution_id"] == "exec_persist"


class TestExecutionTracker:
    """Tests for ExecutionTracker."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_metrics_store()
        # Also reset the global tracker
        import animus_forge.monitoring.tracker as tracker_module

        tracker_module._tracker = None

    def test_execution_tracker_creation(self):
        """ExecutionTracker can be created."""
        tracker = ExecutionTracker()

        assert tracker.store is not None
        assert tracker._current_execution is None

    def test_track_workflow_context_manager(self):
        """track_workflow context manager tracks workflow lifecycle."""
        tracker = ExecutionTracker()

        with tracker.track_workflow("test_wf", "Test Workflow") as exec_id:
            assert exec_id.startswith("exec_")
            assert tracker._current_execution == exec_id

            # Workflow should be active
            status = tracker.get_status()
            assert len(status["active_workflows"]) == 1

        # After context, workflow should be completed
        assert tracker._current_execution is None
        recent = tracker.store.get_recent_executions()
        assert len(recent) == 1
        assert recent[0]["status"] == "completed"

    def test_track_workflow_handles_exception(self):
        """track_workflow marks workflow as failed on exception."""
        tracker = ExecutionTracker()

        try:
            with tracker.track_workflow("test_wf", "Test Workflow"):
                raise ValueError("Test error")
        except ValueError:
            pass

        recent = tracker.store.get_recent_executions()
        assert recent[0]["status"] == "failed"
        assert "Test error" in recent[0]["error"]

    def test_track_step_context_manager(self):
        """track_step context manager tracks step lifecycle."""
        tracker = ExecutionTracker()

        with tracker.track_workflow("test_wf", "Test Workflow"):
            with tracker.track_step("step_1", "transform", "parse") as step:
                step.tokens_used = 100

        recent = tracker.store.get_recent_executions()
        assert recent[0]["total_steps"] == 1
        assert recent[0]["completed_steps"] == 1
        assert recent[0]["total_tokens"] == 100

    def test_track_step_handles_exception(self):
        """track_step marks step as failed on exception."""
        tracker = ExecutionTracker()

        try:
            with tracker.track_workflow("test_wf", "Test Workflow"):
                with tracker.track_step("step_1", "api_call", "fetch"):
                    raise ConnectionError("Network error")
        except ConnectionError:
            pass

        recent = tracker.store.get_recent_executions()
        assert recent[0]["failed_steps"] == 1
        assert recent[0]["steps"][0]["status"] == "failed"

    def test_track_step_requires_active_workflow(self):
        """track_step raises error when no active workflow."""
        tracker = ExecutionTracker()

        try:
            with tracker.track_step("step_1", "transform", "parse"):
                pass
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No active workflow" in str(e)

    def test_track_step_decorator(self):
        """track_step_decorator wraps function for automatic tracking."""
        tracker = ExecutionTracker()

        @tracker.track_step_decorator("transform", "format")
        def format_data(data):
            return data.upper()

        with tracker.track_workflow("test_wf", "Test Workflow"):
            result = format_data("hello")

        assert result == "HELLO"
        recent = tracker.store.get_recent_executions()
        assert recent[0]["total_steps"] == 1

    def test_get_tracker_singleton(self):
        """get_tracker returns global singleton."""
        tracker1 = get_tracker()
        tracker2 = get_tracker()

        assert tracker1 is tracker2

    def test_get_dashboard_data(self):
        """get_dashboard_data returns all dashboard metrics."""
        tracker = ExecutionTracker()

        with tracker.track_workflow("test_wf", "Test Workflow"):
            with tracker.track_step("step_1", "transform", "parse") as step:
                step.tokens_used = 50

        data = tracker.get_dashboard_data()

        assert "summary" in data
        assert "active_workflows" in data
        assert "recent_executions" in data
        assert "step_performance" in data


class TestAgentTracker:
    """Tests for AgentTracker."""

    def test_agent_tracker_creation(self):
        """AgentTracker can be created."""
        tracker = AgentTracker()

        assert len(tracker.get_active_agents()) == 0
        assert len(tracker.get_agent_history()) == 0

    def test_register_agent(self):
        """register_agent adds agent to active list."""
        tracker = AgentTracker()

        tracker.register_agent("agent_1", "planner", "workflow_1")

        agents = tracker.get_active_agents()
        assert len(agents) == 1
        assert agents[0]["agent_id"] == "agent_1"
        assert agents[0]["role"] == "planner"
        assert agents[0]["status"] == "active"

    def test_update_agent(self):
        """update_agent modifies agent properties."""
        tracker = AgentTracker()

        tracker.register_agent("agent_1", "planner")
        tracker.update_agent("agent_1", tasks_completed=5, status="busy")

        agents = tracker.get_active_agents()
        assert agents[0]["tasks_completed"] == 5
        assert agents[0]["status"] == "busy"

    def test_complete_agent(self):
        """complete_agent moves agent to history."""
        tracker = AgentTracker()

        tracker.register_agent("agent_1", "planner")
        tracker.complete_agent("agent_1", "completed")

        # Should not be in active
        assert len(tracker.get_active_agents()) == 0

        # Should be in history
        history = tracker.get_agent_history()
        assert len(history) == 1
        assert history[0]["agent_id"] == "agent_1"
        assert history[0]["status"] == "completed"
        assert "completed_at" in history[0]

    def test_agent_history_limit(self):
        """Agent history respects max_history limit."""
        tracker = AgentTracker()
        tracker._max_history = 5

        # Register and complete more than max
        for i in range(10):
            tracker.register_agent(f"agent_{i}", "worker")
            tracker.complete_agent(f"agent_{i}")

        history = tracker.get_agent_history()
        assert len(history) == 5
        # Most recent should be first
        assert history[0]["agent_id"] == "agent_9"

    def test_get_agent_summary(self):
        """get_agent_summary returns role breakdown."""
        tracker = AgentTracker()

        tracker.register_agent("agent_1", "planner")
        tracker.register_agent("agent_2", "builder")
        tracker.register_agent("agent_3", "planner")
        tracker.complete_agent("agent_1")

        summary = tracker.get_agent_summary()

        assert summary["active_count"] == 2
        assert summary["by_role"]["planner"] == 1
        assert summary["by_role"]["builder"] == 1
        assert summary["recent_count"] == 1

    def test_get_agent_history_with_limit(self):
        """get_agent_history respects limit parameter."""
        tracker = AgentTracker()

        for i in range(10):
            tracker.register_agent(f"agent_{i}", "worker")
            tracker.complete_agent(f"agent_{i}")

        history = tracker.get_agent_history(limit=3)
        assert len(history) == 3


class TestMonitoringIntegration:
    """Integration tests for the monitoring system."""

    def setup_method(self):
        """Reset singletons before each test."""
        reset_metrics_store()
        import animus_forge.monitoring.tracker as tracker_module

        tracker_module._tracker = None

    def test_full_workflow_with_multiple_steps(self):
        """Test complete workflow tracking with multiple steps."""
        tracker = ExecutionTracker()

        with tracker.track_workflow("data_pipeline", "Data Pipeline"):
            with tracker.track_step("fetch", "api_call", "get_data") as step:
                step.tokens_used = 50
                time.sleep(0.01)

            with tracker.track_step("transform", "transform", "parse") as step:
                step.tokens_used = 100
                time.sleep(0.01)

            with tracker.track_step("store", "database", "insert") as step:
                step.tokens_used = 25
                time.sleep(0.01)

        # Verify final state
        recent = tracker.store.get_recent_executions()
        assert len(recent) == 1

        wf = recent[0]
        assert wf["total_steps"] == 3
        assert wf["completed_steps"] == 3
        assert wf["failed_steps"] == 0
        assert wf["total_tokens"] == 175
        assert wf["status"] == "completed"

        # Verify step performance
        perf = tracker.store.get_step_performance()
        assert "api_call:get_data" in perf
        assert "transform:parse" in perf
        assert "database:insert" in perf

    def test_workflow_with_partial_failure(self):
        """Test workflow where one step fails."""
        tracker = ExecutionTracker()

        try:
            with tracker.track_workflow("failing_pipeline", "Failing Pipeline"):
                with tracker.track_step("step_1", "transform", "parse") as step:
                    step.tokens_used = 50

                with tracker.track_step("step_2", "api_call", "fetch"):
                    raise RuntimeError("API unavailable")
        except RuntimeError:
            pass

        recent = tracker.store.get_recent_executions()
        wf = recent[0]

        assert wf["total_steps"] == 2
        assert wf["completed_steps"] == 1
        assert wf["failed_steps"] == 1
        assert wf["status"] == "failed"

    def test_concurrent_workflows(self):
        """Test tracking multiple concurrent workflows."""
        reset_metrics_store()
        store = MetricsStore()

        # Start two workflows
        wf1 = WorkflowMetrics(
            workflow_id="wf_1",
            execution_id="exec_1",
            workflow_name="Workflow 1",
            started_at=datetime.now(UTC),
        )
        wf2 = WorkflowMetrics(
            workflow_id="wf_2",
            execution_id="exec_2",
            workflow_name="Workflow 2",
            started_at=datetime.now(UTC),
        )

        store.start_workflow(wf1)
        store.start_workflow(wf2)

        # Both should be active
        active = store.get_active_workflows()
        assert len(active) == 2

        # Complete one
        store.complete_workflow("exec_1", "completed")

        active = store.get_active_workflows()
        assert len(active) == 1
        assert active[0]["execution_id"] == "exec_2"

        # Complete the other
        store.complete_workflow("exec_2", "completed")

        active = store.get_active_workflows()
        assert len(active) == 0

        recent = store.get_recent_executions()
        assert len(recent) == 2
