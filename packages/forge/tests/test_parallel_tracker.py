"""Tests for parallel execution tracker in monitoring module."""

import sys
import threading
import time

import pytest

sys.path.insert(0, "src")

from animus_forge.monitoring.parallel_tracker import (
    BranchMetrics,
    ParallelExecutionMetrics,
    ParallelExecutionTracker,
    ParallelPatternType,
    RateLimitState,
    get_parallel_tracker,
)


@pytest.fixture(autouse=True)
def reset_global_tracker():
    """Reset global tracker before and after each test."""
    tracker = get_parallel_tracker()
    tracker.reset()
    yield
    tracker.reset()


# =============================================================================
# BranchMetrics Tests
# =============================================================================


class TestBranchMetrics:
    """Tests for BranchMetrics class."""

    def test_branch_creation(self):
        """Branch is created with correct defaults."""
        branch = BranchMetrics(
            branch_id="branch_1",
            parent_id="exec_1",
            item_index=0,
        )
        assert branch.branch_id == "branch_1"
        assert branch.parent_id == "exec_1"
        assert branch.item_index == 0
        assert branch.status == "pending"
        assert branch.started_at is None
        assert branch.duration_ms == 0

    def test_branch_start(self):
        """Branch start sets timestamp and status."""
        branch = BranchMetrics(
            branch_id="branch_1",
            parent_id="exec_1",
            item_index=0,
        )
        branch.start()
        assert branch.status == "running"
        assert branch.started_at is not None

    def test_branch_complete(self):
        """Branch complete calculates duration and sets tokens."""
        branch = BranchMetrics(
            branch_id="branch_1",
            parent_id="exec_1",
            item_index=0,
        )
        branch.start()
        time.sleep(0.01)  # Small delay for duration
        branch.complete(tokens=100)

        assert branch.status == "success"
        assert branch.completed_at is not None
        assert branch.tokens_used == 100
        assert branch.duration_ms > 0

    def test_branch_fail(self):
        """Branch fail records error."""
        branch = BranchMetrics(
            branch_id="branch_1",
            parent_id="exec_1",
            item_index=0,
        )
        branch.start()
        branch.fail("Test error")

        assert branch.status == "failed"
        assert branch.error == "Test error"
        assert branch.completed_at is not None

    def test_branch_cancel(self):
        """Branch cancel sets cancelled status."""
        branch = BranchMetrics(
            branch_id="branch_1",
            parent_id="exec_1",
            item_index=0,
        )
        branch.start()
        branch.cancel()

        assert branch.status == "cancelled"
        assert branch.completed_at is not None

    def test_branch_to_dict(self):
        """Branch serializes to dictionary."""
        branch = BranchMetrics(
            branch_id="branch_1",
            parent_id="exec_1",
            item_index=0,
            item_value="test_item",
        )
        branch.start()
        branch.complete(tokens=50)

        data = branch.to_dict()
        assert data["branch_id"] == "branch_1"
        assert data["parent_id"] == "exec_1"
        assert data["item_index"] == 0
        assert data["item_value"] == "test_item"
        assert data["status"] == "success"
        assert data["tokens_used"] == 50


# =============================================================================
# ParallelExecutionMetrics Tests
# =============================================================================


class TestParallelExecutionMetrics:
    """Tests for ParallelExecutionMetrics class."""

    def test_execution_creation(self):
        """Execution is created with correct defaults."""
        metrics = ParallelExecutionMetrics(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
            total_items=5,
            max_concurrent=3,
        )
        assert metrics.execution_id == "exec_1"
        assert metrics.pattern_type == ParallelPatternType.FAN_OUT
        assert metrics.total_items == 5
        assert metrics.max_concurrent == 3
        assert metrics.status == "pending"

    def test_execution_start(self):
        """Execution start sets timestamp and status."""
        metrics = ParallelExecutionMetrics(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
        )
        metrics.start()
        assert metrics.status == "running"
        assert metrics.started_at is not None

    def test_add_branch(self):
        """Branches can be added to execution."""
        metrics = ParallelExecutionMetrics(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
        )
        branch = metrics.add_branch("branch_1", 0, "item_value")

        assert "branch_1" in metrics.branches
        assert branch.branch_id == "branch_1"
        assert branch.item_value == "item_value"

    def test_execution_complete_aggregates(self):
        """Execution complete aggregates branch metrics."""
        metrics = ParallelExecutionMetrics(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
        )
        metrics.start()

        # Add and complete branches
        b1 = metrics.add_branch("branch_1", 0)
        b1.start()
        b1.complete(tokens=100)

        b2 = metrics.add_branch("branch_2", 1)
        b2.start()
        b2.complete(tokens=200)

        b3 = metrics.add_branch("branch_3", 2)
        b3.start()
        b3.fail("Error")

        metrics.complete()

        assert metrics.status == "completed"
        assert metrics.total_tokens == 300
        assert metrics.successful_count == 2
        assert metrics.failed_count == 1
        assert metrics.duration_ms > 0

    def test_active_branch_count(self):
        """active_branch_count returns running branches."""
        metrics = ParallelExecutionMetrics(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
        )

        b1 = metrics.add_branch("branch_1", 0)
        b1.start()  # Running

        b2 = metrics.add_branch("branch_2", 1)
        b2.start()
        b2.complete()  # Completed

        metrics.add_branch("branch_3", 2)
        # Still pending

        assert metrics.active_branch_count == 1
        assert metrics.pending_branch_count == 1

    def test_completion_ratio(self):
        """completion_ratio calculates correctly."""
        metrics = ParallelExecutionMetrics(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
        )

        b1 = metrics.add_branch("branch_1", 0)
        b1.start()
        b1.complete()

        b2 = metrics.add_branch("branch_2", 1)
        b2.start()
        b2.fail("Error")

        metrics.add_branch("branch_3", 2)
        # Still pending

        # 2 out of 3 completed (success or failed)
        assert metrics.completion_ratio == pytest.approx(2 / 3)

    def test_success_rate(self):
        """success_rate calculates correctly."""
        metrics = ParallelExecutionMetrics(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
        )

        for i in range(3):
            b = metrics.add_branch(f"branch_{i}", i)
            b.start()
            b.complete()

        b_fail = metrics.add_branch("branch_fail", 3)
        b_fail.start()
        b_fail.fail("Error")

        metrics.complete()

        # 3 success out of 4 completed
        assert metrics.success_rate == pytest.approx(0.75)

    def test_rate_limit_recording(self):
        """Rate limit waits are recorded."""
        metrics = ParallelExecutionMetrics(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
        )

        metrics.record_rate_limit_wait(100.0)
        metrics.record_rate_limit_wait(150.0)

        assert metrics.rate_limit_waits == 2
        assert metrics.rate_limit_wait_ms == 250.0

    def test_to_dict(self):
        """Execution serializes to dictionary."""
        metrics = ParallelExecutionMetrics(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.MAP_REDUCE,
            step_id="step_1",
            total_items=3,
            max_concurrent=2,
            workflow_id="wf_1",
        )
        metrics.start()
        metrics.add_branch("branch_1", 0).start()
        metrics.complete()

        data = metrics.to_dict()
        assert data["execution_id"] == "exec_1"
        assert data["pattern_type"] == "map_reduce"
        assert data["total_items"] == 3
        assert data["workflow_id"] == "wf_1"
        assert "branches" in data


# =============================================================================
# RateLimitState Tests
# =============================================================================


class TestRateLimitState:
    """Tests for RateLimitState class."""

    def test_rate_limit_state_creation(self):
        """RateLimitState is created with correct defaults."""
        state = RateLimitState(provider="anthropic")
        assert state.provider == "anthropic"
        assert state.current_concurrent == 0
        assert state.max_concurrent == 0

    def test_rate_limit_state_to_dict(self):
        """RateLimitState serializes correctly."""
        state = RateLimitState(
            provider="openai",
            current_concurrent=3,
            max_concurrent=5,
            waiting_count=2,
            total_requests=100,
            total_wait_ms=5000.0,
        )
        data = state.to_dict()

        assert data["provider"] == "openai"
        assert data["current_concurrent"] == 3
        assert data["max_concurrent"] == 5
        assert data["utilization"] == 60.0  # 3/5 * 100
        assert data["avg_wait_ms"] == 50.0  # 5000/100


# =============================================================================
# ParallelExecutionTracker Tests
# =============================================================================


class TestParallelExecutionTracker:
    """Tests for ParallelExecutionTracker class."""

    def test_tracker_creation(self):
        """Tracker is created with correct defaults."""
        tracker = ParallelExecutionTracker()
        assert len(tracker.get_active_executions()) == 0
        assert len(tracker.get_history()) == 0

    def test_start_execution(self):
        """Execution can be started."""
        tracker = ParallelExecutionTracker()

        metrics = tracker.start_execution(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
            total_items=5,
            max_concurrent=3,
        )

        assert metrics.execution_id == "exec_1"
        assert len(tracker.get_active_executions()) == 1

    def test_start_branch(self):
        """Branch can be started within execution."""
        tracker = ParallelExecutionTracker()

        tracker.start_execution(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
            total_items=3,
            max_concurrent=3,
        )

        branch = tracker.start_branch("exec_1", "branch_1", 0, "item_value")

        assert branch is not None
        assert branch.status == "running"

    def test_complete_branch(self):
        """Branch can be completed."""
        tracker = ParallelExecutionTracker()

        tracker.start_execution(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
            total_items=1,
            max_concurrent=1,
        )
        tracker.start_branch("exec_1", "branch_1", 0)
        tracker.complete_branch("exec_1", "branch_1", tokens=100)

        execution = tracker.get_execution("exec_1")
        assert execution["branches"]["branch_1"]["status"] == "success"
        assert execution["branches"]["branch_1"]["tokens_used"] == 100

    def test_fail_branch(self):
        """Branch can be failed."""
        tracker = ParallelExecutionTracker()

        tracker.start_execution(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
            total_items=1,
            max_concurrent=1,
        )
        tracker.start_branch("exec_1", "branch_1", 0)
        tracker.fail_branch("exec_1", "branch_1", "Test error")

        execution = tracker.get_execution("exec_1")
        assert execution["branches"]["branch_1"]["status"] == "failed"
        assert execution["branches"]["branch_1"]["error"] == "Test error"

    def test_complete_execution(self):
        """Execution can be completed."""
        tracker = ParallelExecutionTracker()

        tracker.start_execution(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
            total_items=1,
            max_concurrent=1,
        )
        tracker.start_branch("exec_1", "branch_1", 0)
        tracker.complete_branch("exec_1", "branch_1", tokens=50)

        result = tracker.complete_execution("exec_1")

        assert result is not None
        assert result.status == "completed"
        assert len(tracker.get_active_executions()) == 0
        assert len(tracker.get_history()) == 1

    def test_fail_execution(self):
        """Execution can be failed."""
        tracker = ParallelExecutionTracker()

        tracker.start_execution(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
            total_items=1,
            max_concurrent=1,
        )

        result = tracker.fail_execution("exec_1", "Test error")

        assert result is not None
        assert result.status == "failed"
        assert len(tracker.get_active_executions()) == 0

    def test_record_rate_limit_wait(self):
        """Rate limit waits are recorded."""
        tracker = ParallelExecutionTracker()

        tracker.start_execution(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
            total_items=1,
            max_concurrent=1,
        )

        tracker.record_rate_limit_wait("exec_1", "anthropic", 100.0)

        execution = tracker.get_execution("exec_1")
        assert execution["rate_limit_waits"] == 1
        assert execution["rate_limit_wait_ms"] == 100.0

    def test_update_rate_limit_state(self):
        """Rate limit state can be updated."""
        tracker = ParallelExecutionTracker()

        tracker.update_rate_limit_state(
            provider="anthropic",
            current_concurrent=3,
            max_concurrent=5,
            waiting_count=2,
        )

        states = tracker.get_rate_limit_states()
        assert "anthropic" in states
        assert states["anthropic"]["current_concurrent"] == 3
        assert states["anthropic"]["max_concurrent"] == 5
        assert states["anthropic"]["utilization"] == 60.0

    def test_get_summary(self):
        """Summary aggregates metrics correctly."""
        tracker = ParallelExecutionTracker()

        # Complete a few executions
        for i in range(3):
            tracker.start_execution(
                execution_id=f"exec_{i}",
                pattern_type=ParallelPatternType.FAN_OUT,
                step_id=f"step_{i}",
                total_items=2,
                max_concurrent=2,
            )
            tracker.start_branch(f"exec_{i}", f"branch_{i}_0", 0)
            tracker.complete_branch(f"exec_{i}", f"branch_{i}_0", tokens=100)
            tracker.complete_execution(f"exec_{i}")

        # Fail one execution
        tracker.start_execution(
            execution_id="exec_fail",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_fail",
            total_items=1,
            max_concurrent=1,
        )
        tracker.fail_execution("exec_fail", "Error")

        summary = tracker.get_summary()

        assert summary["total_executions"] == 4
        assert summary["success_rate"] == 75.0  # 3 out of 4
        assert summary["counters"]["executions_completed"] == 3
        assert summary["counters"]["executions_failed"] == 1

    def test_history_limit(self):
        """History respects max_history limit."""
        tracker = ParallelExecutionTracker(max_history=3)

        for i in range(5):
            tracker.start_execution(
                execution_id=f"exec_{i}",
                pattern_type=ParallelPatternType.FAN_OUT,
                step_id=f"step_{i}",
                total_items=1,
                max_concurrent=1,
            )
            tracker.complete_execution(f"exec_{i}")

        history = tracker.get_history()
        assert len(history) == 3
        # Most recent should be first
        assert history[0]["execution_id"] == "exec_4"

    def test_callbacks(self):
        """Callbacks are called on events."""
        tracker = ParallelExecutionTracker()
        events = []

        def callback(event_type: str, metrics):
            events.append((event_type, metrics.execution_id))

        tracker.register_callback(callback)

        tracker.start_execution(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
            total_items=1,
            max_concurrent=1,
        )
        tracker.complete_execution("exec_1")

        assert ("execution_started", "exec_1") in events
        assert ("execution_completed", "exec_1") in events

    def test_thread_safety(self):
        """Tracker handles concurrent access."""
        tracker = ParallelExecutionTracker()
        errors = []

        def worker(worker_id: int):
            try:
                for i in range(10):
                    exec_id = f"exec_{worker_id}_{i}"
                    tracker.start_execution(
                        execution_id=exec_id,
                        pattern_type=ParallelPatternType.FAN_OUT,
                        step_id=f"step_{worker_id}_{i}",
                        total_items=1,
                        max_concurrent=1,
                    )
                    tracker.start_branch(exec_id, f"branch_{i}", 0)
                    tracker.complete_branch(exec_id, f"branch_{i}", tokens=10)
                    tracker.complete_execution(exec_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(tracker.get_history()) == 50  # 5 workers * 10 executions

    def test_get_dashboard_data(self):
        """get_dashboard_data returns complete data."""
        tracker = ParallelExecutionTracker()

        tracker.start_execution(
            execution_id="exec_active",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
            total_items=2,
            max_concurrent=2,
        )

        tracker.start_execution(
            execution_id="exec_done",
            pattern_type=ParallelPatternType.MAP_REDUCE,
            step_id="step_2",
            total_items=3,
            max_concurrent=3,
        )
        tracker.complete_execution("exec_done")

        tracker.update_rate_limit_state("anthropic", 2, 5, 1)

        data = tracker.get_dashboard_data()

        assert "summary" in data
        assert "active_executions" in data
        assert "recent_executions" in data
        assert "rate_limits" in data
        assert len(data["active_executions"]) == 1
        assert len(data["recent_executions"]) == 1
        assert "anthropic" in data["rate_limits"]

    def test_reset(self):
        """reset clears all data."""
        tracker = ParallelExecutionTracker()

        tracker.start_execution(
            execution_id="exec_1",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="step_1",
            total_items=1,
            max_concurrent=1,
        )
        tracker.complete_execution("exec_1")
        tracker.update_rate_limit_state("anthropic", 1, 5, 0)

        tracker.reset()

        assert len(tracker.get_active_executions()) == 0
        assert len(tracker.get_history()) == 0
        assert len(tracker.get_rate_limit_states()) == 0


# =============================================================================
# Global Tracker Tests
# =============================================================================


class TestGetParallelTracker:
    """Tests for get_parallel_tracker function."""

    def test_returns_singleton(self):
        """get_parallel_tracker returns same instance."""
        tracker1 = get_parallel_tracker()
        tracker2 = get_parallel_tracker()
        assert tracker1 is tracker2

    def test_tracker_is_functional(self):
        """Global tracker is functional."""
        tracker = get_parallel_tracker()
        tracker.reset()  # Clean state

        tracker.start_execution(
            execution_id="global_exec_1",
            pattern_type=ParallelPatternType.AUTO_PARALLEL,
            step_id="step_1",
            total_items=2,
            max_concurrent=2,
        )

        active = tracker.get_active_executions()
        assert any(e["execution_id"] == "global_exec_1" for e in active)

        tracker.complete_execution("global_exec_1")
        tracker.reset()


# =============================================================================
# Integration with Executor Tests
# =============================================================================


class TestTrackerWithExecutor:
    """Tests verifying tracker integration with workflow executor."""

    def test_fan_out_creates_execution(self):
        """fan_out step creates tracked execution."""
        from animus_forge.workflow import StepConfig, WorkflowConfig, WorkflowExecutor

        tracker = get_parallel_tracker()
        tracker.reset()

        workflow = WorkflowConfig(
            name="Tracker Test",
            version="1.0",
            description="",
            steps=[
                StepConfig(
                    id="fan_out_step",
                    type="fan_out",
                    params={
                        "items": ["a", "b", "c"],
                        "step_template": {
                            "type": "shell",
                            "params": {"command": "echo ${item}"},
                        },
                    },
                ),
            ],
        )

        executor = WorkflowExecutor()
        result = executor.execute(workflow)

        assert result.status == "success"

        # Check that execution was tracked
        history = tracker.get_history()
        fan_out_execs = [e for e in history if e["pattern_type"] == "fan_out"]
        assert len(fan_out_execs) >= 1

        # Verify branch tracking
        exec_data = fan_out_execs[0]
        assert exec_data["successful_count"] == 3
        assert exec_data["total_items"] == 3

        tracker.reset()

    def test_auto_parallel_creates_execution(self):
        """auto_parallel creates tracked parallel group execution."""
        from animus_forge.workflow import (
            StepConfig,
            WorkflowConfig,
            WorkflowExecutor,
            WorkflowSettings,
        )

        tracker = get_parallel_tracker()
        tracker.reset()

        workflow = WorkflowConfig(
            name="Auto Parallel Tracker Test",
            version="1.0",
            description="",
            settings=WorkflowSettings(auto_parallel=True, auto_parallel_max_workers=4),
            steps=[
                StepConfig(id="a", type="shell", params={"command": "echo a"}),
                StepConfig(id="b", type="shell", params={"command": "echo b"}),
                StepConfig(id="c", type="shell", params={"command": "echo c"}),
            ],
        )

        executor = WorkflowExecutor()
        result = executor.execute(workflow)

        assert result.status == "success"

        # Check that parallel group was tracked
        history = tracker.get_history()
        parallel_execs = [e for e in history if e["pattern_type"] == "parallel_group"]
        assert len(parallel_execs) >= 1

        tracker.reset()
