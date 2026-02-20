"""Tests for parallel execution module."""

import sys
import time

import pytest

sys.path.insert(0, "src")

from datetime import UTC

from animus_forge.workflow.parallel import (
    ParallelExecutor,
    ParallelResult,
    ParallelTask,
    execute_steps_parallel,
)


def slow_task(duration: float = 0.1) -> str:
    """A task that takes some time."""
    time.sleep(duration)
    return f"completed in {duration}s"


def failing_task() -> str:
    """A task that always fails."""
    raise ValueError("Task failed intentionally")


def add_numbers(a: int, b: int) -> int:
    """Simple task for testing."""
    return a + b


class TestParallelTask:
    """Tests for ParallelTask class."""

    def test_create_task(self):
        """Can create a parallel task."""
        task = ParallelTask(
            id="task-1",
            step_id="step-1",
            handler=add_numbers,
            args=(1, 2),
        )
        assert task.id == "task-1"
        assert task.is_ready is True

    def test_task_with_dependencies(self):
        """Task with dependencies is not ready."""
        task = ParallelTask(
            id="task-1",
            step_id="step-1",
            handler=add_numbers,
            dependencies=["task-0"],
        )
        assert task.is_ready is False

    def test_task_duration(self):
        """Task tracks duration."""
        from datetime import datetime

        task = ParallelTask(
            id="task-1",
            step_id="step-1",
            handler=add_numbers,
        )
        task.started_at = datetime.now(UTC)
        time.sleep(0.01)
        task.completed_at = datetime.now(UTC)

        assert task.duration_ms is not None
        assert task.duration_ms >= 10


class TestParallelResult:
    """Tests for ParallelResult class."""

    def test_empty_result(self):
        """Empty result is successful."""
        result = ParallelResult()
        assert result.all_succeeded is True

    def test_result_with_failure(self):
        """Result with failures is not successful."""
        result = ParallelResult()
        result.failed.append("task-1")
        assert result.all_succeeded is False

    def test_get_result(self):
        """Can get task result."""
        result = ParallelResult()
        task = ParallelTask(id="task-1", step_id="s1", handler=lambda: None)
        task.result = "success"
        result.tasks["task-1"] = task

        assert result.get_result("task-1") == "success"
        assert result.get_result("missing") is None


class TestParallelExecutor:
    """Tests for ParallelExecutor class."""

    def test_analyze_dependencies(self):
        """Can analyze step dependencies."""
        executor = ParallelExecutor()
        steps = [
            {"id": "a"},
            {"id": "b", "depends_on": ["a"]},
            {"id": "c", "depends_on": ["a"]},
            {"id": "d", "depends_on": ["b", "c"]},
        ]
        deps = executor.analyze_dependencies(steps)

        assert deps["a"] == []
        assert deps["b"] == ["a"]
        assert deps["d"] == ["b", "c"]

    def test_find_parallel_groups(self):
        """Can find parallel execution groups."""
        executor = ParallelExecutor()
        deps = {
            "a": [],
            "b": [],
            "c": ["a", "b"],
            "d": ["c"],
        }
        groups = executor.find_parallel_groups(deps)

        assert len(groups) == 3
        assert {"a", "b"} == groups[0]  # a and b can run together
        assert {"c"} == groups[1]
        assert {"d"} == groups[2]

    def test_circular_dependency_raises(self):
        """Circular dependencies raise error."""
        executor = ParallelExecutor()
        deps = {
            "a": ["b"],
            "b": ["a"],
        }
        with pytest.raises(ValueError) as exc:
            executor.find_parallel_groups(deps)
        assert "circular" in str(exc.value).lower()

    def test_execute_independent_tasks(self):
        """Independent tasks run in parallel."""
        executor = ParallelExecutor(max_workers=4)

        tasks = [
            ParallelTask(id=f"task-{i}", step_id=f"s-{i}", handler=slow_task, args=(0.1,))
            for i in range(4)
        ]

        start = time.time()
        result = executor.execute_parallel(tasks)
        elapsed = time.time() - start

        assert result.all_succeeded
        assert len(result.successful) == 4
        # Should complete faster than sequential (4 * 0.1 = 0.4s)
        assert elapsed < 0.3  # Allow some overhead

    def test_execute_with_dependencies(self):
        """Tasks respect dependencies."""
        executor = ParallelExecutor()
        execution_order = []

        def track_order(name):
            def fn():
                execution_order.append(name)
                time.sleep(0.01)
                return name

            return fn

        tasks = [
            ParallelTask(id="a", step_id="a", handler=track_order("a")),
            ParallelTask(id="b", step_id="b", handler=track_order("b"), dependencies=["a"]),
            ParallelTask(id="c", step_id="c", handler=track_order("c"), dependencies=["b"]),
        ]

        result = executor.execute_parallel(tasks)

        assert result.all_succeeded
        # a must come before b, b must come before c
        assert execution_order.index("a") < execution_order.index("b")
        assert execution_order.index("b") < execution_order.index("c")

    def test_execute_handles_failure(self):
        """Failed tasks are tracked."""
        executor = ParallelExecutor()

        tasks = [
            ParallelTask(id="success", step_id="s1", handler=lambda: "ok"),
            ParallelTask(id="failure", step_id="s2", handler=failing_task),
        ]

        result = executor.execute_parallel(tasks)

        assert not result.all_succeeded
        assert "success" in result.successful
        assert "failure" in result.failed
        assert result.get_error("failure") is not None

    def test_callbacks(self):
        """Callbacks are invoked."""
        executor = ParallelExecutor()
        completed = []
        errors = []

        tasks = [
            ParallelTask(id="good", step_id="s1", handler=lambda: "ok"),
            ParallelTask(id="bad", step_id="s2", handler=failing_task),
        ]

        executor.execute_parallel(
            tasks,
            on_complete=lambda id, res: completed.append(id),
            on_error=lambda id, err: errors.append(id),
        )

        assert "good" in completed
        assert "bad" in errors


class TestExecuteStepsParallel:
    """Tests for execute_steps_parallel function."""

    def test_simple_steps(self):
        """Can execute simple steps."""
        steps = [
            {"id": "step1", "value": 1},
            {"id": "step2", "value": 2},
        ]

        def handler(step):
            return step["value"] * 2

        result = execute_steps_parallel(steps, handler)

        assert result.all_succeeded
        assert result.get_result("step1") == 2
        assert result.get_result("step2") == 4

    def test_dependent_steps(self):
        """Steps with dependencies execute in order."""
        execution_order = []
        steps = [
            {"id": "first"},
            {"id": "second", "depends_on": ["first"]},
        ]

        def handler(step):
            execution_order.append(step["id"])
            time.sleep(0.01)
            return step["id"]

        result = execute_steps_parallel(steps, handler)

        assert result.all_succeeded
        assert execution_order[0] == "first"
        assert execution_order[1] == "second"
