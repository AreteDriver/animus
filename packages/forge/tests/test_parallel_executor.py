"""Tests for the parallel execution module."""

import asyncio
import time
from datetime import UTC, datetime

import pytest

from animus_forge.workflow.parallel import (
    ParallelExecutor,
    ParallelResult,
    ParallelStrategy,
    ParallelTask,
    execute_steps_parallel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_handler(*args, **kwargs):
    """Simple handler that returns success."""
    return "ok"


def _slow_handler(seconds=0.05):
    """Handler that sleeps, then returns."""
    time.sleep(seconds)
    return f"slept-{seconds}"


def _fail_handler():
    """Handler that always raises."""
    raise RuntimeError("boom")


def _add_handler(a, b):
    """Handler that adds two numbers."""
    return a + b


async def _async_ok_handler():
    """Async handler that returns success."""
    await asyncio.sleep(0.01)
    return "async-ok"


async def _async_fail_handler():
    """Async handler that always raises."""
    raise RuntimeError("async-boom")


# ---------------------------------------------------------------------------
# ParallelTask dataclass
# ---------------------------------------------------------------------------


class TestParallelTask:
    """Tests for the ParallelTask dataclass."""

    def test_defaults(self):
        task = ParallelTask(id="t1", step_id="s1", handler=_ok_handler)
        assert task.id == "t1"
        assert task.step_id == "s1"
        assert task.args == ()
        assert task.kwargs == {}
        assert task.dependencies == []
        assert task.result is None
        assert task.error is None
        assert task.started_at is None
        assert task.completed_at is None
        assert task.retries == 0

    def test_duration_ms_none_when_not_started(self):
        task = ParallelTask(id="t1", step_id="s1", handler=_ok_handler)
        assert task.duration_ms is None

    def test_duration_ms_calculated(self):
        now = datetime.now(UTC)
        task = ParallelTask(
            id="t1",
            step_id="s1",
            handler=_ok_handler,
            started_at=now,
            completed_at=now,
        )
        assert task.duration_ms == 0

    def test_is_ready_no_deps(self):
        task = ParallelTask(id="t1", step_id="s1", handler=_ok_handler)
        assert task.is_ready is True

    def test_is_ready_with_deps(self):
        task = ParallelTask(id="t1", step_id="s1", handler=_ok_handler, dependencies=["dep1"])
        assert task.is_ready is False


# ---------------------------------------------------------------------------
# ParallelResult dataclass
# ---------------------------------------------------------------------------


class TestParallelResult:
    """Tests for the ParallelResult dataclass."""

    def test_all_succeeded_empty(self):
        result = ParallelResult()
        assert result.all_succeeded is True

    def test_all_succeeded_with_failures(self):
        result = ParallelResult(failed=["t1"])
        assert result.all_succeeded is False

    def test_all_succeeded_with_cancellations(self):
        result = ParallelResult(cancelled=["t1"])
        assert result.all_succeeded is False

    def test_get_result_existing(self):
        task = ParallelTask(id="t1", step_id="s1", handler=_ok_handler)
        task.result = "value"
        result = ParallelResult(tasks={"t1": task})
        assert result.get_result("t1") == "value"

    def test_get_result_missing(self):
        result = ParallelResult()
        assert result.get_result("missing") is None

    def test_get_error_existing(self):
        err = RuntimeError("oops")
        task = ParallelTask(id="t1", step_id="s1", handler=_ok_handler)
        task.error = err
        result = ParallelResult(tasks={"t1": task})
        assert result.get_error("t1") is err

    def test_get_error_missing(self):
        result = ParallelResult()
        assert result.get_error("missing") is None


# ---------------------------------------------------------------------------
# ParallelExecutor — init & helpers
# ---------------------------------------------------------------------------


class TestParallelExecutorInit:
    """Tests for ParallelExecutor initialization."""

    def test_defaults(self):
        executor = ParallelExecutor()
        assert executor.strategy == ParallelStrategy.THREADING
        assert executor.max_workers == 4
        assert executor.timeout == 300.0

    def test_custom_params(self):
        executor = ParallelExecutor(
            strategy=ParallelStrategy.ASYNCIO,
            max_workers=8,
            timeout=60.0,
        )
        assert executor.strategy == ParallelStrategy.ASYNCIO
        assert executor.max_workers == 8
        assert executor.timeout == 60.0


# ---------------------------------------------------------------------------
# analyze_dependencies
# ---------------------------------------------------------------------------


class TestAnalyzeDependencies:
    """Tests for ParallelExecutor.analyze_dependencies."""

    def test_no_deps(self):
        executor = ParallelExecutor()
        steps = [{"id": "a"}, {"id": "b"}]
        deps = executor.analyze_dependencies(steps)
        assert deps == {"a": [], "b": []}

    def test_with_deps_list(self):
        executor = ParallelExecutor()
        steps = [
            {"id": "a"},
            {"id": "b", "depends_on": ["a"]},
        ]
        deps = executor.analyze_dependencies(steps)
        assert deps["a"] == []
        assert deps["b"] == ["a"]

    def test_with_deps_string(self):
        """String depends_on is converted to list."""
        executor = ParallelExecutor()
        steps = [
            {"id": "a"},
            {"id": "b", "depends_on": "a"},
        ]
        deps = executor.analyze_dependencies(steps)
        assert deps["b"] == ["a"]


# ---------------------------------------------------------------------------
# find_parallel_groups
# ---------------------------------------------------------------------------


class TestFindParallelGroups:
    """Tests for ParallelExecutor.find_parallel_groups."""

    def test_all_independent(self):
        executor = ParallelExecutor()
        deps = {"a": [], "b": [], "c": []}
        groups = executor.find_parallel_groups(deps)
        assert len(groups) == 1
        assert groups[0] == {"a", "b", "c"}

    def test_linear_chain(self):
        executor = ParallelExecutor()
        deps = {"a": [], "b": ["a"], "c": ["b"]}
        groups = executor.find_parallel_groups(deps)
        assert len(groups) == 3
        assert groups[0] == {"a"}
        assert groups[1] == {"b"}
        assert groups[2] == {"c"}

    def test_diamond(self):
        """Diamond pattern: a -> b,c -> d."""
        executor = ParallelExecutor()
        deps = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
        groups = executor.find_parallel_groups(deps)
        assert len(groups) == 3
        assert groups[0] == {"a"}
        assert groups[1] == {"b", "c"}
        assert groups[2] == {"d"}

    def test_circular_dependency_raises(self):
        executor = ParallelExecutor()
        deps = {"a": ["b"], "b": ["a"]}
        with pytest.raises(ValueError, match="[Cc]ircular"):
            executor.find_parallel_groups(deps)


# ---------------------------------------------------------------------------
# _get_ready_tasks
# ---------------------------------------------------------------------------


class TestGetReadyTasks:
    """Tests for ParallelExecutor._get_ready_tasks."""

    def test_no_deps_all_ready(self):
        executor = ParallelExecutor()
        t1 = ParallelTask(id="t1", step_id="s1", handler=_ok_handler)
        t2 = ParallelTask(id="t2", step_id="s2", handler=_ok_handler)
        ready = executor._get_ready_tasks({"t1": t1, "t2": t2}, set())
        assert len(ready) == 2

    def test_deps_not_met(self):
        executor = ParallelExecutor()
        t1 = ParallelTask(id="t1", step_id="s1", handler=_ok_handler, dependencies=["dep1"])
        ready = executor._get_ready_tasks({"t1": t1}, set())
        assert len(ready) == 0

    def test_deps_met(self):
        executor = ParallelExecutor()
        t1 = ParallelTask(id="t1", step_id="s1", handler=_ok_handler, dependencies=["dep1"])
        ready = executor._get_ready_tasks({"t1": t1}, {"dep1"})
        assert len(ready) == 1


# ---------------------------------------------------------------------------
# execute_parallel — THREADING strategy
# ---------------------------------------------------------------------------


class TestExecuteThreaded:
    """Tests for threaded parallel execution."""

    def test_single_task(self):
        executor = ParallelExecutor(strategy=ParallelStrategy.THREADING)
        tasks = [ParallelTask(id="t1", step_id="s1", handler=_ok_handler)]
        result = executor.execute_parallel(tasks)
        assert result.all_succeeded
        assert result.get_result("t1") == "ok"
        assert "t1" in result.successful

    def test_multiple_independent_tasks(self):
        executor = ParallelExecutor(strategy=ParallelStrategy.THREADING)
        tasks = [
            ParallelTask(id="t1", step_id="s1", handler=_ok_handler),
            ParallelTask(id="t2", step_id="s2", handler=_ok_handler),
            ParallelTask(id="t3", step_id="s3", handler=_ok_handler),
        ]
        result = executor.execute_parallel(tasks)
        assert result.all_succeeded
        assert len(result.successful) == 3

    def test_task_with_args(self):
        executor = ParallelExecutor(strategy=ParallelStrategy.THREADING)
        tasks = [
            ParallelTask(id="t1", step_id="s1", handler=_add_handler, args=(3, 4)),
        ]
        result = executor.execute_parallel(tasks)
        assert result.get_result("t1") == 7

    def test_task_failure(self):
        executor = ParallelExecutor(strategy=ParallelStrategy.THREADING)
        tasks = [ParallelTask(id="t1", step_id="s1", handler=_fail_handler)]
        result = executor.execute_parallel(tasks)
        assert not result.all_succeeded
        assert "t1" in result.failed
        assert isinstance(result.get_error("t1"), RuntimeError)

    def test_dependency_ordering(self):
        """Tasks with dependencies run after their dependencies."""
        execution_order = []

        def handler_a():
            execution_order.append("a")
            return "a"

        def handler_b():
            execution_order.append("b")
            return "b"

        executor = ParallelExecutor(strategy=ParallelStrategy.THREADING, max_workers=1)
        tasks = [
            ParallelTask(id="a", step_id="sa", handler=handler_a),
            ParallelTask(id="b", step_id="sb", handler=handler_b, dependencies=["a"]),
        ]
        result = executor.execute_parallel(tasks)
        assert result.all_succeeded
        assert execution_order == ["a", "b"]

    def test_fail_fast(self):
        """fail_fast=True cancels remaining tasks on first failure."""
        executor = ParallelExecutor(strategy=ParallelStrategy.THREADING, max_workers=1)
        tasks = [
            ParallelTask(id="t1", step_id="s1", handler=_fail_handler),
            ParallelTask(id="t2", step_id="s2", handler=_ok_handler, dependencies=["t1"]),
        ]
        result = executor.execute_parallel(tasks, fail_fast=True)
        assert not result.all_succeeded
        assert "t1" in result.failed
        # t2 should be cancelled since t1 failed and fail_fast is on
        assert "t2" in result.cancelled or "t2" not in result.successful

    def test_on_complete_callback(self):
        completed = []
        executor = ParallelExecutor(strategy=ParallelStrategy.THREADING)
        tasks = [ParallelTask(id="t1", step_id="s1", handler=_ok_handler)]
        result = executor.execute_parallel(
            tasks, on_complete=lambda tid, res: completed.append((tid, res))
        )
        assert result.all_succeeded
        assert ("t1", "ok") in completed

    def test_on_error_callback(self):
        errors = []
        executor = ParallelExecutor(strategy=ParallelStrategy.THREADING)
        tasks = [ParallelTask(id="t1", step_id="s1", handler=_fail_handler)]
        _ = executor.execute_parallel(
            tasks, on_error=lambda tid, err: errors.append((tid, str(err)))
        )
        assert len(errors) == 1
        assert errors[0][0] == "t1"
        assert "boom" in errors[0][1]

    def test_total_duration_tracked(self):
        executor = ParallelExecutor(strategy=ParallelStrategy.THREADING)
        tasks = [ParallelTask(id="t1", step_id="s1", handler=_ok_handler)]
        result = executor.execute_parallel(tasks)
        assert result.total_duration_ms >= 0

    def test_timestamps_set(self):
        executor = ParallelExecutor(strategy=ParallelStrategy.THREADING)
        tasks = [ParallelTask(id="t1", step_id="s1", handler=_ok_handler)]
        result = executor.execute_parallel(tasks)
        task = result.tasks["t1"]
        assert task.started_at is not None
        assert task.completed_at is not None


# ---------------------------------------------------------------------------
# execute_parallel — ASYNCIO strategy
# ---------------------------------------------------------------------------


class TestExecuteAsync:
    """Tests for asyncio parallel execution."""

    def test_single_task(self):
        executor = ParallelExecutor(strategy=ParallelStrategy.ASYNCIO)
        tasks = [ParallelTask(id="t1", step_id="s1", handler=_ok_handler)]
        result = executor.execute_parallel(tasks)
        assert result.all_succeeded
        assert result.get_result("t1") == "ok"

    def test_async_handler(self):
        executor = ParallelExecutor(strategy=ParallelStrategy.ASYNCIO)
        tasks = [ParallelTask(id="t1", step_id="s1", handler=_async_ok_handler)]
        result = executor.execute_parallel(tasks)
        assert result.all_succeeded
        assert result.get_result("t1") == "async-ok"

    def test_multiple_tasks(self):
        executor = ParallelExecutor(strategy=ParallelStrategy.ASYNCIO)
        tasks = [
            ParallelTask(id="t1", step_id="s1", handler=_ok_handler),
            ParallelTask(id="t2", step_id="s2", handler=_ok_handler),
        ]
        result = executor.execute_parallel(tasks)
        assert result.all_succeeded
        assert len(result.successful) == 2

    def test_failure(self):
        executor = ParallelExecutor(strategy=ParallelStrategy.ASYNCIO)
        tasks = [ParallelTask(id="t1", step_id="s1", handler=_fail_handler)]
        result = executor.execute_parallel(tasks)
        assert not result.all_succeeded
        assert "t1" in result.failed

    def test_dependency_ordering(self):
        execution_order = []

        def handler_a():
            execution_order.append("a")
            return "a"

        def handler_b():
            execution_order.append("b")
            return "b"

        executor = ParallelExecutor(strategy=ParallelStrategy.ASYNCIO, max_workers=1)
        tasks = [
            ParallelTask(id="a", step_id="sa", handler=handler_a),
            ParallelTask(id="b", step_id="sb", handler=handler_b, dependencies=["a"]),
        ]
        result = executor.execute_parallel(tasks)
        assert result.all_succeeded
        assert execution_order == ["a", "b"]


# ---------------------------------------------------------------------------
# execute_parallel — PROCESS strategy
# ---------------------------------------------------------------------------


class TestExecuteProcess:
    """Tests for process pool parallel execution."""

    def test_single_task(self):
        """Process pool can execute a simple picklable handler."""
        executor = ParallelExecutor(strategy=ParallelStrategy.PROCESS, max_workers=1)
        tasks = [ParallelTask(id="t1", step_id="s1", handler=_ok_handler)]
        result = executor.execute_parallel(tasks)
        assert result.all_succeeded
        assert result.get_result("t1") == "ok"

    def test_task_with_args(self):
        executor = ParallelExecutor(strategy=ParallelStrategy.PROCESS, max_workers=1)
        tasks = [
            ParallelTask(id="t1", step_id="s1", handler=_add_handler, args=(10, 20)),
        ]
        result = executor.execute_parallel(tasks)
        assert result.get_result("t1") == 30

    def test_failure(self):
        executor = ParallelExecutor(strategy=ParallelStrategy.PROCESS, max_workers=1)
        tasks = [ParallelTask(id="t1", step_id="s1", handler=_fail_handler)]
        result = executor.execute_parallel(tasks)
        assert not result.all_succeeded
        assert "t1" in result.failed


# ---------------------------------------------------------------------------
# _cancel_pending_tasks
# ---------------------------------------------------------------------------


class TestCancelPendingTasks:
    """Tests for the _cancel_pending_tasks helper."""

    def test_marks_pending_as_cancelled(self):
        executor = ParallelExecutor()
        result = ParallelResult()
        t1 = ParallelTask(id="t1", step_id="s1", handler=_ok_handler)
        t2 = ParallelTask(id="t2", step_id="s2", handler=_ok_handler)
        pending = {"t1": t1, "t2": t2}
        executor._cancel_pending_tasks(pending, result)
        assert "t1" in result.cancelled
        assert "t2" in result.cancelled
        assert "t1" in result.tasks
        assert "t2" in result.tasks

    def test_skips_already_tracked(self):
        executor = ParallelExecutor()
        t1 = ParallelTask(id="t1", step_id="s1", handler=_ok_handler)
        result = ParallelResult(tasks={"t1": t1})
        pending = {"t1": t1}
        executor._cancel_pending_tasks(pending, result)
        # Should not add to cancelled since it's already tracked
        assert "t1" not in result.cancelled


# ---------------------------------------------------------------------------
# Deadlock detection
# ---------------------------------------------------------------------------


class TestDeadlockDetection:
    """Tests for deadlock detection during execution."""

    def test_deadlock_raises_threaded(self):
        executor = ParallelExecutor(strategy=ParallelStrategy.THREADING)
        # Tasks with unresolvable circular deps
        tasks = [
            ParallelTask(id="a", step_id="sa", handler=_ok_handler, dependencies=["b"]),
            ParallelTask(id="b", step_id="sb", handler=_ok_handler, dependencies=["a"]),
        ]
        with pytest.raises(ValueError, match="[Dd]eadlock"):
            executor.execute_parallel(tasks)

    def test_deadlock_raises_process(self):
        executor = ParallelExecutor(strategy=ParallelStrategy.PROCESS, max_workers=1)
        tasks = [
            ParallelTask(id="a", step_id="sa", handler=_ok_handler, dependencies=["b"]),
            ParallelTask(id="b", step_id="sb", handler=_ok_handler, dependencies=["a"]),
        ]
        with pytest.raises(ValueError, match="[Dd]eadlock"):
            executor.execute_parallel(tasks)


# ---------------------------------------------------------------------------
# execute_steps_parallel convenience function
# ---------------------------------------------------------------------------


class TestExecuteStepsParallel:
    """Tests for the execute_steps_parallel convenience function."""

    def test_basic(self):
        steps = [
            {"id": "a"},
            {"id": "b"},
        ]

        def handler(step):
            return {"step_id": step["id"]}

        result = execute_steps_parallel(steps, handler)
        assert result.all_succeeded
        assert len(result.successful) == 2

    def test_with_dependencies(self):
        execution_order = []

        def handler(step):
            execution_order.append(step["id"])
            return {"step_id": step["id"]}

        steps = [
            {"id": "a"},
            {"id": "b", "depends_on": ["a"]},
        ]
        result = execute_steps_parallel(steps, handler, max_workers=1)
        assert result.all_succeeded
        assert execution_order.index("a") < execution_order.index("b")

    def test_with_strategy(self):
        steps = [{"id": "a"}]

        def handler(step):
            return step["id"]

        result = execute_steps_parallel(steps, handler, strategy=ParallelStrategy.THREADING)
        assert result.all_succeeded


# ---------------------------------------------------------------------------
# ParallelStrategy enum
# ---------------------------------------------------------------------------


class TestParallelStrategy:
    """Tests for the ParallelStrategy enum."""

    def test_values(self):
        assert ParallelStrategy.THREADING.value == "threading"
        assert ParallelStrategy.ASYNCIO.value == "asyncio"
        assert ParallelStrategy.PROCESS.value == "process"
