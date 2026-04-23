"""Tests for the rate_limited_executor livelock fix.

Previously lines 493-499 of `execute_parallel_rate_limited` had two
defensive `continue` branches that skipped task booking, causing the
outer `while pending` loop to spin forever if:
  - `_run_task_with_rate_limit` returned None, or
  - it returned a task_id that didn't match any pending task.

These tests now pass with the fix in place — they would have hung
indefinitely against the pre-fix code.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from animus_forge.workflow.parallel import ParallelTask
from animus_forge.workflow.rate_limited_executor import RateLimitedParallelExecutor


class TestLivelockFixNoneItem:
    """Line 493-494 fix: orphan async_task is booked when coro returns None."""

    @pytest.mark.asyncio
    async def test_none_result_is_booked_as_failed(self):
        ex = RateLimitedParallelExecutor(max_workers=2, timeout=5.0)

        async def none_run(task, semaphores):
            return None

        ex._run_task_with_rate_limit = none_run

        async def handler(**kwargs):
            return {}

        t1 = ParallelTask(id="t1", step_id="s1", handler=handler, kwargs={})

        # Pre-fix this hung forever. With fix, the orphan is booked as
        # failed and the outer loop terminates.
        result = await ex.execute_parallel_rate_limited(tasks=[t1])

        assert "t1" in result.tasks
        assert "t1" in result.failed
        assert isinstance(result.tasks["t1"].error, RuntimeError)
        assert "None" in str(result.tasks["t1"].error)

    @pytest.mark.asyncio
    async def test_none_result_fires_on_error_callback(self):
        ex = RateLimitedParallelExecutor(max_workers=1, timeout=5.0)

        async def none_run(task, semaphores):
            return None

        ex._run_task_with_rate_limit = none_run

        errors = []

        def on_error(task_id, exc):
            errors.append((task_id, type(exc).__name__))

        async def handler(**kwargs):
            return {}

        t1 = ParallelTask(id="t1", step_id="s1", handler=handler, kwargs={})
        await ex.execute_parallel_rate_limited(tasks=[t1], on_error=on_error)

        assert errors == [("t1", "RuntimeError")]


class TestLivelockFixMismatchedTaskId:
    """Line 497-499 fix: orphan async_task is booked when the returned
    task_id doesn't match any pending task."""

    @pytest.mark.asyncio
    async def test_mismatched_id_books_real_task_as_failed(self):
        ex = RateLimitedParallelExecutor(max_workers=1, timeout=5.0)

        async def ghost_run(task, semaphores):
            # Return a task_id that doesn't match — simulates a handler
            # that mangles ids or a stale retry result leaking through.
            return ("ghost-id-not-in-pending", "result", None)

        ex._run_task_with_rate_limit = ghost_run

        async def handler(**kwargs):
            return {}

        t1 = ParallelTask(id="t1", step_id="s1", handler=handler, kwargs={})

        # Pre-fix this hung forever. With fix, the real pending task (t1)
        # gets booked as failed with an "id mismatch" error.
        result = await ex.execute_parallel_rate_limited(tasks=[t1])

        assert "t1" in result.tasks
        assert "t1" in result.failed
        assert isinstance(result.tasks["t1"].error, RuntimeError)
        assert "mismatch" in str(result.tasks["t1"].error).lower()
        # The ghost id never gets a real task record but is tracked as
        # completed so the outer loop can exit.
        assert "ghost-id-not-in-pending" not in result.tasks
