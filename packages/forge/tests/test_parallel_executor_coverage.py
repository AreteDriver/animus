"""Coverage for parallel.py gaps on 2c8cde2.

Targets:
- Line 309: threaded deadlock raise
- Lines 362-363: _run_single_task CancelledError branch
- Lines 383-389: _cancel_async_tasks body
"""

from __future__ import annotations

import asyncio
import sys

import pytest

sys.path.insert(0, "src")

from animus_forge.workflow.parallel import (
    ParallelExecutor,
    ParallelResult,
    ParallelStrategy,
    ParallelTask,
)


def _task(task_id: str, handler, **kwargs):
    return ParallelTask(id=task_id, step_id=task_id, handler=handler, kwargs=kwargs)


class TestThreadedDeadlock:
    def test_unresolvable_dependency_raises(self):
        ex = ParallelExecutor(strategy=ParallelStrategy.THREADING, max_workers=2)

        def handler(**kwargs):
            return {}

        # t2 depends on t1, but t1 isn't in the list → deadlock on first iter.
        t2 = ParallelTask(
            id="t2",
            step_id="t2",
            handler=handler,
            kwargs={},
            dependencies=["missing-t1"],
        )
        with pytest.raises(ValueError, match="Deadlock"):
            ex.execute_parallel([t2])


class TestRunSingleTaskCancelled:
    """Lines 362-363: CancelledError path in _run_single_task."""

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        ex = ParallelExecutor(strategy=ParallelStrategy.ASYNCIO, max_workers=1)

        async def slow_handler(**kwargs):
            await asyncio.sleep(10)
            return {}

        task = _task("t1", slow_handler)

        # Start the coro, then cancel it immediately.
        async_task = asyncio.create_task(ex._run_single_task(task))
        await asyncio.sleep(0)  # let it start
        async_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await async_task
        # completed_at stamped even on cancel.
        assert task.completed_at is not None


class TestCancelAsyncTasks:
    """Lines 383-389: _cancel_async_tasks cancels everything except the
    excluded task and records them in result.cancelled."""

    @pytest.mark.asyncio
    async def test_cancels_other_tasks(self):
        ex = ParallelExecutor(strategy=ParallelStrategy.ASYNCIO, max_workers=4)

        async def slow_handler(**kwargs):
            await asyncio.sleep(10)

        t1 = _task("t1", slow_handler)
        t2 = _task("t2", slow_handler)
        t3 = _task("t3", slow_handler)

        # Build a dict of live asyncio tasks → ParallelTask.
        async_tasks = {}
        for pt in (t1, t2, t3):

            async def _run():
                await asyncio.sleep(10)

            async_tasks[asyncio.create_task(_run())] = pt

        pending = {t1.id: t1, t2.id: t2, t3.id: t3}
        result = ParallelResult()

        # Exclude t1 — cancel t2 and t3.
        ex._cancel_async_tasks(async_tasks, exclude_id="t1", pending=pending, result=result)

        # t1 stayed in pending, t2/t3 removed.
        assert "t1" in pending
        assert "t2" not in pending
        assert "t3" not in pending
        assert set(result.cancelled) == {"t2", "t3"}
        assert "t2" in result.tasks
        assert "t3" in result.tasks

        # Clean up the still-running asyncio tasks to avoid warnings.
        for async_task in async_tasks:
            if not async_task.done():
                async_task.cancel()
        await asyncio.gather(*async_tasks.keys(), return_exceptions=True)
