"""Parallel group execution logic for workflow executor.

Mixin class providing auto-parallel analysis and parallel group execution.
"""

from __future__ import annotations

import asyncio
import logging
import time

from animus_forge.monitoring.parallel_tracker import (
    ParallelPatternType,
    get_parallel_tracker,
)

from .auto_parallel import build_dependency_graph, find_parallel_groups
from .executor_results import StepResult, StepStatus
from .loader import StepConfig
from .parallel import ParallelExecutor, ParallelStrategy, ParallelTask
from .rate_limited_executor import RateLimitedParallelExecutor

logger = logging.getLogger(__name__)


class ParallelGroupMixin:
    """Mixin providing parallel group execution.

    Expects the following attributes from the host class:
    - _execute_step(step, workflow_id) -> StepResult
    - _execute_step_async(step, workflow_id) -> StepResult
    - _check_budget_exceeded(step, result) -> bool
    - _record_step_completion(step, step_result, result) -> None
    - _handle_step_failure(step, step_result, result, workflow_id) -> str
    - _handle_step_failure_async(step, step_result, result, workflow_id) -> str
    - _store_step_outputs(step, step_result) -> None
    """

    def _execute_with_auto_parallel(
        self,
        workflow,
        start_index: int,
        workflow_id: str | None,
        result,
    ) -> None:
        """Execute workflow with auto-parallel optimization.

        Analyzes step dependencies and executes independent steps
        concurrently for improved performance.

        Args:
            workflow: WorkflowConfig to execute
            start_index: Index to start from
            workflow_id: Current workflow ID
            result: ExecutionResult to update
        """
        steps = workflow.steps[start_index:]
        if not steps:
            result.status = "success"
            return

        # Build dependency graph and find parallel groups
        graph = build_dependency_graph(steps)
        groups = find_parallel_groups(graph)

        max_workers = workflow.settings.auto_parallel_max_workers
        step_map = {step.id: step for step in steps}
        completed: set[str] = set()

        logger.info(
            f"Auto-parallel: {len(steps)} steps in {len(groups)} groups (max_workers={max_workers})"
        )

        for group in groups:
            group_steps = [step_map[step_id] for step_id in group.step_ids]

            # Check budget for all steps in group
            for step in group_steps:
                if self._check_budget_exceeded(step, result):
                    return

            if len(group_steps) == 1:
                # Single step - execute directly
                step = group_steps[0]
                step_result = self._execute_step(step, workflow_id)
                self._record_step_completion(step, step_result, result)

                if step_result.status == StepStatus.FAILED:
                    action = self._handle_step_failure(step, step_result, result, workflow_id)
                    if action == "abort":
                        return
                    if action != "skip":
                        self._store_step_outputs(step, step_result)
                else:
                    self._store_step_outputs(step, step_result)

                completed.add(step.id)
            else:
                # Multiple steps - execute in parallel
                self._execute_parallel_group(group_steps, workflow_id, result, max_workers)
                completed.update(step.id for step in group_steps)

                # Check if any step in the group failed fatally
                if result.status == "failed":
                    return

        result.status = "success"

    def _execute_parallel_group(
        self,
        steps: list[StepConfig],
        workflow_id: str | None,
        result,
        max_workers: int,
    ) -> None:
        """Execute a group of steps in parallel.

        Args:
            steps: Steps to execute concurrently
            workflow_id: Current workflow ID
            result: ExecutionResult to update
            max_workers: Maximum concurrent workers
        """
        logger.debug(
            f"Executing parallel group: {[s.id for s in steps]} (max_workers={max_workers})"
        )

        # Start parallel execution tracking
        tracker = get_parallel_tracker()
        group_id = "_".join(s.id for s in steps[:3])  # First 3 step IDs
        execution_id = f"parallel_group_{group_id}_{int(time.time() * 1000)}"
        tracker.start_execution(
            execution_id=execution_id,
            pattern_type=ParallelPatternType.PARALLEL_GROUP,
            step_id=group_id,
            total_items=len(steps),
            max_concurrent=max_workers,
            workflow_id=workflow_id,
        )

        # Detect AI step types for rate limiting
        ai_step_types = {"claude_code", "openai"}
        has_ai_steps = any(step.type in ai_step_types for step in steps)
        max_timeout = max(step.timeout_seconds for step in steps)

        if has_ai_steps:
            # Use rate-limited executor with adaptive configuration
            executor = RateLimitedParallelExecutor(
                strategy=ParallelStrategy.ASYNCIO,
                max_workers=max_workers,
                timeout=max_timeout,
                adaptive=True,  # Enable adaptive rate limiting
            )
        else:
            executor = ParallelExecutor(
                strategy=ParallelStrategy.THREADING,
                max_workers=max_workers,
                timeout=max_timeout,
            )

        step_results: dict[str, StepResult] = {}

        def make_handler(step: StepConfig, idx: int):
            def handler(**kwargs):
                # Track branch start
                tracker.start_branch(execution_id, step.id, idx, step.id)
                try:
                    step_result = self._execute_step(step, workflow_id)
                    tokens = step_result.tokens_used if step_result else 0
                    if step_result and step_result.status == StepStatus.FAILED:
                        tracker.fail_branch(
                            execution_id, step.id, step_result.error or "Unknown error"
                        )
                    else:
                        tracker.complete_branch(execution_id, step.id, tokens)
                    return step_result
                except Exception as e:
                    tracker.fail_branch(execution_id, step.id, str(e))
                    raise

            return handler

        tasks = [
            ParallelTask(
                id=step.id,
                step_id=step.id,
                handler=make_handler(step, idx),
                kwargs={"step_type": step.type},
            )
            for idx, step in enumerate(steps)
        ]

        def on_complete(task_id: str, step_result: StepResult):
            step_results[task_id] = step_result

        def on_error(task_id: str, error: Exception):
            step_results[task_id] = StepResult(
                step_id=task_id,
                status=StepStatus.FAILED,
                error=str(error),
            )

        executor.execute_parallel(
            tasks=tasks,
            on_complete=on_complete,
            on_error=on_error,
            fail_fast=False,
        )

        # Capture and track rate limit stats for AI executors
        if has_ai_steps and hasattr(executor, "get_provider_stats"):
            provider_stats = executor.get_provider_stats()
            for provider, stats in provider_stats.items():
                if stats.get("total_429s", 0) > 0 or stats.get("is_throttled", False):
                    tracker.update_rate_limit_state(
                        provider=provider,
                        current_limit=stats.get("current_limit", 0),
                        base_limit=stats.get("base_limit", 0),
                        total_429s=stats.get("total_429s", 0),
                        is_throttled=stats.get("is_throttled", False),
                    )

        # Process results
        step_map = {step.id: step for step in steps}
        abort = False
        has_failures = False

        for step_id, step_result in step_results.items():
            step = step_map[step_id]
            self._record_step_completion(step, step_result, result)

            if step_result.status == StepStatus.FAILED:
                has_failures = True
                action = self._handle_step_failure(step, step_result, result, workflow_id)
                if action == "abort":
                    abort = True
                elif action != "skip":
                    self._store_step_outputs(step, step_result)
            else:
                self._store_step_outputs(step, step_result)

        # Complete execution tracking
        if has_failures:
            tracker.fail_execution(execution_id)
        else:
            tracker.complete_execution(execution_id)

        if abort:
            result.status = "failed"

    async def _execute_with_auto_parallel_async(
        self,
        workflow,
        start_index: int,
        workflow_id: str | None,
        result,
    ) -> None:
        """Execute workflow with auto-parallel optimization (async version)."""
        steps = workflow.steps[start_index:]
        if not steps:
            result.status = "success"
            return

        graph = build_dependency_graph(steps)
        groups = find_parallel_groups(graph)

        max_workers = workflow.settings.auto_parallel_max_workers
        step_map = {step.id: step for step in steps}

        logger.info(
            f"Auto-parallel async: {len(steps)} steps in {len(groups)} groups "
            f"(max_workers={max_workers})"
        )

        for group in groups:
            group_steps = [step_map[step_id] for step_id in group.step_ids]

            for step in group_steps:
                if self._check_budget_exceeded(step, result):
                    return

            if len(group_steps) == 1:
                step = group_steps[0]
                step_result = await self._execute_step_async(step, workflow_id)
                self._record_step_completion(step, step_result, result)

                if step_result.status == StepStatus.FAILED:
                    action = await self._handle_step_failure_async(
                        step, step_result, result, workflow_id
                    )
                    if action == "abort":
                        return
                    if action != "skip":
                        self._store_step_outputs(step, step_result)
                else:
                    self._store_step_outputs(step, step_result)
            else:
                await self._execute_parallel_group_async(
                    group_steps, workflow_id, result, max_workers
                )
                if result.status == "failed":
                    return

        result.status = "success"

    async def _execute_parallel_group_async(
        self,
        steps: list[StepConfig],
        workflow_id: str | None,
        result,
        max_workers: int,
    ) -> None:
        """Execute a group of steps in parallel (async version)."""
        logger.debug(
            f"Executing parallel group async: {[s.id for s in steps]} (max_workers={max_workers})"
        )

        semaphore = asyncio.Semaphore(max_workers)
        step_results: dict[str, StepResult] = {}

        async def execute_step(step: StepConfig):
            async with semaphore:
                return step.id, await self._execute_step_async(step, workflow_id)

        tasks = [execute_step(step) for step in steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for item in results:
            if isinstance(item, Exception):
                continue
            step_id, step_result = item
            step_results[step_id] = step_result

        step_map = {step.id: step for step in steps}
        abort = False

        for step_id, step_result in step_results.items():
            step = step_map[step_id]
            self._record_step_completion(step, step_result, result)

            if step_result.status == StepStatus.FAILED:
                action = await self._handle_step_failure_async(
                    step, step_result, result, workflow_id
                )
                if action == "abort":
                    abort = True
                elif action != "skip":
                    self._store_step_outputs(step, step_result)
            else:
                self._store_step_outputs(step, step_result)

        if abort:
            result.status = "failed"
