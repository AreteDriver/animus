"""Async sub-agent lifecycle manager.

Provides non-blocking agent spawning with run IDs, timeouts,
cascade cancellation, and concurrency limits. Replaces the
sequential delegation loop with parallel execution when agents
are independent.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from animus_forge.agents.agent_config import AgentConfig, get_agent_config

logger = logging.getLogger(__name__)


class RunStatus(StrEnum):
    """Status of a sub-agent run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass
class AgentRun:
    """Tracks a single sub-agent execution.

    Attributes:
        run_id: Unique identifier for this run.
        agent: Agent role name.
        task: Task description.
        config: Agent configuration.
        status: Current run status.
        result: Agent output (set on completion).
        error: Error message (set on failure).
        started_at: Epoch time when execution started.
        completed_at: Epoch time when execution finished.
        parent_id: Run ID of the spawning agent (for cascade).
        children: Run IDs of agents spawned by this one.
        task_handle: asyncio.Task reference for cancellation.
    """

    run_id: str
    agent: str
    task: str
    config: AgentConfig
    status: RunStatus = RunStatus.PENDING
    result: str | None = None
    error: str | None = None
    started_at: float = 0.0
    completed_at: float = 0.0
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    task_handle: asyncio.Task | None = field(default=None, repr=False)

    @property
    def duration_ms(self) -> int:
        """Execution duration in milliseconds."""
        if not self.started_at:
            return 0
        end = self.completed_at or time.time()
        return int((end - self.started_at) * 1000)

    def to_dict(self) -> dict[str, Any]:
        """Serialize run state (excludes task_handle)."""
        return {
            "run_id": self.run_id,
            "agent": self.agent,
            "task": self.task[:200],
            "status": self.status.value,
            "result": self.result[:500] if self.result else None,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "parent_id": self.parent_id,
            "children": self.children,
        }


class SubAgentManager:
    """Manages async sub-agent lifecycle.

    Provides:
    - Non-blocking spawn with run IDs
    - Timeout enforcement per agent
    - Concurrency limits (global lane)
    - Cascade cancellation (stop parent → stop children)
    - Run tracking and status queries

    Args:
        max_concurrent: Maximum agents running simultaneously.
        max_depth: Maximum nesting depth for sub-agent spawning.
        agent_configs: Optional per-role config overrides.
    """

    def __init__(
        self,
        max_concurrent: int = 8,
        max_depth: int = 3,
        agent_configs: dict[str, AgentConfig] | None = None,
    ):
        self._max_concurrent = max_concurrent
        self._max_depth = max_depth
        self._agent_configs = agent_configs
        self._runs: dict[str, AgentRun] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0

    @property
    def active_count(self) -> int:
        """Number of currently running agents."""
        return self._active_count

    @property
    def runs(self) -> dict[str, AgentRun]:
        """All tracked runs."""
        return self._runs

    def get_run(self, run_id: str) -> AgentRun | None:
        """Get run by ID."""
        return self._runs.get(run_id)

    def list_runs(self, status: RunStatus | None = None) -> list[AgentRun]:
        """List runs, optionally filtered by status."""
        runs = list(self._runs.values())
        if status is not None:
            runs = [r for r in runs if r.status == status]
        return runs

    async def spawn(
        self,
        agent: str,
        task: str,
        execute_fn: Any,
        parent_id: str | None = None,
        config_override: AgentConfig | None = None,
    ) -> AgentRun:
        """Spawn a sub-agent asynchronously.

        Returns immediately with an AgentRun in PENDING status.
        The agent executes in the background, respecting concurrency
        limits and timeouts.

        Args:
            agent: Agent role name.
            task: Task for the agent to perform.
            execute_fn: Async callable(agent, task, config) -> str.
                The actual agent execution function.
            parent_id: Optional parent run ID (for nesting).
            config_override: Optional config override for this specific run.

        Returns:
            AgentRun with run_id for tracking.

        Raises:
            RuntimeError: If max depth exceeded.
        """
        # Check nesting depth
        depth = self._get_depth(parent_id)
        if depth >= self._max_depth:
            raise RuntimeError(
                f"Max sub-agent depth ({self._max_depth}) exceeded. Current depth: {depth}"
            )

        config = config_override or get_agent_config(agent, self._agent_configs)
        run_id = f"run-{uuid.uuid4().hex[:12]}"

        run = AgentRun(
            run_id=run_id,
            agent=agent,
            task=task,
            config=config,
            parent_id=parent_id,
        )
        self._runs[run_id] = run

        # Register as child of parent
        if parent_id and parent_id in self._runs:
            self._runs[parent_id].children.append(run_id)

        # Launch in background with semaphore gating
        run.task_handle = asyncio.create_task(
            self._execute_run(run, execute_fn),
            name=f"subagent-{agent}-{run_id[:8]}",
        )

        return run

    async def spawn_batch(
        self,
        delegations: list[dict[str, str]],
        execute_fn: Any,
        parent_id: str | None = None,
    ) -> list[AgentRun]:
        """Spawn multiple agents and wait for all to complete.

        Agents run concurrently up to the concurrency limit.
        Results are gathered with exception handling per agent.

        Args:
            delegations: List of {"agent": str, "task": str} dicts.
            execute_fn: Async callable(agent, task, config) -> str.
            parent_id: Optional parent run ID.

        Returns:
            List of AgentRun objects (completed or failed).
        """
        runs = []
        for d in delegations:
            agent = d.get("agent", "unknown")
            task = d.get("task", "")
            run = await self.spawn(agent, task, execute_fn, parent_id=parent_id)
            runs.append(run)

        # Wait for all to complete
        tasks = [r.task_handle for r in runs if r.task_handle is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return runs

    async def cancel(self, run_id: str, cascade: bool = True) -> bool:
        """Cancel a running agent.

        Args:
            run_id: Run to cancel.
            cascade: If True, also cancel all children.

        Returns:
            True if cancellation was initiated.
        """
        run = self._runs.get(run_id)
        if run is None:
            return False

        if run.status not in (RunStatus.PENDING, RunStatus.RUNNING):
            return False

        # Cancel children first (depth-first)
        if cascade:
            for child_id in run.children:
                await self.cancel(child_id, cascade=True)

        # Cancel the task
        if run.task_handle and not run.task_handle.done():
            run.task_handle.cancel()

        run.status = RunStatus.CANCELLED
        run.completed_at = time.time()
        logger.info("Cancelled agent run %s (%s)", run_id, run.agent)
        return True

    async def cancel_all(self) -> int:
        """Cancel all active runs. Returns count cancelled."""
        count = 0
        for run_id, run in list(self._runs.items()):
            if run.status in (RunStatus.PENDING, RunStatus.RUNNING):
                await self.cancel(run_id, cascade=False)
                count += 1
        return count

    def cleanup(self, max_age_seconds: float = 3600.0) -> int:
        """Remove completed runs older than max_age.

        Args:
            max_age_seconds: Max age for completed runs.

        Returns:
            Number of runs removed.
        """
        cutoff = time.time() - max_age_seconds
        to_remove = [
            run_id
            for run_id, run in self._runs.items()
            if run.status
            in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.TIMED_OUT)
            and run.completed_at < cutoff
        ]
        for run_id in to_remove:
            del self._runs[run_id]
        return len(to_remove)

    def _get_depth(self, parent_id: str | None) -> int:
        """Calculate nesting depth from parent chain."""
        depth = 0
        current = parent_id
        while current and current in self._runs:
            depth += 1
            current = self._runs[current].parent_id
        return depth

    async def _execute_run(self, run: AgentRun, execute_fn: Any) -> None:
        """Execute an agent run with semaphore gating and timeout."""
        async with self._semaphore:
            run.status = RunStatus.RUNNING
            run.started_at = time.time()
            self._active_count += 1

            try:
                result = await asyncio.wait_for(
                    execute_fn(run.agent, run.task, run.config),
                    timeout=run.config.timeout_seconds,
                )

                # Truncate if needed
                if len(result) > run.config.max_output_chars:
                    result = result[: run.config.max_output_chars] + "\n[truncated]"

                run.result = result
                run.status = RunStatus.COMPLETED
                logger.debug(
                    "Agent %s completed in %dms (run %s)",
                    run.agent,
                    run.duration_ms,
                    run.run_id,
                )

            except TimeoutError:
                run.status = RunStatus.TIMED_OUT
                run.error = f"Agent {run.agent} timed out after {run.config.timeout_seconds}s"
                logger.warning(run.error)

                # Cancel children on timeout
                for child_id in run.children:
                    await self.cancel(child_id, cascade=True)

            except asyncio.CancelledError:
                run.status = RunStatus.CANCELLED
                run.error = "Cancelled"
                logger.info("Agent %s cancelled (run %s)", run.agent, run.run_id)

            except Exception as e:
                run.status = RunStatus.FAILED
                run.error = str(e)
                logger.error("Agent %s failed: %s (run %s)", run.agent, e, run.run_id)

            finally:
                run.completed_at = time.time()
                self._active_count = max(0, self._active_count - 1)
