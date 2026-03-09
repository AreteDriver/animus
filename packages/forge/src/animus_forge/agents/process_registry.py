"""Unified background process registry.

Provides a single view across all task management systems
(SubAgentManager, JobManager, ScheduleManager) for monitoring,
cancellation, and lifecycle control.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ProcessType(StrEnum):
    """Type of background process."""

    AGENT = "agent"
    JOB = "job"
    SCHEDULE = "schedule"


class ProcessState(StrEnum):
    """Unified process state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ProcessInfo:
    """Unified process information.

    Attributes:
        id: Process identifier (run_id, job_id, or schedule_id).
        type: Process type (agent, job, schedule).
        state: Current state.
        name: Human-readable name.
        started_at: Epoch time when started.
        duration_ms: Duration in milliseconds.
        metadata: Additional process-specific data.
    """

    id: str
    type: ProcessType
    state: ProcessState
    name: str
    started_at: float = 0.0
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "state": self.state.value,
            "name": self.name,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


# Maps from source-specific statuses to unified ProcessState
_AGENT_STATE_MAP = {
    "pending": ProcessState.PENDING,
    "running": ProcessState.RUNNING,
    "completed": ProcessState.COMPLETED,
    "failed": ProcessState.FAILED,
    "cancelled": ProcessState.CANCELLED,
    "timed_out": ProcessState.FAILED,
}

_JOB_STATE_MAP = {
    "pending": ProcessState.PENDING,
    "running": ProcessState.RUNNING,
    "completed": ProcessState.COMPLETED,
    "failed": ProcessState.FAILED,
    "cancelled": ProcessState.CANCELLED,
}

_SCHEDULE_STATE_MAP = {
    "active": ProcessState.RUNNING,
    "paused": ProcessState.PAUSED,
    "disabled": ProcessState.CANCELLED,
}


class ProcessRegistry:
    """Unified registry across all background task systems.

    Aggregates state from SubAgentManager, JobManager, and
    ScheduleManager into a single queryable interface.

    Args:
        subagent_manager: Optional SubAgentManager instance.
        job_manager: Optional JobManager instance.
        schedule_manager: Optional ScheduleManager instance.
    """

    def __init__(
        self,
        subagent_manager: Any | None = None,
        job_manager: Any | None = None,
        schedule_manager: Any | None = None,
    ):
        self._subagent_manager = subagent_manager
        self._job_manager = job_manager
        self._schedule_manager = schedule_manager

    def list_all(
        self,
        state: ProcessState | None = None,
        process_type: ProcessType | None = None,
    ) -> list[ProcessInfo]:
        """List all tracked processes.

        Args:
            state: Optional filter by state.
            process_type: Optional filter by type.

        Returns:
            List of ProcessInfo sorted by started_at (newest first).
        """
        processes = []

        if process_type is None or process_type == ProcessType.AGENT:
            processes.extend(self._collect_agents())

        if process_type is None or process_type == ProcessType.JOB:
            processes.extend(self._collect_jobs())

        if process_type is None or process_type == ProcessType.SCHEDULE:
            processes.extend(self._collect_schedules())

        if state is not None:
            processes = [p for p in processes if p.state == state]

        processes.sort(key=lambda p: p.started_at, reverse=True)
        return processes

    def get(self, process_id: str) -> ProcessInfo | None:
        """Get a specific process by ID.

        Searches across all managers.

        Args:
            process_id: Process identifier.

        Returns:
            ProcessInfo or None if not found.
        """
        # Check agents
        if self._subagent_manager:
            run = self._subagent_manager.get_run(process_id)
            if run:
                return self._agent_to_process(run)

        # Check jobs
        if self._job_manager:
            try:
                job = self._job_manager.get_job(process_id)
                if job:
                    return self._job_to_process(job)
            except Exception:
                logger.debug("Failed to get job %s", process_id, exc_info=True)

        # Check schedules
        if self._schedule_manager:
            try:
                schedule = self._schedule_manager.get_schedule(process_id)
                if schedule:
                    return self._schedule_to_process(schedule)
            except Exception:
                logger.debug("Failed to get schedule %s", process_id, exc_info=True)

        return None

    async def cancel(self, process_id: str) -> bool:
        """Cancel a process by ID.

        Attempts cancellation across all managers.

        Args:
            process_id: Process to cancel.

        Returns:
            True if cancellation was initiated.
        """
        # Try agents first (async cancel)
        if self._subagent_manager:
            result = await self._subagent_manager.cancel(process_id)
            if result:
                return True

        # Try jobs (sync cancel)
        if self._job_manager:
            try:
                result = self._job_manager.cancel_job(process_id)
                if result:
                    return True
            except Exception:
                logger.debug("Failed to cancel job %s", process_id, exc_info=True)

        # Try schedules (pause = cancel)
        if self._schedule_manager:
            try:
                result = self._schedule_manager.pause_schedule(process_id)
                if result:
                    return True
            except Exception:
                logger.debug("Failed to cancel schedule %s", process_id, exc_info=True)

        return False

    @property
    def active_count(self) -> int:
        """Number of currently active processes."""
        return len(self.list_all(state=ProcessState.RUNNING))

    @property
    def total_count(self) -> int:
        """Total tracked processes."""
        return len(self.list_all())

    def summary(self) -> dict[str, Any]:
        """Get a summary of all processes.

        Returns:
            Dict with counts by type and state.
        """
        all_procs = self.list_all()
        by_type: dict[str, int] = {}
        by_state: dict[str, int] = {}

        for p in all_procs:
            by_type[p.type.value] = by_type.get(p.type.value, 0) + 1
            by_state[p.state.value] = by_state.get(p.state.value, 0) + 1

        return {
            "total": len(all_procs),
            "by_type": by_type,
            "by_state": by_state,
        }

    def _collect_agents(self) -> list[ProcessInfo]:
        """Collect processes from SubAgentManager."""
        if not self._subagent_manager:
            return []
        return [self._agent_to_process(run) for run in self._subagent_manager.list_runs()]

    def _collect_jobs(self) -> list[ProcessInfo]:
        """Collect processes from JobManager."""
        if not self._job_manager:
            return []
        try:
            jobs = self._job_manager.list_jobs()
            return [self._job_to_process(job) for job in jobs]
        except Exception:
            logger.debug("Failed to collect jobs", exc_info=True)
            return []

    def _collect_schedules(self) -> list[ProcessInfo]:
        """Collect processes from ScheduleManager."""
        if not self._schedule_manager:
            return []
        try:
            schedules = self._schedule_manager.list_schedules()
            return [self._schedule_to_process(s) for s in schedules]
        except Exception:
            logger.debug("Failed to collect schedules", exc_info=True)
            return []

    @staticmethod
    def _agent_to_process(run: Any) -> ProcessInfo:
        """Convert AgentRun to ProcessInfo."""
        state = _AGENT_STATE_MAP.get(run.status.value, ProcessState.PENDING)
        return ProcessInfo(
            id=run.run_id,
            type=ProcessType.AGENT,
            state=state,
            name=f"{run.agent}: {run.task[:80]}",
            started_at=run.started_at,
            duration_ms=run.duration_ms,
            metadata={
                "agent": run.agent,
                "parent_id": run.parent_id,
                "children": run.children,
                "error": run.error,
            },
        )

    @staticmethod
    def _job_to_process(job: Any) -> ProcessInfo:
        """Convert Job to ProcessInfo."""
        status_str = getattr(job, "status", "pending")
        if hasattr(status_str, "value"):
            status_str = status_str.value
        state = _JOB_STATE_MAP.get(str(status_str), ProcessState.PENDING)

        started = getattr(job, "started_at", 0.0)
        if hasattr(started, "timestamp"):
            started = started.timestamp()

        return ProcessInfo(
            id=getattr(job, "id", getattr(job, "job_id", "")),
            type=ProcessType.JOB,
            state=state,
            name=getattr(job, "workflow_id", "unknown"),
            started_at=started,
            duration_ms=getattr(job, "duration_ms", 0),
            metadata={
                "workflow_id": getattr(job, "workflow_id", ""),
                "progress": getattr(job, "progress", 0),
            },
        )

    @staticmethod
    def _schedule_to_process(schedule: Any) -> ProcessInfo:
        """Convert Schedule to ProcessInfo."""
        status_str = getattr(schedule, "status", "active")
        if hasattr(status_str, "value"):
            status_str = status_str.value
        state = _SCHEDULE_STATE_MAP.get(str(status_str), ProcessState.RUNNING)

        return ProcessInfo(
            id=getattr(schedule, "id", getattr(schedule, "schedule_id", "")),
            type=ProcessType.SCHEDULE,
            state=state,
            name=getattr(schedule, "name", getattr(schedule, "workflow_id", "unknown")),
            started_at=time.time(),
            metadata={
                "cron": getattr(schedule, "cron", ""),
                "next_run": str(getattr(schedule, "next_run", "")),
            },
        )
