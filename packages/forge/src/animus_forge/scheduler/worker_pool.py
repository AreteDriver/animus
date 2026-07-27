"""CitizenWorkerPool — manage a bounded number of worker processes for missions.

Each worker is a short-lived process that loads a citizen, executes one task,
returns the CitizenOutput, and exits.  The pool ensures we don't exceed
system resources (memory, CPU, file descriptors) and provides lifecycle hooks
for monitoring and recovery.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from animus_forge.citizens.base import Citizen
from animus_forge.citizens.builder import BuilderCitizen
from animus_forge.citizens.planner import PlannerCitizen
from animus_forge.citizens.reviewer import ReviewerCitizen
from animus_forge.missions.domain import CitizenOutput, Task, TaskContext

if TYPE_CHECKING:
    from animus_forge.scheduler.lease import LeaseManager
    from animus_forge.scheduler.containers import ContainerManager

logger = logging.getLogger(__name__)

# Map of role → citizen class
_CITIZEN_REGISTRY: dict[str, type[Citizen]] = {
    "planner": PlannerCitizen,
    "builder": BuilderCitizen,
    "reviewer": ReviewerCitizen,
}


def _run_task_worker(
    task_id: str,
    mission_id: str,
    citizen_role: str,
    description: str,
    context_json: dict[str, Any],
) -> dict[str, Any]:
    """Entry point executed inside a fresh process.

    It imports, instantiates, and runs the citizen for a single task.
    All state is returned serialised as JSON-able dict.
    """
    # Re-instantiate the context (lightweight dataclass, serialised by dict)
    ctx = TaskContext(**context_json)

    # Build a minimal Task object from what the lease holds
    task = Task(
        task_id=uuid.UUID(task_id),
        mission_id=uuid.UUID(mission_id),
        citizen_role=citizen_role,
        description=description or "",
    )

    citizen_cls = _CITIZEN_REGISTRY.get(citizen_role)
    if citizen_cls is None:
        return {
            "status": "failed",
            "summary": f"Unknown citizen role: {citizen_role}",
            "changed_files": [],
            "evidence": [],
            "risks": [{"severity": "high", "description": f"Role {citizen_role} not in registry"}],
            "confidence": 0.0,
        }

    citizen = citizen_cls()
    try:
        output = citizen.run(task=task, context=ctx)
        return output.model_dump(mode="json")
    except Exception as exc:
        logger.error("Worker exception for task %s: %s", task_id, exc)
        return {
            "status": "failed",
            "summary": f"Worker crashed: {exc}",
            "changed_files": [],
            "evidence": [{"type": "exception", "detail": str(exc)}],
            "risks": [{"severity": "critical", "description": str(exc)}],
            "confidence": 0.0,
        }


@dataclass
class WorkerSlot:
    """A slot in the pool, either idle or currently running a task."""

    slot_id: str
    lease_id: str | None = None
    task_id: str | None = None
    citizen_role: str | None = None
    started_at: float | None = None


@dataclass
class PoolConfig:
    """Tuning knobs for the worker pool."""

    max_workers: int = field(default_factory=lambda: max(2, os.cpu_count() or 2))
    worker_timeout_seconds: int = 300  # hard kill after this long
    poll_interval_seconds: float = 2.0
    isolation_mode: str = "process"  # "process" or "container"


class CitizenWorkerPool:
    """Bounded pool of worker processes executing citizen tasks.

    Usage::

        pool = CitizenWorkerPool(lease_manager, config=PoolConfig(max_workers=4))
        await pool.start()
        # Submit tasks from the scheduler loop
        await pool.submit(task_id="...", citizen_role="builder", context=ctx)
        # Later …
        await pool.stop()
    """

    def __init__(
        self,
        lease_manager: LeaseManager,
        *,
        config: PoolConfig | None = None,
        container_manager: ContainerManager | None = None,
    ):
        self.lease = lease_manager
        self.config = config or PoolConfig()
        self.container = container_manager
        self._slots: dict[str, WorkerSlot] = {}
        self._executor: ProcessPoolExecutor | None = None
        self._shutdown_event = asyncio.Event()
        self._results_queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self._pending: dict[str, asyncio.Future] = {}  # task_id → Future
        self._background_tasks: set[asyncio.Task] = set()
        self._initialised = False

    async def start(self) -> None:
        if self._initialised:
            return
        self._executor = ProcessPoolExecutor(
            max_workers=self.config.max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        for i in range(self.config.max_workers):
            self._slots[str(i)] = WorkerSlot(slot_id=str(i))
        self._initialised = True
        logger.info("CitizenWorkerPool started with %d slots", self.config.max_workers)

    async def stop(self) -> None:
        if not self._initialised:
            return
        self._shutdown_event.set()
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        # Cancel any running asyncio tasks we spawned
        for t in list(self._background_tasks):
            t.cancel()
        self._initialised = False
        logger.info("CitizenWorkerPool stopped")

    async def submit(
        self,
        task_id: str,
        citizen_role: str,
        context: TaskContext,
        *,
        mission_id: str = "unknown",
        ttl_seconds: int | None = None,
    ) -> str | None:
        """Try to submit a task to the pool.

        1. Acquire a lease.
        2. If successful, dispatch to the process pool.
        3. Return the lease_id so the scheduler can track it.

        Returns:
            Lease ID on success, ``None`` if no slot available or lease
            acquisition failed.
        """
        # Find a free slot
        free_slot = next(
            (s for s in self._slots.values() if s.lease_id is None),
            None,
        )
        if not free_slot:
            logger.debug("No free worker slot for task %s", task_id)
            return None

        lease = self.lease.acquire(
            task_id=task_id,
            mission_id=mission_id,
            citizen_role=citizen_role,
            worker_id=free_slot.slot_id,
            ttl_seconds=ttl_seconds or self.config.worker_timeout_seconds,
        )
        if not lease:
            return None

        free_slot.lease_id = lease.lease_id
        free_slot.task_id = task_id
        free_slot.citizen_role = citizen_role
        free_slot.started_at = time.time()

        if self.config.isolation_mode == "container" and self.container is not None:
            # Dispatch via container manager (asyncio.to_thread so it doesn't block)
            af = asyncio.create_task(
                self._run_in_container(task_id, mission_id, citizen_role, context)
            )
            self._pending[task_id] = af
            af.add_done_callback(
                lambda fut, tid=task_id: asyncio.create_task(
                    self._on_task_done(tid, fut)
                )
            )
        else:
            # Dispatch to executor (runs in a separate process)
            future = self._executor.submit(
                _run_task_worker,
                task_id,
                mission_id,
                citizen_role,
                context.task_description or "",
                context.model_dump(mode="json"),
            )
            # Wrap the concurrent.futures.Future in an asyncio.Future so we can await it
            af = asyncio.wrap_future(future)
            self._pending[task_id] = af

            # Attach a callback that enqueues the result and cleans the slot
            af.add_done_callback(
                lambda fut, tid=task_id: asyncio.create_task(
                    self._on_task_done(tid, fut)
                )
            )

        logger.info(
            "Task %s (%s) dispatched to slot %s, lease %s",
            task_id,
            citizen_role,
            free_slot.slot_id,
            lease.lease_id,
        )
        return lease.lease_id

    async def _run_in_container(
        self,
        task_id: str,
        mission_id: str,
        citizen_role: str,
        context: TaskContext,
    ) -> dict[str, Any]:
        """Offload a citizen task to a container via asyncio.to_thread."""
        if self.container is None:
            return {
                "status": "failed",
                "summary": "Container mode requested but no ContainerManager provided",
                "confidence": 0.0,
            }
        return await asyncio.to_thread(
            self.container.run_task,
            task_id=task_id,
            mission_id=mission_id,
            citizen_role=citizen_role,
            description=context.task_description or "",
            context_json=context.model_dump(mode="json"),
        )

    async def _on_task_done(self, task_id: str, future: asyncio.Future) -> None:
        """Callback when a worker process finishes (success or crash)."""
        try:
            result = future.result()
        except Exception as exc:
            logger.error("Worker future for task %s raised: %s", task_id, exc)
            result = {
                "status": "failed",
                "summary": f"Pool future exception: {exc}",
                "changed_files": [],
                "evidence": [{"type": "pool_error", "detail": str(exc)}],
                "risks": [{"severity": "critical", "description": str(exc)}],
                "confidence": 0.0,
            }

        # Free the slot
        for slot in self._slots.values():
            if slot.task_id == task_id:
                slot.lease_id = None
                slot.task_id = None
                slot.citizen_role = None
                slot.started_at = None
                break

        self._pending.pop(task_id, None)
        await self._results_queue.put((task_id, result))

    async def results(self) -> asyncio.Queue[tuple[str, dict]]:
        """Return the queue of completed task results.

        The scheduler loop reads from this queue to process CitizenOutput
        and update task / mission state.
        """
        return self._results_queue

    def active_count(self) -> int:
        """Number of currently running workers."""
        return sum(1 for s in self._slots.values() if s.lease_id is not None)

    def free_count(self) -> int:
        return sum(1 for s in self._slots.values() if s.lease_id is None)

    def kill_slot(self, slot_id: str) -> bool:
        """Hard-kill a worker slot (used for stuck / zombie recovery).

        Returns:
            ``True`` if the slot was occupied and reset.
        """
        slot = self._slots.get(slot_id)
        if not slot or not slot.lease_id:
            return False

        # Release lease so task becomes recoverable
        self.lease.release(slot.lease_id, outcome="killed")

        # Cancel the future if still pending
        if slot.task_id and slot.task_id in self._pending:
            self._pending[slot.task_id].cancel()

        slot.lease_id = None
        slot.task_id = None
        slot.citizen_role = None
        slot.started_at = None
        logger.warning("Slot %s force-killed", slot_id)
        return True

    async def run_recovery_loop(self, interval: float | None = None) -> None:
        """Background coroutine that recovers expired leases.

        Should be ``await``ed or ``asyncio.create_task``-ed by the
        scheduler during its own run loop.
        """
        interval = interval or self.config.poll_interval_seconds
        while not self._shutdown_event.is_set():
            recovered = self.lease.recover_expired()
            if recovered:
                logger.warning("Recovered %d expired tasks: %s", len(recovered), recovered)
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)
