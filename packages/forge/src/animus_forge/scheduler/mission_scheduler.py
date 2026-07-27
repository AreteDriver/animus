"""MissionScheduler — continuous run loop for autonomous mission execution.

Orchestrates READY tasks → lease → worker → result → state update.
Runs as an asyncio long-running task (or a background thread).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

from animus_forge.missions.domain import CitizenOutput, MissionStatus, Task, TaskContext, TaskStatus
from animus_forge.missions.store import MissionLedger
from animus_forge.scheduler.cost_enforcer import CostEnforcer
from animus_forge.scheduler.lease import LeaseManager
from animus_forge.scheduler.metrics import LEASE_EXPIRED, MISSION_COMPLETED, MISSION_FAILED, RESULT_PROCESSED, SchedulerMetrics, TASK_DISPATCHED
from animus_forge.scheduler.worker_pool import CitizenWorkerPool, PoolConfig
from animus_forge.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """Tuning for the mission scheduler."""

    poll_interval_seconds: float = 5.0
    max_concurrent_missions: int = 3
    default_task_ttl_seconds: int = 300
    default_mission_cap_usd: Decimal = Decimal("10.00")
    global_cap_usd: Decimal = Decimal("100.00")
    enable_recovery: bool = True


class MissionScheduler:
    """Continuous scheduler for citizen-driven missions.

    Lifecycle::

        await scheduler.start()   # starts pool, recovery loops
        await scheduler.run_once()  # single tick (for testing)
        await scheduler.stop()     # graceful shutdown

    The scheduler maintains a run loop that:

    1. Queries the ledger for READY tasks.
    2. Checks cost enforcer (mission + global budget).
    3. Acquires a lease for each ready task.
    4. Submits to the worker pool.
    5. Collects results from the worker pool.
    6. Transitions tasks and missions based on deterministic state machine.
    7. Runs recovery for expired leases / zombie workers.
    """

    def __init__(
        self,
        ledger: MissionLedger,
        lease_manager: LeaseManager,
        worker_pool: CitizenWorkerPool,
        cost_enforcer: CostEnforcer,
        workspace: WorkspaceManager | None = None,
        metrics: SchedulerMetrics | None = None,
        *,
        config: SchedulerConfig | None = None,
    ):
        self.ledger = ledger
        self.lease = lease_manager
        self.pool = worker_pool
        self.cost = cost_enforcer
        self.workspace = workspace
        self.metrics = metrics
        self.config = config or SchedulerConfig()

        self._run_task: asyncio.Task | None = None
        self._recovery_task: asyncio.Task | None = None
        self._result_consumer_task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self.pool.start()
        self._stopped.clear()
        self._run_task = asyncio.create_task(self._run_loop())
        self._result_consumer_task = asyncio.create_task(self._consume_results())
        if self.config.enable_recovery:
            self._recovery_task = asyncio.create_task(self.pool.run_recovery_loop())
        logger.info("MissionScheduler started")

    async def stop(self) -> None:
        self._stopped.set()
        for t in (self._run_task, self._result_consumer_task, self._recovery_task):
            if t:
                t.cancel()
        await self.pool.stop()
        logger.info("MissionScheduler stopped")

    async def run_once(self) -> int:
        """Execute a single scheduler tick and return number of tasks dispatched."""
        return await self._tick()

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        while not self._stopped.is_set():
            try:
                dispatched = await self._tick()
                if dispatched:
                    logger.info("Tick dispatched %d task(s)", dispatched)
            except Exception:
                logger.exception("Scheduler tick failed")
            await asyncio.wait_for(self._stopped.wait(), timeout=self.config.poll_interval_seconds)

    async def _tick(self) -> int:
        """Core tick: find READY tasks, budget-check, lease, dispatch."""
        dispatched = 0

        # 1. Limit concurrent missions
        active_missions = self.ledger.count_active_missions()
        if active_missions >= self.config.max_concurrent_missions:
            logger.debug("Active mission limit reached (%d)", active_missions)
            return 0

        # 2. Get running missions and find their ready tasks
        running = self.ledger.list_missions(status=MissionStatus.RUNNING)
        ready_tasks: list[Task] = []
        for mission in running:
            ready_tasks.extend(self.ledger.get_ready_tasks(mission.mission_id))

        if not ready_tasks:
            return 0

        for task in ready_tasks:
            if self._stopped.is_set():
                break

            # 3. Cost gate
            ok, reason = self.cost.can_start_task(
                str(task.mission_id),
                estimated_cost=Decimal("0.10"),
                mission_cap=self.config.default_mission_cap_usd,
            )
            if not ok:
                logger.warning("Cost gate blocked task %s: %s", task.task_id, reason)
                continue

            # 4. Transition task to LEASED
            try:
                self.ledger.transition_task(
                    task_id=task.task_id,
                    to_status=TaskStatus.LEASED,
                )
            except Exception:
                logger.warning("Task %s no longer eligible for lease (race?)", task.task_id)
                continue

            # 5. Build context (include latest checkpoint if present)
            mission = self.ledger.get_mission(task.mission_id)
            latest_checkpoint = self.ledger.get_latest_checkpoint(task.task_id)
            ctx = TaskContext(
                mission_objective=mission.objective if mission else "",
                task_description=task.description,
                repository=mission.repository if mission else "",
                base_commit=None,
                allowed_paths=mission.allowed_paths if mission else [],
                protected_paths=mission.protected_paths if mission else [],
                relevant_files=[],
                prior_attempts=[],
                checkpoint=latest_checkpoint.model_dump(mode="json") if latest_checkpoint else None,
                budget_remaining_usd=self.cost.mission_remaining(
                    str(task.mission_id), self.config.default_mission_cap_usd
                ),
                output_schema=None,
            )

            # 6. Submit to pool (lease acquired inside submit)
            lease_id = await self.pool.submit(
                task_id=str(task.task_id),
                citizen_role=task.citizen_role,
                context=ctx,
                mission_id=str(task.mission_id),
                ttl_seconds=self.config.default_task_ttl_seconds,
            )
            if not lease_id:
                # Pool full or lease race — revert task to READY
                try:
                    self.ledger.transition_task(
                        task_id=task.task_id,
                        to_status=TaskStatus.READY,
                    )
                except Exception:
                    logger.warning("Failed to revert task %s to READY", task.task_id)
                continue

            # 7. Transition to RUNNING now that worker is dispatched
            try:
                self.ledger.transition_task(
                    task_id=task.task_id,
                    to_status=TaskStatus.RUNNING,
                )
            except Exception:
                logger.warning("Failed to transition task %s to RUNNING", task.task_id)

            dispatched += 1
            if self.metrics:
                self.metrics.record(
                    TASK_DISPATCHED,
                    mission_id=str(task.mission_id),
                    task_id=str(task.task_id),
                )

        return dispatched

    # ------------------------------------------------------------------
    # Result consumer
    # ------------------------------------------------------------------

    async def _consume_results(self) -> None:
        """Background coroutine that processes completed task results."""
        while not self._stopped.is_set():
            try:
                queue = await self.pool.results()
                task_id_str, result_dict = await asyncio.wait_for(
                    queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            await self._process_result(task_id_str, result_dict)

    async def _process_result(self, task_id_str: str, result_dict: dict[str, Any]) -> None:
        """Process a single completed task result."""
        try:
            output = CitizenOutput(**result_dict)
        except Exception as exc:
            logger.error("Failed to parse CitizenOutput for task %s: %s", task_id_str, exc)
            output = CitizenOutput(
                status="failed",
                summary=f"Result parse error: {exc}",
                risks=[{"severity": "critical", "description": str(exc)}],
            )

        # Release lease
        lease = self.lease.get_lease_for_task(task_id_str)
        if lease:
            self.lease.release(lease.lease_id, outcome=output.status)

        # Find task in ledger
        task = self.ledger.get_task_by_id(task_id_str)
        if not task:
            logger.warning("Result received for unknown task %s", task_id_str)
            return

        # Record cost (placeholder until real cost plumbing exists)
        self.cost.record(
            mission_id=str(task.mission_id),
            task_id=task_id_str,
            operation="citizen_task",
        )

        # Save checkpoint with result data
        try:
            self.ledger.save_checkpoint(
                task_id=task.task_id,
                attempt_id=task.task_id,
                stage=output.status,
                inputs={},
                outputs={"summary": output.summary, "confidence": output.confidence},
                artifacts=[a.model_dump(mode="json") for a in output.artifacts],
            )
        except Exception:
            logger.exception("Failed to save checkpoint for task %s", task.task_id)

        # Transition based on citizen output
        if output.status == "completed":
            try:
                self.ledger.transition_task(
                    task_id=task.task_id,
                    to_status=TaskStatus.COMPLETED,
                )
            except Exception:
                logger.exception("Failed to transition task %s to COMPLETED", task.task_id)
        else:
            # failed or needs_repair — retry if attempts remain
            if task.current_attempt < task.max_attempts:
                try:
                    self.ledger.transition_task(
                        task_id=task.task_id,
                        to_status=TaskStatus.READY,
                    )
                    self.ledger.increment_attempt(task.task_id)
                except Exception:
                    logger.exception("Failed to retry task %s", task.task_id)
            else:
                try:
                    self.ledger.transition_task(
                        task_id=task.task_id,
                        to_status=TaskStatus.FAILED,
                    )
                except Exception:
                    logger.exception("Failed to transition task %s to FAILED", task.task_id)

        if self.metrics:
            self.metrics.record(
                RESULT_PROCESSED,
                mission_id=str(task.mission_id),
                task_id=task_id_str,
                value=output.status,
            )

        # Check mission completion
        await self._check_mission_completion(task.mission_id)

    async def _check_mission_completion(self, mission_id: UUID) -> None:
        """Evaluate whether a mission is fully complete or should move to REVIEW."""
        tasks = self.ledger.list_tasks(mission_id=mission_id)
        if not tasks:
            return

        all_done = all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            for t in tasks
        )
        any_failed = any(t.status == TaskStatus.FAILED for t in tasks)

        if not all_done:
            return

        mission = self.ledger.get_mission(mission_id)
        if not mission:
            return

        if mission.status in (MissionStatus.COMPLETED, MissionStatus.FAILED, MissionStatus.CANCELLED):
            return

        if any_failed:
            try:
                self.ledger.transition_mission(
                    mission_id=mission_id,
                    to_status=MissionStatus.FAILED,
                )
                if self.metrics:
                    self.metrics.record(
                        MISSION_FAILED,
                        mission_id=str(mission_id),
                    )
            except Exception:
                logger.exception("Failed to transition mission %s to FAILED", mission_id)
        else:
            try:
                # Route through REVIEW → COMPLETED per state machine
                self.ledger.transition_mission(
                    mission_id=mission_id,
                    to_status=MissionStatus.REVIEW,
                )
                self.ledger.transition_mission(
                    mission_id=mission_id,
                    to_status=MissionStatus.COMPLETED,
                )
                if self.metrics:
                    self.metrics.record(
                        MISSION_COMPLETED,
                        mission_id=str(mission_id),
                    )
            except Exception:
                logger.exception("Failed to transition mission %s to COMPLETED", mission_id)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Snapshot of scheduler health for observability."""
        snap: dict[str, Any] = {
            "running": not self._stopped.is_set(),
            "active_workers": self.pool.active_count(),
            "free_slots": self.pool.free_count(),
            "global_spend_usd": str(self.cost.global_spend()),
            "global_cap_usd": str(self.cost.global_cap),
        }
        if self.metrics:
            snap["metrics_summary"] = self.metrics.summary()
        return snap
