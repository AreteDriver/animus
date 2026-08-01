"""MissionScheduler — continuous run loop for autonomous mission execution.

Orchestrates READY tasks → lease → worker → result → state update.
Runs as an asyncio long-running task (or a background thread).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import UUID

from animus_forge.missions.domain import CitizenOutput, MissionStatus, Task, TaskContext, TaskStatus
from animus_forge.missions.store import MissionLedger
from animus_forge.scheduler.cost_enforcer import CostEnforcer
from animus_forge.scheduler.lease import LeaseManager
from animus_forge.scheduler.lifecycle import (
    LoopSupervisor,
    RestartConfig,
    RestartPolicy,
    SchedulerLifecycleState,
    SchedulerStatusSnapshot,
)
from animus_forge.scheduler.metrics import (
    MISSION_COMPLETED,
    MISSION_FAILED,
    RESULT_PROCESSED,
    TASK_DISPATCHED,
    SchedulerMetrics,
)
from animus_forge.scheduler.worker_pool import CitizenWorkerPool
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

        self._supervisor = LoopSupervisor(
            restart_config=RestartConfig(
                policy=RestartPolicy.ON_FAILURE,
                max_restarts=3,
                delay_seconds=0.5,
            ),
        )
        self._supervisor.register("dispatcher", self._run_loop)
        self._supervisor.register("result_consumer", self._consume_results)
        if self.config.enable_recovery:
            self._supervisor.register("recovery", self.pool.run_recovery_loop)

    # ------------------------------------------------------------------
    # Public lifecycle interface
    # ------------------------------------------------------------------

    @property
    def lifecycle_state(self) -> SchedulerLifecycleState:
        return self._supervisor.state

    @property
    def is_running(self) -> bool:
        return self._supervisor.is_running

    @property
    def is_ready(self) -> bool:
        return self.lifecycle_state == SchedulerLifecycleState.RUNNING

    @property
    def is_healthy(self) -> bool:
        return self._supervisor.is_healthy

    async def start(self) -> None:
        """Start the scheduler and its supervised loops.

        Idempotent: returns immediately if the scheduler is already running
        or starting.  Safe to call after ``stop()`` to restart.
        """
        if self.is_running:
            logger.debug("MissionScheduler.start() called while in state %s", self.lifecycle_state)
            return

        await self.pool.start()
        await self._supervisor.start()
        logger.info("MissionScheduler started (state=%s)", self.lifecycle_state.value)

    async def stop(self) -> None:
        """Stop the scheduler gracefully and await supervised loop cleanup."""
        if not self.is_running:
            logger.debug("MissionScheduler.stop() called while in state %s", self.lifecycle_state)
            return

        await self._supervisor.stop()
        await self.pool.stop()
        logger.info("MissionScheduler stopped")

    async def run_once(self) -> int:
        """Execute a single scheduler tick and return number of tasks dispatched."""
        return await self._tick()

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def _wait_for_stop_or_timeout(self, timeout: float) -> None:
        """Wait for the stop signal, swallowing the normal timeout."""
        try:
            await asyncio.wait_for(self._supervisor.stop_requested.wait(), timeout=timeout)
        except TimeoutError:
            pass

    async def _run_loop(self) -> None:
        while self._supervisor.should_continue:
            try:
                dispatched = await self._tick()
                if dispatched:
                    logger.info("Tick dispatched %d task(s)", dispatched)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler tick failed")
                self._supervisor.record_error("dispatcher", "tick failed")
            self._supervisor.mark_tick("dispatcher")
            await self._wait_for_stop_or_timeout(self.config.poll_interval_seconds)

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
            if not self._supervisor.should_continue:
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
        while self._supervisor.should_continue:
            try:
                queue = await self.pool.results()
                task_id_str, result_dict = await asyncio.wait_for(
                    queue.get(),
                    timeout=1.0,
                )
            except TimeoutError:
                self._supervisor.mark_tick("result_consumer")
                continue
            except asyncio.CancelledError:
                break

            await self._process_result(task_id_str, result_dict)
            self._supervisor.mark_tick("result_consumer")

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
        """Public snapshot of scheduler health for observability."""
        snap = SchedulerStatusSnapshot(
            lifecycle_state=self.lifecycle_state,
            loops=self._supervisor.snapshot(),
            active_workers=self.pool.active_count(),
            free_slots=self.pool.free_count(),
            global_spend_usd=str(self.cost.global_spend()),
            global_cap_usd=str(self.cost.global_cap),
            last_tick_at=self._supervisor.last_tick_at,
            last_error=self._supervisor.last_error,
            restart_count=self._supervisor.restart_count,
            metrics_summary=self.metrics.summary() if self.metrics else None,
        )
        return snap.to_dict()
