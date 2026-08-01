"""MissionSystem — orchestrates citizen missions and manages their lifecycle.

Core responsibilities:
1. Issue MissionOrders to AgentRuntimes
2. Track mission status and enforce constraints
3. Handle timeout and failure recovery
4. Debrief completed missions and reintegrate results into Animus memory

Governance rules:
- Citizens may leave Animus core only through issued MissionOrders
- All missions have bounded authority, objectives, and return conditions
- MissionSystem respects the Constitution: no autonomous deployment without human approval
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from animus.logging import get_logger
from animus.mission.order import MissionOrder, MissionResult, MissionStatus
from animus.mission.runtime import AgentRuntime, LocalRuntime

logger = get_logger("mission.system")


@dataclass
class MissionConfig:
    """Configuration for MissionSystem behavior."""

    default_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    max_concurrent_missions: int = 10
    auto_debrief: bool = True
    persistence_dir: Path | None = None
    # If True, MissionSystem rejects orders that exceed runtime capabilities
    enforce_runtime_caps: bool = True


class MissionSystem:
    """Orchestrates citizen missions across AgentRuntimes.

    Usage:
        system = MissionSystem(config)
        order = MissionOrder(citizen_id="architect-001", mission_type="scan")
        system.issue(order, runtime=LocalRuntime())
        # ... monitor ...
        result = system.debrief(order.id)
    """

    def __init__(self, config: MissionConfig | None = None):
        self.config = config or MissionConfig()
        self._missions: dict[str, MissionOrder] = {}
        self._handles: dict[str, str] = {}  # order_id -> runtime handle
        self._runtimes: dict[str, AgentRuntime] = {}  # order_id -> runtime instance
        self._history: list[dict[str, Any]] = []
        self._load_state()

    # ------------------------------------------------------------------
    # Issue
    # ------------------------------------------------------------------

    def issue(
        self,
        order: MissionOrder,
        runtime: AgentRuntime | None = None,
    ) -> MissionOrder:
        """Issue a MissionOrder to an AgentRuntime.

        Args:
            order: The mission order to issue.
            runtime: The runtime to execute on. Defaults to LocalRuntime.

        Returns:
            Updated order with status=ISSUED and issued_at timestamp.

        Raises:
            RuntimeError: If max concurrent missions exceeded or runtime caps violated.
        """
        active = sum(1 for m in self._missions.values() if m.is_active)
        if active >= self.config.max_concurrent_missions:
            raise RuntimeError(
                f"Max concurrent missions ({self.config.max_concurrent_missions}) exceeded. "
                f"Active: {active}."
            )

        if runtime is None:
            runtime = LocalRuntime()

        # Runtime capability check
        if self.config.enforce_runtime_caps:
            caps = runtime.capabilities
            if not caps.can_handle(order.mission_type):
                raise RuntimeError(
                    f"Runtime '{runtime.name}' cannot handle mission type '{order.mission_type}'"
                )
            for constraint in order.constraints:
                if constraint.name == "max_concurrent" and isinstance(constraint.value, int):
                    if constraint.value > caps.max_concurrent_missions:
                        raise RuntimeError(
                            f"Runtime '{runtime.name}' max concurrent ({caps.max_concurrent_missions}) "
                            f"below mission requirement ({constraint.value})"
                        )

        order.status = MissionStatus.ISSUED
        order.issued_at = datetime.now()

        # Spawn on runtime
        try:
            handle = runtime.spawn(order)
        except Exception as e:
            order.status = MissionStatus.FAILED
            order.result = MissionResult(success=False, errors=[f"Spawn failed: {e}"])
            self._missions[order.id] = order
            self._persist_state()
            raise RuntimeError(f"Failed to spawn mission {order.id}: {e}") from e

        self._missions[order.id] = order
        self._handles[order.id] = handle
        self._runtimes[order.id] = runtime

        logger.info(f"Mission {order.id} issued to {runtime.name} (handle={handle})")
        self._persist_state()
        return order

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, order_id: str) -> MissionOrder:
        """Mark a mission as started (RUNNING).

        Typically called by the runtime when the mission begins execution.
        """
        order = self._get_mission(order_id)
        if order and order.status == MissionStatus.ISSUED:
            order.status = MissionStatus.RUNNING
            order.started_at = datetime.now()
            logger.info(f"Mission {order_id} started")
            self._persist_state()
        return order

    def pause(self, order_id: str) -> MissionOrder:
        """Pause an active mission."""
        order = self._get_mission(order_id)
        if order and order.status == MissionStatus.RUNNING:
            order.status = MissionStatus.PAUSED
            logger.info(f"Mission {order_id} paused")
            self._persist_state()
        return order

    def resume(self, order_id: str) -> MissionOrder:
        """Resume a paused mission."""
        order = self._get_mission(order_id)
        if order and order.status == MissionStatus.PAUSED:
            order.status = MissionStatus.RUNNING
            logger.info(f"Mission {order_id} resumed")
            self._persist_state()
        return order

    def report_result(self, order_id: str, result: MissionResult) -> MissionOrder:
        """Report a result for a mission (completes or fails it).

        Called by the runtime or the citizen when the mission finishes.
        """
        order = self._get_mission(order_id)
        if not order:
            return None

        order.result = result
        order.completed_at = datetime.now()

        if result.success:
            order.status = MissionStatus.COMPLETED
            logger.info(f"Mission {order_id} completed successfully")
        else:
            order.status = MissionStatus.FAILED
            logger.warning(
                f"Mission {order_id} failed: {result.errors[:3] if result.errors else 'unknown'}"
            )

        self._persist_state()

        if self.config.auto_debrief:
            self.debrief(order_id)

        return order

    def timeout(self, order_id: str) -> MissionOrder:
        """Force-timeout a mission."""
        order = self._get_mission(order_id)
        if not order:
            return None

        order.status = MissionStatus.TIMED_OUT
        order.completed_at = datetime.now()
        order.result = MissionResult(
            success=False,
            errors=[f"Mission timed out after {order.timeout.total_seconds()}s"],
        )

        # Terminate on runtime
        runtime = self._runtimes.get(order_id)
        handle = self._handles.get(order_id)
        if runtime and handle:
            try:
                runtime.terminate(handle, reason="timeout")
            except Exception as e:
                logger.warning(f"Runtime terminate on timeout failed: {e}")

        logger.warning(f"Mission {order_id} timed out")
        self._persist_state()
        return order

    def cancel(self, order_id: str, reason: str = "cancelled") -> MissionOrder:
        """Cancel a mission before it completes."""
        order = self._get_mission(order_id)
        if not order:
            return None

        runtime = self._runtimes.get(order_id)
        handle = self._handles.get(order_id)
        if runtime and handle:
            try:
                runtime.terminate(handle, reason=reason)
            except Exception as e:
                logger.warning(f"Runtime terminate on cancel failed: {e}")

        order.status = MissionStatus.FAILED
        order.completed_at = datetime.now()
        order.result = MissionResult(success=False, errors=[f"Cancelled: {reason}"])
        logger.info(f"Mission {order_id} cancelled: {reason}")
        self._persist_state()
        return order

    # ------------------------------------------------------------------
    # Debrief / Reintegration
    # ------------------------------------------------------------------

    def debrief(self, order_id: str) -> MissionResult | None:
        """Debrief a completed mission and reintegrate results into memory.

        This is the critical handoff: mission results become part of
        Animus memory so the citizen's learnings are preserved.

        Returns:
            The mission result, or None if order not found.
        """
        order = self._get_mission(order_id)
        if not order:
            return None

        if order.status not in (
            MissionStatus.COMPLETED,
            MissionStatus.FAILED,
            MissionStatus.TIMED_OUT,
        ):
            logger.warning(f"Cannot debrief mission {order_id} in status {order.status.name}")
            return None

        result = order.result
        if result is None:
            return None

        # Capture checkpoint from runtime if available
        runtime = self._runtimes.get(order_id)
        handle = self._handles.get(order_id)
        if runtime and handle:
            try:
                checkpoint = runtime.checkpoint(handle)
                result.outputs["final_checkpoint"] = checkpoint
            except Exception as e:
                logger.warning(f"Checkpoint capture during debrief failed: {e}")

        order.status = MissionStatus.DEBRIEFED

        # Log to history
        self._history.append(
            {
                "order_id": order_id,
                "citizen_id": order.citizen_id,
                "mission_type": order.mission_type,
                "status": order.status.name,
                "success": result.success,
                "metrics": result.metrics,
                "debriefed_at": datetime.now().isoformat(),
            }
        )

        logger.info(f"Mission {order_id} debriefed (success={result.success})")
        self._persist_state()
        return result

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def status(self, order_id: str) -> MissionStatus | None:
        """Get the status of a mission."""
        order = self._missions.get(order_id)
        return order.status if order else None

    def get(self, order_id: str) -> MissionOrder | None:
        """Get a mission order by ID."""
        return self._missions.get(order_id)

    def list_active(self) -> list[MissionOrder]:
        """List all active missions."""
        return [m for m in self._missions.values() if m.is_active]

    def list_completed(self) -> list[MissionOrder]:
        """List all completed/debriefed missions."""
        return [
            m
            for m in self._missions.values()
            if m.status in (MissionStatus.COMPLETED, MissionStatus.DEBRIEFED)
        ]

    def list_failed(self) -> list[MissionOrder]:
        """List all failed/timed-out missions."""
        return [
            m
            for m in self._missions.values()
            if m.status in (MissionStatus.FAILED, MissionStatus.TIMED_OUT)
        ]

    def history(self, citizen_id: str | None = None) -> list[dict[str, Any]]:
        """Return debrief history, optionally filtered by citizen."""
        if citizen_id is None:
            return list(self._history)
        return [h for h in self._history if h.get("citizen_id") == citizen_id]

    def stats(self) -> dict[str, Any]:
        """Aggregate mission statistics."""
        total = len(self._missions)
        active = len(self.list_active())
        completed = len(self.list_completed())
        failed = len(self.list_failed())
        debriefed = sum(1 for m in self._missions.values() if m.has_debriefed)

        success_rate = 0.0
        if completed + failed > 0:
            success_rate = completed / (completed + failed)

        return {
            "total": total,
            "active": active,
            "completed": completed,
            "failed": failed,
            "debriefed": debriefed,
            "success_rate": round(success_rate, 3),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _state_path(self) -> Path:
        if self.config.persistence_dir:
            return self.config.persistence_dir / "mission_state.json"
        return Path.home() / ".config" / "animus" / "mission_state.json"

    def _persist_state(self) -> None:
        path = self._state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "missions": [m.to_dict() for m in self._missions.values()],
                "history": self._history,
                "updated_at": datetime.now().isoformat(),
            }
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Mission state persistence failed: {e}")

    def _load_state(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for m_data in data.get("missions", []):
                try:
                    order = MissionOrder.from_dict(m_data)
                    self._missions[order.id] = order
                except Exception:
                    continue
            self._history = data.get("history", [])
        except Exception as e:
            logger.warning(f"Mission state load failed: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_mission(self, order_id: str) -> MissionOrder | None:
        order = self._missions.get(order_id)
        if not order:
            logger.warning(f"Mission {order_id} not found")
        return order

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"MissionSystem(total={s['total']}, active={s['active']}, "
            f"completed={s['completed']}, failed={s['failed']}, "
            f"success_rate={s['success_rate']})"
        )
