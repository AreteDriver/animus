"""CitizenMissionMixin — optional mixin for citizens that support mission deployment.

Not all citizens need to deploy missions. This mixin adds the `deploy_mission()`
method that wraps MissionSystem.issue() with citizen-specific defaults.

Usage:
    class ArchitectCitizen(CitizenMissionMixin):
        def __init__(self):
            super().__init__(citizen_id="architect-001")

        def deploy_code_review(self, repo_url: str):
            return self.deploy_mission(
                mission_type="code_review",
                objectives={"target_repo": repo_url},
                constraints=[MissionConstraint("readonly", True, "authority")],
            )
"""

from __future__ import annotations

from typing import Any

from animus.mission.order import MissionConstraint, MissionOrder
from animus.mission.runtime import AgentRuntime
from animus.mission.system import MissionSystem


class CitizenMissionMixin:
    """Optional mixin adding mission deployment capability to citizens."""

    def __init__(self, citizen_id: str, mission_system: MissionSystem | None = None):
        self._citizen_id = citizen_id
        self._mission_system = mission_system or MissionSystem()

    def deploy_mission(
        self,
        mission_type: str,
        objectives: dict[str, Any],
        constraints: list[MissionConstraint] | None = None,
        runtime: AgentRuntime | None = None,
        timeout_minutes: int = 30,
        priority: int = 5,
    ) -> MissionOrder:
        """Deploy a mission order for this citizen.

        Args:
            mission_type: Type of mission (e.g., "code_review", "threat_scan").
            objectives: Mission objectives as a dict.
            constraints: Optional list of constraints.
            runtime: Target runtime. Defaults to LocalRuntime.
            timeout_minutes: Mission timeout.
            priority: Priority 1–10 (1=highest).

        Returns:
            Issued MissionOrder.
        """
        order = MissionOrder(
            citizen_id=self._citizen_id,
            mission_type=mission_type,
            objectives=objectives,
            constraints=constraints or [],
            timeout=__import__("datetime").timedelta(minutes=timeout_minutes),
            priority=priority,
        )
        return self._mission_system.issue(order, runtime=runtime)

    def mission_history(self) -> list[dict[str, Any]]:
        """Return debrief history for this citizen."""
        return self._mission_system.history(citizen_id=self._citizen_id)
