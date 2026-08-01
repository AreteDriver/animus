"""Animus Mission System — bounded deployment for citizens to remote bodies.

Implements ADL-20260712-001: Mission System + AgentRuntime abstraction.
Key design:
- Citizens leave Animus core only through issued MissionOrders
- AgentRuntime is a Protocol (duck-typed), not a base class
- LocalRuntime is the default adapter; other adapters (ADK, LangGraph, etc.) are external
- MissionSystem tracks status, enforces constraints, handles debrief/reintegration

Usage:
    mission = MissionSystem()
    order = MissionOrder(
        citizen_id="architect-001",
        mission_type="code_review",
        objectives={"target_repo": "github.com/org/repo"},
        constraints={"max_tokens": 10000, "readonly": True},
    )
    mission.issue(order, runtime=LocalRuntime())
    status = mission.status(order.id)
    # ... mission executes ...
    result = mission.debrief(order.id)
"""

from animus.mission.citizen_mixin import CitizenMissionMixin
from animus.mission.order import (
    MissionConstraint,
    MissionOrder,
    MissionResult,
    MissionStatus,
)
from animus.mission.runtime import (
    AgentRuntime,
    LocalRuntime,
    RuntimeCapabilities,
)
from animus.mission.system import (
    MissionConfig,
    MissionSystem,
)

__all__ = [
    "MissionOrder",
    "MissionStatus",
    "MissionResult",
    "MissionConstraint",
    "AgentRuntime",
    "LocalRuntime",
    "RuntimeCapabilities",
    "MissionSystem",
    "MissionConfig",
    "CitizenMissionMixin",
]
