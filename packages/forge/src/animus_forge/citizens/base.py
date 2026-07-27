"""Base class for all citizens.

A citizen is a durable logical identity with a role, capabilities, and
bounded autonomy.  A worker is a disposable execution process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from animus_forge.missions.domain import CitizenOutput, Task, TaskContext


class Citizen(ABC):
    """Durable citizen identity — survives worker replacement, model changes,
    and context exhaustion.

    Attributes:
        role: Canonical role name (e.g. ``"planner"``).
        capabilities: Set of capability tags.
        can_modify_code: Whether this citizen may write repository files.
        can_approve: Whether this citizen may approve work for merge.
    """

    role: str = "abstract"
    capabilities: set[str] = set()
    can_modify_code: bool = False
    can_approve: bool = False

    @abstractmethod
    def run(self, task: Task, context: TaskContext) -> CitizenOutput:
        """Execute a single task and return structured output.

        Args:
            task: The task to execute.
            context: Bounded context packet containing mission contract,
                relevant files, budget, and output schema.

        Returns:
            A ``CitizenOutput`` conforming to the expected schema.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize citizen identity (not runtime state)."""
        return {
            "role": self.role,
            "capabilities": sorted(self.capabilities),
            "can_modify_code": self.can_modify_code,
            "can_approve": self.can_approve,
        }
