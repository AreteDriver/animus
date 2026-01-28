"""Protocol for safety guardrail systems."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SafetyGuard(Protocol):
    """Structural interface for safety guardrail systems."""

    def check_action(self, action: dict[str, Any]) -> tuple[bool, str | None]: ...
    def check_learning(
        self, content: str, category: str
    ) -> tuple[bool, str | None]: ...
