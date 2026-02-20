"""Animus Forge — Multi-agent orchestration engine.

Declarative YAML workflows with token budgets, quality gates,
and SQLite checkpoint/resume.
"""

from animus.forge.models import (
    AgentConfig,
    ForgeError,
    GateConfig,
    ReviseRequestedError,
    StepResult,
    WorkflowConfig,
    WorkflowState,
)

__all__ = [
    "AgentConfig",
    "ForgeEngine",
    "ForgeError",
    "GateConfig",
    "ReviseRequestedError",
    "StepResult",
    "WorkflowConfig",
    "WorkflowState",
]


def __getattr__(name: str):
    """Lazy import ForgeEngine to avoid circular imports."""
    if name == "ForgeEngine":
        from animus.forge.engine import ForgeEngine

        return ForgeEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
