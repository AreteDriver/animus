"""Coordination module — bridges Forge monitoring to Quorum intent graph."""

from __future__ import annotations

from .consciousness_bridge import ConsciousnessBridge, ConsciousnessConfig
from .workflow_evolution import WorkflowEvolution, WorkflowPatch, WorkflowPatchInvalid

__all__ = [
    "ConsciousnessBridge",
    "ConsciousnessConfig",
    "WorkflowEvolution",
    "WorkflowPatch",
    "WorkflowPatchInvalid",
]
