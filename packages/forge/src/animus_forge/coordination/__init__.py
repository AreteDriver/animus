"""Coordination module — bridges Forge monitoring to Quorum intent graph."""

from __future__ import annotations

from .consciousness_bridge import ConsciousnessBridge, ConsciousnessConfig
from .evolution_loop import EvolutionConfig, EvolutionLoop
from .identity_anchor import DriftResult, IdentityAnchor
from .identity_patch import IdentityPatch, IdentityPatchGate
from .workflow_evolution import WorkflowEvolution, WorkflowPatch, WorkflowPatchInvalid

__all__ = [
    "ConsciousnessBridge",
    "ConsciousnessConfig",
    "DriftResult",
    "EvolutionConfig",
    "EvolutionLoop",
    "IdentityAnchor",
    "IdentityPatch",
    "IdentityPatchGate",
    "WorkflowEvolution",
    "WorkflowPatch",
    "WorkflowPatchInvalid",
]
