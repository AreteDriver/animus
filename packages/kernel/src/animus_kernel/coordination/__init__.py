"""Coordination module — workflow evolution and auto-promotion."""

from __future__ import annotations

from .evolution_loop import EvolutionConfig, EvolutionLoop
from .workflow_evolution import WorkflowEvolution, WorkflowPatch, WorkflowPatchInvalid

__all__ = [
    "EvolutionConfig",
    "EvolutionLoop",
    "WorkflowEvolution",
    "WorkflowPatch",
    "WorkflowPatchInvalid",
]
