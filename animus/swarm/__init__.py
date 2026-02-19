"""Animus Swarm — Parallel agent orchestration with stigmergic coordination."""

from animus.swarm.models import (
    CyclicDependencyError,
    IntentEntry,
    SwarmConfig,
    SwarmError,
    SwarmStage,
)

__all__ = [
    "CyclicDependencyError",
    "IntentEntry",
    "IntentGraph",
    "IntentResolver",
    "SwarmConfig",
    "SwarmEngine",
    "SwarmError",
    "SwarmStage",
    "build_dag",
    "derive_stages",
]


def __getattr__(name: str):
    """Lazy import heavy modules to avoid circular imports."""
    if name == "SwarmEngine":
        from animus.swarm.engine import SwarmEngine

        return SwarmEngine
    if name == "IntentGraph":
        from animus.swarm.intent import IntentGraph

        return IntentGraph
    if name == "IntentResolver":
        from animus.swarm.intent import IntentResolver

        return IntentResolver
    if name in ("build_dag", "derive_stages"):
        from animus.swarm import graph

        return getattr(graph, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
