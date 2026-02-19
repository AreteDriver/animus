"""Animus Swarm — Parallel agent orchestration with stigmergic coordination."""

from animus.swarm.engine import SwarmEngine
from animus.swarm.graph import build_dag, derive_stages
from animus.swarm.intent import IntentGraph, IntentResolver
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
