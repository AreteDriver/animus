"""Data models for Animus Swarm coordination."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from animus.forge.models import ForgeError


class SwarmError(ForgeError):
    """Base exception for Swarm operations."""


class CyclicDependencyError(SwarmError):
    """Raised when the agent dependency graph contains a cycle."""


@dataclass
class SwarmConfig:
    """Configuration for parallel execution behavior."""

    max_workers: int = 4
    stage_timeout_seconds: float = 300.0


@dataclass
class SwarmStage:
    """A group of agents that can execute in parallel.

    All agents in a stage have their input dependencies satisfied
    by agents in prior stages.
    """

    index: int
    agent_names: list[str] = field(default_factory=list)


@dataclass
class IntentEntry:
    """A single agent's declared intent in the shared intent graph.

    Fields mirror the whitepaper Section 5.3 specification.
    Stability ranges from 0.0 (speculative) to 1.0 (committed/verified).
    """

    agent: str
    action: str = "execute_step"
    provides: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    stability: float = 0.0
    evidence: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"
