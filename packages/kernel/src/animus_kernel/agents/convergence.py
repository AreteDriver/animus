"""Adapter between Convergent's IntentResolver and Gorgon's delegation pipeline.

Optional integration — Gorgon works without Convergent installed.
When available, checks delegations for coherence before parallel execution:
overlapping tasks, conflicting agents, redundant work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    from convergent import (
        Intent,
        InterfaceKind,
        InterfaceSpec,
        create_delegation_checker,
    )

    HAS_CONVERGENT = True
except ImportError:
    HAS_CONVERGENT = False


@dataclass
class ConvergenceResult:
    """Result of checking delegations for coherence."""

    adjustments: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    dropped_agents: set[str] = field(default_factory=set)

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0


class DelegationConvergenceChecker:
    """Checks delegations for coherence using Convergent's IntentResolver.

    No-ops gracefully when Convergent is not installed.
    """

    def __init__(self, resolver: Any | None = None) -> None:
        self._resolver = resolver
        self._enabled = HAS_CONVERGENT and resolver is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def check_delegations(self, delegations: list[dict[str, str]]) -> ConvergenceResult:
        """Check a list of delegations for overlap and conflicts.

        Each delegation is {"agent": str, "task": str}. Publishes each as
        an Intent, then resolves each against the graph.

        Returns:
            ConvergenceResult with any adjustments, conflicts, or agents to drop.
        """
        if not self._enabled:
            return ConvergenceResult()

        result = ConvergenceResult()

        # Publish all delegations as intents
        intents: list[tuple[str, Any]] = []
        for delegation in delegations:
            intent = self._delegation_to_intent(delegation)
            self._resolver.publish(intent)
            intents.append((delegation.get("agent", "unknown"), intent))

        # Resolve each against the graph
        for agent_name, intent in intents:
            resolution = self._resolver.resolve(intent)

            for adj in resolution.adjustments:
                result.adjustments.append(
                    {
                        "agent": agent_name,
                        "kind": adj.kind,
                        "description": adj.description,
                        "confidence": adj.confidence,
                    }
                )
                # If told to consume instead, the agent is redundant
                if adj.kind == "ConsumeInstead" and adj.confidence >= 0.7:
                    result.dropped_agents.add(agent_name)

            for conflict in resolution.conflicts:
                result.conflicts.append(
                    {
                        "agent": agent_name,
                        "description": conflict.description,
                        "their_stability": conflict.their_stability,
                        "confidence": conflict.confidence,
                    }
                )

        return result

    @staticmethod
    def _delegation_to_intent(delegation: dict[str, str]) -> Any:
        """Convert a Gorgon delegation dict to a Convergent Intent."""
        agent = delegation.get("agent", "unknown")
        task = delegation.get("task", "")

        # Infer tags from the agent role
        role_tags = {
            "planner": ["planning", "architecture", "design"],
            "builder": ["implementation", "code", "feature"],
            "tester": ["testing", "qa", "coverage"],
            "reviewer": ["review", "security", "quality"],
            "architect": ["architecture", "design", "system"],
            "documenter": ["documentation", "docs", "guide"],
            "analyst": ["analysis", "data", "metrics"],
        }
        tags = role_tags.get(agent, [agent])

        return Intent(
            agent_id=agent,
            intent=task,
            provides=[
                InterfaceSpec(
                    name=f"{agent}_output",
                    kind=InterfaceKind.FUNCTION,
                    signature="(task: str) -> str",
                    tags=tags,
                ),
            ],
        )


def create_checker() -> DelegationConvergenceChecker:
    """Create a DelegationConvergenceChecker with a fresh resolver.

    Returns a disabled checker if Convergent is not installed.
    """
    if not HAS_CONVERGENT:
        logger.info("Convergent not installed — delegation coherence checking disabled")
        return DelegationConvergenceChecker(resolver=None)

    resolver = create_delegation_checker(min_stability=0.0)
    logger.info("Convergent delegation coherence checker enabled")
    return DelegationConvergenceChecker(resolver=resolver)


def format_convergence_alert(result: ConvergenceResult) -> str:
    """Format a ConvergenceResult into a human-readable alert string.

    Returns empty string if no conflicts or dropped agents.
    """
    parts: list[str] = []

    if result.conflicts:
        parts.append(f"Conflicts ({len(result.conflicts)}):")
        for c in result.conflicts:
            parts.append(f"  - {c.get('agent', '?')}: {c.get('description', '?')}")

    if result.dropped_agents:
        agents = ", ".join(sorted(result.dropped_agents))
        parts.append(f"Dropped agents ({len(result.dropped_agents)}): {agents}")

    if result.adjustments:
        parts.append(f"Adjustments ({len(result.adjustments)}):")
        for a in result.adjustments:
            parts.append(f"  - {a.get('agent', '?')}: {a.get('description', '?')}")

    return "\n".join(parts)


def create_bridge(db_path: str | None = None) -> Any:
    """Create a GorgonBridge for coordination protocol features.

    Returns None if Convergent is not installed.
    """
    if not HAS_CONVERGENT:
        logger.info("Convergent not installed — coordination bridge disabled")
        return None
    try:
        from pathlib import Path

        from convergent import CoordinationConfig, GorgonBridge

        if db_path is None:
            db_dir = Path.home() / ".gorgon"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "coordination.db")

        bridge = GorgonBridge(CoordinationConfig(db_path=db_path))
        logger.info("Convergent coordination bridge enabled (db=%s)", db_path)
        return bridge
    except Exception as e:
        logger.warning("Failed to create coordination bridge: %s", e)
        return None


def create_event_log(db_path: str | None = None) -> Any:
    """Create a Convergent EventLog for coordination event tracking.

    Returns None if Convergent is not installed.

    Args:
        db_path: Path to SQLite database. Defaults to ~/.gorgon/coordination.events.db.

    Returns:
        EventLog instance or None.
    """
    if not HAS_CONVERGENT:
        logger.info("Convergent not installed — coordination event log disabled")
        return None
    try:
        from pathlib import Path

        from convergent import EventLog

        if db_path is None:
            db_dir = Path.home() / ".gorgon"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "coordination.events.db")

        event_log = EventLog(db_path)
        logger.info("Convergent event log enabled (db=%s)", db_path)
        return event_log
    except Exception as e:
        logger.warning("Failed to create coordination event log: %s", e)
        return None


def get_coordination_health(bridge: Any) -> dict[str, Any]:
    """Run a coordination health check via Convergent's HealthChecker.

    Args:
        bridge: A GorgonBridge instance.

    Returns:
        Dict with grade, issues, and subsystem metrics. Empty dict on failure.
    """
    if not HAS_CONVERGENT or bridge is None:
        return {}
    try:
        from dataclasses import asdict

        from convergent import HealthChecker

        checker = HealthChecker.from_bridge(bridge)
        health = checker.check()
        return asdict(health)
    except Exception as e:
        logger.warning("Coordination health check failed: %s", e)
        return {}


def check_dependency_cycles(resolver: Any) -> list[dict[str, Any]]:
    """Check the intent graph for dependency cycles.

    Args:
        resolver: An IntentResolver instance.

    Returns:
        List of cycle dicts with intent_ids and agent_ids. Empty on failure.
    """
    if not HAS_CONVERGENT or resolver is None:
        return []
    try:
        from convergent import find_cycles

        cycles = find_cycles(resolver)
        return [
            {
                "intent_ids": list(c.intent_ids),
                "agent_ids": list(c.agent_ids),
                "display": str(c),
            }
            for c in cycles
        ]
    except Exception as e:
        logger.warning("Dependency cycle check failed: %s", e)
        return []


def get_execution_order(resolver: Any) -> list[str]:
    """Get topological execution order for intents.

    Args:
        resolver: An IntentResolver instance.

    Returns:
        List of intent IDs in dependency-first order. Empty on failure.
    """
    if not HAS_CONVERGENT or resolver is None:
        return []
    try:
        from convergent import topological_order

        return topological_order(resolver)
    except Exception as e:
        logger.warning("Execution order computation failed: %s", e)
        return []
