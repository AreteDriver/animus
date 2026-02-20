"""Coordination health dashboard — aggregated metrics from all subsystems.

Provides a holistic view of multi-agent coordination health by reading
from intent graph, stigmergy, phi scores, voting, and signal bus. Useful
for monitoring production deployments and diagnosing coordination issues.

Each subsystem contributes metrics to a ``CoordinationHealth`` dataclass.
A configurable grading system (A-F) summarizes overall health.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from convergent.resolver import IntentResolver
from convergent.score_store import ScoreStore
from convergent.stigmergy import StigmergyField

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentGraphHealth:
    """Metrics for the intent graph layer."""

    total_intents: int = 0
    agent_count: int = 0
    avg_stability: float = 0.0
    min_stability: float = 0.0
    max_stability: float = 0.0
    conflict_count: int = 0
    provides_count: int = 0
    requires_count: int = 0


@dataclass(frozen=True)
class StigmergyHealth:
    """Metrics for stigmergy markers."""

    total_markers: int = 0
    markers_by_type: dict[str, int] = field(default_factory=dict)
    avg_strength: float = 0.0
    unique_agents: int = 0
    unique_targets: int = 0


@dataclass(frozen=True)
class ScoringHealth:
    """Metrics for phi-weighted scoring."""

    total_agents: int = 0
    total_outcomes: int = 0
    avg_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    scores_by_agent: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VotingHealth:
    """Metrics for the Triumvirate voting system."""

    total_decisions: int = 0
    approval_rate: float = 0.0
    avg_confidence: float = 0.0
    escalation_count: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CoordinationHealth:
    """Aggregated health metrics from all coordination subsystems."""

    intent_graph: IntentGraphHealth = field(default_factory=IntentGraphHealth)
    stigmergy: StigmergyHealth = field(default_factory=StigmergyHealth)
    scoring: ScoringHealth = field(default_factory=ScoringHealth)
    voting: VotingHealth = field(default_factory=VotingHealth)
    grade: str = "A"
    issues: list[str] = field(default_factory=list)


class HealthChecker:
    """Aggregates health metrics from coordination subsystems.

    Accepts individual subsystem components or a GorgonBridge. Call
    ``check()`` to produce a ``CoordinationHealth`` report.

    Args:
        resolver: IntentResolver for intent graph metrics.
        stigmergy: StigmergyField for marker metrics.
        store: ScoreStore for scoring and voting metrics.
    """

    def __init__(
        self,
        resolver: IntentResolver | None = None,
        stigmergy: StigmergyField | None = None,
        store: ScoreStore | None = None,
    ) -> None:
        self._resolver = resolver
        self._stigmergy = stigmergy
        self._store = store

    @classmethod
    def from_bridge(cls, bridge: object) -> HealthChecker:
        """Create a HealthChecker from a GorgonBridge.

        Args:
            bridge: A ``GorgonBridge`` instance (uses duck typing to avoid
                circular imports).

        Returns:
            Configured HealthChecker.
        """
        stigmergy = getattr(bridge, "stigmergy", None)
        store = getattr(bridge, "_store", None)
        return cls(resolver=None, stigmergy=stigmergy, store=store)

    def check(self) -> CoordinationHealth:
        """Run health checks across all configured subsystems.

        Returns:
            CoordinationHealth with metrics, grade, and issues.
        """
        issues: list[str] = []

        ig_health = self._check_intent_graph(issues)
        stig_health = self._check_stigmergy(issues)
        score_health = self._check_scoring(issues)
        vote_health = self._check_voting(issues)

        grade = self._compute_grade(ig_health, stig_health, score_health, vote_health, issues)

        return CoordinationHealth(
            intent_graph=ig_health,
            stigmergy=stig_health,
            scoring=score_health,
            voting=vote_health,
            grade=grade,
            issues=issues,
        )

    def _check_intent_graph(self, issues: list[str]) -> IntentGraphHealth:
        """Collect intent graph metrics."""
        if self._resolver is None:
            return IntentGraphHealth()

        intents = self._resolver.backend.query_all()
        if not intents:
            return IntentGraphHealth()

        stabilities = [i.compute_stability() for i in intents]
        agents = {i.agent_id for i in intents}
        provides = sum(len(i.provides) for i in intents)
        requires = sum(len(i.requires) for i in intents)

        # Count conflicts by checking overlap between intents
        conflict_count = 0
        for i, a in enumerate(intents):
            for b in intents[i + 1 :]:
                if a.agent_id == b.agent_id:
                    continue
                a_specs = a.provides + a.requires
                b_specs = b.provides + b.requires
                for a_spec in a_specs:
                    if any(a_spec.structurally_overlaps(b_spec) for b_spec in b_specs):
                        conflict_count += 1
                        break

        avg_stab = sum(stabilities) / len(stabilities)
        if avg_stab < 0.3:
            issues.append(f"Low average intent stability: {avg_stab:.2f}")

        return IntentGraphHealth(
            total_intents=len(intents),
            agent_count=len(agents),
            avg_stability=avg_stab,
            min_stability=min(stabilities),
            max_stability=max(stabilities),
            conflict_count=conflict_count,
            provides_count=provides,
            requires_count=requires,
        )

    def _check_stigmergy(self, issues: list[str]) -> StigmergyHealth:
        """Collect stigmergy metrics."""
        if self._stigmergy is None:
            return StigmergyHealth()

        total = self._stigmergy.count()
        if total == 0:
            return StigmergyHealth()

        # Query all markers by type to get distribution
        conn = self._stigmergy._conn
        cursor = conn.execute(
            "SELECT marker_type, COUNT(*) as cnt FROM stigmergy_markers GROUP BY marker_type"
        )
        markers_by_type = {row["marker_type"]: row["cnt"] for row in cursor}

        cursor = conn.execute("SELECT AVG(strength) as avg_s FROM stigmergy_markers")
        avg_strength = cursor.fetchone()["avg_s"] or 0.0

        cursor = conn.execute("SELECT COUNT(DISTINCT agent_id) as cnt FROM stigmergy_markers")
        unique_agents = cursor.fetchone()["cnt"]

        cursor = conn.execute("SELECT COUNT(DISTINCT target) as cnt FROM stigmergy_markers")
        unique_targets = cursor.fetchone()["cnt"]

        if avg_strength < 0.2:
            issues.append(f"Low average marker strength: {avg_strength:.2f} (markers may be stale)")

        return StigmergyHealth(
            total_markers=total,
            markers_by_type=markers_by_type,
            avg_strength=avg_strength,
            unique_agents=unique_agents,
            unique_targets=unique_targets,
        )

    def _check_scoring(self, issues: list[str]) -> ScoringHealth:
        """Collect phi scoring metrics."""
        if self._store is None:
            return ScoringHealth()

        conn = self._store._conn

        # Get all scores
        cursor = conn.execute("SELECT agent_id, skill_domain, phi_score FROM scores")
        rows = cursor.fetchall()
        if not rows:
            return ScoringHealth()

        scores = [row["phi_score"] for row in rows]
        agents: dict[str, float] = {}
        for row in rows:
            agent = row["agent_id"]
            # Average across domains for each agent
            if agent not in agents:
                agents[agent] = 0.0
            agents[agent] = max(agents[agent], row["phi_score"])

        # Count total outcomes
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM outcomes")
        total_outcomes = cursor.fetchone()["cnt"]

        avg = sum(scores) / len(scores)
        low_agents = [a for a, s in agents.items() if s < 0.3]
        if low_agents:
            issues.append(f"Low trust agents: {', '.join(low_agents)}")

        return ScoringHealth(
            total_agents=len(agents),
            total_outcomes=total_outcomes,
            avg_score=avg,
            min_score=min(scores),
            max_score=max(scores),
            scores_by_agent=agents,
        )

    def _check_voting(self, issues: list[str]) -> VotingHealth:
        """Collect voting metrics."""
        if self._store is None:
            return VotingHealth()

        conn = self._store._conn

        # Decision stats
        cursor = conn.execute("SELECT outcome, COUNT(*) as cnt FROM decisions GROUP BY outcome")
        outcomes: dict[str, int] = {}
        total = 0
        approved = 0
        for row in cursor:
            outcomes[row["outcome"]] = row["cnt"]
            total += row["cnt"]
            if row["outcome"] == "approved":
                approved += row["cnt"]

        if total == 0:
            return VotingHealth()

        # Average confidence
        cursor = conn.execute("SELECT AVG(confidence) as avg_c FROM vote_records")
        row = cursor.fetchone()
        avg_conf = row["avg_c"] if row and row["avg_c"] is not None else 0.0

        # Escalation count
        escalation_count = outcomes.get("escalated", 0)
        approval_rate = approved / total if total > 0 else 0.0

        if approval_rate < 0.5 and total >= 5:
            issues.append(f"Low approval rate: {approval_rate:.0%} ({approved}/{total})")
        if escalation_count > total * 0.3 and total >= 5:
            issues.append(f"High escalation rate: {escalation_count}/{total}")

        return VotingHealth(
            total_decisions=total,
            approval_rate=approval_rate,
            avg_confidence=avg_conf,
            escalation_count=escalation_count,
            outcomes=outcomes,
        )

    def _compute_grade(
        self,
        ig: IntentGraphHealth,
        stig: StigmergyHealth,
        score: ScoringHealth,
        vote: VotingHealth,
        issues: list[str],
    ) -> str:
        """Compute overall health grade (A-F) based on issues and metrics.

        Grading:
            A: 0 issues
            B: 1 issue
            C: 2 issues
            D: 3 issues
            F: 4+ issues
        """
        grades = {0: "A", 1: "B", 2: "C", 3: "D"}
        return grades.get(len(issues), "F")


def health_report(health: CoordinationHealth) -> str:
    """Render a CoordinationHealth as a human-readable text report.

    Args:
        health: The health metrics to render.

    Returns:
        Multi-line formatted text report.
    """
    lines: list[str] = []
    lines.append(f"=== Coordination Health Report [Grade: {health.grade}] ===")
    lines.append("")

    # Intent Graph
    ig = health.intent_graph
    lines.append("## Intent Graph")
    if ig.total_intents == 0:
        lines.append("  (no data)")
    else:
        lines.append(f"  Intents: {ig.total_intents} ({ig.agent_count} agents)")
        lines.append(
            f"  Stability: avg={ig.avg_stability:.2f}"
            f" min={ig.min_stability:.2f} max={ig.max_stability:.2f}"
        )
        lines.append(f"  Interfaces: {ig.provides_count} provides, {ig.requires_count} requires")
        lines.append(f"  Conflicts: {ig.conflict_count}")
    lines.append("")

    # Stigmergy
    stig = health.stigmergy
    lines.append("## Stigmergy")
    if stig.total_markers == 0:
        lines.append("  (no data)")
    else:
        lines.append(f"  Markers: {stig.total_markers} (avg strength={stig.avg_strength:.2f})")
        lines.append(f"  Agents: {stig.unique_agents}, Targets: {stig.unique_targets}")
        if stig.markers_by_type:
            type_str = ", ".join(f"{t}={c}" for t, c in sorted(stig.markers_by_type.items()))
            lines.append(f"  Types: {type_str}")
    lines.append("")

    # Scoring
    sc = health.scoring
    lines.append("## Phi Scoring")
    if sc.total_agents == 0:
        lines.append("  (no data)")
    else:
        lines.append(f"  Agents: {sc.total_agents}, Outcomes: {sc.total_outcomes}")
        lines.append(
            f"  Scores: avg={sc.avg_score:.2f} min={sc.min_score:.2f} max={sc.max_score:.2f}"
        )
    lines.append("")

    # Voting
    vt = health.voting
    lines.append("## Voting")
    if vt.total_decisions == 0:
        lines.append("  (no data)")
    else:
        lines.append(f"  Decisions: {vt.total_decisions}")
        lines.append(f"  Approval rate: {vt.approval_rate:.0%}")
        lines.append(f"  Avg confidence: {vt.avg_confidence:.2f}")
        if vt.outcomes:
            outcome_str = ", ".join(f"{o}={c}" for o, c in sorted(vt.outcomes.items()))
            lines.append(f"  Outcomes: {outcome_str}")
    lines.append("")

    # Issues
    if health.issues:
        lines.append("## Issues")
        for issue in health.issues:
            lines.append(f"  ! {issue}")
    else:
        lines.append("## Issues")
        lines.append("  None detected")

    return "\n".join(lines)
