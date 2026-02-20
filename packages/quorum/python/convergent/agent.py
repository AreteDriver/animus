"""
Simulated Agent — demonstrates convergent behavior.

Each agent has a plan (a sequence of intents to publish over time).
Before publishing each intent, the agent runs it through the resolver
and adjusts based on the result. This simulates real agent behavior
without requiring actual Claude Code sessions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from convergent.intent import (
    Adjustment,
    Evidence,
    Intent,
    ResolutionResult,
)
from convergent.resolver import IntentResolver

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """A single step in an agent's plan."""

    intent: Intent
    on_adjust: Callable[[Intent, list[Adjustment]], Intent] | None = None
    post_evidence: list[Evidence] = field(default_factory=list)


@dataclass
class AgentLog:
    """Record of what an agent did during simulation."""

    agent_id: str
    published_intents: list[Intent] = field(default_factory=list)
    resolutions: list[ResolutionResult] = field(default_factory=list)
    adjustments_applied: list[Adjustment] = field(default_factory=list)
    conflicts_encountered: list[str] = field(default_factory=list)

    @property
    def total_adjustments(self) -> int:
        return len(self.adjustments_applied)

    @property
    def total_conflicts(self) -> int:
        return len(self.conflicts_encountered)

    @property
    def converged(self) -> bool:
        """Agent converged if it completed with no unresolved conflicts."""
        return self.total_conflicts == 0


class SimulatedAgent:
    """A simulated agent that publishes intents and resolves against the graph.

    Usage:
        agent = SimulatedAgent("agent-a", resolver)
        agent.plan([
            AgentAction(intent=Intent(...), post_evidence=[...]),
            AgentAction(intent=Intent(...), post_evidence=[...]),
        ])
        log = agent.execute_all()
    """

    def __init__(self, agent_id: str, resolver: IntentResolver) -> None:
        self.agent_id = agent_id
        self.resolver = resolver
        self.actions: list[AgentAction] = []
        self.log = AgentLog(agent_id=agent_id)
        self._step = 0

    def plan(self, actions: list[AgentAction]) -> None:
        """Set the agent's plan — a sequence of actions to execute."""
        self.actions = actions
        self._step = 0

    @property
    def has_next(self) -> bool:
        return self._step < len(self.actions)

    def execute_step(self) -> ResolutionResult | None:
        """Execute the next step of the agent's plan.

        1. Take the planned intent
        2. Resolve it against the current graph
        3. Apply adjustments
        4. Publish the (possibly adjusted) intent
        5. Add post-execution evidence

        Returns the resolution result, or None if plan is complete.
        """
        if not self.has_next:
            return None

        action = self.actions[self._step]
        intent = action.intent

        logger.info(f"[{self.agent_id}] Step {self._step}: Resolving '{intent.intent}'")

        # 1. Resolve against current graph state
        result = self.resolver.resolve(intent)
        self.log.resolutions.append(result)

        # 2. Apply adjustments
        if result.has_adjustments:
            for adj in result.adjustments:
                logger.info(f"[{self.agent_id}]   Adjustment ({adj.kind}): {adj.description}")
                self.log.adjustments_applied.append(adj)

            # Apply ConsumeInstead: remove duplicate provisions
            consume_names = set()
            for adj in result.adjustments:
                if adj.kind == "ConsumeInstead":
                    # Extract the provision name being dropped
                    for prov in intent.provides:
                        consume_names.add(prov.name)

            if consume_names:
                intent.provides = [p for p in intent.provides if p.name not in consume_names]

            # Apply AdoptConstraint: add constraints to intent
            for constraint in result.adopted_constraints:
                if constraint not in intent.constraints:
                    intent.constraints.append(constraint)

            # Let custom callback modify intent if provided
            if action.on_adjust:
                intent = action.on_adjust(intent, result.adjustments)

        # 3. Record conflicts
        for conflict in result.conflicts:
            self.log.conflicts_encountered.append(conflict.description)
            logger.warning(f"[{self.agent_id}]   CONFLICT: {conflict.description}")

        # 4. Publish to graph
        computed_stability = self.resolver.publish(intent)
        self.log.published_intents.append(intent)

        logger.info(
            f"[{self.agent_id}]   Published '{intent.intent}' (stability: {computed_stability:.2f})"
        )

        # 5. Add post-execution evidence (simulates tests passing, code committed)
        for evidence in action.post_evidence:
            intent.add_evidence(evidence)

        self._step += 1
        return result

    def execute_all(self) -> AgentLog:
        """Execute all planned steps. Returns the complete log."""
        while self.has_next:
            self.execute_step()
        return self.log


class SimulationRunner:
    """Runs multiple agents in round-robin to simulate parallel execution.

    In real life, agents run truly in parallel. In simulation, we
    interleave their steps to show how they observe and adapt to
    each other's intents over time.
    """

    def __init__(self, resolver: IntentResolver) -> None:
        self.resolver = resolver
        self.agents: list[SimulatedAgent] = []

    def add_agent(self, agent: SimulatedAgent) -> None:
        self.agents.append(agent)

    def run(self) -> SimulationResult:
        """Run all agents in round-robin until all complete."""
        round_num = 0
        rounds: list[RoundLog] = []

        while any(a.has_next for a in self.agents):
            round_num += 1
            round_log = RoundLog(round_number=round_num)

            for agent in self.agents:
                if agent.has_next:
                    result = agent.execute_step()
                    if result:
                        round_log.resolutions.append((agent.agent_id, result))

            rounds.append(round_log)
            logger.info(
                f"--- Round {round_num} complete. "
                f"Graph size: {self.resolver.intent_count} intents ---"
            )

        # Compute convergence metrics
        all_converged = all(a.log.converged for a in self.agents)
        total_adjustments = sum(a.log.total_adjustments for a in self.agents)
        total_conflicts = sum(a.log.total_conflicts for a in self.agents)

        return SimulationResult(
            rounds=rounds,
            agent_logs={a.agent_id: a.log for a in self.agents},
            all_converged=all_converged,
            total_adjustments=total_adjustments,
            total_conflicts=total_conflicts,
            total_intents=self.resolver.intent_count,
        )


@dataclass
class RoundLog:
    """Record of a single simulation round."""

    round_number: int
    resolutions: list[tuple[str, ResolutionResult]] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Final result of a simulation run."""

    rounds: list[RoundLog]
    agent_logs: dict[str, AgentLog]
    all_converged: bool
    total_adjustments: int
    total_conflicts: int
    total_intents: int

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "CONVERGENT SIMULATION RESULTS",
            "=" * 60,
            f"Agents:           {len(self.agent_logs)}",
            f"Rounds:           {len(self.rounds)}",
            f"Total intents:    {self.total_intents}",
            f"Adjustments made: {self.total_adjustments}",
            f"Conflicts:        {self.total_conflicts}",
            f"All converged:    {'YES' if self.all_converged else 'NO'}",
            "",
        ]

        for agent_id, log in self.agent_logs.items():
            lines.append(f"  [{agent_id}]")
            lines.append(f"    Intents published:  {len(log.published_intents)}")
            lines.append(f"    Adjustments:        {log.total_adjustments}")
            lines.append(f"    Conflicts:          {log.total_conflicts}")
            lines.append(f"    Converged:          {'YES' if log.converged else 'NO'}")

            if log.adjustments_applied:
                lines.append("    Adjustments applied:")
                for adj in log.adjustments_applied:
                    lines.append(f"      - [{adj.kind}] {adj.description}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
