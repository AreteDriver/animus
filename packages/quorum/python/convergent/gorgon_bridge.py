"""Gorgon integration bridge — single entry point for coordination protocol.

Gorgon's pipeline calls this bridge to enrich agent prompts with stigmergy
context and flocking constraints, run Triumvirate consensus votes, record
task outcomes, and update phi scores. Complements the existing
``create_delegation_checker()`` factory which handles intent graph delegation.

All subsystems (scorer, triumvirate, stigmergy, flocking, signal bus) are
initialized lazily and degrade gracefully if not configured.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from convergent.coordination_config import CoordinationConfig
from convergent.flocking import FlockingCoordinator
from convergent.protocol import (
    AgentIdentity,
    Decision,
    Signal,
    Vote,
    VoteChoice,
)
from convergent.score_store import ScoreStore
from convergent.scoring import PhiScorer
from convergent.signal_bus import SignalBus
from convergent.sqlite_signal_backend import SQLiteSignalBackend
from convergent.stigmergy import StigmergyField
from convergent.triumvirate import Triumvirate

logger = logging.getLogger(__name__)


class GorgonBridge:
    """Single entry point for Gorgon to use Convergent's coordination protocol.

    Initializes all Phase 3 subsystems from a CoordinationConfig and exposes
    a clean API for Gorgon's pipeline to call with minimal changes.

    Signal bus backend selection:
    - ``db_path == ":memory:"`` → no signal bus (no filesystem/persistence)
    - ``signal_bus_type == "sqlite"`` → SQLiteSignalBackend (.signals.db)
    - ``signal_bus_type == "filesystem"`` → FilesystemSignalBackend

    Args:
        config: Coordination configuration. Uses defaults if not provided.
    """

    def __init__(self, config: CoordinationConfig | None = None) -> None:
        self._config = config or CoordinationConfig()

        # Scoring
        self._store = ScoreStore(self._config.db_path)
        self._scorer = PhiScorer(self._store)

        # Voting
        self._triumvirate = Triumvirate(self._scorer, self._config, store=self._store)

        # Stigmergy (separate DB to keep concerns isolated)
        stigmergy_db = self._config.db_path
        if stigmergy_db != ":memory:":
            stigmergy_db = str(Path(stigmergy_db).with_suffix(".stigmergy.db"))
        self._stigmergy = StigmergyField(
            db_path=stigmergy_db,
            evaporation_rate=self._config.stigmergy_evaporation_rate,
        )

        # Flocking
        self._flocking = FlockingCoordinator(self._stigmergy)

        # Signal bus — smart backend selection
        self._signal_bus: SignalBus | None = None
        if self._config.db_path != ":memory:":
            if self._config.signal_bus_type == "sqlite":
                signal_db = str(Path(self._config.db_path).with_suffix(".signals.db"))
                backend = SQLiteSignalBackend(signal_db)
                self._signal_bus = SignalBus(backend=backend)
            else:
                signals_dir = Path(self._config.db_path).parent / "signals"
                self._signal_bus = SignalBus(signals_dir=signals_dir)

    def enrich_prompt(
        self,
        agent_id: str,
        task_description: str,
        file_paths: list[str],
        current_work: str = "",
    ) -> str:
        """Build additional context to inject into an agent's prompt.

        Combines stigmergy markers, flocking constraints, and the agent's
        phi score context into a single string.

        Args:
            agent_id: The agent about to work.
            task_description: The original task description.
            file_paths: Files the agent plans to work on.
            current_work: Description of current work (for cohesion check).

        Returns:
            Multi-line context string, or empty string if nothing relevant.
        """
        sections: list[str] = []

        # Stigmergy context
        stigmergy_ctx = self._stigmergy.get_context_for_agent(file_paths)
        if stigmergy_ctx:
            sections.append(stigmergy_ctx)

        # Flocking constraints
        flocking_ctx = self._flocking.generate_constraints(
            agent_id=agent_id,
            task_description=task_description,
            current_work=current_work or task_description,
            file_paths=file_paths,
        )
        if flocking_ctx:
            sections.append(flocking_ctx)

        # Agent score context
        scores = self._scorer.get_all_scores(agent_id)
        if scores:
            lines = ["## Your Trust Scores", ""]
            for domain, score in sorted(scores.items()):
                lines.append(f"- {domain}: {score:.2f}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def request_consensus(
        self,
        task_id: str,
        question: str,
        context: str,
        quorum: str | None = None,
        artifacts: list[str] | None = None,
    ) -> str:
        """Create a consensus request for agents to vote on.

        Args:
            task_id: Gorgon task this relates to.
            question: What are we deciding?
            context: Relevant information for voters.
            quorum: Quorum level string (e.g. "majority"). Uses config default if None.
            artifacts: File paths, PR URLs, etc.

        Returns:
            The request_id for submitting votes.
        """
        from convergent.protocol import QuorumLevel

        q = QuorumLevel(quorum) if quorum else None
        request = self._triumvirate.create_request(
            task_id=task_id,
            question=question,
            context=context,
            quorum=q,
            artifacts=artifacts,
        )
        return request.request_id

    def submit_agent_vote(
        self,
        request_id: str,
        agent_id: str,
        role: str,
        model: str,
        choice: str,
        confidence: float,
        reasoning: str,
    ) -> None:
        """Submit a vote from an agent.

        Args:
            request_id: The consensus request to vote on.
            agent_id: The voting agent's ID.
            role: The agent's role (e.g. "reviewer", "tester").
            model: The agent's model (e.g. "claude:sonnet").
            choice: Vote choice string (e.g. "approve", "reject").
            confidence: How confident the agent is (0.0-1.0).
            reasoning: Why the agent voted this way.
        """
        phi = self._scorer.get_score(agent_id, role)
        agent = AgentIdentity(
            agent_id=agent_id,
            role=role,
            model=model,
            phi_score=phi,
        )
        vote = Vote(
            agent=agent,
            choice=VoteChoice(choice),
            confidence=confidence,
            reasoning=reasoning,
        )
        self._triumvirate.submit_vote(request_id, vote)

    def evaluate(self, request_id: str) -> Decision:
        """Evaluate votes and produce a decision.

        Args:
            request_id: The consensus request to evaluate.

        Returns:
            The Decision outcome.
        """
        return self._triumvirate.evaluate(request_id)

    def get_decision(self, request_id: str) -> Decision | None:
        """Get the decision for a request, if one has been made.

        Args:
            request_id: The request to look up.

        Returns:
            The Decision, or None if not yet evaluated.
        """
        return self._triumvirate.get_decision(request_id)

    def record_task_outcome(
        self,
        agent_id: str,
        skill_domain: str,
        outcome: str,
        file_paths: list[str] | None = None,
    ) -> float:
        """Record a task outcome and update the agent's phi score.

        Also leaves stigmergy markers for modified files.

        Args:
            agent_id: The agent that completed the task.
            skill_domain: The skill domain (e.g. "code_review", "testing").
            outcome: The outcome ("approved", "rejected", "failed").
            file_paths: Files the agent modified (leaves file_modified markers).

        Returns:
            The agent's updated phi score for this domain.
        """
        score = self._scorer.record_outcome(agent_id, skill_domain, outcome)

        # Leave stigmergy markers for modified files
        if file_paths:
            for path in file_paths:
                self._stigmergy.leave_marker(
                    agent_id=agent_id,
                    marker_type="file_modified",
                    target=path,
                    content=f"{outcome} by {agent_id} in {skill_domain}",
                )

        # Publish signal
        if self._signal_bus is not None:
            self._signal_bus.publish(
                Signal(
                    signal_type="task_outcome",
                    source_agent=agent_id,
                    payload=json.dumps({"skill_domain": skill_domain, "outcome": outcome}),
                )
            )

        logger.info(
            "Recorded %s for %s in %s (phi=%.3f)",
            outcome,
            agent_id,
            skill_domain,
            score,
        )
        return score

    def get_agent_score(self, agent_id: str, skill_domain: str) -> float:
        """Get an agent's current phi score.

        Args:
            agent_id: The agent to look up.
            skill_domain: The skill domain.

        Returns:
            The phi score (0.5 prior if no history).
        """
        return self._scorer.get_score(agent_id, skill_domain)

    def get_vote_history(self, task_id: str) -> list[Decision]:
        """Get all decisions for a given task.

        Args:
            task_id: The task to look up.

        Returns:
            List of Decisions for this task.
        """
        return self._triumvirate.get_vote_history(task_id)

    def get_decision_history(
        self,
        task_id: str | None = None,
        outcome: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query persisted decision history.

        Args:
            task_id: Filter by task. None for all.
            outcome: Filter by outcome string (e.g. "approved"). None for all.
            since: ISO 8601 timestamp cutoff. None for all.
            limit: Maximum results (default 100).

        Returns:
            List of decision summary dicts.
        """
        return self._store.get_decision_history(
            task_id=task_id, outcome=outcome, since=since, limit=limit
        )

    def get_agent_vote_stats(self, agent_id: str) -> dict:
        """Get voting statistics for an agent.

        Args:
            agent_id: The agent to look up.

        Returns:
            Dict with total, approve/reject/abstain/escalate counts,
            avg_confidence.
        """
        return self._store.get_agent_vote_stats(agent_id)

    def leave_marker(
        self,
        agent_id: str,
        marker_type: str,
        target: str,
        content: str,
    ) -> None:
        """Leave a stigmergy marker directly.

        Args:
            agent_id: The agent leaving the marker.
            marker_type: Marker type (e.g. "known_issue", "pattern_found").
            target: What this refers to (file path, module name).
            content: The information to convey.
        """
        self._stigmergy.leave_marker(agent_id, marker_type, target, content)

    def evaporate_markers(self) -> int:
        """Run stigmergy evaporation to decay old markers.

        Returns:
            Count of markers removed.
        """
        return self._stigmergy.evaporate()

    @property
    def scorer(self) -> PhiScorer:
        """Access the underlying PhiScorer."""
        return self._scorer

    @property
    def triumvirate(self) -> Triumvirate:
        """Access the underlying Triumvirate."""
        return self._triumvirate

    @property
    def stigmergy(self) -> StigmergyField:
        """Access the underlying StigmergyField."""
        return self._stigmergy

    @property
    def flocking(self) -> FlockingCoordinator:
        """Access the underlying FlockingCoordinator."""
        return self._flocking

    @property
    def signal_bus(self) -> SignalBus | None:
        """Access the underlying SignalBus (None if in-memory mode)."""
        return self._signal_bus

    def close(self) -> None:
        """Close all database connections and stop signal bus."""
        self._store.close()
        self._stigmergy.close()
        if self._signal_bus is not None:
            self._signal_bus.close()
