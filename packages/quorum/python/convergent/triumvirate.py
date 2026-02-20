"""Triumvirate voting engine for multi-agent consensus.

Collects votes from agents, applies phi-weighted scoring, evaluates
quorum rules, and produces a Decision. Works with 2-5 agents (not
hardcoded to exactly 3).

The name "Triumvirate" references the classical Roman governance model
of three rulers sharing authority — here, multiple agents share
decision-making power weighted by their track records.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from convergent.coordination_config import CoordinationConfig
from convergent.protocol import (
    ConsensusRequest,
    Decision,
    DecisionOutcome,
    QuorumLevel,
    Vote,
    VoteChoice,
)
from convergent.scoring import PhiScorer

logger = logging.getLogger(__name__)


class Triumvirate:
    """Voting engine that manages consensus requests, votes, and decisions.

    Args:
        scorer: PhiScorer for weighting votes by agent trust.
        config: Coordination configuration.
    """

    def __init__(
        self,
        scorer: PhiScorer,
        config: CoordinationConfig,
        store: object | None = None,
    ) -> None:
        self._scorer = scorer
        self._config = config
        self._store = store  # Optional ScoreStore for decision persistence
        self._requests: dict[str, ConsensusRequest] = {}
        self._votes: dict[str, list[Vote]] = {}
        self._decisions: dict[str, Decision] = {}

    def create_request(
        self,
        task_id: str,
        question: str,
        context: str,
        quorum: QuorumLevel | None = None,
        artifacts: list[str] | None = None,
    ) -> ConsensusRequest:
        """Create a new consensus request for agents to vote on.

        Args:
            task_id: Gorgon task this relates to.
            question: What are we deciding?
            context: Relevant information for voters.
            quorum: Required agreement level. Defaults to config default.
            artifacts: File paths, PR URLs, etc.

        Returns:
            The created ConsensusRequest.
        """
        request_id = str(uuid.uuid4())
        request = ConsensusRequest(
            request_id=request_id,
            task_id=task_id,
            question=question,
            context=context,
            quorum=quorum or self._config.default_quorum,
            artifacts=artifacts or [],
            timeout_seconds=self._config.vote_timeout_seconds,
        )
        self._requests[request_id] = request
        self._votes[request_id] = []
        logger.info("Created consensus request %s for task %s", request_id, task_id)
        return request

    def submit_vote(self, request_id: str, vote: Vote) -> None:
        """Submit a vote for a consensus request.

        Args:
            request_id: The request to vote on.
            vote: The vote to submit.

        Raises:
            KeyError: If request_id is not found.
            ValueError: If a decision has already been made.
        """
        if request_id not in self._requests:
            raise KeyError(f"Unknown request: {request_id}")
        if request_id in self._decisions:
            raise ValueError(f"Decision already made for request: {request_id}")

        weighted_vote = self._scorer.apply_vote_weight(vote)
        self._votes[request_id].append(weighted_vote)
        logger.info(
            "Vote submitted for %s by %s: %s (weight=%.3f)",
            request_id,
            vote.agent.agent_id,
            vote.choice.value,
            weighted_vote.weighted_score,
        )

    def evaluate(self, request_id: str) -> Decision:
        """Evaluate votes and produce a decision for a consensus request.

        Args:
            request_id: The request to evaluate.

        Returns:
            The Decision outcome.

        Raises:
            KeyError: If request_id is not found.
        """
        if request_id not in self._requests:
            raise KeyError(f"Unknown request: {request_id}")

        request = self._requests[request_id]
        votes = self._votes[request_id]

        # Check for timeout
        requested_at = datetime.fromisoformat(request.requested_at)
        if requested_at.tzinfo is None:
            requested_at = requested_at.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        elapsed = (now - requested_at).total_seconds()

        if not votes or elapsed > request.timeout_seconds:
            decision = Decision(
                request=request,
                votes=votes,
                outcome=DecisionOutcome.DEADLOCK,
                reasoning_summary="No votes received or timeout exceeded",
            )
            self._decisions[request_id] = decision
            self._persist_decision(decision)
            return decision

        # Check for escalations first
        if any(v.choice is VoteChoice.ESCALATE for v in votes):
            decision = Decision(
                request=request,
                votes=votes,
                outcome=DecisionOutcome.ESCALATED,
                reasoning_summary=self._summarize_reasoning(votes),
            )
            self._decisions[request_id] = decision
            self._persist_decision(decision)
            return decision

        # Calculate weighted totals
        total_approve = sum(v.weighted_score for v in votes if v.choice is VoteChoice.APPROVE)
        total_reject = sum(v.weighted_score for v in votes if v.choice is VoteChoice.REJECT)

        # Evaluate quorum
        outcome = self._evaluate_quorum(request.quorum, votes, total_approve, total_reject)

        decision = Decision(
            request=request,
            votes=votes,
            outcome=outcome,
            total_weighted_approve=total_approve,
            total_weighted_reject=total_reject,
            reasoning_summary=self._summarize_reasoning(votes),
        )
        self._decisions[request_id] = decision
        self._persist_decision(decision)
        logger.info(
            "Decision for %s: %s (approve=%.3f, reject=%.3f)",
            request_id,
            outcome.value,
            total_approve,
            total_reject,
        )
        return decision

    def get_decision(self, request_id: str) -> Decision | None:
        """Get the decision for a request, if one has been made.

        Args:
            request_id: The request to look up.

        Returns:
            The Decision, or None if not yet evaluated.
        """
        return self._decisions.get(request_id)

    def get_vote_history(self, task_id: str) -> list[Decision]:
        """Get all decisions for a given task.

        Args:
            task_id: The task to look up.

        Returns:
            List of Decisions for this task, ordered by creation time.
        """
        return [d for d in self._decisions.values() if d.request.task_id == task_id]

    def _persist_decision(self, decision: Decision) -> None:
        """Persist a decision to the score store (graceful degradation)."""
        if self._store is None:
            return
        try:
            self._store.record_decision(decision)  # type: ignore[attr-defined]
        except Exception:
            logger.warning(
                "Failed to persist decision %s",
                decision.request.request_id,
                exc_info=True,
            )

    @staticmethod
    def _evaluate_quorum(
        quorum: QuorumLevel,
        votes: list[Vote],
        total_approve: float,
        total_reject: float,
    ) -> DecisionOutcome:
        """Evaluate whether quorum rules are satisfied.

        Args:
            quorum: The required quorum level.
            votes: All submitted votes.
            total_approve: Sum of weighted approve scores.
            total_reject: Sum of weighted reject scores.

        Returns:
            The decision outcome based on quorum rules.
        """
        if quorum is QuorumLevel.ANY:
            # At least one agent approves
            if total_approve > 0:
                return DecisionOutcome.APPROVED
            return DecisionOutcome.REJECTED

        if quorum is QuorumLevel.MAJORITY:
            # Weighted approve > weighted reject
            if total_approve > total_reject:
                return DecisionOutcome.APPROVED
            if total_reject > total_approve:
                return DecisionOutcome.REJECTED
            # Tie: highest individual weighted_score breaks it
            return _break_tie(votes)

        if quorum in (QuorumLevel.UNANIMOUS, QuorumLevel.UNANIMOUS_HUMAN):
            # All non-abstain votes must be approve
            substantive = [v for v in votes if v.choice not in (VoteChoice.ABSTAIN,)]
            if not substantive:
                return DecisionOutcome.DEADLOCK
            if all(v.choice is VoteChoice.APPROVE for v in substantive):
                return DecisionOutcome.APPROVED
            return DecisionOutcome.REJECTED

        return DecisionOutcome.DEADLOCK  # pragma: no cover

    @staticmethod
    def _summarize_reasoning(votes: list[Vote]) -> str:
        """Build a summary of vote reasoning.

        Args:
            votes: All votes cast.

        Returns:
            Newline-separated summary of each vote's reasoning.
        """
        parts = []
        for v in votes:
            parts.append(f"[{v.choice.value}] {v.agent.agent_id}: {v.reasoning}")
        return "\n".join(parts)


def _break_tie(votes: list[Vote]) -> DecisionOutcome:
    """Break a tie using the highest individual weighted_score.

    Args:
        votes: All submitted votes.

    Returns:
        APPROVED if highest-weighted vote is approve, else REJECTED.
        DEADLOCK if no substantive votes exist.
    """
    substantive = [v for v in votes if v.choice in (VoteChoice.APPROVE, VoteChoice.REJECT)]
    if not substantive:
        return DecisionOutcome.DEADLOCK
    best = max(substantive, key=lambda v: v.weighted_score)
    if best.choice is VoteChoice.APPROVE:
        return DecisionOutcome.APPROVED
    return DecisionOutcome.REJECTED
