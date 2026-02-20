"""Phi-weighted scoring engine for agent trust scores.

Tracks agent performance over time and weights their votes accordingly.
An agent that consistently produces approved work earns higher vote weight.

The "phi" in phi-weighted references the golden ratio as an aspirational
principle — the system seeks the optimal balance between exploitation
(trusting proven agents) and exploration (giving newer agents a chance).
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timezone

from convergent.protocol import Vote
from convergent.score_store import ScoreStore


class PhiScorer:
    """Calculates and manages phi-weighted trust scores for agents.

    Scores are per-agent AND per-skill-domain: an agent good at code
    review might be bad at testing. Recent outcomes are weighted more
    heavily than old ones via exponential decay.

    Args:
        store: Persistence layer for outcomes and scores.
        decay_rate: How fast old outcomes lose influence (higher = faster).
        prior_score: Starting assumption for new agents (0.5 = neutral).
        min_score: Floor — agents are never fully distrusted.
        max_score: Ceiling — agents are never fully trusted.
    """

    def __init__(
        self,
        store: ScoreStore,
        decay_rate: float = 0.05,
        prior_score: float = 0.5,
        min_score: float = 0.1,
        max_score: float = 0.95,
    ) -> None:
        self._store = store
        self._decay_rate = decay_rate
        self._prior_score = prior_score
        self._min_score = min_score
        self._max_score = max_score

    @staticmethod
    def calculate_phi_score(
        outcomes: list[tuple[str, float]],
        decay_rate: float = 0.05,
        prior_score: float = 0.5,
        min_score: float = 0.1,
        max_score: float = 0.95,
    ) -> float:
        """Calculate phi-weighted trust score for an agent in a skill domain.

        Each outcome is weighted by recency: weight = e^(-decay_rate * age_days)
        Recent outcomes count more. Very old outcomes fade toward zero influence.

        The prior_score (0.5 = neutral) is used when few observations exist,
        creating a Bayesian-style "pull toward center" that prevents extreme
        scores from small sample sizes.

        Args:
            outcomes: List of ("approved"/"rejected"/"failed", age_in_days) tuples.
            decay_rate: How fast old outcomes lose influence (higher = faster decay).
            prior_score: Starting assumption (0.5 = neutral).
            min_score: Floor — agents are never fully distrusted.
            max_score: Ceiling — agents are never fully trusted.

        Returns:
            Phi score between min_score and max_score.
        """
        if not outcomes:
            return prior_score

        weighted_successes = 0.0
        weighted_total = 0.0
        prior_weight = 2.0  # Equivalent to 2 "virtual" neutral observations

        for outcome, age_days in outcomes:
            weight = math.exp(-decay_rate * age_days)
            weighted_total += weight
            if outcome == "approved":
                weighted_successes += weight

        # Bayesian smoothing: mix observed rate with prior
        raw_score = (weighted_successes + prior_weight * prior_score) / (
            weighted_total + prior_weight
        )
        return max(min_score, min(max_score, raw_score))

    def record_outcome(
        self,
        agent_id: str,
        skill_domain: str,
        outcome: str,
    ) -> float:
        """Record a task outcome and recalculate the agent's phi score.

        Args:
            agent_id: The agent whose outcome is being recorded.
            skill_domain: The skill domain (e.g. "code_review", "testing").
            outcome: The outcome ("approved", "rejected", "failed").

        Returns:
            The newly calculated phi score.
        """
        self._store.record_outcome(agent_id, skill_domain, outcome)
        score = self._recalculate(agent_id, skill_domain)
        self._store.save_score(agent_id, skill_domain, score)
        return score

    def get_score(self, agent_id: str, skill_domain: str) -> float:
        """Get the current phi score for an agent in a skill domain.

        Returns the stored score if available, otherwise the prior score.

        Args:
            agent_id: The agent.
            skill_domain: The skill domain.

        Returns:
            The phi score (between min_score and max_score).
        """
        stored = self._store.get_score(agent_id, skill_domain)
        return stored if stored is not None else self._prior_score

    def get_all_scores(self, agent_id: str) -> dict[str, float]:
        """Get all phi scores for an agent across all skill domains.

        Args:
            agent_id: The agent.

        Returns:
            Dict mapping skill_domain to phi_score.
        """
        return self._store.get_all_scores(agent_id)

    def apply_vote_weight(self, vote: Vote) -> Vote:
        """Return a new Vote with weighted_score set based on phi score.

        weighted_score = phi_score * confidence

        The phi score is looked up from the store using the agent's ID and
        role, NOT taken from the self-reported ``vote.agent.phi_score``.
        This prevents agents from inflating their own vote weight.

        Args:
            vote: The vote to weight.

        Returns:
            New Vote instance with weighted_score calculated.
        """
        server_phi = self.get_score(vote.agent.agent_id, vote.agent.role)
        weighted = server_phi * vote.confidence
        return replace(vote, weighted_score=weighted)

    def _recalculate(self, agent_id: str, skill_domain: str) -> float:
        """Recalculate phi score from all outcomes. Idempotent.

        Args:
            agent_id: The agent.
            skill_domain: The skill domain.

        Returns:
            The newly calculated phi score.
        """
        raw_outcomes = self._store.get_outcomes(agent_id, skill_domain)
        now = datetime.now(timezone.utc)

        outcomes_with_age: list[tuple[str, float]] = []
        for outcome, timestamp_str in raw_outcomes:
            ts = datetime.fromisoformat(timestamp_str)
            # Ensure timezone-aware comparison
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (now - ts).total_seconds() / 86400.0
            outcomes_with_age.append((outcome, age_days))

        return self.calculate_phi_score(
            outcomes_with_age,
            decay_rate=self._decay_rate,
            prior_score=self._prior_score,
            min_score=self._min_score,
            max_score=self._max_score,
        )
