"""Reward aggregation from rubric scores.

Converts structured dimension scores into training signals for
reinforcement finetuning (RFT). Per ATLAS paper, rubric-based rewards
outperform scalar judge rewards.

Supports:
- Weighted combination across dimensions
- Task-type-specific reward shaping
- Min/max thresholds for pass/fail gating
- Selective reward (only reward dimensions above threshold)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from animus.eval.rubric import Score, ScoreLevel
from animus.logging import get_logger

logger = get_logger("eval.rewards")


@dataclass
class RewardConfig:
    """Configuration for reward aggregation."""

    # Minimum weighted score to receive any reward
    min_threshold: float = 0.3
    # Maximum reward value
    max_reward: float = 1.0
    # Reward shaping: linear, sigmoid, or step
    shaping: str = "linear"  # linear, sigmoid, step
    # Per-task-type dimension weights override
    task_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    # If True, only reward dimensions that score >= threshold individually
    selective_reward: bool = False
    # Per-dimension minimum for selective reward
    selective_threshold: float = 0.5
    # Penalty multiplier for critical failures
    critical_penalty: float = -1.0


class RewardAggregator:
    """Aggregate rubric scores into scalar rewards for RFT.

        Unlike raw weighted_score, this applies shaping and thresholds
    to produce training-suitable reward signals.
    """

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()

    def _apply_shaping(self, score: float) -> float:
        """Apply reward shaping function."""
        if self.config.shaping == "linear":
            return score
        elif self.config.shaping == "sigmoid":
            import math

            # Sigmoid centered at 0.5 with steepness 10
            return 1.0 / (1.0 + math.exp(-10 * (score - 0.5)))
        elif self.config.shaping == "step":
            if score >= 0.7:
                return 1.0
            elif score >= 0.5:
                return 0.5
            return 0.0
        return score

    def aggregate(self, score: Score, task_type: str = "general") -> float:
        """Aggregate rubric Score into a scalar reward.

        Args:
            score: The rubric Score to aggregate.
            task_type: Task type for weight override lookup.

        Returns:
            Scalar reward value (can be negative for critical failures).
        """
        # Check for critical failures on required dimensions
        if score.has_critical_failure:
            logger.debug("Critical failure detected, applying penalty")
            return self.config.critical_penalty

        # Get base weighted score
        base_score = score.weighted_score

        # Apply min threshold gate
        if base_score < self.config.min_threshold:
            return 0.0

        # Apply shaping
        shaped = self._apply_shaping(base_score)

        # Apply task-type weight overrides
        weights = self.config.task_weights.get(task_type, {})
        if weights and score.dimension_scores:
            # Recalculate with task-specific weights
            total = 0.0
            weight_sum = 0.0
            for ds in score.dimension_scores:
                weight = weights.get(ds.dimension_name, 1.0)
                total += ds.raw_score * weight
                weight_sum += weight
            if weight_sum > 0:
                shaped = self._apply_shaping(total / weight_sum)

        # Selective reward: only reward if all dimensions above threshold
        if self.config.selective_reward:
            all_above = all(
                ds.raw_score >= self.config.selective_threshold for ds in score.dimension_scores
            )
            if not all_above:
                return 0.0

        # Clamp to max_reward
        return min(shaped, self.config.max_reward)

    def per_dimension_rewards(self, score: Score) -> dict[str, float]:
        """Return individual rewards per dimension.

        Useful for diagnostic feedback and targeted improvement.
        """
        rewards = {}
        for ds in score.dimension_scores:
            if ds.level == ScoreLevel.CRITICAL:
                rewards[ds.dimension_name] = self.config.critical_penalty
            elif ds.raw_score < self.config.min_threshold:
                rewards[ds.dimension_name] = 0.0
            else:
                rewards[ds.dimension_name] = self._apply_shaping(ds.raw_score)
        return rewards

    def improvement_direction(self, score: Score) -> list[str]:
        """Generate actionable improvement directions from failures.

        Returns list of concrete suggestions based on lowest-scoring dimensions.
        """
        suggestions: list[str] = []
        failures = score.failures()

        if not failures:
            return ["No critical failures — focus on raising GOOD to EXCELLENT"]

        # Sort by raw_score ascending
        failures.sort(key=lambda ds: ds.raw_score)

        for ds in failures:
            if ds.level == ScoreLevel.CRITICAL:
                suggestions.append(f"CRITICAL: {ds.dimension_name} — {ds.justification}")
            else:
                suggestions.append(f"POOR: {ds.dimension_name} — {ds.justification}")

        return suggestions
