"""
Layer 3: Economics — Optimization.

Cost-of-rework estimation, token budget tracking, and confidence-based
escalation. Makes escalation economic rather than conversational.

The key insight: escalation decisions should be cost-optimized, not
status-based. Escalate when:
    P(wrong_auto_resolve) * cost_of_rework > cost_of_escalation

This turns the "should we ask a human?" question into a pure
expected-value calculation.

Cost model components:
  - Token costs: per-resolve, per-escalation LLM calls
  - Rework costs: redoing work after a wrong auto-resolve
  - Human costs: interrupting a human for a decision
  - Opportunity costs: blocking other agents while waiting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Economic model for coordination decisions.

    All costs are in abstract "cost units" — teams can map these to
    dollars, tokens, or time. The ratios matter more than absolutes.
    """

    # Per-operation costs
    token_cost_per_resolve: float = 0.001
    token_cost_per_escalation: float = 0.01
    rework_cost_per_conflict: float = 0.10
    human_escalation_cost: float = 1.00

    # Risk multipliers
    rework_probability_at_low_confidence: float = 0.5
    rework_probability_at_high_confidence: float = 0.05
    confidence_threshold: float = 0.7


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    """Token/cost budget for a coordination session.

    Tracks resource consumption and enforces limits.
    """

    max_tokens: int = 100_000
    max_cost: float = 10.0
    tokens_used: int = 0
    cost_incurred: float = 0.0
    resolves_performed: int = 0
    escalations_performed: int = 0

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def remaining_cost(self) -> float:
        return max(0.0, self.max_cost - self.cost_incurred)

    @property
    def exhausted(self) -> bool:
        return self.remaining_tokens == 0 or self.remaining_cost <= 0.0

    def can_afford(self, cost: float) -> bool:
        """Check if the budget can absorb this cost."""
        return self.cost_incurred + cost <= self.max_cost

    def charge(self, cost: float, tokens: int = 0) -> bool:
        """Charge cost to the budget. Returns False if over budget."""
        if not self.can_afford(cost):
            return False
        self.cost_incurred += cost
        self.tokens_used += tokens
        return True

    def record_resolve(self, cost: float) -> bool:
        """Record a resolution operation. Returns False if over budget."""
        charged = self.charge(cost)
        self.resolves_performed += 1
        return charged

    def record_escalation(self, cost: float) -> bool:
        """Record an escalation operation. Returns False if over budget."""
        charged = self.charge(cost)
        self.escalations_performed += 1
        return charged

    @property
    def utilization(self) -> float:
        """Budget utilization as a fraction (0.0–1.0)."""
        if self.max_cost <= 0:
            return 1.0
        return min(1.0, self.cost_incurred / self.max_cost)


# ---------------------------------------------------------------------------
# Escalation decisions
# ---------------------------------------------------------------------------


class EscalationAction(str, Enum):
    """What to do when a conflict is detected."""

    AUTO_RESOLVE = "auto_resolve"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    DEFER = "defer"
    BLOCK = "block"


@dataclass
class EscalationDecision:
    """The economic recommendation for handling a conflict.

    Includes the expected cost of each option and the recommended action.
    """

    action: EscalationAction
    expected_cost_auto: float
    expected_cost_escalate: float
    confidence: float
    reasoning: str

    @property
    def savings(self) -> float:
        """How much the recommended action saves vs the alternative."""
        if self.action == EscalationAction.AUTO_RESOLVE:
            return self.expected_cost_escalate - self.expected_cost_auto
        elif self.action == EscalationAction.ESCALATE_TO_HUMAN:
            return self.expected_cost_auto - self.expected_cost_escalate
        return 0.0


class EscalationPolicy:
    """Determines when to escalate based on economics, not conversation.

    The core formula:
        expected_cost_auto = P(rework) * rework_cost + resolve_cost
        expected_cost_escalate = escalation_cost + human_cost

        if expected_cost_auto > expected_cost_escalate:
            escalate (cheaper to ask a human)
        else:
            auto-resolve (cheaper to take the risk)

    This replaces the conversational pattern of "let me ask my coordinator"
    with a pure expected-value calculation.
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.budget = budget or Budget()

    def evaluate(
        self,
        confidence: float,
        stability_gap: float,
        num_affected_agents: int = 1,
    ) -> EscalationDecision:
        """Evaluate whether to auto-resolve or escalate.

        Args:
            confidence: Confidence in the auto-resolution (0.0–1.0).
            stability_gap: Difference in stability between conflicting intents.
            num_affected_agents: How many agents are affected by this decision.

        Returns:
            EscalationDecision with the economically optimal action.
        """
        cm = self.cost_model

        # Probability of rework if we auto-resolve
        if confidence >= cm.confidence_threshold:
            p_rework = cm.rework_probability_at_high_confidence
        else:
            p_rework = cm.rework_probability_at_low_confidence

        # Scale rework cost by affected agents
        rework_cost = cm.rework_cost_per_conflict * num_affected_agents

        # Expected costs
        expected_auto = (p_rework * rework_cost) + cm.token_cost_per_resolve
        expected_escalate = cm.token_cost_per_escalation + cm.human_escalation_cost

        # Budget check: if we can't afford escalation, auto-resolve
        if not self.budget.can_afford(expected_escalate):
            return EscalationDecision(
                action=EscalationAction.AUTO_RESOLVE,
                expected_cost_auto=expected_auto,
                expected_cost_escalate=expected_escalate,
                confidence=confidence,
                reasoning="Budget insufficient for escalation; auto-resolving",
            )

        # Budget check: if budget is nearly exhausted, defer
        if self.budget.utilization > 0.95:
            return EscalationDecision(
                action=EscalationAction.DEFER,
                expected_cost_auto=expected_auto,
                expected_cost_escalate=expected_escalate,
                confidence=confidence,
                reasoning="Budget nearly exhausted; deferring decision",
            )

        # Economic comparison
        if expected_auto <= expected_escalate:
            action = EscalationAction.AUTO_RESOLVE
            reasoning = (
                f"Auto-resolve is cheaper: ${expected_auto:.4f} vs "
                f"${expected_escalate:.4f} (confidence={confidence:.2f}, "
                f"P(rework)={p_rework:.2f})"
            )
        else:
            action = EscalationAction.ESCALATE_TO_HUMAN
            reasoning = (
                f"Escalation is cheaper: ${expected_escalate:.4f} vs "
                f"${expected_auto:.4f} (confidence={confidence:.2f}, "
                f"P(rework)={p_rework:.2f})"
            )

        return EscalationDecision(
            action=action,
            expected_cost_auto=expected_auto,
            expected_cost_escalate=expected_escalate,
            confidence=confidence,
            reasoning=reasoning,
        )

    def evaluate_batch(
        self,
        conflicts: list[dict[str, float]],
    ) -> list[EscalationDecision]:
        """Evaluate multiple conflicts economically.

        Each conflict dict should have: confidence, stability_gap,
        and optionally num_affected_agents.
        """
        return [
            self.evaluate(
                confidence=c.get("confidence", 0.5),
                stability_gap=c.get("stability_gap", 0.0),
                num_affected_agents=int(c.get("num_affected_agents", 1)),
            )
            for c in conflicts
        ]


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


@dataclass
class CoordinationCostReport:
    """Summary of coordination costs for a session.

    Used to prove that coordination overhead is sublinear
    relative to the work done.
    """

    total_resolves: int = 0
    total_escalations: int = 0
    total_auto_resolved: int = 0
    total_blocked: int = 0
    total_deferred: int = 0
    total_cost: float = 0.0
    total_rework_avoided: float = 0.0
    decisions: list[EscalationDecision] = field(default_factory=list)

    @property
    def escalation_rate(self) -> float:
        """Fraction of conflicts that required escalation."""
        total = self.total_resolves + self.total_escalations
        if total == 0:
            return 0.0
        return self.total_escalations / total

    @property
    def auto_resolve_rate(self) -> float:
        """Fraction of conflicts auto-resolved."""
        total = self.total_resolves + self.total_escalations
        if total == 0:
            return 1.0
        return self.total_auto_resolved / total

    @property
    def cost_per_decision(self) -> float:
        """Average cost per coordination decision."""
        total = self.total_resolves + self.total_escalations
        if total == 0:
            return 0.0
        return self.total_cost / total

    def record(self, decision: EscalationDecision) -> None:
        """Record a decision and its costs."""
        self.decisions.append(decision)
        if decision.action == EscalationAction.AUTO_RESOLVE:
            self.total_auto_resolved += 1
            self.total_resolves += 1
            self.total_cost += decision.expected_cost_auto
        elif decision.action == EscalationAction.ESCALATE_TO_HUMAN:
            self.total_escalations += 1
            self.total_cost += decision.expected_cost_escalate
        elif decision.action == EscalationAction.BLOCK:
            self.total_blocked += 1
        elif decision.action == EscalationAction.DEFER:
            self.total_deferred += 1
        # Track rework avoided when escalating instead of auto-resolving
        if decision.action == EscalationAction.ESCALATE_TO_HUMAN:
            self.total_rework_avoided += max(
                0, decision.expected_cost_auto - decision.expected_cost_escalate
            )
