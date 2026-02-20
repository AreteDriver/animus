"""Token budget tracking for Forge workflows."""

from dataclasses import dataclass, field

from animus.forge.models import BudgetExhaustedError, WorkflowConfig
from animus.logging import get_logger

logger = get_logger("forge.budget")


@dataclass
class BudgetTracker:
    """Tracks token usage per agent and total workflow cost."""

    agent_budgets: dict[str, int] = field(default_factory=dict)
    agent_usage: dict[str, int] = field(default_factory=dict)
    max_cost_usd: float = 1.0
    total_cost: float = 0.0

    @classmethod
    def from_config(cls, config: WorkflowConfig) -> "BudgetTracker":
        """Create a tracker from a workflow config."""
        return cls(
            agent_budgets={a.name: a.budget_tokens for a in config.agents},
            agent_usage={a.name: 0 for a in config.agents},
            max_cost_usd=config.max_cost_usd,
        )

    def record(self, agent_name: str, tokens: int, cost: float) -> None:
        """Record token usage for an agent.

        Raises:
            BudgetExhaustedError: If recording would exceed agent or workflow budget.
        """
        current = self.agent_usage.get(agent_name, 0)
        budget = self.agent_budgets.get(agent_name, 0)

        if current + tokens > budget:
            raise BudgetExhaustedError(
                f"Agent {agent_name!r} would exceed budget: {current + tokens} > {budget} tokens"
            )

        if self.total_cost + cost > self.max_cost_usd:
            raise BudgetExhaustedError(
                f"Workflow would exceed cost ceiling: "
                f"${self.total_cost + cost:.4f} > ${self.max_cost_usd:.2f}"
            )

        self.agent_usage[agent_name] = current + tokens
        self.total_cost += cost
        logger.debug(
            f"Budget: {agent_name} used {tokens} tokens "
            f"({current + tokens}/{budget}), cost ${cost:.4f}"
        )

    def check(self, agent_name: str) -> bool:
        """Return True if the agent is under budget."""
        current = self.agent_usage.get(agent_name, 0)
        budget = self.agent_budgets.get(agent_name, 0)
        return current < budget and self.total_cost < self.max_cost_usd

    def remaining(self, agent_name: str) -> int:
        """Return remaining tokens for an agent."""
        current = self.agent_usage.get(agent_name, 0)
        budget = self.agent_budgets.get(agent_name, 0)
        return max(0, budget - current)

    def summary(self) -> dict[str, dict]:
        """Return a summary of budget usage."""
        result = {}
        for name, budget in self.agent_budgets.items():
            used = self.agent_usage.get(name, 0)
            result[name] = {
                "budget": budget,
                "used": used,
                "remaining": max(0, budget - used),
                "pct": round(used / budget * 100, 1) if budget > 0 else 0.0,
            }
        result["_total"] = {
            "cost_usd": round(self.total_cost, 4),
            "max_cost_usd": self.max_cost_usd,
        }
        return result
