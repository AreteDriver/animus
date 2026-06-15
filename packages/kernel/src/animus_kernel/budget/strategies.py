"""Budget Allocation Strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AllocationResult:
    """Result of a budget allocation calculation."""

    allocations: dict[str, int]  # agent_id -> tokens
    total_allocated: int
    unallocated: int = 0
    notes: list[str] = field(default_factory=list)


class AllocationStrategy(ABC):
    """Base class for budget allocation strategies."""

    @abstractmethod
    def allocate(
        self,
        total_budget: int,
        agents: list[dict],
        context: dict = None,
    ) -> AllocationResult:
        """Allocate budget across agents.

        Args:
            total_budget: Total available tokens
            agents: List of agent configs with 'id' and optional 'priority', 'estimate'
            context: Additional context for allocation

        Returns:
            AllocationResult with per-agent allocations
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Get strategy name."""
        pass


class EqualAllocation(AllocationStrategy):
    """Divide budget equally among all agents."""

    def name(self) -> str:
        return "equal"

    def allocate(
        self,
        total_budget: int,
        agents: list[dict],
        context: dict = None,
    ) -> AllocationResult:
        """Allocate budget equally."""
        if not agents:
            return AllocationResult(allocations={}, total_allocated=0, unallocated=total_budget)

        per_agent = total_budget // len(agents)
        allocations = {agent["id"]: per_agent for agent in agents}
        total_allocated = per_agent * len(agents)

        return AllocationResult(
            allocations=allocations,
            total_allocated=total_allocated,
            unallocated=total_budget - total_allocated,
        )


class PriorityAllocation(AllocationStrategy):
    """Allocate based on agent priority levels.

    Higher priority agents get larger allocations.
    Priority is specified as 'priority' field (1-10, higher = more budget).
    """

    def __init__(self, base_share: float = 0.1):
        """Initialize strategy.

        Args:
            base_share: Minimum share each agent gets (0.0-1.0)
        """
        self.base_share = base_share

    def name(self) -> str:
        return "priority"

    def allocate(
        self,
        total_budget: int,
        agents: list[dict],
        context: dict = None,
    ) -> AllocationResult:
        """Allocate based on priority."""
        if not agents:
            return AllocationResult(allocations={}, total_allocated=0, unallocated=total_budget)

        # Calculate base allocation
        base_per_agent = int(total_budget * self.base_share / len(agents))
        base_total = base_per_agent * len(agents)
        remaining = total_budget - base_total

        # Calculate priority weights
        priorities = []
        for agent in agents:
            priority = agent.get("priority", 5)
            priorities.append(max(1, min(10, priority)))  # Clamp 1-10

        total_priority = sum(priorities)

        # Allocate remaining based on priority
        allocations = {}
        notes = []
        for agent, priority in zip(agents, priorities):
            agent_id = agent["id"]
            priority_share = (
                int((priority / total_priority) * remaining) if total_priority > 0 else 0
            )
            allocation = base_per_agent + priority_share
            allocations[agent_id] = allocation
            notes.append(f"{agent_id}: priority={priority}, tokens={allocation}")

        total_allocated = sum(allocations.values())

        return AllocationResult(
            allocations=allocations,
            total_allocated=total_allocated,
            unallocated=total_budget - total_allocated,
            notes=notes,
        )


class AdaptiveAllocation(AllocationStrategy):
    """Adaptive allocation based on estimates and historical usage.

    Uses agent estimates if provided, falls back to historical averages,
    and adjusts based on actual performance.
    """

    def __init__(
        self,
        buffer_percent: float = 0.2,
        history: list[dict] = None,
    ):
        """Initialize strategy.

        Args:
            buffer_percent: Extra buffer to add to estimates (0.0-1.0)
            history: Historical usage data for agents
        """
        self.buffer_percent = buffer_percent
        self.history = history or []
        self._historical_averages: dict[str, int] = {}
        self._calculate_averages()

    def _calculate_averages(self):
        """Calculate historical averages from history data."""
        agent_totals: dict[str, list[int]] = {}
        for record in self.history:
            agent_id = record.get("agent_id", "unknown")
            tokens = record.get("tokens", 0)
            if agent_id not in agent_totals:
                agent_totals[agent_id] = []
            agent_totals[agent_id].append(tokens)

        for agent_id, values in agent_totals.items():
            self._historical_averages[agent_id] = sum(values) // len(values) if values else 5000

    def name(self) -> str:
        return "adaptive"

    def allocate(
        self,
        total_budget: int,
        agents: list[dict],
        context: dict = None,
    ) -> AllocationResult:
        """Allocate based on estimates and history."""
        if not agents:
            return AllocationResult(allocations={}, total_allocated=0, unallocated=total_budget)

        # Calculate estimated needs per agent
        estimates = {}
        notes = []
        for agent in agents:
            agent_id = agent["id"]
            estimate = agent.get("estimate")

            if estimate:
                source = "provided"
            elif agent_id in self._historical_averages:
                estimate = self._historical_averages[agent_id]
                source = "historical"
            else:
                # Default estimate based on role
                role = agent.get("role", "")
                defaults = {
                    "planner": 5000,
                    "builder": 20000,
                    "tester": 10000,
                    "reviewer": 5000,
                }
                estimate = defaults.get(role, 10000)
                source = "default"

            # Add buffer
            buffered = int(estimate * (1 + self.buffer_percent))
            estimates[agent_id] = buffered
            notes.append(f"{agent_id}: {source} estimate={estimate}, buffered={buffered}")

        # Scale if total exceeds budget
        total_estimated = sum(estimates.values())
        if total_estimated > total_budget:
            scale_factor = total_budget / total_estimated
            notes.append(f"Scaling down by {scale_factor:.2f} (over budget)")
            estimates = {k: int(v * scale_factor) for k, v in estimates.items()}

        total_allocated = sum(estimates.values())

        return AllocationResult(
            allocations=estimates,
            total_allocated=total_allocated,
            unallocated=total_budget - total_allocated,
            notes=notes,
        )

    def add_history(self, agent_id: str, tokens: int):
        """Add a usage record to history.

        Args:
            agent_id: Agent identifier
            tokens: Tokens used
        """
        self.history.append({"agent_id": agent_id, "tokens": tokens})
        self._calculate_averages()


class ReservePoolAllocation(AllocationStrategy):
    """Allocate with a shared reserve pool for overflow.

    Gives each agent a guaranteed minimum, with remaining budget
    in a shared pool for agents that need more.
    """

    def __init__(
        self,
        guaranteed_percent: float = 0.6,
        reserve_percent: float = 0.2,
    ):
        """Initialize strategy.

        Args:
            guaranteed_percent: Percent of budget guaranteed to agents
            reserve_percent: Percent kept in reserve pool
        """
        self.guaranteed_percent = guaranteed_percent
        self.reserve_percent = reserve_percent

    def name(self) -> str:
        return "reserve_pool"

    def allocate(
        self,
        total_budget: int,
        agents: list[dict],
        context: dict = None,
    ) -> AllocationResult:
        """Allocate with reserve pool."""
        if not agents:
            return AllocationResult(allocations={}, total_allocated=0, unallocated=total_budget)

        # Split budget into pools
        guaranteed_pool = int(total_budget * self.guaranteed_percent)
        reserve = int(total_budget * self.reserve_percent)
        flexible_pool = total_budget - guaranteed_pool - reserve

        # Guaranteed allocation
        guaranteed_per_agent = guaranteed_pool // len(agents)

        # Distribute flexible pool based on estimates
        estimates = {}
        for agent in agents:
            agent_id = agent["id"]
            estimate = agent.get("estimate", 5000)
            estimates[agent_id] = estimate

        total_estimated = sum(estimates.values())

        allocations = {}
        notes = [
            f"Guaranteed pool: {guaranteed_pool} ({guaranteed_per_agent}/agent)",
            f"Flexible pool: {flexible_pool}",
            f"Reserve: {reserve}",
        ]

        for agent in agents:
            agent_id = agent["id"]
            estimate = estimates[agent_id]

            # Guaranteed + share of flexible
            flexible_share = (
                int((estimate / total_estimated) * flexible_pool) if total_estimated > 0 else 0
            )
            total_allocation = guaranteed_per_agent + flexible_share
            allocations[agent_id] = total_allocation
            notes.append(
                f"{agent_id}: guaranteed={guaranteed_per_agent} + flexible={flexible_share}"
            )

        total_allocated = sum(allocations.values())

        return AllocationResult(
            allocations=allocations,
            total_allocated=total_allocated,
            unallocated=reserve,
            notes=notes,
        )


def get_strategy(name: str, **kwargs) -> AllocationStrategy:
    """Get an allocation strategy by name.

    Args:
        name: Strategy name (equal, priority, adaptive, reserve_pool)
        **kwargs: Strategy-specific configuration

    Returns:
        AllocationStrategy instance

    Raises:
        ValueError: If strategy name is unknown
    """
    strategies = {
        "equal": EqualAllocation,
        "priority": PriorityAllocation,
        "adaptive": AdaptiveAllocation,
        "reserve_pool": ReservePoolAllocation,
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name](**kwargs)
