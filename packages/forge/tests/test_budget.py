"""Tests for the budget module."""

import sys

import pytest

sys.path.insert(0, "src")

from animus_forge.budget import (
    AdaptiveAllocation,
    BudgetConfig,
    BudgetManager,
    EqualAllocation,
    PriorityAllocation,
    UsageRecord,
)
from animus_forge.budget.manager import BudgetStatus
from animus_forge.budget.strategies import ReservePoolAllocation, get_strategy


class TestBudgetConfig:
    """Tests for BudgetConfig class."""

    def test_default_values(self):
        """Default configuration values."""
        config = BudgetConfig()
        assert config.total_budget == 100000
        assert config.warning_threshold == 0.75
        assert config.critical_threshold == 0.90
        assert config.reserve_tokens == 5000

    def test_custom_values(self):
        """Custom configuration values."""
        config = BudgetConfig(
            total_budget=50000,
            per_agent_limit=10000,
            per_step_limit=5000,
        )
        assert config.total_budget == 50000
        assert config.per_agent_limit == 10000


class TestBudgetManager:
    """Tests for BudgetManager class."""

    def test_initial_state(self):
        """Initial budget state."""
        manager = BudgetManager()
        assert manager.used == 0
        assert manager.remaining == manager.total_budget
        assert manager.status == BudgetStatus.OK

    def test_record_usage(self):
        """Recording usage updates totals."""
        manager = BudgetManager(BudgetConfig(total_budget=10000))
        record = manager.record_usage("agent1", 1000, "test operation")

        assert isinstance(record, UsageRecord)
        assert manager.used == 1000
        assert manager.remaining == 9000

    def test_can_allocate_within_budget(self):
        """Can allocate within budget."""
        manager = BudgetManager(BudgetConfig(total_budget=10000, reserve_tokens=1000))
        assert manager.can_allocate(5000) is True
        assert manager.can_allocate(9000) is True
        assert manager.can_allocate(9001) is False  # Exceeds available (10000 - 1000 reserve)

    def test_can_allocate_per_agent_limit(self):
        """Per-agent limit is enforced."""
        config = BudgetConfig(total_budget=100000, per_agent_limit=5000)
        manager = BudgetManager(config)

        manager.record_usage("agent1", 3000)
        assert manager.can_allocate(2000, "agent1") is True
        assert manager.can_allocate(2001, "agent1") is False  # Would exceed 5000

    def test_can_allocate_per_step_limit(self):
        """Per-step limit is enforced."""
        config = BudgetConfig(total_budget=100000, per_step_limit=5000)
        manager = BudgetManager(config)

        assert manager.can_allocate(5000) is True
        assert manager.can_allocate(5001) is False

    def test_status_thresholds(self):
        """Status changes at thresholds."""
        config = BudgetConfig(
            total_budget=10000,
            warning_threshold=0.5,
            critical_threshold=0.8,
        )
        manager = BudgetManager(config)

        assert manager.status == BudgetStatus.OK

        manager.record_usage("agent", 5001)  # Just over 50%
        assert manager.status == BudgetStatus.WARNING

        manager.record_usage("agent", 3000)  # Over 80%
        assert manager.status == BudgetStatus.CRITICAL

        manager.record_usage("agent", 2500)  # Over 100%
        assert manager.status == BudgetStatus.EXCEEDED

    def test_threshold_callback(self):
        """Callback is invoked on threshold crossing."""
        events = []

        def callback(status, info):
            events.append((status, info))

        config = BudgetConfig(total_budget=10000, warning_threshold=0.5)
        manager = BudgetManager(config, on_threshold_callback=callback)

        manager.record_usage("agent", 6000)  # Cross warning threshold

        assert len(events) == 1
        assert events[0][0] == BudgetStatus.WARNING

    def test_get_agent_usage(self):
        """Can get usage per agent."""
        manager = BudgetManager()
        manager.record_usage("agent1", 1000)
        manager.record_usage("agent2", 2000)
        manager.record_usage("agent1", 500)

        assert manager.get_agent_usage("agent1") == 1500
        assert manager.get_agent_usage("agent2") == 2000
        assert manager.get_agent_usage("unknown") == 0

    def test_get_agent_remaining(self):
        """Can get remaining budget per agent."""
        config = BudgetConfig(total_budget=100000, per_agent_limit=5000)
        manager = BudgetManager(config)

        manager.record_usage("agent1", 3000)
        assert manager.get_agent_remaining("agent1") == 2000

    def test_get_usage_history(self):
        """Can get usage history."""
        manager = BudgetManager()
        manager.record_usage("agent1", 100)
        manager.record_usage("agent2", 200)

        history = manager.get_usage_history()
        assert len(history) == 2

        agent1_history = manager.get_usage_history("agent1")
        assert len(agent1_history) == 1

    def test_get_stats(self):
        """Can get budget statistics."""
        manager = BudgetManager(BudgetConfig(total_budget=10000))
        manager.record_usage("agent1", 2000)
        manager.record_usage("agent2", 1000)

        stats = manager.get_stats()
        assert stats["total_budget"] == 10000
        assert stats["used"] == 3000
        assert stats["remaining"] == 7000
        assert stats["total_operations"] == 2
        assert "agent1" in stats["agents"]

    def test_estimate_cost(self):
        """Cost estimation works."""
        manager = BudgetManager()
        cost = manager.estimate_cost(1000000, "claude-3-opus")
        assert cost > 0

        # Different models have different costs
        opus_cost = manager.estimate_cost(1000, "claude-3-opus")
        haiku_cost = manager.estimate_cost(1000, "claude-3-haiku")
        assert opus_cost > haiku_cost

    def test_reset(self):
        """Can reset budget tracking."""
        manager = BudgetManager()
        manager.record_usage("agent", 5000)
        manager.reset()

        assert manager.used == 0
        assert len(manager.get_usage_history()) == 0

    def test_set_budget(self):
        """Can update total budget."""
        manager = BudgetManager()
        manager.set_budget(50000)
        assert manager.total_budget == 50000


class TestEqualAllocation:
    """Tests for EqualAllocation strategy."""

    def test_equal_division(self):
        """Budget is divided equally."""
        strategy = EqualAllocation()
        agents = [
            {"id": "agent1"},
            {"id": "agent2"},
            {"id": "agent3"},
        ]
        result = strategy.allocate(30000, agents)

        assert result.allocations["agent1"] == 10000
        assert result.allocations["agent2"] == 10000
        assert result.allocations["agent3"] == 10000
        assert result.total_allocated == 30000

    def test_empty_agents(self):
        """Empty agents list returns empty allocations."""
        strategy = EqualAllocation()
        result = strategy.allocate(10000, [])
        assert result.allocations == {}
        assert result.unallocated == 10000


class TestPriorityAllocation:
    """Tests for PriorityAllocation strategy."""

    def test_priority_weighting(self):
        """Higher priority gets more budget."""
        strategy = PriorityAllocation(base_share=0.0)
        agents = [
            {"id": "low", "priority": 1},
            {"id": "high", "priority": 9},
        ]
        result = strategy.allocate(10000, agents)

        assert result.allocations["high"] > result.allocations["low"]

    def test_base_share(self):
        """Base share is distributed first."""
        strategy = PriorityAllocation(base_share=0.5)
        agents = [
            {"id": "agent1", "priority": 5},
            {"id": "agent2", "priority": 5},
        ]
        result = strategy.allocate(10000, agents)

        # Each gets at least 25% (half of 50% base)
        assert result.allocations["agent1"] >= 2500


class TestAdaptiveAllocation:
    """Tests for AdaptiveAllocation strategy."""

    def test_uses_estimates(self):
        """Uses provided estimates."""
        strategy = AdaptiveAllocation(buffer_percent=0.0)
        agents = [
            {"id": "small", "estimate": 1000},
            {"id": "large", "estimate": 9000},
        ]
        result = strategy.allocate(20000, agents)

        assert result.allocations["large"] > result.allocations["small"]

    def test_buffer_added(self):
        """Buffer is added to estimates."""
        strategy = AdaptiveAllocation(buffer_percent=0.2)
        agents = [{"id": "agent", "estimate": 1000}]
        result = strategy.allocate(10000, agents)

        assert result.allocations["agent"] == 1200  # 1000 + 20%

    def test_scaling_when_over_budget(self):
        """Scales down when estimates exceed budget."""
        strategy = AdaptiveAllocation(buffer_percent=0.0)
        agents = [
            {"id": "a", "estimate": 6000},
            {"id": "b", "estimate": 6000},
        ]
        result = strategy.allocate(10000, agents)

        assert result.total_allocated <= 10000


class TestReservePoolAllocation:
    """Tests for ReservePoolAllocation strategy."""

    def test_reserve_pool(self):
        """Reserve pool is unallocated."""
        strategy = ReservePoolAllocation(
            guaranteed_percent=0.5,
            reserve_percent=0.2,
        )
        agents = [{"id": "agent"}]
        result = strategy.allocate(10000, agents)

        assert result.unallocated == 2000  # 20% reserve


class TestGetStrategy:
    """Tests for get_strategy function."""

    def test_get_all_strategies(self):
        """Can get all strategy types."""
        assert isinstance(get_strategy("equal"), EqualAllocation)
        assert isinstance(get_strategy("priority"), PriorityAllocation)
        assert isinstance(get_strategy("adaptive"), AdaptiveAllocation)
        assert isinstance(get_strategy("reserve_pool"), ReservePoolAllocation)

    def test_unknown_strategy_raises(self):
        """Unknown strategy raises ValueError."""
        with pytest.raises(ValueError):
            get_strategy("unknown")
