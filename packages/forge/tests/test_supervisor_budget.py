"""Tests for SupervisorAgent budget passthrough (TODO 3)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from animus_forge.agents.supervisor import SupervisorAgent
from animus_forge.budget.manager import BudgetConfig, BudgetManager


class TestSupervisorBudgetKwarg:
    """Test that SupervisorAgent accepts and stores budget_manager."""

    def test_accepts_budget_manager(self):
        provider = MagicMock()
        bm = BudgetManager()
        sup = SupervisorAgent(provider=provider, budget_manager=bm)
        assert sup._budget_manager is bm

    def test_works_without_budget_manager(self):
        provider = MagicMock()
        sup = SupervisorAgent(provider=provider)
        assert sup._budget_manager is None


class TestBudgetContextInjection:
    """Test that budget context is injected into agent prompts."""

    def test_budget_context_added_to_prompt(self):
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="agent result")
        bm = BudgetManager(config=BudgetConfig(total_budget=50000))

        sup = SupervisorAgent(provider=provider, budget_manager=bm)
        asyncio.run(sup._run_agent("builder", "Build auth", []))

        call_args = provider.complete.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "Budget Constraint" in system_msg
        assert "50,000" in system_msg

    def test_no_budget_context_when_unlimited(self):
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="agent result")
        bm = BudgetManager(config=BudgetConfig(total_budget=0))

        sup = SupervisorAgent(provider=provider, budget_manager=bm)
        asyncio.run(sup._run_agent("builder", "Build auth", []))

        call_args = provider.complete.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "Budget Constraint" not in system_msg

    def test_no_budget_context_without_manager(self):
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="agent result")

        sup = SupervisorAgent(provider=provider)
        asyncio.run(sup._run_agent("builder", "Build auth", []))

        call_args = provider.complete.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "Budget Constraint" not in system_msg

    def test_budget_exception_does_not_break_agent(self):
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="agent result")
        bm = MagicMock()
        bm.get_budget_context.side_effect = RuntimeError("DB error")

        sup = SupervisorAgent(provider=provider, budget_manager=bm)
        result = asyncio.run(sup._run_agent("builder", "Build auth", []))
        assert result == "agent result"


class TestBudgetGateDelegation:
    """Test that delegations are skipped when budget is critical."""

    def test_delegation_skipped_when_budget_critical(self):
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="done")
        bm = BudgetManager(config=BudgetConfig(total_budget=10000))
        # Use up almost all budget
        bm.record_usage("setup", 9500)

        sup = SupervisorAgent(provider=provider, budget_manager=bm)
        results = asyncio.run(
            sup._execute_delegations(
                [{"agent": "builder", "task": "Build it"}],
                [],
                lambda _: None,
            )
        )
        assert "builder" in results
        assert "skipped" in results["builder"].lower()
        assert "budget" in results["builder"].lower()
        # Provider should NOT have been called for the delegation
        provider.complete.assert_not_called()

    def test_delegation_proceeds_when_budget_ok(self):
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="completed task")
        bm = BudgetManager(config=BudgetConfig(total_budget=100000))

        sup = SupervisorAgent(provider=provider, budget_manager=bm)
        results = asyncio.run(
            sup._execute_delegations(
                [{"agent": "builder", "task": "Build it"}],
                [],
                lambda _: None,
            )
        )
        assert results["builder"] == "completed task"
        provider.complete.assert_called_once()

    def test_delegation_proceeds_without_budget_manager(self):
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="done")

        sup = SupervisorAgent(provider=provider)
        results = asyncio.run(
            sup._execute_delegations(
                [{"agent": "builder", "task": "Build it"}],
                [],
                lambda _: None,
            )
        )
        assert results["builder"] == "done"


class TestBudgetRecording:
    """Test that token usage is recorded after delegation completion."""

    def test_usage_recorded_after_delegation(self):
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="x" * 400)  # ~100 tokens
        bm = BudgetManager(config=BudgetConfig(total_budget=100000))

        sup = SupervisorAgent(provider=provider, budget_manager=bm)
        asyncio.run(
            sup._execute_delegations(
                [{"agent": "builder", "task": "Build it"}],
                [],
                lambda _: None,
            )
        )
        assert bm.used > 0
        assert bm.get_agent_usage("builder") > 0

    def test_usage_recorded_for_multiple_agents(self):
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="result text here")
        bm = BudgetManager(config=BudgetConfig(total_budget=100000))

        sup = SupervisorAgent(provider=provider, budget_manager=bm)
        asyncio.run(
            sup._execute_delegations(
                [
                    {"agent": "builder", "task": "Build it"},
                    {"agent": "tester", "task": "Test it"},
                ],
                [],
                lambda _: None,
            )
        )
        assert bm.get_agent_usage("builder") > 0
        assert bm.get_agent_usage("tester") > 0

    def test_recording_exception_does_not_break_results(self):
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="done")
        bm = MagicMock()
        bm.can_allocate.return_value = True
        bm.record_usage.side_effect = RuntimeError("DB error")

        sup = SupervisorAgent(provider=provider, budget_manager=bm)
        results = asyncio.run(
            sup._execute_delegations(
                [{"agent": "builder", "task": "Build it"}],
                [],
                lambda _: None,
            )
        )
        assert results["builder"] == "done"
