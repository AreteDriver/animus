"""Integration tests for budget passthrough end-to-end flows.

Tests persistence round-trips, daily limit enforcement with SQLite,
budget context in supervisor prompts, budget gate in delegations,
and bot commands with persisted data.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, "src")

from animus_forge.budget.manager import BudgetConfig, BudgetManager

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def budget_backend():
    """Create a temp SQLite backend with migrations 010+011 applied."""
    from animus_forge.state.backends import SQLiteBackend

    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "integration.db")
        b = SQLiteBackend(db_path=db_path)
        migrations_dir = os.path.join(os.path.dirname(__file__), "..", "migrations")
        for migration in ("010_task_history.sql", "011_budget_session_usage.sql"):
            with open(os.path.join(migrations_dir, migration)) as f:
                sql = f.read()
            b.executescript(sql)
        yield b
        b.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# TestBudgetPersistenceRoundTrip
# =============================================================================


class TestBudgetPersistenceRoundTrip:
    """End-to-end: record usage → destroy manager → new manager restores state."""

    def test_full_round_trip(self, budget_backend):
        bm1 = BudgetManager(
            config=BudgetConfig(total_budget=50000),
            backend=budget_backend,
            session_id="integration-1",
        )
        bm1.record_usage("planner", 3000, "analysis")
        bm1.record_usage("builder", 12000, "code_gen")
        bm1.record_usage("tester", 5000, "test_gen")
        assert bm1.used == 20000
        assert bm1.remaining == 30000

        # Simulate restart — new manager, same session
        bm2 = BudgetManager(
            config=BudgetConfig(total_budget=50000),
            backend=budget_backend,
            session_id="integration-1",
        )
        assert bm2.used == 20000
        assert bm2.remaining == 30000
        assert bm2.get_agent_usage("planner") == 3000
        assert bm2.get_agent_usage("builder") == 12000
        assert bm2.get_agent_usage("tester") == 5000

        # Continue recording on restored manager
        bm2.record_usage("reviewer", 8000, "review")
        assert bm2.used == 28000

        # Third restart confirms cumulative state
        bm3 = BudgetManager(
            config=BudgetConfig(total_budget=50000),
            backend=budget_backend,
            session_id="integration-1",
        )
        assert bm3.used == 28000
        assert bm3.get_agent_usage("reviewer") == 8000


# =============================================================================
# TestDailyLimitWithPersistence
# =============================================================================


class TestDailyLimitWithPersistence:
    """Daily limit enforcement via budget_log table + persistent BudgetManager."""

    def test_daily_limit_blocks_after_threshold(self, budget_backend):
        """Record enough usage in budget_log to trigger daily limit block."""
        from animus_forge.db import TaskStore
        from animus_forge.workflow.executor import WorkflowExecutor
        from animus_forge.workflow.executor_results import ExecutionResult
        from animus_forge.workflow.loader import StepConfig

        store = TaskStore(budget_backend)

        bm = BudgetManager(
            config=BudgetConfig(total_budget=100000, daily_token_limit=10000),
            backend=budget_backend,
            session_id="daily-test",
        )

        # Seed daily usage in budget_log
        today = datetime.now().strftime("%Y-%m-%d")
        budget_backend.execute(
            "INSERT INTO budget_log (date, agent_role, task_count, total_tokens, total_cost_usd) "
            "VALUES (?, ?, ?, ?, ?)",
            (today, "builder", 5, 11000, 1.10),
        )

        executor = WorkflowExecutor.__new__(WorkflowExecutor)
        executor.budget_manager = bm
        executor.dry_run = False
        executor.memory_manager = None
        executor.feedback_engine = None
        executor.emit_callback = None

        step = StepConfig(id="s1", type="claude_code", params={"estimated_tokens": 100})
        result = ExecutionResult(workflow_name="wf-daily")

        with patch("animus_forge.db.get_task_store", return_value=store):
            exceeded = executor._check_budget_exceeded(step, result)
        assert exceeded is True
        assert "Daily" in result.error

    def test_daily_limit_passes_when_under(self, budget_backend):
        """Under daily limit, execution proceeds."""
        from animus_forge.db import TaskStore
        from animus_forge.workflow.executor import WorkflowExecutor
        from animus_forge.workflow.executor_results import ExecutionResult
        from animus_forge.workflow.loader import StepConfig

        store = TaskStore(budget_backend)

        bm = BudgetManager(
            config=BudgetConfig(total_budget=100000, daily_token_limit=50000),
            backend=budget_backend,
            session_id="daily-pass",
        )

        today = datetime.now().strftime("%Y-%m-%d")
        budget_backend.execute(
            "INSERT INTO budget_log (date, agent_role, task_count, total_tokens, total_cost_usd) "
            "VALUES (?, ?, ?, ?, ?)",
            (today, "builder", 2, 5000, 0.50),
        )

        executor = WorkflowExecutor.__new__(WorkflowExecutor)
        executor.budget_manager = bm
        executor.dry_run = False
        executor.memory_manager = None
        executor.feedback_engine = None
        executor.emit_callback = None

        step = StepConfig(id="s1", type="claude_code", params={"estimated_tokens": 100})
        result = ExecutionResult(workflow_name="wf-daily-ok")

        with patch("animus_forge.db.get_task_store", return_value=store):
            exceeded = executor._check_budget_exceeded(step, result)
        assert exceeded is False


# =============================================================================
# TestBudgetContextInSupervisorPrompt
# =============================================================================


class TestBudgetContextInSupervisorPrompt:
    """Verify budget context is injected into agent prompts by supervisor."""

    def test_budget_context_injected(self, budget_backend):
        from animus_forge.agents.supervisor import SupervisorAgent

        bm = BudgetManager(
            config=BudgetConfig(total_budget=80000),
            backend=budget_backend,
            session_id="supervisor-ctx",
        )
        bm.record_usage("builder", 20000)

        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="Done.")

        sup = SupervisorAgent(provider=provider, budget_manager=bm)

        asyncio.run(sup._run_agent("builder", "Write a test", []))

        # Verify budget context was in the system prompt
        call_args = provider.complete.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "Budget Constraint" in system_msg
        assert "60,000" in system_msg  # remaining
        assert "80,000" in system_msg  # total

    def test_no_budget_context_without_manager(self):
        from animus_forge.agents.supervisor import SupervisorAgent

        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="Done.")

        sup = SupervisorAgent(provider=provider)

        asyncio.run(sup._run_agent("builder", "Write a test", []))

        call_args = provider.complete.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "Budget Constraint" not in system_msg


# =============================================================================
# TestBudgetGateInDelegations
# =============================================================================


class TestBudgetGateInDelegations:
    """Verify budget gate prevents delegation when budget is exhausted."""

    def test_gate_blocks_delegation(self, budget_backend):
        from animus_forge.agents.supervisor import SupervisorAgent

        bm = BudgetManager(
            config=BudgetConfig(total_budget=6000, reserve_tokens=2000),
            backend=budget_backend,
            session_id="gate-test",
        )
        # Use enough tokens that can_allocate(5000) fails
        # total_budget=6000, reserve=2000 → available=4000
        # After 2000 used → available=2000, can_allocate(5000) → False
        bm.record_usage("builder", 2000)

        provider = AsyncMock()
        sup = SupervisorAgent(provider=provider, budget_manager=bm)

        delegations = [
            {"agent": "tester", "task": "Write tests"},
            {"agent": "reviewer", "task": "Review code"},
        ]

        results = asyncio.run(sup._execute_delegations(delegations, [], lambda *a, **kw: None))

        # Both delegations should be skipped
        assert "skipped" in results.get("tester", "").lower()
        assert "skipped" in results.get("reviewer", "").lower()

    def test_gate_allows_delegation_with_budget(self, budget_backend):
        from animus_forge.agents.supervisor import SupervisorAgent

        bm = BudgetManager(
            config=BudgetConfig(total_budget=100000, reserve_tokens=5000),
            backend=budget_backend,
            session_id="gate-pass",
        )

        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="Task completed successfully.")

        sup = SupervisorAgent(provider=provider, budget_manager=bm)

        delegations = [{"agent": "builder", "task": "Build feature"}]

        results = asyncio.run(sup._execute_delegations(delegations, [], lambda *a, **kw: None))

        # Delegation should proceed
        assert "completed" in results.get("builder", "").lower()


# =============================================================================
