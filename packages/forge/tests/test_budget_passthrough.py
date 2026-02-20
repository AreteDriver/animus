"""Tests for TODO 3 — Session Budget Passthrough (Jidoka Pattern).

Tests budget context generation, daily limit enforcement,
prompt injection (executor + supervisor), CLI daily command,
and bot /budget command.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import datetime
from unittest.mock import patch

import pytest

sys.path.insert(0, "src")

from animus_forge.budget.manager import BudgetConfig, BudgetManager

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def backend():
    """Create a temp SQLite backend with migrations 010+011 applied."""
    from animus_forge.state.backends import SQLiteBackend

    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "test.db")
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


@pytest.fixture
def store(backend):
    """Create a TaskStore with the test backend."""
    from animus_forge.db import TaskStore

    return TaskStore(backend)


# =============================================================================
# TestBudgetContext
# =============================================================================


class TestBudgetContext:
    """Tests for BudgetManager.get_budget_context()."""

    def test_returns_formatted_context(self):
        bm = BudgetManager(config=BudgetConfig(total_budget=100000))
        ctx = bm.get_budget_context()
        assert "[Budget Constraint]" in ctx
        assert "100,000" in ctx
        assert "concise" in ctx.lower()

    def test_shows_remaining_after_usage(self):
        bm = BudgetManager(config=BudgetConfig(total_budget=100000))
        bm.record_usage("agent", 40000)
        ctx = bm.get_budget_context()
        assert "60,000" in ctx  # remaining
        assert "100,000" in ctx  # total

    def test_empty_when_unlimited(self):
        bm = BudgetManager(config=BudgetConfig(total_budget=0))
        assert bm.get_budget_context() == ""

    def test_empty_when_negative(self):
        bm = BudgetManager(config=BudgetConfig(total_budget=-1))
        assert bm.get_budget_context() == ""


# =============================================================================
# TestDailyTokenLimit
# =============================================================================


class TestDailyTokenLimit:
    """Tests for daily_token_limit field on BudgetConfig."""

    def test_default_disabled(self):
        config = BudgetConfig()
        assert config.daily_token_limit == 0

    def test_can_set(self):
        config = BudgetConfig(daily_token_limit=50000)
        assert config.daily_token_limit == 50000


# =============================================================================
# TestDailyBudgetCheck
# =============================================================================


class TestDailyBudgetCheck:
    """Tests for daily limit enforcement in _check_budget_exceeded."""

    def _make_executor(self, daily_limit=0):
        """Create a minimal executor with budget_manager."""
        from animus_forge.workflow.executor import WorkflowExecutor

        bm = BudgetManager(config=BudgetConfig(total_budget=1000000, daily_token_limit=daily_limit))
        executor = WorkflowExecutor.__new__(WorkflowExecutor)
        executor.budget_manager = bm
        executor.dry_run = False
        executor.memory_manager = None
        executor.feedback_engine = None
        executor.emit_callback = None
        return executor

    def _make_step_and_result(self):
        from animus_forge.workflow.executor_results import ExecutionResult
        from animus_forge.workflow.loader import StepConfig

        step = StepConfig(
            id="step-1",
            type="claude_code",
            params={"estimated_tokens": 100},
        )
        result = ExecutionResult(workflow_name="wf-1")
        return step, result

    def test_daily_limit_blocks_when_exceeded(self, store):
        executor = self._make_executor(daily_limit=5000)
        step, result = self._make_step_and_result()

        # Insert enough daily usage
        today = datetime.now().strftime("%Y-%m-%d")
        store.backend.execute(
            "INSERT INTO budget_log (date, agent_role, task_count, total_tokens, total_cost_usd) "
            "VALUES (?, ?, ?, ?, ?)",
            (today, "builder", 5, 6000, 0.50),
        )

        with patch("animus_forge.db.get_task_store", return_value=store):
            exceeded = executor._check_budget_exceeded(step, result)
        assert exceeded is True
        assert "Daily" in result.error

    def test_daily_limit_passes_when_under(self, store):
        executor = self._make_executor(daily_limit=50000)
        step, result = self._make_step_and_result()

        today = datetime.now().strftime("%Y-%m-%d")
        store.backend.execute(
            "INSERT INTO budget_log (date, agent_role, task_count, total_tokens, total_cost_usd) "
            "VALUES (?, ?, ?, ?, ?)",
            (today, "builder", 2, 1000, 0.10),
        )

        with patch("animus_forge.db.get_task_store", return_value=store):
            exceeded = executor._check_budget_exceeded(step, result)
        assert exceeded is False

    def test_daily_limit_disabled_when_zero(self):
        executor = self._make_executor(daily_limit=0)
        step, result = self._make_step_and_result()
        exceeded = executor._check_budget_exceeded(step, result)
        assert exceeded is False

    def test_daily_check_error_does_not_crash(self):
        executor = self._make_executor(daily_limit=5000)
        step, result = self._make_step_and_result()

        with patch("animus_forge.db.get_task_store", side_effect=RuntimeError("DB unavailable")):
            exceeded = executor._check_budget_exceeded(step, result)
        assert exceeded is False  # Graceful degradation

    def test_daily_sums_across_agents(self, store):
        executor = self._make_executor(daily_limit=10000)
        step, result = self._make_step_and_result()

        today = datetime.now().strftime("%Y-%m-%d")
        store.backend.execute(
            "INSERT INTO budget_log (date, agent_role, task_count, total_tokens, total_cost_usd) "
            "VALUES (?, ?, ?, ?, ?)",
            (today, "builder", 3, 6000, 0.30),
        )
        store.backend.execute(
            "INSERT INTO budget_log (date, agent_role, task_count, total_tokens, total_cost_usd) "
            "VALUES (?, ?, ?, ?, ?)",
            (today, "tester", 2, 5000, 0.20),
        )

        with patch("animus_forge.db.get_task_store", return_value=store):
            exceeded = executor._check_budget_exceeded(step, result)
        assert exceeded is True


# =============================================================================
# TestExecutorPromptInjection
# =============================================================================


class TestExecutorPromptInjection:
    """Tests for budget context injection in executor_ai prompts."""

    def test_budget_context_in_executor_prompt(self):
        """Budget context appears in the prompt sent to Claude."""

        class FakeExecutor:
            dry_run = True
            memory_manager = None
            budget_manager = BudgetManager(config=BudgetConfig(total_budget=80000))

        from animus_forge.workflow.executor_ai import AIHandlersMixin
        from animus_forge.workflow.loader import StepConfig

        obj = FakeExecutor()
        AIHandlersMixin._execute_claude_code = AIHandlersMixin._execute_claude_code

        step = StepConfig(
            id="s1",
            type="claude_code",
            params={"prompt": "Do something", "use_memory": False},
        )
        result = AIHandlersMixin._execute_claude_code(obj, step, {})
        assert "Budget Constraint" in result["prompt"]
        assert "80,000" in result["prompt"]

    def test_no_budget_context_without_manager(self):
        """No budget context when budget_manager is absent."""

        class FakeExecutor:
            dry_run = True
            memory_manager = None
            # No budget_manager attribute

        from animus_forge.workflow.executor_ai import AIHandlersMixin
        from animus_forge.workflow.loader import StepConfig

        obj = FakeExecutor()
        step = StepConfig(
            id="s1",
            type="claude_code",
            params={"prompt": "Do something", "use_memory": False},
        )
        result = AIHandlersMixin._execute_claude_code(obj, step, {})
        assert "Budget Constraint" not in result["prompt"]

    def test_no_budget_context_when_unlimited(self):
        """No budget context when total_budget is 0."""

        class FakeExecutor:
            dry_run = True
            memory_manager = None
            budget_manager = BudgetManager(config=BudgetConfig(total_budget=0))

        from animus_forge.workflow.executor_ai import AIHandlersMixin
        from animus_forge.workflow.loader import StepConfig

        obj = FakeExecutor()
        step = StepConfig(
            id="s1",
            type="claude_code",
            params={"prompt": "Do something", "use_memory": False},
        )
        result = AIHandlersMixin._execute_claude_code(obj, step, {})
        assert "Budget Constraint" not in result["prompt"]


# =============================================================================
# TestBudgetDailyCommand
# =============================================================================


class TestBudgetDailyCommand:
    """Tests for 'budget daily' CLI command."""

    def test_daily_shows_table(self, store):
        today = datetime.now().strftime("%Y-%m-%d")
        store.backend.execute(
            "INSERT INTO budget_log (date, agent_role, task_count, total_tokens, total_cost_usd) "
            "VALUES (?, ?, ?, ?, ?)",
            (today, "builder", 5, 12000, 1.20),
        )

        import typer
        from typer.testing import CliRunner

        from animus_forge.cli.commands.budget import budget_app

        app = typer.Typer()
        app.add_typer(budget_app)

        runner = CliRunner()
        with patch("animus_forge.db.get_task_store", return_value=store):
            result = runner.invoke(app, ["daily"])
        assert result.exit_code == 0
        assert "12,000" in result.output

    def test_daily_empty(self, store):
        import typer
        from typer.testing import CliRunner

        from animus_forge.cli.commands.budget import budget_app

        app = typer.Typer()
        app.add_typer(budget_app)

        runner = CliRunner()
        with patch("animus_forge.db.get_task_store", return_value=store):
            result = runner.invoke(app, ["daily"])
        assert result.exit_code == 0
        assert "No daily budget data" in result.output

    def test_daily_agent_filter(self, store):
        today = datetime.now().strftime("%Y-%m-%d")
        store.backend.execute(
            "INSERT INTO budget_log (date, agent_role, task_count, total_tokens, total_cost_usd) "
            "VALUES (?, ?, ?, ?, ?)",
            (today, "builder", 3, 8000, 0.80),
        )
        store.backend.execute(
            "INSERT INTO budget_log (date, agent_role, task_count, total_tokens, total_cost_usd) "
            "VALUES (?, ?, ?, ?, ?)",
            (today, "tester", 2, 4000, 0.40),
        )

        import typer
        from typer.testing import CliRunner

        from animus_forge.cli.commands.budget import budget_app

        app = typer.Typer()
        app.add_typer(budget_app)

        runner = CliRunner()
        with patch("animus_forge.db.get_task_store", return_value=store):
            result = runner.invoke(app, ["daily", "--agent", "builder"])
        assert result.exit_code == 0
        assert "8,000" in result.output

    def test_daily_json_output(self, store):
        today = datetime.now().strftime("%Y-%m-%d")
        store.backend.execute(
            "INSERT INTO budget_log (date, agent_role, task_count, total_tokens, total_cost_usd) "
            "VALUES (?, ?, ?, ?, ?)",
            (today, "builder", 3, 8000, 0.80),
        )

        import json

        import typer
        from typer.testing import CliRunner

        from animus_forge.cli.commands.budget import budget_app

        app = typer.Typer()
        app.add_typer(budget_app)

        runner = CliRunner()
        with patch("animus_forge.db.get_task_store", return_value=store):
            result = runner.invoke(app, ["daily", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["total_tokens"] == 8000


# =============================================================================

# =============================================================================
# TestPersistentBudget
# =============================================================================


class TestPersistentBudget:
    """Tests for BudgetManager SQLite persistence via backend/session_id."""

    def test_restore_on_init(self, backend):
        """Usage recorded in DB is restored when a new BudgetManager is created."""
        bm1 = BudgetManager(
            config=BudgetConfig(total_budget=100000),
            backend=backend,
            session_id="sess-1",
        )
        bm1.record_usage("builder", 5000, "step-1")
        bm1.record_usage("tester", 3000, "step-2")

        # Create a new manager with the same session — should restore
        bm2 = BudgetManager(
            config=BudgetConfig(total_budget=100000),
            backend=backend,
            session_id="sess-1",
        )
        assert bm2.used == 8000
        assert bm2.get_agent_usage("builder") == 5000
        assert bm2.get_agent_usage("tester") == 3000

    def test_persist_on_record_usage(self, backend):
        """record_usage inserts a row into budget_session_usage."""
        bm = BudgetManager(
            config=BudgetConfig(total_budget=100000),
            backend=backend,
            session_id="sess-2",
        )
        bm.record_usage("builder", 7000, "compilation")

        rows = backend.fetchall(
            "SELECT * FROM budget_session_usage WHERE session_id = ?",
            ("sess-2",),
        )
        assert len(rows) == 1
        assert rows[0]["agent_id"] == "builder"
        assert rows[0]["tokens"] == 7000
        assert rows[0]["operation"] == "compilation"

    def test_reset_clears_db(self, backend):
        """reset() deletes session rows from budget_session_usage."""
        bm = BudgetManager(
            config=BudgetConfig(total_budget=100000),
            backend=backend,
            session_id="sess-3",
        )
        bm.record_usage("builder", 5000)
        bm.record_usage("tester", 3000)
        assert bm.used == 8000

        bm.reset()
        assert bm.used == 0

        rows = backend.fetchall(
            "SELECT * FROM budget_session_usage WHERE session_id = ?",
            ("sess-3",),
        )
        assert len(rows) == 0

    def test_survives_reinit(self, backend):
        """Data persists across multiple BudgetManager instances."""
        bm1 = BudgetManager(
            config=BudgetConfig(total_budget=200000),
            backend=backend,
            session_id="sess-4",
        )
        bm1.record_usage("builder", 10000)

        bm2 = BudgetManager(
            config=BudgetConfig(total_budget=200000),
            backend=backend,
            session_id="sess-4",
        )
        bm2.record_usage("builder", 5000)

        bm3 = BudgetManager(
            config=BudgetConfig(total_budget=200000),
            backend=backend,
            session_id="sess-4",
        )
        assert bm3.used == 15000
        assert bm3.get_agent_usage("builder") == 15000

    def test_no_backend_fallback(self):
        """Without backend, BudgetManager works in-memory only."""
        bm = BudgetManager(config=BudgetConfig(total_budget=100000))
        bm.record_usage("builder", 5000)
        assert bm.used == 5000

        # No crash, no persistence
        bm.reset()
        assert bm.used == 0

    def test_sessions_isolated(self, backend):
        """Different session_ids don't interfere with each other."""
        bm_a = BudgetManager(
            config=BudgetConfig(total_budget=100000),
            backend=backend,
            session_id="sess-A",
        )
        bm_b = BudgetManager(
            config=BudgetConfig(total_budget=100000),
            backend=backend,
            session_id="sess-B",
        )

        bm_a.record_usage("builder", 5000)
        bm_b.record_usage("builder", 9000)

        # Re-create and verify isolation
        bm_a2 = BudgetManager(
            config=BudgetConfig(total_budget=100000),
            backend=backend,
            session_id="sess-A",
        )
        bm_b2 = BudgetManager(
            config=BudgetConfig(total_budget=100000),
            backend=backend,
            session_id="sess-B",
        )
        assert bm_a2.used == 5000
        assert bm_b2.used == 9000

    def test_restore_graceful_on_missing_table(self):
        """If the table doesn't exist, init doesn't crash."""
        from animus_forge.state.backends import SQLiteBackend

        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "empty.db")
            b = SQLiteBackend(db_path=db_path)
            # No migration applied — table doesn't exist
            bm = BudgetManager(
                config=BudgetConfig(total_budget=100000),
                backend=b,
                session_id="sess-x",
            )
            assert bm.used == 0
            b.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_persist_graceful_on_missing_table(self):
        """record_usage doesn't crash if table is missing."""
        from animus_forge.state.backends import SQLiteBackend

        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "empty.db")
            b = SQLiteBackend(db_path=db_path)
            bm = BudgetManager(
                config=BudgetConfig(total_budget=100000),
                backend=b,
                session_id="sess-x",
            )
            # Should not crash — graceful degradation
            bm.record_usage("builder", 5000)
            assert bm.used == 5000  # In-memory still works
            b.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_reset_budget_tracker_function(self):
        """reset_budget_tracker() clears the singleton."""
        from animus_forge.budget import get_budget_tracker, reset_budget_tracker

        tracker = get_budget_tracker()
        tracker.record_usage("agent", 1000)
        assert tracker.used == 1000

        reset_budget_tracker()
        tracker2 = get_budget_tracker()
        assert tracker2.used == 0
        assert tracker2 is not tracker

        # Clean up
        reset_budget_tracker()
