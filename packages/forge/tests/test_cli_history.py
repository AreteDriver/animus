"""Tests for the history CLI commands."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from animus_forge.cli.main import app

runner = CliRunner()


def _mock_store(tasks=None, stats=None, budget=None, summary=None):
    """Create a mock TaskStore with canned responses."""
    store = MagicMock()
    store.query_tasks.return_value = tasks or []
    store.get_agent_stats.return_value = stats or []
    store.get_daily_budget.return_value = budget or []
    store.get_summary.return_value = summary or {
        "total_tasks": 0,
        "successful": 0,
        "failed": 0,
        "success_rate": 0.0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "top_agents": [],
    }
    return store


SAMPLE_TASKS = [
    {
        "id": 1,
        "job_id": "step-1",
        "workflow_id": "wf-alpha",
        "status": "completed",
        "agent_role": "builder",
        "model": "claude-sonnet",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 1000,
        "cost_usd": 0.01,
        "duration_ms": 500,
        "error": None,
        "metadata": None,
        "created_at": "2026-02-14T10:00:00",
        "completed_at": "2026-02-14T10:00:00",
    },
]

SAMPLE_STATS = [
    {
        "agent_role": "builder",
        "total_tasks": 5,
        "successful_tasks": 4,
        "failed_tasks": 1,
        "total_tokens": 5000,
        "total_cost_usd": 0.05,
        "avg_duration_ms": 600.0,
        "success_rate": 80.0,
    },
]

SAMPLE_BUDGET = [
    {
        "date": "2026-02-14",
        "agent_role": "builder",
        "total_tokens": 3000,
        "total_cost_usd": 0.03,
        "task_count": 3,
    },
]


# =============================================================================
# history list
# =============================================================================


class TestHistoryList:
    @patch("animus_forge.db.get_task_store")
    def test_list_shows_tasks(self, mock_get):
        mock_get.return_value = _mock_store(tasks=SAMPLE_TASKS)
        result = runner.invoke(app, ["history", "list"])
        assert result.exit_code == 0
        assert "Task History" in result.output
        assert "step-1" in result.output or "wf-alpha" in result.output
        assert "completed" in result.output

    @patch("animus_forge.db.get_task_store")
    def test_list_empty(self, mock_get):
        mock_get.return_value = _mock_store(tasks=[])
        result = runner.invoke(app, ["history", "list"])
        assert result.exit_code == 0
        assert "No tasks found" in result.output

    @patch("animus_forge.db.get_task_store")
    def test_list_with_filters(self, mock_get):
        store = _mock_store(tasks=SAMPLE_TASKS)
        mock_get.return_value = store
        result = runner.invoke(
            app, ["history", "list", "--status", "completed", "--agent", "builder"]
        )
        assert result.exit_code == 0
        store.query_tasks.assert_called_once_with(
            status="completed", agent_role="builder", limit=10
        )

    @patch("animus_forge.db.get_task_store")
    def test_list_custom_limit(self, mock_get):
        store = _mock_store(tasks=SAMPLE_TASKS)
        mock_get.return_value = store
        result = runner.invoke(app, ["history", "list", "--limit", "5"])
        assert result.exit_code == 0
        store.query_tasks.assert_called_once_with(status=None, agent_role=None, limit=5)

    @patch("animus_forge.db.get_task_store")
    def test_list_error_handling(self, mock_get):
        mock_get.side_effect = Exception("DB error")
        result = runner.invoke(app, ["history", "list"])
        assert result.exit_code == 1
        assert "Error" in result.output


# =============================================================================
# history stats
# =============================================================================


class TestHistoryStats:
    @patch("animus_forge.db.get_task_store")
    def test_stats_shows_table(self, mock_get):
        mock_get.return_value = _mock_store(stats=SAMPLE_STATS)
        result = runner.invoke(app, ["history", "stats"])
        assert result.exit_code == 0
        assert "Agent Performance" in result.output
        assert "builder" in result.output
        assert "80.0%" in result.output

    @patch("animus_forge.db.get_task_store")
    def test_stats_empty(self, mock_get):
        mock_get.return_value = _mock_store(stats=[])
        result = runner.invoke(app, ["history", "stats"])
        assert result.exit_code == 0
        assert "No agent stats found" in result.output

    @patch("animus_forge.db.get_task_store")
    def test_stats_single_agent(self, mock_get):
        store = _mock_store(stats=SAMPLE_STATS)
        mock_get.return_value = store
        result = runner.invoke(app, ["history", "stats", "--agent", "builder"])
        assert result.exit_code == 0
        store.get_agent_stats.assert_called_once_with(agent_role="builder")

    @patch("animus_forge.db.get_task_store")
    def test_stats_error_handling(self, mock_get):
        mock_get.side_effect = Exception("DB error")
        result = runner.invoke(app, ["history", "stats"])
        assert result.exit_code == 1


# =============================================================================
# history budget
# =============================================================================


class TestHistoryBudget:
    @patch("animus_forge.db.get_task_store")
    def test_budget_shows_table(self, mock_get):
        mock_get.return_value = _mock_store(budget=SAMPLE_BUDGET)
        result = runner.invoke(app, ["history", "budget"])
        assert result.exit_code == 0
        assert "Budget" in result.output
        assert "2026-02-14" in result.output

    @patch("animus_forge.db.get_task_store")
    def test_budget_empty(self, mock_get):
        mock_get.return_value = _mock_store(budget=[])
        result = runner.invoke(app, ["history", "budget"])
        assert result.exit_code == 0
        assert "No budget data found" in result.output

    @patch("animus_forge.db.get_task_store")
    def test_budget_custom_days(self, mock_get):
        store = _mock_store(budget=SAMPLE_BUDGET)
        mock_get.return_value = store
        result = runner.invoke(app, ["history", "budget", "--days", "14"])
        assert result.exit_code == 0
        store.get_daily_budget.assert_called_once_with(days=14)

    @patch("animus_forge.db.get_task_store")
    def test_budget_error_handling(self, mock_get):
        mock_get.side_effect = Exception("DB error")
        result = runner.invoke(app, ["history", "budget"])
        assert result.exit_code == 1
