"""Tests for analytics data collectors."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from animus_forge.analytics.collectors import (
    APIClientMetricsCollector,
    BudgetMetricsCollector,
    CollectedData,
    ExecutionMetricsCollector,
    HistoricalMetricsCollector,
)


class TestCollectedDataContextString:
    """Extended tests for CollectedData.to_context_string."""

    def test_scalar_value(self):
        cd = CollectedData(
            source="test",
            collected_at=datetime.now(UTC),
            data={"scalar": 42},
            metadata={},
        )
        result = cd.to_context_string()
        assert "42" in result
        assert "scalar" in result

    def test_list_over_10_items(self):
        cd = CollectedData(
            source="test",
            collected_at=datetime.now(UTC),
            data={"items": list(range(15))},
            metadata={},
        )
        result = cd.to_context_string()
        assert "... and 5 more" in result


class TestExecutionMetricsCollector:
    """Tests for ExecutionMetricsCollector."""

    @patch("animus_forge.monitoring.tracker.get_tracker")
    def test_collect(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {
                "total_executions": 10,
                "failed_executions": 2,
                "active_workflows": 1,
                "total_steps_executed": 30,
                "total_tokens_used": 5000,
                "avg_duration_ms": 500,
                "success_rate": 80,
            },
            "active_workflows": [{"id": "wf1"}],
            "recent_executions": [{"id": "ex1"}, {"id": "ex2"}],
            "step_performance": {
                "llm_call": {"avg_ms": 200, "count": 20, "failure_rate": 5},
            },
        }
        mock_get_tracker.return_value = mock_tracker

        collector = ExecutionMetricsCollector()
        result = collector.collect(None, {})

        assert result.source == "execution_tracker"
        assert result.data["summary"]["total_executions"] == 10
        assert result.metadata["active_count"] == 1
        assert result.metadata["recent_count"] == 2

    @patch("animus_forge.monitoring.tracker.get_tracker")
    def test_collect_with_config(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {"avg_duration_ms": 0},
            "active_workflows": [],
            "recent_executions": [{"id": f"ex{i}"} for i in range(30)],
            "step_performance": {},
        }
        mock_get_tracker.return_value = mock_tracker

        collector = ExecutionMetricsCollector()
        result = collector.collect(None, {"db_path": "/tmp/test.db", "recent_limit": 5})

        assert len(result.data["recent_executions"]) == 5
        mock_get_tracker.assert_called_with("/tmp/test.db")

    @patch("animus_forge.monitoring.tracker.get_tracker")
    def test_build_timing_metrics(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {"avg_duration_ms": 400, "total_executions": 5},
            "active_workflows": [],
            "recent_executions": [],
            "step_performance": {
                "llm_call": {"avg_ms": 150, "count": 10},
            },
        }
        mock_get_tracker.return_value = mock_tracker

        collector = ExecutionMetricsCollector()
        result = collector.collect(None, {})

        timing = result.data["metrics"]["timing"]
        assert "llm_call" in timing
        assert "workflow_execution" in timing
        assert timing["workflow_execution"]["avg_ms"] == 400

    @patch("animus_forge.monitoring.tracker.get_tracker")
    def test_build_counter_metrics(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {
                "total_executions": 20,
                "failed_executions": 3,
                "active_workflows": 2,
                "total_steps_executed": 60,
                "total_tokens_used": 8000,
            },
            "active_workflows": [],
            "recent_executions": [],
            "step_performance": {},
        }
        mock_get_tracker.return_value = mock_tracker

        collector = ExecutionMetricsCollector()
        result = collector.collect(None, {})

        counters = result.data["metrics"]["counters"]
        assert counters["total_executions"] == 20
        assert counters["error_count"] == 3


class TestHistoricalMetricsCollector:
    """Tests for HistoricalMetricsCollector."""

    @patch("animus_forge.monitoring.metrics.MetricsStore")
    def test_collect(self, mock_store_cls):
        mock_store = MagicMock()
        mock_store.get_historical_data.return_value = [
            {
                "started_at": "2025-01-15T10:00:00",
                "status": "completed",
                "duration_ms": 500,
                "total_tokens": 100,
                "total_steps": 3,
            },
            {
                "started_at": "2025-01-15T10:30:00",
                "status": "failed",
                "duration_ms": 200,
                "total_tokens": 50,
                "total_steps": 2,
            },
            {
                "started_at": "2025-01-15T11:00:00",
                "status": "completed",
                "duration_ms": 300,
                "total_tokens": 80,
                "total_steps": 3,
            },
        ]
        mock_store_cls.return_value = mock_store

        collector = HistoricalMetricsCollector()
        result = collector.collect(None, {"hours": 48})

        assert result.source == "historical_metrics"
        assert result.metadata["hours"] == 48
        assert result.metadata["record_count"] == 3
        assert result.data["total_records"] == 3

        # Check hourly aggregation
        hourly = result.data["hourly_stats"]
        assert "2025-01-15T10" in hourly
        assert hourly["2025-01-15T10"]["count"] == 2
        assert hourly["2025-01-15T10"]["failed"] == 1

        # Check timing
        timing = result.data["metrics"]["timing"]
        assert "workflow_execution" in timing
        assert timing["workflow_execution"]["count"] == 3

        # Check counters
        counters = result.data["metrics"]["counters"]
        assert counters["total_executions"] == 3
        assert counters["failed_executions"] == 1
        assert counters["total_tokens_used"] == 230

    @patch("animus_forge.monitoring.metrics.MetricsStore")
    def test_collect_empty(self, mock_store_cls):
        mock_store = MagicMock()
        mock_store.get_historical_data.return_value = []
        mock_store_cls.return_value = mock_store

        collector = HistoricalMetricsCollector()
        result = collector.collect(None, {})

        assert result.data["total_records"] == 0
        assert result.data["metrics"]["timing"] == {}

    @patch("animus_forge.monitoring.metrics.MetricsStore")
    def test_aggregate_by_hour_averages(self, mock_store_cls):
        mock_store = MagicMock()
        mock_store.get_historical_data.return_value = [
            {
                "started_at": "2025-01-15T10:00:00",
                "status": "completed",
                "duration_ms": 400,
                "total_tokens": 100,
                "total_steps": 2,
            },
            {
                "started_at": "2025-01-15T10:30:00",
                "status": "completed",
                "duration_ms": 600,
                "total_tokens": 200,
                "total_steps": 3,
            },
        ]
        mock_store_cls.return_value = mock_store

        collector = HistoricalMetricsCollector()
        result = collector.collect(None, {})

        hourly = result.data["hourly_stats"]["2025-01-15T10"]
        assert hourly["avg_duration_ms"] == 500.0
        assert hourly["success_rate"] == 100.0

    @patch("animus_forge.monitoring.metrics.MetricsStore")
    def test_no_duration_data(self, mock_store_cls):
        mock_store = MagicMock()
        mock_store.get_historical_data.return_value = [
            {"started_at": "2025-01-15T10:00:00", "status": "completed"}
        ]
        mock_store_cls.return_value = mock_store

        collector = HistoricalMetricsCollector()
        result = collector.collect(None, {})

        assert result.data["metrics"]["timing"] == {}


class TestAPIClientMetricsCollector:
    """Tests for APIClientMetricsCollector."""

    @patch("animus_forge.utils.circuit_breaker.get_all_circuit_stats")
    @patch("animus_forge.api_clients.resilience.get_all_provider_stats")
    def test_collect(self, mock_provider, mock_circuit):
        mock_provider.return_value = {
            "openai": {
                "rate_limit": {"requests_allowed": 100, "requests_denied": 5},
                "bulkhead": {"active": 3, "waiting": 1},
            },
        }
        mock_circuit.return_value = {
            "openai_chat": {"failure_count": 2, "state": "closed"},
        }

        collector = APIClientMetricsCollector()
        result = collector.collect(None, {})

        assert result.source == "api_client_metrics"
        counters = result.data["metrics"]["counters"]
        assert counters["openai_requests_allowed"] == 100
        assert counters["openai_requests_denied"] == 5
        assert counters["circuit_openai_chat_failures"] == 2
        assert counters["circuit_openai_chat_state"] == 0  # closed = 0

    @patch("animus_forge.utils.circuit_breaker.get_all_circuit_stats")
    @patch("animus_forge.api_clients.resilience.get_all_provider_stats")
    def test_collect_with_filter(self, mock_provider, mock_circuit):
        mock_provider.return_value = {
            "openai": {"rate_limit": {}, "bulkhead": {}},
            "anthropic": {"rate_limit": {}, "bulkhead": {}},
        }
        mock_circuit.return_value = {}

        collector = APIClientMetricsCollector()
        result = collector.collect(None, {"providers": ["openai"]})

        assert "openai" in result.data["provider_stats"]
        assert "anthropic" not in result.data["provider_stats"]

    @patch("animus_forge.utils.circuit_breaker.get_all_circuit_stats")
    @patch("animus_forge.api_clients.resilience.get_all_provider_stats")
    def test_circuit_open_state(self, mock_provider, mock_circuit):
        mock_provider.return_value = {}
        mock_circuit.return_value = {
            "test_circuit": {"failure_count": 10, "state": "open"},
        }

        collector = APIClientMetricsCollector()
        result = collector.collect(None, {})

        counters = result.data["metrics"]["counters"]
        assert counters["circuit_test_circuit_state"] == 1  # open = 1


class TestBudgetMetricsCollector:
    """Tests for BudgetMetricsCollector."""

    @patch("animus_forge.budget.get_budget_tracker")
    def test_collect(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_stats.return_value = {
            "total_budget": 100.0,
            "used": 45.0,
            "remaining": 55.0,
            "total_operations": 200,
            "percent_used": 45.0,
        }
        mock_get_tracker.return_value = mock_tracker

        collector = BudgetMetricsCollector()
        result = collector.collect(None, {})

        assert result.source == "budget_tracker"
        counters = result.data["metrics"]["counters"]
        assert counters["budget_total"] == 100.0
        assert counters["budget_spent"] == 45.0
        assert result.metadata["utilization_pct"] == 45.0

    @patch("animus_forge.budget.get_budget_tracker")
    def test_collect_with_history(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_stats.return_value = {
            "total_budget": 100.0,
            "used": 0,
            "remaining": 100.0,
            "total_operations": 0,
            "percent_used": 0,
        }
        mock_tracker.get_usage_history.return_value = [{"ts": "2025-01-01", "cost": 1.0}]
        mock_get_tracker.return_value = mock_tracker

        collector = BudgetMetricsCollector()
        result = collector.collect(None, {"include_history": True})

        assert "history" in result.data
        mock_tracker.get_usage_history.assert_called_once()

    @patch("animus_forge.budget.get_budget_tracker")
    def test_collect_zero_budget(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_tracker.get_stats.return_value = {
            "total_budget": 0,
            "used": 0,
            "remaining": 0,
            "total_operations": 0,
            "percent_used": 0,
        }
        mock_get_tracker.return_value = mock_tracker

        collector = BudgetMetricsCollector()
        result = collector.collect(None, {})

        assert result.metadata["utilization_pct"] == 0
