"""Data Collectors for Analytics Pipelines.

Provides modular data collection components that can be used in analytics pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class CollectedData:
    """Container for collected data."""

    source: str
    collected_at: datetime
    data: dict[str, Any]
    metadata: dict[str, Any]

    def to_context_string(self) -> str:
        """Convert to string format for AI agent context."""
        lines = [
            f"# Data Collection: {self.source}",
            f"Collected: {self.collected_at.isoformat()}",
            "",
        ]

        for key, value in self.data.items():
            lines.append(f"## {key}")
            if isinstance(value, dict):
                for k, v in value.items():
                    lines.append(f"- {k}: {v}")
            elif isinstance(value, list):
                for item in value[:10]:  # Limit to 10 items
                    lines.append(f"- {item}")
                if len(value) > 10:
                    lines.append(f"- ... and {len(value) - 10} more")
            else:
                lines.append(str(value))
            lines.append("")

        return "\n".join(lines)


class DataCollector(ABC):
    """Abstract base class for data collectors."""

    @abstractmethod
    def collect(self, context: Any, config: dict) -> CollectedData:
        """Collect data from the source.

        Args:
            context: Previous stage output or initial context
            config: Collector configuration

        Returns:
            CollectedData with collected information
        """
        pass


class JSONCollector(DataCollector):
    """Collector that accepts JSON data directly (for testing/manual input)."""

    def collect(self, context: Any, config: dict) -> CollectedData:
        """Pass through JSON data.

        Config options:
            data: dict - The data to pass through
            source_name: str - Name for the data source
        """
        data = config.get("data", context if isinstance(context, dict) else {})
        source_name = config.get("source_name", "json_input")

        return CollectedData(
            source=source_name,
            collected_at=datetime.now(UTC),
            data=data,
            metadata={"type": "json_passthrough"},
        )


class AggregateCollector(DataCollector):
    """Collector that aggregates data from multiple collectors."""

    def __init__(self, collectors: list[DataCollector]):
        self.collectors = collectors

    def collect(self, context: Any, config: dict) -> CollectedData:
        """Collect and aggregate data from all child collectors.

        Config options:
            collector_configs: dict[int, dict] - Config for each collector by index
        """
        collector_configs = config.get("collector_configs", {})

        aggregated_data = {}

        for i, collector in enumerate(self.collectors):
            collector_config = collector_configs.get(i, {})
            result = collector.collect(context, collector_config)
            aggregated_data[result.source] = result.data

        return CollectedData(
            source="aggregate",
            collected_at=datetime.now(UTC),
            data=aggregated_data,
            metadata={"collector_count": len(self.collectors)},
        )


class ExecutionMetricsCollector(DataCollector):
    """Collector for workflow execution metrics from ExecutionTracker.

    Collects:
    - Summary statistics (success rate, avg duration, total executions)
    - Active workflows
    - Recent executions
    - Step performance by type
    """

    def collect(self, context: Any, config: dict) -> CollectedData:
        """Collect execution metrics from the global tracker.

        Config options:
            db_path: str - Optional database path for tracker
            recent_limit: int - Number of recent executions to include (default: 20)
        """
        from animus_forge.monitoring.tracker import get_tracker

        db_path = config.get("db_path")
        recent_limit = config.get("recent_limit", 20)

        tracker = get_tracker(db_path)
        dashboard_data = tracker.get_dashboard_data()

        # Transform into analytics format
        data = {
            "summary": dashboard_data["summary"],
            "metrics": {
                "timing": self._build_timing_metrics(dashboard_data),
                "counters": self._build_counter_metrics(dashboard_data),
            },
            "active_workflows": dashboard_data["active_workflows"],
            "recent_executions": dashboard_data["recent_executions"][:recent_limit],
            "step_performance": dashboard_data["step_performance"],
        }

        return CollectedData(
            source="execution_tracker",
            collected_at=datetime.now(UTC),
            data=data,
            metadata={
                "type": "execution_metrics",
                "active_count": len(dashboard_data["active_workflows"]),
                "recent_count": len(dashboard_data["recent_executions"]),
            },
        )

    def _build_timing_metrics(self, dashboard_data: dict) -> dict:
        """Build timing metrics in analyzer-compatible format."""
        timing = {}

        # Add step performance as timing metrics
        for step_key, stats in dashboard_data.get("step_performance", {}).items():
            timing[step_key] = {
                "avg_ms": stats.get("avg_ms", 0),
                "count": stats.get("count", 0),
                "max_ms": stats.get("avg_ms", 0) * 1.5,  # Estimate
            }

        # Add workflow duration
        summary = dashboard_data.get("summary", {})
        if summary.get("avg_duration_ms"):
            timing["workflow_execution"] = {
                "avg_ms": summary["avg_duration_ms"],
                "count": summary.get("total_executions", 0),
                "max_ms": summary["avg_duration_ms"] * 2,  # Estimate
            }

        return timing

    def _build_counter_metrics(self, dashboard_data: dict) -> dict:
        """Build counter metrics in analyzer-compatible format."""
        summary = dashboard_data.get("summary", {})
        return {
            "total_executions": summary.get("total_executions", 0),
            "failed_executions": summary.get("failed_executions", 0),
            "active_workflows": summary.get("active_workflows", 0),
            "total_steps_executed": summary.get("total_steps_executed", 0),
            "total_tokens_used": summary.get("total_tokens_used", 0),
            "error_count": summary.get("failed_executions", 0),
        }


class HistoricalMetricsCollector(DataCollector):
    """Collector for historical execution data from MetricsStore.

    Useful for trend analysis over longer time periods.
    """

    def collect(self, context: Any, config: dict) -> CollectedData:
        """Collect historical metrics from the metrics store.

        Config options:
            db_path: str - Database path for metrics store
            hours: int - Hours of history to retrieve (default: 24)
        """
        from animus_forge.monitoring.metrics import MetricsStore

        db_path = config.get("db_path")
        hours = config.get("hours", 24)

        store = MetricsStore(db_path)
        historical = store.get_historical_data(hours=hours)

        # Build time series data
        hourly_stats = self._aggregate_by_hour(historical)

        data = {
            "raw_executions": historical,
            "hourly_stats": hourly_stats,
            "metrics": {
                "timing": self._build_historical_timing(historical),
                "counters": self._build_historical_counters(historical),
            },
            "period_hours": hours,
            "total_records": len(historical),
        }

        return CollectedData(
            source="historical_metrics",
            collected_at=datetime.now(UTC),
            data=data,
            metadata={
                "type": "historical",
                "hours": hours,
                "record_count": len(historical),
            },
        )

    def _aggregate_by_hour(self, executions: list[dict]) -> dict:
        """Aggregate executions by hour for trend analysis."""
        hourly: dict[str, dict] = {}

        for exec_data in executions:
            started = exec_data.get("started_at", "")
            if started:
                # Extract hour from ISO timestamp
                hour_key = started[:13]  # YYYY-MM-DDTHH
                if hour_key not in hourly:
                    hourly[hour_key] = {
                        "count": 0,
                        "failed": 0,
                        "total_duration_ms": 0,
                        "total_tokens": 0,
                    }
                hourly[hour_key]["count"] += 1
                if exec_data.get("status") == "failed":
                    hourly[hour_key]["failed"] += 1
                hourly[hour_key]["total_duration_ms"] += exec_data.get("duration_ms", 0)
                hourly[hour_key]["total_tokens"] += exec_data.get("total_tokens", 0)

        # Calculate averages
        for stats in hourly.values():
            if stats["count"] > 0:
                stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
                stats["success_rate"] = ((stats["count"] - stats["failed"]) / stats["count"]) * 100

        return hourly

    def _build_historical_timing(self, executions: list[dict]) -> dict:
        """Build timing metrics from historical data."""
        if not executions:
            return {}

        durations = [e.get("duration_ms", 0) for e in executions if e.get("duration_ms")]
        if not durations:
            return {}

        return {
            "workflow_execution": {
                "avg_ms": sum(durations) / len(durations),
                "max_ms": max(durations),
                "min_ms": min(durations),
                "count": len(durations),
            }
        }

    def _build_historical_counters(self, executions: list[dict]) -> dict:
        """Build counter metrics from historical data."""
        total = len(executions)
        failed = sum(1 for e in executions if e.get("status") == "failed")
        total_tokens = sum(e.get("total_tokens", 0) for e in executions)
        total_steps = sum(e.get("total_steps", 0) for e in executions)

        return {
            "total_executions": total,
            "failed_executions": failed,
            "error_count": failed,
            "total_tokens_used": total_tokens,
            "total_steps_executed": total_steps,
        }


class APIClientMetricsCollector(DataCollector):
    """Collector for API client resilience metrics.

    Collects rate limiting, bulkhead, and circuit breaker stats.
    """

    def collect(self, context: Any, config: dict) -> CollectedData:
        """Collect API client resilience metrics.

        Config options:
            providers: list[str] - Specific providers to collect (default: all)
        """
        from animus_forge.api_clients.resilience import get_all_provider_stats
        from animus_forge.utils.circuit_breaker import get_all_circuit_stats

        providers = config.get("providers")

        provider_stats = get_all_provider_stats()
        circuit_stats = get_all_circuit_stats()

        # Filter by providers if specified
        if providers:
            provider_stats = {k: v for k, v in provider_stats.items() if k in providers}

        # Build metrics format
        timing = {}
        counters = {}

        for provider, stats in provider_stats.items():
            rate_limit = stats.get("rate_limit", {})
            bulkhead = stats.get("bulkhead", {})

            # Rate limit metrics
            counters[f"{provider}_requests_allowed"] = rate_limit.get("requests_allowed", 0)
            counters[f"{provider}_requests_denied"] = rate_limit.get("requests_denied", 0)

            # Bulkhead metrics
            counters[f"{provider}_concurrent_active"] = bulkhead.get("active", 0)
            counters[f"{provider}_concurrent_waiting"] = bulkhead.get("waiting", 0)

        # Circuit breaker metrics
        for circuit_name, circuit_state in circuit_stats.items():
            counters[f"circuit_{circuit_name}_failures"] = circuit_state.get("failure_count", 0)
            counters[f"circuit_{circuit_name}_state"] = (
                1 if circuit_state.get("state") == "open" else 0
            )

        data = {
            "provider_stats": provider_stats,
            "circuit_breakers": circuit_stats,
            "metrics": {
                "timing": timing,
                "counters": counters,
            },
        }

        return CollectedData(
            source="api_client_metrics",
            collected_at=datetime.now(UTC),
            data=data,
            metadata={
                "type": "resilience_metrics",
                "provider_count": len(provider_stats),
                "circuit_count": len(circuit_stats),
            },
        )


class BudgetMetricsCollector(DataCollector):
    """Collector for budget and cost tracking metrics."""

    def collect(self, context: Any, config: dict) -> CollectedData:
        """Collect budget tracking metrics.

        Config options:
            include_history: bool - Include spending history (default: False)
        """
        from animus_forge.budget import get_budget_tracker

        include_history = config.get("include_history", False)

        tracker = get_budget_tracker()
        stats = tracker.get_stats()

        counters = {
            "budget_total": stats.get("total_budget", 0),
            "budget_spent": stats.get("used", 0),
            "budget_remaining": stats.get("remaining", 0),
            "total_operations": stats.get("total_operations", 0),
            "percent_used": stats.get("percent_used", 0),
        }

        data = {
            "status": stats,
            "metrics": {
                "timing": {},
                "counters": counters,
            },
        }

        if include_history:
            data["history"] = tracker.get_usage_history()

        total_budget = stats.get("total_budget", 1)
        used = stats.get("used", 0)
        utilization = (used / total_budget) * 100 if total_budget > 0 else 0

        return CollectedData(
            source="budget_tracker",
            collected_at=datetime.now(UTC),
            data=data,
            metadata={
                "type": "budget_metrics",
                "utilization_pct": utilization,
            },
        )
