"""Analytics Pipeline Orchestration.

Provides a flexible pipeline framework for chaining data collection,
analysis, visualization, and reporting stages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class PipelineStage(str, Enum):
    """Pipeline stage types."""

    COLLECT = "collect"
    CLEAN = "clean"
    ANALYZE = "analyze"
    VISUALIZE = "visualize"
    REPORT = "report"
    ALERT = "alert"


@dataclass
class StageResult:
    """Result from a pipeline stage."""

    stage: PipelineStage
    status: str  # "success", "failed", "skipped"
    output: Any
    duration_ms: float
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""

    pipeline_id: str
    status: str  # "completed", "failed", "partial"
    started_at: datetime
    completed_at: datetime | None = None
    stages: list[StageResult] = field(default_factory=list)
    final_output: Any = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "stages": [
                {
                    "stage": s.stage.value,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "error": s.error,
                }
                for s in self.stages
            ],
            "errors": self.errors,
        }


class AnalyticsPipeline:
    """Orchestrates analytics workflows through modular stages.

    Example usage:
        pipeline = AnalyticsPipeline("daily_analysis")
        pipeline.add_stage(PipelineStage.COLLECT, json_collector.collect)
        pipeline.add_stage(PipelineStage.ANALYZE, threshold_analyzer.analyze)
        pipeline.add_stage(PipelineStage.VISUALIZE, chart_generator.generate)
        pipeline.add_stage(PipelineStage.REPORT, report_generator.generate)

        result = pipeline.execute({"date": "2025-01-18"})
    """

    def __init__(self, pipeline_id: str, use_agents: bool = True):
        """Initialize analytics pipeline.

        Args:
            pipeline_id: Unique identifier for this pipeline
            use_agents: Whether to use Claude agents for AI-powered stages
        """
        self.pipeline_id = pipeline_id
        self.use_agents = use_agents
        self._stages: list[tuple[PipelineStage, Callable, dict]] = []
        self._claude_client = None
        if use_agents:
            from animus_forge.api_clients import ClaudeCodeClient

            self._claude_client = ClaudeCodeClient()

    def add_stage(
        self,
        stage_type: PipelineStage,
        handler: Callable[[Any, dict], Any],
        config: dict = None,
    ) -> AnalyticsPipeline:
        """Add a stage to the pipeline.

        Args:
            stage_type: Type of pipeline stage
            handler: Function to execute for this stage
            config: Optional configuration for the stage

        Returns:
            Self for method chaining
        """
        self._stages.append((stage_type, handler, config or {}))
        return self

    def add_agent_stage(
        self,
        stage_type: PipelineStage,
        agent_role: str,
        task_template: str,
        config: dict = None,
    ) -> AnalyticsPipeline:
        """Add an AI agent-powered stage.

        Args:
            stage_type: Type of pipeline stage
            agent_role: Role for the Claude agent (e.g., "analyst", "visualizer")
            task_template: Task template with {{context}} placeholder
            config: Optional configuration

        Returns:
            Self for method chaining
        """
        if not self._claude_client:
            raise ValueError("Pipeline not configured for agent usage")

        def agent_handler(context: Any, cfg: dict) -> Any:
            task = task_template.replace("{{context}}", str(context))
            result = self._claude_client.execute_agent(
                role=agent_role,
                task=task,
                context=str(context),
            )
            if not result.get("success"):
                raise RuntimeError(result.get("error", "Agent execution failed"))
            output = result.get("output", result)
            if result.get("pending_user_confirmation"):
                return {
                    "output": output,
                    "pending_user_confirmation": True,
                    "consensus": result.get("consensus"),
                }
            return output

        self._stages.append((stage_type, agent_handler, config or {}))
        return self

    def execute(self, initial_context: dict = None) -> PipelineResult:
        """Execute the pipeline.

        Args:
            initial_context: Initial context/parameters for the pipeline

        Returns:
            PipelineResult with all stage outputs
        """
        import time

        result = PipelineResult(
            pipeline_id=self.pipeline_id,
            status="running",
            started_at=datetime.now(UTC),
        )

        context = initial_context or {}
        current_output = context

        for stage_type, handler, config in self._stages:
            stage_start = time.time()

            try:
                # Merge config into context
                stage_context = {**context, **config}

                # Execute stage
                output = handler(current_output, stage_context)

                duration_ms = (time.time() - stage_start) * 1000

                stage_result = StageResult(
                    stage=stage_type,
                    status="success",
                    output=output,
                    duration_ms=duration_ms,
                    metadata={"config": config},
                )

                result.stages.append(stage_result)
                current_output = output

                # Update context with stage output
                context[f"{stage_type.value}_output"] = output

            except Exception as e:
                duration_ms = (time.time() - stage_start) * 1000

                stage_result = StageResult(
                    stage=stage_type,
                    status="failed",
                    output=None,
                    duration_ms=duration_ms,
                    error=str(e),
                )

                result.stages.append(stage_result)
                result.errors.append(f"Stage {stage_type.value} failed: {e}")
                result.status = "failed"
                break

        if result.status != "failed":
            result.status = "completed"
            result.final_output = current_output

        result.completed_at = datetime.now(UTC)
        return result


class PipelineBuilder:
    """Fluent builder for creating common pipeline configurations."""

    @staticmethod
    def trend_analysis_pipeline() -> AnalyticsPipeline:
        """Create a pipeline for analyzing metrics trends."""
        from .analyzers import TrendAnalyzer
        from .collectors import JSONCollector
        from .reporters import ReportGenerator

        pipeline = AnalyticsPipeline("trend_analysis")

        collector = JSONCollector()
        analyzer = TrendAnalyzer()
        reporter = ReportGenerator()

        return (
            pipeline.add_stage(PipelineStage.COLLECT, collector.collect)
            .add_stage(PipelineStage.ANALYZE, analyzer.analyze)
            .add_stage(PipelineStage.REPORT, reporter.generate)
        )

    @staticmethod
    def threshold_alert_pipeline() -> AnalyticsPipeline:
        """Create a pipeline for threshold-based alerting."""
        from .analyzers import ThresholdAnalyzer
        from .collectors import JSONCollector
        from .reporters import AlertGenerator

        pipeline = AnalyticsPipeline("threshold_alerts")

        collector = JSONCollector()
        analyzer = ThresholdAnalyzer()
        alerter = AlertGenerator()

        return (
            pipeline.add_stage(PipelineStage.COLLECT, collector.collect)
            .add_stage(PipelineStage.ANALYZE, analyzer.analyze)
            .add_stage(PipelineStage.ALERT, alerter.generate)
        )

    @staticmethod
    def full_analysis_pipeline() -> AnalyticsPipeline:
        """Create a comprehensive analysis pipeline with AI-powered reporting."""
        from .analyzers import CompositeAnalyzer, ThresholdAnalyzer, TrendAnalyzer
        from .collectors import JSONCollector
        from .reporters import ReportGenerator

        pipeline = AnalyticsPipeline("full_analysis")

        collector = JSONCollector()
        analyzer = CompositeAnalyzer([TrendAnalyzer(), ThresholdAnalyzer()])
        reporter = ReportGenerator()

        return (
            pipeline.add_stage(PipelineStage.COLLECT, collector.collect)
            .add_stage(PipelineStage.ANALYZE, analyzer.analyze)
            .add_agent_stage(
                PipelineStage.VISUALIZE,
                "visualizer",
                "Create visualization recommendations for this analysis data:\n\n{{context}}",
            )
            .add_stage(PipelineStage.REPORT, reporter.generate)
        )

    @staticmethod
    def workflow_metrics_pipeline() -> AnalyticsPipeline:
        """Create a pipeline for real-time workflow metrics analysis.

        Collects data from ExecutionTracker and analyzes trends.
        """
        from .analyzers import CompositeAnalyzer, ThresholdAnalyzer, TrendAnalyzer
        from .collectors import ExecutionMetricsCollector
        from .reporters import ReportGenerator
        from .visualizers import ChartGenerator

        pipeline = AnalyticsPipeline("workflow_metrics", use_agents=False)

        collector = ExecutionMetricsCollector()
        analyzer = CompositeAnalyzer([TrendAnalyzer(), ThresholdAnalyzer()])
        visualizer = ChartGenerator()
        reporter = ReportGenerator()

        return (
            pipeline.add_stage(PipelineStage.COLLECT, collector.collect)
            .add_stage(
                PipelineStage.ANALYZE,
                analyzer.analyze,
                config={
                    "thresholds": {
                        "metrics.counters.error_count": {"warning": 5, "critical": 10},
                        "summary.success_rate": {
                            "warning": 90,
                            "critical": 80,
                            "direction": "below",
                        },
                    }
                },
            )
            .add_stage(PipelineStage.VISUALIZE, visualizer.generate)
            .add_stage(PipelineStage.REPORT, reporter.generate)
        )

    @staticmethod
    def historical_trends_pipeline(hours: int = 24) -> AnalyticsPipeline:
        """Create a pipeline for historical trend analysis.

        Args:
            hours: Hours of history to analyze (default: 24)
        """
        from .analyzers import TrendAnalyzer
        from .collectors import HistoricalMetricsCollector
        from .reporters import ReportGenerator
        from .visualizers import ChartGenerator

        pipeline = AnalyticsPipeline("historical_trends", use_agents=False)

        collector = HistoricalMetricsCollector()
        analyzer = TrendAnalyzer()
        visualizer = ChartGenerator()
        reporter = ReportGenerator()

        return (
            pipeline.add_stage(
                PipelineStage.COLLECT,
                collector.collect,
                config={"hours": hours},
            )
            .add_stage(PipelineStage.ANALYZE, analyzer.analyze)
            .add_stage(PipelineStage.VISUALIZE, visualizer.generate)
            .add_stage(PipelineStage.REPORT, reporter.generate)
        )

    @staticmethod
    def api_health_pipeline() -> AnalyticsPipeline:
        """Create a pipeline for API client health monitoring.

        Monitors rate limiting, circuit breakers, and bulkhead stats.
        """
        from .analyzers import ThresholdAnalyzer
        from .collectors import APIClientMetricsCollector
        from .reporters import AlertGenerator

        pipeline = AnalyticsPipeline("api_health", use_agents=False)

        collector = APIClientMetricsCollector()
        analyzer = ThresholdAnalyzer()
        alerter = AlertGenerator()

        return (
            pipeline.add_stage(PipelineStage.COLLECT, collector.collect)
            .add_stage(
                PipelineStage.ANALYZE,
                analyzer.analyze,
                config={
                    "thresholds": {
                        # Alert if any circuit is open
                        "metrics.counters.circuit_openai_state": {
                            "warning": 1,
                            "critical": 1,
                        },
                        "metrics.counters.circuit_anthropic_state": {
                            "warning": 1,
                            "critical": 1,
                        },
                        # Alert on high denial rates
                        "metrics.counters.openai_requests_denied": {
                            "warning": 10,
                            "critical": 50,
                        },
                    }
                },
            )
            .add_stage(PipelineStage.ALERT, alerter.generate)
        )

    @staticmethod
    def operations_dashboard_pipeline() -> AnalyticsPipeline:
        """Create a comprehensive operations dashboard pipeline.

        Combines workflow metrics, API health, and budget tracking.
        """
        from .analyzers import CompositeAnalyzer, ThresholdAnalyzer, TrendAnalyzer
        from .collectors import (
            AggregateCollector,
            APIClientMetricsCollector,
            BudgetMetricsCollector,
            ExecutionMetricsCollector,
        )
        from .reporters import ReportGenerator
        from .visualizers import DashboardBuilder

        pipeline = AnalyticsPipeline("operations_dashboard", use_agents=False)

        # Aggregate multiple data sources
        collector = AggregateCollector(
            [
                ExecutionMetricsCollector(),
                APIClientMetricsCollector(),
                BudgetMetricsCollector(),
            ]
        )

        analyzer = CompositeAnalyzer([TrendAnalyzer(), ThresholdAnalyzer()])
        dashboard = DashboardBuilder()
        reporter = ReportGenerator()

        return (
            pipeline.add_stage(PipelineStage.COLLECT, collector.collect)
            .add_stage(
                PipelineStage.ANALYZE,
                analyzer.analyze,
                config={
                    "thresholds": {
                        "execution_tracker.metrics.counters.error_count": {
                            "warning": 5,
                            "critical": 10,
                        },
                        "budget_tracker.metrics.counters.budget_remaining": {
                            "warning": 10,
                            "critical": 5,
                            "direction": "below",
                        },
                    }
                },
            )
            .add_stage(PipelineStage.VISUALIZE, dashboard.build)
            .add_stage(PipelineStage.REPORT, reporter.generate)
        )
