"""Tests for Analytics Pipeline components.

Tests for:
- Pipeline execution flow
- Data collectors
- Analyzers
- Pre-built pipelines
"""

import sys
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from animus_forge.analytics.analyzers import (
    AnalysisResult,
    CompositeAnalyzer,
    ThresholdAnalyzer,
    TrendAnalyzer,
)
from animus_forge.analytics.collectors import (
    AggregateCollector,
    CollectedData,
    JSONCollector,
)
from animus_forge.analytics.pipeline import (
    AnalyticsPipeline,
    PipelineBuilder,
    PipelineResult,
    PipelineStage,
    StageResult,
)


class TestPipelineCore:
    """Tests for core pipeline functionality."""

    def test_pipeline_creation(self):
        """Pipeline can be created with an ID."""
        pipeline = AnalyticsPipeline("test_pipeline", use_agents=False)
        assert pipeline.pipeline_id == "test_pipeline"
        assert len(pipeline._stages) == 0

    def test_add_stage_returns_self(self):
        """add_stage returns self for chaining."""
        pipeline = AnalyticsPipeline("test", use_agents=False)

        def handler(context, config):
            return context

        result = pipeline.add_stage(PipelineStage.COLLECT, handler)
        assert result is pipeline

    def test_pipeline_method_chaining(self):
        """Pipeline supports method chaining."""
        pipeline = AnalyticsPipeline("test", use_agents=False)

        def handler(context, config):
            return context

        result = (
            pipeline.add_stage(PipelineStage.COLLECT, handler)
            .add_stage(PipelineStage.ANALYZE, handler)
            .add_stage(PipelineStage.REPORT, handler)
        )

        assert result is pipeline
        assert len(pipeline._stages) == 3

    def test_pipeline_executes_stages_in_order(self):
        """Pipeline executes stages in the order they were added."""
        pipeline = AnalyticsPipeline("test", use_agents=False)
        execution_order = []

        def stage1(context, config):
            execution_order.append("stage1")
            return {"from": "stage1"}

        def stage2(context, config):
            execution_order.append("stage2")
            return {"from": "stage2"}

        def stage3(context, config):
            execution_order.append("stage3")
            return {"from": "stage3"}

        pipeline.add_stage(PipelineStage.COLLECT, stage1)
        pipeline.add_stage(PipelineStage.ANALYZE, stage2)
        pipeline.add_stage(PipelineStage.REPORT, stage3)

        result = pipeline.execute()

        assert execution_order == ["stage1", "stage2", "stage3"]
        assert result.status == "completed"
        assert result.final_output == {"from": "stage3"}

    def test_pipeline_passes_output_between_stages(self):
        """Each stage receives the previous stage's output."""
        pipeline = AnalyticsPipeline("test", use_agents=False)
        received_inputs = []

        def stage1(context, config):
            # Copy to avoid mutation issues
            received_inputs.append(dict(context))
            return {"value": 1}

        def stage2(context, config):
            received_inputs.append(dict(context))
            return {"value": context.get("value", 0) + 1}

        def stage3(context, config):
            received_inputs.append(dict(context))
            return {"value": context.get("value", 0) + 1}

        pipeline.add_stage(PipelineStage.COLLECT, stage1)
        pipeline.add_stage(PipelineStage.ANALYZE, stage2)
        pipeline.add_stage(PipelineStage.REPORT, stage3)

        result = pipeline.execute({"initial": True})

        # Stage 1 receives initial context
        assert received_inputs[0] == {"initial": True}
        # Stage 2 receives output from stage 1
        assert received_inputs[1]["value"] == 1
        # Stage 3 receives output from stage 2
        assert received_inputs[2]["value"] == 2
        # Final output is from stage 3
        assert result.final_output == {"value": 3}

    def test_pipeline_stops_on_error(self):
        """Pipeline stops executing when a stage fails."""
        pipeline = AnalyticsPipeline("test", use_agents=False)
        execution_count = 0

        def stage1(context, config):
            nonlocal execution_count
            execution_count += 1
            raise ValueError("Stage 1 failed")

        def stage2(context, config):
            nonlocal execution_count
            execution_count += 1
            return context

        pipeline.add_stage(PipelineStage.COLLECT, stage1)
        pipeline.add_stage(PipelineStage.ANALYZE, stage2)

        result = pipeline.execute()

        assert execution_count == 1
        assert result.status == "failed"
        assert len(result.errors) == 1
        assert "Stage 1 failed" in result.errors[0]

    def test_pipeline_result_contains_stage_results(self):
        """Pipeline result includes results from each stage."""
        pipeline = AnalyticsPipeline("test", use_agents=False)

        def handler(context, config):
            return {"processed": True}

        pipeline.add_stage(PipelineStage.COLLECT, handler)
        pipeline.add_stage(PipelineStage.ANALYZE, handler)

        result = pipeline.execute()

        assert len(result.stages) == 2
        assert result.stages[0].stage == PipelineStage.COLLECT
        assert result.stages[0].status == "success"
        assert result.stages[1].stage == PipelineStage.ANALYZE
        assert result.stages[1].status == "success"

    def test_pipeline_result_tracks_timing(self):
        """Pipeline result includes duration for each stage."""
        pipeline = AnalyticsPipeline("test", use_agents=False)

        def handler(context, config):
            return context

        pipeline.add_stage(PipelineStage.COLLECT, handler)
        result = pipeline.execute()

        assert result.stages[0].duration_ms >= 0
        assert result.completed_at is not None
        assert result.started_at is not None

    def test_pipeline_result_to_dict(self):
        """Pipeline result can be serialized to dict."""
        result = PipelineResult(
            pipeline_id="test",
            status="completed",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            stages=[
                StageResult(
                    stage=PipelineStage.COLLECT,
                    status="success",
                    output={"data": "test"},
                    duration_ms=10.5,
                )
            ],
            errors=[],
        )

        data = result.to_dict()

        assert data["pipeline_id"] == "test"
        assert data["status"] == "completed"
        assert len(data["stages"]) == 1
        assert data["stages"][0]["stage"] == "collect"


class TestJSONCollector:
    """Tests for JSONCollector."""

    def test_json_collector_passthrough(self):
        """JSONCollector passes through JSON data."""
        collector = JSONCollector()

        result = collector.collect(
            {"existing": "context"},
            {"data": {"key": "value"}, "source_name": "test_source"},
        )

        assert isinstance(result, CollectedData)
        assert result.source == "test_source"
        assert result.data == {"key": "value"}

    def test_json_collector_uses_context_as_fallback(self):
        """JSONCollector uses context if no data in config."""
        collector = JSONCollector()

        result = collector.collect({"context_key": "context_value"}, {})

        assert result.data == {"context_key": "context_value"}

    def test_collected_data_to_context_string(self):
        """CollectedData can be converted to string for AI context."""
        data = CollectedData(
            source="test",
            collected_at=datetime.now(UTC),
            data={"metrics": {"count": 10}, "items": ["a", "b", "c"]},
            metadata={"type": "test"},
        )

        context_str = data.to_context_string()

        assert "# Data Collection: test" in context_str
        assert "metrics" in context_str
        assert "items" in context_str


class TestAggregateCollector:
    """Tests for AggregateCollector."""

    def test_aggregate_collector_combines_sources(self):
        """AggregateCollector combines data from multiple collectors."""
        collector1 = JSONCollector()
        collector2 = JSONCollector()

        aggregate = AggregateCollector([collector1, collector2])

        result = aggregate.collect(
            {},
            {
                "collector_configs": {
                    0: {"data": {"source1": True}, "source_name": "first"},
                    1: {"data": {"source2": True}, "source_name": "second"},
                }
            },
        )

        assert result.source == "aggregate"
        assert "first" in result.data
        assert "second" in result.data
        assert result.metadata["collector_count"] == 2


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer."""

    def test_trend_analyzer_with_collected_data(self):
        """TrendAnalyzer processes CollectedData input."""
        analyzer = TrendAnalyzer()

        collected = CollectedData(
            source="test",
            collected_at=datetime.now(UTC),
            data={
                "metrics": {
                    "counters": {"request_count": 100, "error_count": 5},
                    "timing": {"response_time": {"avg_ms": 50}},
                }
            },
            metadata={},
        )

        result = analyzer.analyze(collected, {})

        assert isinstance(result, AnalysisResult)
        assert result.analyzer == "trends"  # plural form

    def test_trend_analyzer_severity_info_by_default(self):
        """TrendAnalyzer returns info severity when no issues."""
        analyzer = TrendAnalyzer()

        result = analyzer.analyze(
            {"metrics": {"counters": {"ok": 100}}},
            {},
        )

        assert result.severity == "info"


class TestThresholdAnalyzer:
    """Tests for ThresholdAnalyzer."""

    def test_threshold_analyzer_detects_above_threshold(self):
        """ThresholdAnalyzer detects values above threshold."""
        analyzer = ThresholdAnalyzer()

        result = analyzer.analyze(
            {"metrics": {"counters": {"error_count": 15}}},
            {"thresholds": {"metrics.counters.error_count": {"warning": 5, "critical": 10}}},
        )

        assert result.severity == "critical"
        assert len(result.findings) > 0

    def test_threshold_analyzer_detects_below_threshold(self):
        """ThresholdAnalyzer detects values below threshold with direction."""
        analyzer = ThresholdAnalyzer()

        # With direction "below": value 75 <= critical 80, so it's critical
        # To get warning, value must be <= warning but > critical
        result = analyzer.analyze(
            {"summary": {"success_rate": 85}},  # 85 <= 90 but > 80
            {
                "thresholds": {
                    "summary.success_rate": {
                        "warning": 90,
                        "critical": 80,
                        "direction": "below",
                    }
                }
            },
        )

        assert result.severity == "warning"
        assert len(result.findings) > 0

    def test_threshold_analyzer_ok_when_within_limits(self):
        """ThresholdAnalyzer returns info when values are OK."""
        analyzer = ThresholdAnalyzer()

        result = analyzer.analyze(
            {"metrics": {"counters": {"error_count": 2}}},
            {"thresholds": {"metrics.counters.error_count": {"warning": 5, "critical": 10}}},
        )

        assert result.severity == "info"
        # ThresholdAnalyzer always adds an info finding when all metrics are OK
        assert len(result.findings) == 1
        assert result.findings[0]["severity"] == "info"
        assert "within acceptable" in result.findings[0]["message"]


class TestCompositeAnalyzer:
    """Tests for CompositeAnalyzer."""

    def test_composite_analyzer_runs_all_analyzers(self):
        """CompositeAnalyzer runs all child analyzers."""
        trend = TrendAnalyzer()
        threshold = ThresholdAnalyzer()

        composite = CompositeAnalyzer([trend, threshold])

        result = composite.analyze(
            {"metrics": {"counters": {"count": 10}}},
            {"thresholds": {"metrics.counters.count": {"warning": 5}}},
        )

        assert isinstance(result, AnalysisResult)
        assert result.analyzer == "composite"

    def test_composite_analyzer_uses_worst_severity(self):
        """CompositeAnalyzer uses the worst severity from children."""
        trend = TrendAnalyzer()
        threshold = ThresholdAnalyzer()

        composite = CompositeAnalyzer([trend, threshold])

        # CompositeAnalyzer passes config to children via analyzer_configs
        # If not specified, it passes {} to each, so we need to use analyzer_configs
        result = composite.analyze(
            {"metrics": {"counters": {"errors": 100}}},
            {
                "analyzer_configs": {
                    1: {"thresholds": {"metrics.counters.errors": {"warning": 5, "critical": 10}}}
                }
            },
        )

        assert result.severity == "critical"


class TestPreBuiltPipelines:
    """Tests for PipelineBuilder pre-built pipelines."""

    def test_trend_analysis_pipeline_creates(self):
        """trend_analysis_pipeline creates a valid pipeline."""
        pipeline = PipelineBuilder.trend_analysis_pipeline()

        assert pipeline.pipeline_id == "trend_analysis"
        assert len(pipeline._stages) == 3

    def test_threshold_alert_pipeline_creates(self):
        """threshold_alert_pipeline creates a valid pipeline."""
        pipeline = PipelineBuilder.threshold_alert_pipeline()

        assert pipeline.pipeline_id == "threshold_alerts"
        assert len(pipeline._stages) == 3

    def test_full_analysis_pipeline_creates(self):
        """full_analysis_pipeline creates a valid pipeline."""
        pipeline = PipelineBuilder.full_analysis_pipeline()

        assert pipeline.pipeline_id == "full_analysis"
        assert len(pipeline._stages) >= 3

    @patch("animus_forge.monitoring.tracker.get_tracker")
    def test_workflow_metrics_pipeline_creates(self, mock_get_tracker):
        """workflow_metrics_pipeline creates a valid pipeline."""
        # Mock the tracker
        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {
                "total_executions": 10,
                "failed_executions": 1,
                "active_workflows": 0,
                "total_steps_executed": 50,
                "total_tokens_used": 1000,
                "success_rate": 90,
                "avg_duration_ms": 100,
            },
            "active_workflows": [],
            "recent_executions": [],
            "step_performance": {},
        }
        mock_get_tracker.return_value = mock_tracker

        pipeline = PipelineBuilder.workflow_metrics_pipeline()

        assert pipeline.pipeline_id == "workflow_metrics"
        assert len(pipeline._stages) == 4

    @patch("animus_forge.monitoring.metrics.MetricsStore")
    def test_historical_trends_pipeline_creates(self, mock_store):
        """historical_trends_pipeline creates a valid pipeline."""
        mock_store.return_value.get_historical_data.return_value = []

        pipeline = PipelineBuilder.historical_trends_pipeline(hours=48)

        assert pipeline.pipeline_id == "historical_trends"
        assert len(pipeline._stages) == 4

    @patch("animus_forge.api_clients.resilience.get_all_provider_stats")
    @patch("animus_forge.utils.circuit_breaker.get_all_circuit_stats")
    def test_api_health_pipeline_creates(self, mock_circuit, mock_provider):
        """api_health_pipeline creates a valid pipeline."""
        mock_provider.return_value = {}
        mock_circuit.return_value = {}

        pipeline = PipelineBuilder.api_health_pipeline()

        assert pipeline.pipeline_id == "api_health"
        assert len(pipeline._stages) == 3

    def test_operations_dashboard_pipeline_creates(self):
        """operations_dashboard_pipeline creates a valid pipeline."""
        # This test just verifies the pipeline can be constructed
        # The actual collectors are tested separately
        pipeline = PipelineBuilder.operations_dashboard_pipeline()

        assert pipeline.pipeline_id == "operations_dashboard"
        assert len(pipeline._stages) == 4


class TestPipelineExecution:
    """Integration tests for full pipeline execution."""

    def test_simple_pipeline_end_to_end(self):
        """Simple pipeline executes end to end."""
        pipeline = AnalyticsPipeline("test_e2e", use_agents=False)

        collector = JSONCollector()
        analyzer = TrendAnalyzer()

        pipeline.add_stage(PipelineStage.COLLECT, collector.collect)
        pipeline.add_stage(PipelineStage.ANALYZE, analyzer.analyze)

        result = pipeline.execute({"data": {"metrics": {"counters": {"requests": 100}}}})

        assert result.status == "completed"
        assert len(result.stages) == 2
        assert result.stages[0].status == "success"
        assert result.stages[1].status == "success"

    def test_pipeline_with_config(self):
        """Pipeline passes config to stages."""
        pipeline = AnalyticsPipeline("test_config", use_agents=False)

        received_configs = []

        def capture_config(context, config):
            received_configs.append(config)
            return context

        pipeline.add_stage(PipelineStage.COLLECT, capture_config, {"custom_key": "custom_value"})

        pipeline.execute({})

        assert len(received_configs) == 1
        assert received_configs[0].get("custom_key") == "custom_value"


# ---------------------------------------------------------------------------
# Additional coverage: PipelineStage enum completeness
# ---------------------------------------------------------------------------


class TestPipelineStageValues:
    """Tests for PipelineStage enum values."""

    def test_all_values(self):
        assert PipelineStage.COLLECT.value == "collect"
        assert PipelineStage.CLEAN.value == "clean"
        assert PipelineStage.ANALYZE.value == "analyze"
        assert PipelineStage.VISUALIZE.value == "visualize"
        assert PipelineStage.REPORT.value == "report"
        assert PipelineStage.ALERT.value == "alert"

    def test_string_enum(self):
        """PipelineStage is a string enum."""
        assert isinstance(PipelineStage.COLLECT, str)


# ---------------------------------------------------------------------------
# Additional coverage: StageResult
# ---------------------------------------------------------------------------


class TestStageResultExtended:
    """Extended StageResult tests."""

    def test_defaults(self):
        result = StageResult(
            stage=PipelineStage.COLLECT,
            status="success",
            output={"data": [1, 2, 3]},
            duration_ms=50.0,
        )
        assert result.error is None
        assert result.metadata == {}

    def test_with_error(self):
        result = StageResult(
            stage=PipelineStage.ANALYZE,
            status="failed",
            output=None,
            duration_ms=10.0,
            error="Analysis failed",
        )
        assert result.status == "failed"
        assert result.error == "Analysis failed"

    def test_with_metadata(self):
        result = StageResult(
            stage=PipelineStage.REPORT,
            status="success",
            output="report",
            duration_ms=100.0,
            metadata={"format": "pdf"},
        )
        assert result.metadata == {"format": "pdf"}


# ---------------------------------------------------------------------------
# Additional coverage: PipelineResult serialization edge cases
# ---------------------------------------------------------------------------


class TestPipelineResultSerialization:
    """Extended PipelineResult serialization tests."""

    def test_to_dict_no_completed_at(self):
        result = PipelineResult(
            pipeline_id="test",
            status="running",
            started_at=datetime.now(UTC),
        )
        d = result.to_dict()
        assert d["completed_at"] is None

    def test_to_dict_with_errors(self):
        result = PipelineResult(
            pipeline_id="test",
            status="failed",
            started_at=datetime.now(UTC),
            errors=["Error 1", "Error 2"],
        )
        d = result.to_dict()
        assert d["errors"] == ["Error 1", "Error 2"]

    def test_to_dict_with_multiple_stages(self):
        result = PipelineResult(
            pipeline_id="test",
            status="completed",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            stages=[
                StageResult(
                    stage=PipelineStage.COLLECT,
                    status="success",
                    output={},
                    duration_ms=10.0,
                ),
                StageResult(
                    stage=PipelineStage.ANALYZE,
                    status="success",
                    output={},
                    duration_ms=20.0,
                ),
            ],
        )
        d = result.to_dict()
        assert len(d["stages"]) == 2
        assert d["stages"][0]["stage"] == "collect"
        assert d["stages"][1]["stage"] == "analyze"


# ---------------------------------------------------------------------------
# Additional coverage: add_agent_stage
# ---------------------------------------------------------------------------


class TestAddAgentStageExtended:
    """Extended tests for add_agent_stage."""

    def test_raises_without_agent_client(self):
        pipeline = AnalyticsPipeline("test", use_agents=False)
        with pytest.raises(ValueError, match="not configured for agent"):
            pipeline.add_agent_stage(
                PipelineStage.ANALYZE,
                "analyst",
                "Analyze: {{context}}",
            )

    @patch("animus_forge.api_clients.ClaudeCodeClient")
    def test_agent_handler_success(self, mock_cls):
        mock_client = MagicMock()
        mock_client.execute_agent.return_value = {
            "success": True,
            "output": "analysis result",
        }
        mock_cls.return_value = mock_client

        pipeline = AnalyticsPipeline("test", use_agents=True)
        pipeline.add_agent_stage(
            PipelineStage.ANALYZE,
            "analyst",
            "Analyze: {{context}}",
        )

        _, handler, _ = pipeline._stages[0]
        result = handler("test data", {})
        assert result == "analysis result"

    @patch("animus_forge.api_clients.ClaudeCodeClient")
    def test_agent_handler_failure_raises(self, mock_cls):
        mock_client = MagicMock()
        mock_client.execute_agent.return_value = {
            "success": False,
            "error": "Agent failed",
        }
        mock_cls.return_value = mock_client

        pipeline = AnalyticsPipeline("test", use_agents=True)
        pipeline.add_agent_stage(PipelineStage.ANALYZE, "analyst", "{{context}}")

        _, handler, _ = pipeline._stages[0]
        with pytest.raises(RuntimeError, match="Agent.*failed"):
            handler("test data", {})

    @patch("animus_forge.api_clients.ClaudeCodeClient")
    def test_agent_handler_pending_confirmation(self, mock_cls):
        mock_client = MagicMock()
        mock_client.execute_agent.return_value = {
            "success": True,
            "output": "partial",
            "pending_user_confirmation": True,
            "consensus": {"agreed": True},
        }
        mock_cls.return_value = mock_client

        pipeline = AnalyticsPipeline("test", use_agents=True)
        pipeline.add_agent_stage(PipelineStage.ANALYZE, "analyst", "{{context}}")

        _, handler, _ = pipeline._stages[0]
        result = handler("test data", {})
        assert isinstance(result, dict)
        assert result["pending_user_confirmation"] is True
        assert result["consensus"] == {"agreed": True}


# ---------------------------------------------------------------------------
# Additional coverage: pipeline execution edge cases
# ---------------------------------------------------------------------------


class TestPipelineExecutionEdgeCases:
    """Edge case tests for pipeline execution."""

    def test_empty_pipeline(self):
        pipeline = AnalyticsPipeline("empty", use_agents=False)
        result = pipeline.execute()
        assert result.status == "completed"
        assert result.final_output == {}
        assert result.stages == []

    def test_none_initial_context(self):
        pipeline = AnalyticsPipeline("test", use_agents=False)
        pipeline.add_stage(
            PipelineStage.COLLECT,
            lambda data, ctx: {"ok": True},
        )
        result = pipeline.execute(initial_context=None)
        assert result.status == "completed"

    def test_handler_returning_none(self):
        pipeline = AnalyticsPipeline("test", use_agents=False)
        pipeline.add_stage(PipelineStage.COLLECT, lambda d, c: None)
        result = pipeline.execute()
        assert result.status == "completed"
        assert result.final_output is None

    def test_first_stage_fails(self):
        pipeline = AnalyticsPipeline("test", use_agents=False)

        def failing(data, ctx):
            raise RuntimeError("Startup failure")

        pipeline.add_stage(PipelineStage.COLLECT, failing)
        result = pipeline.execute()
        assert result.status == "failed"
        assert len(result.stages) == 1
        assert result.stages[0].status == "failed"

    def test_long_pipeline_chain(self):
        """Pipeline with many stages processes correctly."""
        pipeline = AnalyticsPipeline("test", use_agents=False)
        for i in range(10):
            pipeline.add_stage(
                PipelineStage.CLEAN,
                lambda data, ctx, idx=i: (
                    {"step": idx, **data} if isinstance(data, dict) else {"step": idx}
                ),
            )
        result = pipeline.execute(initial_context={"start": True})
        assert result.status == "completed"
        assert len(result.stages) == 10

    def test_stage_config_merged(self):
        """Stage-level config is merged into the context dict."""
        received = {}

        def handler(data, ctx):
            received.update(ctx)
            return data

        pipeline = AnalyticsPipeline("test", use_agents=False)
        pipeline.add_stage(
            PipelineStage.ANALYZE,
            handler,
            config={"threshold": 0.5},
        )
        pipeline.execute(initial_context={"base": "val"})
        assert received.get("threshold") == 0.5
        assert received.get("base") == "val"

    def test_stage_output_stored_in_context(self):
        """Each stage output is stored in context keyed by stage type."""
        context_seen = {}

        def analyze(data, ctx):
            context_seen.update(ctx)
            return {"analyzed": True}

        pipeline = AnalyticsPipeline("test", use_agents=False)
        pipeline.add_stage(
            PipelineStage.COLLECT,
            lambda data, ctx: {"raw_data": [1, 2]},
        )
        pipeline.add_stage(PipelineStage.ANALYZE, analyze)
        pipeline.execute()
        assert "collect_output" in context_seen


# ---------------------------------------------------------------------------
# Additional coverage: CollectedData.to_context_string
# ---------------------------------------------------------------------------


class TestCollectedDataContextString:
    """Tests for CollectedData.to_context_string formatting."""

    def test_with_dict_data(self):
        data = CollectedData(
            source="test",
            collected_at=datetime.now(UTC),
            data={"section": {"key1": "val1", "key2": "val2"}},
            metadata={},
        )
        s = data.to_context_string()
        assert "# Data Collection: test" in s
        assert "key1: val1" in s
        assert "key2: val2" in s

    def test_with_list_data(self):
        data = CollectedData(
            source="test",
            collected_at=datetime.now(UTC),
            data={"items": ["a", "b", "c"]},
            metadata={},
        )
        s = data.to_context_string()
        assert "- a" in s
        assert "- b" in s

    def test_with_list_data_truncated(self):
        """Lists longer than 10 items are truncated."""
        data = CollectedData(
            source="test",
            collected_at=datetime.now(UTC),
            data={"items": list(range(15))},
            metadata={},
        )
        s = data.to_context_string()
        assert "and 5 more" in s

    def test_with_scalar_data(self):
        data = CollectedData(
            source="test",
            collected_at=datetime.now(UTC),
            data={"count": 42},
            metadata={},
        )
        s = data.to_context_string()
        assert "42" in s


# ---------------------------------------------------------------------------
# AnalysisResult.to_context_string
# ---------------------------------------------------------------------------


class TestAnalysisResultContextString:
    """Tests for AnalysisResult.to_context_string formatting."""

    def test_basic_output(self):
        result = AnalysisResult(
            analyzer="test_analyzer",
            analyzed_at=datetime.now(UTC),
            findings=[
                {"severity": "warning", "message": "High error rate"},
            ],
            metrics={"error_rate": 0.15},
            recommendations=["Investigate errors"],
            severity="warning",
        )
        s = result.to_context_string()
        assert "# Analysis Results: test_analyzer" in s
        assert "[WARNING] High error rate" in s
        assert "error_rate: 0.15" in s
        assert "Investigate errors" in s

    def test_empty_recommendations(self):
        result = AnalysisResult(
            analyzer="test",
            analyzed_at=datetime.now(UTC),
            findings=[],
            metrics={},
            recommendations=[],
            severity="info",
        )
        s = result.to_context_string()
        assert "Recommendations" not in s
