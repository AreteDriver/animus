"""Tests for the analytics pipeline module."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

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
from animus_forge.analytics.reporters import (
    Alert,
    AlertBatch,
    AlertGenerator,
    GeneratedReport,
    ReportGenerator,
    ReportSection,
)


class TestCollectedData:
    """Tests for CollectedData dataclass."""

    def test_basic_creation(self):
        """Test basic CollectedData creation."""
        data = CollectedData(
            source="test_source",
            collected_at=datetime.now(UTC),
            data={"key": "value"},
            metadata={"type": "test"},
        )
        assert data.source == "test_source"
        assert data.data == {"key": "value"}
        assert data.metadata == {"type": "test"}

    def test_to_context_string_with_dict(self):
        """Test context string generation with dict data."""
        data = CollectedData(
            source="metrics",
            collected_at=datetime(2025, 1, 18, 12, 0, tzinfo=UTC),
            data={"cpu": {"usage": 50, "temp": 70}},
            metadata={},
        )
        context = data.to_context_string()
        assert "# Data Collection: metrics" in context
        assert "cpu" in context

    def test_to_context_string_with_list(self):
        """Test context string with list data (truncated to 10)."""
        data = CollectedData(
            source="events",
            collected_at=datetime.now(UTC),
            data={"items": list(range(15))},
            metadata={},
        )
        context = data.to_context_string()
        assert "items" in context
        assert "... and 5 more" in context


class TestJSONCollector:
    """Tests for JSONCollector."""

    def test_collect_from_config(self):
        """Test collecting data from config."""
        collector = JSONCollector()
        result = collector.collect(
            context={},
            config={"data": {"test": "data"}, "source_name": "my_source"},
        )
        assert result.source == "my_source"
        assert result.data == {"test": "data"}
        assert result.metadata["type"] == "json_passthrough"

    def test_collect_from_context(self):
        """Test collecting data from context when no config data."""
        collector = JSONCollector()
        result = collector.collect(
            context={"from_context": True},
            config={},
        )
        assert result.source == "json_input"
        assert result.data == {"from_context": True}

    def test_collect_empty_context(self):
        """Test with empty non-dict context."""
        collector = JSONCollector()
        result = collector.collect(context="not a dict", config={})
        assert result.data == {}


class TestAggregateCollector:
    """Tests for AggregateCollector."""

    def test_aggregate_multiple_collectors(self):
        """Test aggregating from multiple collectors."""
        collector1 = JSONCollector()
        collector2 = JSONCollector()

        aggregate = AggregateCollector([collector1, collector2])
        result = aggregate.collect(
            context={},
            config={
                "collector_configs": {
                    0: {"data": {"a": 1}, "source_name": "source_a"},
                    1: {"data": {"b": 2}, "source_name": "source_b"},
                }
            },
        )

        assert result.source == "aggregate"
        assert result.metadata["collector_count"] == 2
        assert "source_a" in result.data
        assert "source_b" in result.data


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_basic_creation(self):
        """Test basic AnalysisResult creation."""
        result = AnalysisResult(
            analyzer="test",
            analyzed_at=datetime.now(UTC),
            findings=[{"severity": "warning", "message": "Test finding"}],
            metrics={"count": 10},
            recommendations=["Fix the issue"],
            severity="warning",
        )
        assert result.analyzer == "test"
        assert len(result.findings) == 1
        assert result.severity == "warning"

    def test_to_context_string(self):
        """Test context string generation."""
        result = AnalysisResult(
            analyzer="test_analyzer",
            analyzed_at=datetime(2025, 1, 18, 12, 0, tzinfo=UTC),
            findings=[{"severity": "critical", "message": "Critical issue"}],
            metrics={"errors": 5},
            recommendations=["Fix immediately"],
            severity="critical",
        )
        context = result.to_context_string()
        assert "# Analysis Results: test_analyzer" in context
        assert "CRITICAL" in context
        assert "Critical issue" in context
        assert "errors: 5" in context
        assert "Fix immediately" in context


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer."""

    def test_analyze_with_slow_operation(self):
        """Test detection of slow operations."""
        analyzer = TrendAnalyzer()
        data = {
            "metrics": {"timing": {"slow_query": {"avg_ms": 2000, "max_ms": 5000, "count": 10}}}
        }
        result = analyzer.analyze(data, {})
        assert result.severity == "warning"
        assert any("slow" in f["message"].lower() for f in result.findings)
        assert len(result.recommendations) > 0

    def test_analyze_with_high_variance(self):
        """Test detection of high variance."""
        analyzer = TrendAnalyzer()
        data = {"metrics": {"timing": {"variable_op": {"avg_ms": 100, "max_ms": 500, "count": 10}}}}
        result = analyzer.analyze(data, {})
        assert any("variance" in f.get("category", "") for f in result.findings)

    def test_analyze_with_error_counters(self):
        """Test detection of error counters."""
        analyzer = TrendAnalyzer()
        data = {
            "metrics": {
                "counters": {"request_errors": 5},
                "timing": {},
            }
        }
        result = analyzer.analyze(data, {})
        assert result.severity == "warning"
        assert any("error" in f["message"].lower() for f in result.findings)

    def test_analyze_healthy_system(self):
        """Test analysis of healthy metrics."""
        analyzer = TrendAnalyzer()
        data = {"metrics": {"timing": {"fast_op": {"avg_ms": 10, "max_ms": 20, "count": 100}}}}
        result = analyzer.analyze(data, {})
        assert result.severity == "info"

    def test_analyze_with_collected_data_object(self):
        """Test with CollectedData input."""
        analyzer = TrendAnalyzer()
        collected = CollectedData(
            source="test",
            collected_at=datetime.now(UTC),
            data={"metrics": {"timing": {}}},
            metadata={},
        )
        result = analyzer.analyze(collected, {})
        assert result.analyzer == "trends"


class TestThresholdAnalyzer:
    """Tests for ThresholdAnalyzer."""

    def test_above_critical_threshold(self):
        """Test detection of critical threshold breach (above)."""
        analyzer = ThresholdAnalyzer()
        data = {"cpu.usage": 95}
        config = {
            "thresholds": {"cpu.usage": {"warning": 80, "critical": 90, "direction": "above"}}
        }
        result = analyzer.analyze(data, config)
        assert result.severity == "critical"
        assert any("critical" in f["severity"] for f in result.findings)

    def test_above_warning_threshold(self):
        """Test detection of warning threshold breach (above)."""
        analyzer = ThresholdAnalyzer()
        data = {"cpu.usage": 85}
        config = {
            "thresholds": {"cpu.usage": {"warning": 80, "critical": 90, "direction": "above"}}
        }
        result = analyzer.analyze(data, config)
        assert result.severity == "warning"

    def test_below_threshold(self):
        """Test detection of threshold breach (below)."""
        analyzer = ThresholdAnalyzer()
        data = {"disk.free_gb": 5}
        config = {
            "thresholds": {"disk.free_gb": {"warning": 20, "critical": 10, "direction": "below"}}
        }
        result = analyzer.analyze(data, config)
        assert result.severity == "critical"

    def test_within_thresholds(self):
        """Test when metrics are within acceptable thresholds."""
        analyzer = ThresholdAnalyzer()
        data = {"cpu.usage": 50}
        config = {
            "thresholds": {"cpu.usage": {"warning": 80, "critical": 90, "direction": "above"}}
        }
        result = analyzer.analyze(data, config)
        assert result.severity == "info"
        assert any("within acceptable" in f["message"] for f in result.findings)

    def test_nested_path_access(self):
        """Test nested metric path access."""
        analyzer = ThresholdAnalyzer()
        data = {"system": {"memory": {"percent": 95}}}
        config = {
            "thresholds": {
                "system.memory.percent": {
                    "warning": 80,
                    "critical": 90,
                    "direction": "above",
                }
            }
        }
        result = analyzer.analyze(data, config)
        assert result.severity == "critical"


class TestCompositeAnalyzer:
    """Tests for CompositeAnalyzer."""

    def test_combines_results(self):
        """Test combining results from multiple analyzers."""
        trend_analyzer = TrendAnalyzer()
        threshold_analyzer = ThresholdAnalyzer()

        composite = CompositeAnalyzer([trend_analyzer, threshold_analyzer])
        data = {
            "metrics": {"timing": {"op": {"avg_ms": 2000, "max_ms": 3000, "count": 10}}},
            "cpu.usage": 95,
        }
        config = {
            "analyzer_configs": {
                1: {
                    "thresholds": {
                        "cpu.usage": {
                            "warning": 80,
                            "critical": 90,
                            "direction": "above",
                        }
                    }
                }
            }
        }
        result = composite.analyze(data, config)

        assert result.analyzer == "composite"
        assert result.metadata["analyzer_count"] == 2
        assert result.severity == "critical"  # Highest severity
        assert len(result.findings) >= 2

    def test_deduplicates_recommendations(self):
        """Test that recommendations are deduplicated."""
        analyzer1 = MagicMock()
        analyzer1.analyze.return_value = AnalysisResult(
            analyzer="a1",
            analyzed_at=datetime.now(UTC),
            findings=[],
            metrics={},
            recommendations=["Fix A", "Fix B"],
            severity="info",
        )
        analyzer2 = MagicMock()
        analyzer2.analyze.return_value = AnalysisResult(
            analyzer="a2",
            analyzed_at=datetime.now(UTC),
            findings=[],
            metrics={},
            recommendations=["Fix A", "Fix C"],  # "Fix A" is duplicate
            severity="info",
        )

        composite = CompositeAnalyzer([analyzer1, analyzer2])
        result = composite.analyze({}, {})

        assert result.recommendations == ["Fix A", "Fix B", "Fix C"]


class TestReportSection:
    """Tests for ReportSection dataclass."""

    def test_creation(self):
        """Test ReportSection creation."""
        section = ReportSection(title="Test", content="Content", priority=50)
        assert section.title == "Test"
        assert section.priority == 50


class TestGeneratedReport:
    """Tests for GeneratedReport dataclass."""

    def test_to_context_string(self):
        """Test context string generation."""
        report = GeneratedReport(
            reporter="test",
            generated_at=datetime.now(UTC),
            title="Test Report",
            sections=[
                ReportSection("High Priority", "Important content", 100),
                ReportSection("Low Priority", "Other content", 10),
            ],
            summary="Test summary",
            format="markdown",
        )
        context = report.to_context_string()
        assert "Test Report" in context
        assert "Test summary" in context
        # High priority should come first
        high_idx = context.find("High Priority")
        low_idx = context.find("Low Priority")
        assert high_idx < low_idx

    def test_to_markdown(self):
        """Test markdown export."""
        report = GeneratedReport(
            reporter="test",
            generated_at=datetime(2025, 1, 18, 12, 0, tzinfo=UTC),
            title="Monthly Report",
            sections=[ReportSection("Section 1", "Content here", 50)],
            summary="Executive summary",
            format="markdown",
        )
        md = report.to_markdown()
        assert "# Monthly Report" in md
        assert "Executive Summary" in md
        assert "2025-01-18" in md


class TestAlert:
    """Tests for Alert dataclass."""

    def test_to_dict(self):
        """Test converting alert to dict."""
        alert = Alert(
            alert_id="ALT-001",
            severity="critical",
            title="High CPU",
            message="CPU usage above 90%",
            source="monitoring",
            timestamp=datetime(2025, 1, 18, 12, 0, tzinfo=UTC),
            data={"cpu": 95},
        )
        d = alert.to_dict()
        assert d["alert_id"] == "ALT-001"
        assert d["severity"] == "critical"
        assert d["acknowledged"] is False
        assert "2025-01-18" in d["timestamp"]


class TestAlertBatch:
    """Tests for AlertBatch dataclass."""

    def test_to_context_string(self):
        """Test context string with alerts by severity."""
        batch = AlertBatch(
            generator="test",
            generated_at=datetime.now(UTC),
            alerts=[
                Alert(
                    "ALT-001",
                    "critical",
                    "Critical",
                    "msg",
                    "src",
                    datetime.now(UTC),
                ),
                Alert(
                    "ALT-002",
                    "warning",
                    "Warning",
                    "msg",
                    "src",
                    datetime.now(UTC),
                ),
            ],
            summary="2 alerts",
        )
        context = batch.to_context_string()
        assert "Total Alerts: 2" in context
        assert "CRITICAL" in context
        assert "WARNING" in context


class TestReportGenerator:
    """Tests for ReportGenerator."""

    def test_generate_from_analysis_result(self):
        """Test generating report from AnalysisResult."""
        generator = ReportGenerator()
        analysis = AnalysisResult(
            analyzer="test",
            analyzed_at=datetime.now(UTC),
            findings=[
                {
                    "severity": "critical",
                    "category": "performance",
                    "message": "Slow response",
                },
                {"severity": "warning", "category": "memory", "message": "High memory"},
            ],
            metrics={"response_time": 2500},
            recommendations=["Optimize queries"],
            severity="critical",
        )
        report = generator.generate(analysis, {"title": "Performance Report"})

        assert report.title == "Performance Report"
        assert report.reporter == "report_generator"
        assert "critical" in report.summary.lower()
        assert any("Status" in s.title for s in report.sections)
        assert any("Findings" in s.title for s in report.sections)

    def test_generate_from_dict(self):
        """Test generating report from dict data."""
        generator = ReportGenerator()
        data = {
            "findings": [{"severity": "info", "message": "All good"}],
            "metrics": {"uptime": 99.9},
            "recommendations": [],
            "severity": "info",
        }
        report = generator.generate(data, {})
        assert report.summary == "Status: INFO."

    def test_generate_without_recommendations(self):
        """Test report without recommendations section."""
        generator = ReportGenerator()
        report = generator.generate({}, {"include_recommendations": False})
        assert not any("Recommended" in s.title for s in report.sections)

    def test_format_metrics_table(self):
        """Test metrics formatting."""
        generator = ReportGenerator()
        data = {"metrics": {"cpu_usage": 45.5, "memory_used": 1024}}
        report = generator.generate(data, {})
        metrics_section = next((s for s in report.sections if "Metrics" in s.title), None)
        assert metrics_section is not None
        assert "45.50" in metrics_section.content  # Float formatting


class TestAlertGenerator:
    """Tests for AlertGenerator."""

    def test_generate_alerts_from_findings(self):
        """Test generating alerts from findings."""
        generator = AlertGenerator()
        analysis = AnalysisResult(
            analyzer="test",
            analyzed_at=datetime.now(UTC),
            findings=[
                {"severity": "critical", "category": "cpu", "message": "CPU critical"},
                {
                    "severity": "warning",
                    "category": "memory",
                    "message": "Memory warning",
                },
                {"severity": "info", "category": "disk", "message": "Disk OK"},
            ],
            metrics={},
            recommendations=[],
            severity="critical",
        )
        batch = generator.generate(analysis, {"min_severity": "warning"})

        # Should only include critical and warning, not info
        assert len(batch.alerts) == 2
        assert any(a.severity == "critical" for a in batch.alerts)
        assert any(a.severity == "warning" for a in batch.alerts)

    def test_generate_no_alerts(self):
        """Test when no findings meet threshold."""
        generator = AlertGenerator()
        batch = generator.generate(
            {"findings": [{"severity": "info", "message": "OK"}]},
            {"min_severity": "critical"},
        )
        assert len(batch.alerts) == 0
        assert "No alerts" in batch.summary

    def test_alert_counter_increments(self):
        """Test that alert IDs increment."""
        generator = AlertGenerator()
        batch1 = generator.generate({"findings": [{"severity": "warning", "message": "A"}]}, {})
        batch2 = generator.generate({"findings": [{"severity": "warning", "message": "B"}]}, {})
        assert batch1.alerts[0].alert_id != batch2.alerts[0].alert_id


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_creation(self):
        """Test StageResult creation."""
        result = StageResult(
            stage=PipelineStage.ANALYZE,
            status="success",
            output={"data": "test"},
            duration_ms=100.5,
        )
        assert result.stage == PipelineStage.ANALYZE
        assert result.status == "success"
        assert result.error is None


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dict."""
        result = PipelineResult(
            pipeline_id="test_pipeline",
            status="completed",
            started_at=datetime(2025, 1, 18, 12, 0, tzinfo=UTC),
            completed_at=datetime(2025, 1, 18, 12, 5, tzinfo=UTC),
            stages=[
                StageResult(PipelineStage.COLLECT, "success", {}, 100.0),
                StageResult(PipelineStage.ANALYZE, "success", {}, 200.0),
            ],
        )
        d = result.to_dict()
        assert d["pipeline_id"] == "test_pipeline"
        assert d["status"] == "completed"
        assert len(d["stages"]) == 2
        assert d["stages"][0]["stage"] == "collect"


class TestAnalyticsPipeline:
    """Tests for AnalyticsPipeline."""

    def test_add_stage(self):
        """Test adding stages with method chaining."""
        pipeline = AnalyticsPipeline("test", use_agents=False)

        def handler(data, config):
            return data

        result = pipeline.add_stage(PipelineStage.COLLECT, handler)
        assert result is pipeline  # Method chaining
        assert len(pipeline._stages) == 1

    def test_execute_simple_pipeline(self):
        """Test executing a simple pipeline."""
        pipeline = AnalyticsPipeline("test", use_agents=False)

        def collect(data, config):
            return {"collected": True}

        def analyze(data, config):
            return {"analyzed": True, "input": data}

        pipeline.add_stage(PipelineStage.COLLECT, collect)
        pipeline.add_stage(PipelineStage.ANALYZE, analyze)

        result = pipeline.execute({})
        assert result.status == "completed"
        assert len(result.stages) == 2
        assert result.stages[0].status == "success"
        assert result.stages[1].status == "success"
        assert result.final_output["analyzed"] is True

    def test_execute_with_failure(self):
        """Test pipeline execution with stage failure."""
        pipeline = AnalyticsPipeline("test", use_agents=False)

        def failing_handler(data, config):
            raise ValueError("Test error")

        pipeline.add_stage(PipelineStage.COLLECT, failing_handler)
        pipeline.add_stage(PipelineStage.ANALYZE, lambda d, c: d)

        result = pipeline.execute({})
        assert result.status == "failed"
        assert len(result.stages) == 1  # Second stage not executed
        assert result.stages[0].status == "failed"
        assert "Test error" in result.stages[0].error
        assert len(result.errors) > 0

    def test_execute_passes_context(self):
        """Test that context is passed between stages."""
        pipeline = AnalyticsPipeline("test", use_agents=False)
        context_log = []

        def stage1(data, config):
            context_log.append(("stage1", config))
            return "stage1_output"

        def stage2(data, config):
            context_log.append(("stage2", config, data))
            return "stage2_output"

        pipeline.add_stage(PipelineStage.COLLECT, stage1, {"custom": "config1"})
        pipeline.add_stage(PipelineStage.ANALYZE, stage2)

        result = pipeline.execute({"initial": "context"})

        assert result.status == "completed"
        # Stage 1 should have initial context + custom config
        assert context_log[0][1]["custom"] == "config1"
        # Stage 2 should receive stage 1 output
        assert context_log[1][2] == "stage1_output"

    @patch("animus_forge.api_clients.ClaudeCodeClient")
    def test_add_agent_stage(self, mock_client_class):
        """Test adding an agent-powered stage."""
        mock_client = MagicMock()
        mock_client.execute_agent.return_value = {
            "success": True,
            "output": "AI generated analysis",
        }
        mock_client_class.return_value = mock_client

        pipeline = AnalyticsPipeline("test", use_agents=True)
        pipeline.add_agent_stage(
            PipelineStage.ANALYZE,
            "analyst",
            "Analyze this: {{context}}",
        )

        result = pipeline.execute({"data": "test"})

        assert result.status == "completed"
        mock_client.execute_agent.assert_called_once()

    def test_add_agent_stage_without_agents_enabled(self):
        """Test that agent stage fails without agents enabled."""
        pipeline = AnalyticsPipeline("test", use_agents=False)

        with pytest.raises(ValueError, match="not configured"):
            pipeline.add_agent_stage(PipelineStage.ANALYZE, "analyst", "task")


class TestPipelineBuilder:
    """Tests for PipelineBuilder factory methods."""

    def test_trend_analysis_pipeline(self):
        """Test creating trend analysis pipeline."""
        pipeline = PipelineBuilder.trend_analysis_pipeline()
        assert pipeline.pipeline_id == "trend_analysis"
        assert len(pipeline._stages) == 3

    def test_threshold_alert_pipeline(self):
        """Test creating threshold alert pipeline."""
        pipeline = PipelineBuilder.threshold_alert_pipeline()
        assert pipeline.pipeline_id == "threshold_alerts"
        assert len(pipeline._stages) == 3

    @patch("animus_forge.api_clients.ClaudeCodeClient")
    def test_full_analysis_pipeline(self, mock_client_class):
        """Test creating full analysis pipeline with agents."""
        mock_client_class.return_value = MagicMock()
        pipeline = PipelineBuilder.full_analysis_pipeline()
        assert pipeline.pipeline_id == "full_analysis"
        assert len(pipeline._stages) == 4  # collect, analyze, visualize (agent), report


class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_stage_values(self):
        """Test that all expected stages exist."""
        assert PipelineStage.COLLECT.value == "collect"
        assert PipelineStage.CLEAN.value == "clean"
        assert PipelineStage.ANALYZE.value == "analyze"
        assert PipelineStage.VISUALIZE.value == "visualize"
        assert PipelineStage.REPORT.value == "report"
        assert PipelineStage.ALERT.value == "alert"
