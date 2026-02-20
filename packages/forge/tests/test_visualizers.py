"""Tests for analytics visualizers module."""

import sys
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, "src")

from animus_forge.analytics.visualizers import (
    ChartGenerator,
    ChartSpec,
    DashboardBuilder,
    DashboardSpec,
    VisualizationResult,
)


class TestChartSpec:
    """Tests for ChartSpec dataclass."""

    def test_create_chart_spec(self):
        """Can create ChartSpec with required fields."""
        spec = ChartSpec(
            chart_type="bar",
            title="Test Chart",
            data={"a": 1, "b": 2},
        )
        assert spec.chart_type == "bar"
        assert spec.title == "Test Chart"
        assert spec.data == {"a": 1, "b": 2}

    def test_default_config_and_code(self):
        """Default config is empty dict and code is empty string."""
        spec = ChartSpec(chart_type="line", title="Test", data={})
        assert spec.config == {}
        assert spec.code == ""

    def test_custom_config(self):
        """Can set custom config."""
        spec = ChartSpec(
            chart_type="bar",
            title="Test",
            data={},
            config={"color": "blue", "size": 100},
        )
        assert spec.config["color"] == "blue"
        assert spec.config["size"] == 100

    def test_to_dict(self):
        """to_dict returns correct structure."""
        spec = ChartSpec(
            chart_type="gauge",
            title="Status",
            data={"value": 75},
            config={"max": 100},
        )
        result = spec.to_dict()

        assert result["type"] == "gauge"
        assert result["title"] == "Status"
        assert result["data"] == {"value": 75}
        assert result["config"] == {"max": 100}

    def test_to_dict_excludes_code(self):
        """to_dict does not include code field."""
        spec = ChartSpec(
            chart_type="bar",
            title="Test",
            data={},
            code="st.bar_chart(data)",
        )
        result = spec.to_dict()
        assert "code" not in result


class TestDashboardSpec:
    """Tests for DashboardSpec dataclass."""

    def test_create_dashboard_spec(self):
        """Can create DashboardSpec with required fields."""
        chart = ChartSpec(chart_type="bar", title="Chart 1", data={})
        spec = DashboardSpec(
            title="My Dashboard",
            charts=[chart],
            layout="grid",
        )
        assert spec.title == "My Dashboard"
        assert len(spec.charts) == 1
        assert spec.layout == "grid"

    def test_default_refresh_interval(self):
        """Default refresh interval is 0 (no auto-refresh)."""
        spec = DashboardSpec(title="Test", charts=[], layout="vertical")
        assert spec.refresh_interval == 0

    def test_custom_refresh_interval(self):
        """Can set custom refresh interval."""
        spec = DashboardSpec(
            title="Test",
            charts=[],
            layout="vertical",
            refresh_interval=300,
        )
        assert spec.refresh_interval == 300

    def test_multiple_charts(self):
        """Can include multiple charts."""
        charts = [
            ChartSpec(chart_type="bar", title="Chart 1", data={}),
            ChartSpec(chart_type="line", title="Chart 2", data={}),
            ChartSpec(chart_type="pie", title="Chart 3", data={}),
        ]
        spec = DashboardSpec(title="Multi", charts=charts, layout="grid")
        assert len(spec.charts) == 3


class TestVisualizationResult:
    """Tests for VisualizationResult dataclass."""

    def test_create_result(self):
        """Can create VisualizationResult with required fields."""
        result = VisualizationResult(
            visualizer="test",
            generated_at=datetime.now(UTC),
            charts=[],
            dashboard=None,
            streamlit_code="",
        )
        assert result.visualizer == "test"
        assert result.charts == []
        assert result.dashboard is None

    def test_with_charts(self):
        """Can include charts in result."""
        charts = [
            ChartSpec(chart_type="bar", title="Test", data={"a": 1}),
        ]
        result = VisualizationResult(
            visualizer="chart_gen",
            generated_at=datetime.now(UTC),
            charts=charts,
            dashboard=None,
            streamlit_code="st.bar_chart()",
        )
        assert len(result.charts) == 1
        assert result.charts[0].title == "Test"

    def test_with_dashboard(self):
        """Can include dashboard in result."""
        chart = ChartSpec(chart_type="bar", title="Test", data={})
        dashboard = DashboardSpec(title="Dashboard", charts=[chart], layout="vertical")

        result = VisualizationResult(
            visualizer="dashboard",
            generated_at=datetime.now(UTC),
            charts=[chart],
            dashboard=dashboard,
            streamlit_code="",
        )
        assert result.dashboard is not None
        assert result.dashboard.title == "Dashboard"

    def test_to_context_string_basic(self):
        """to_context_string generates markdown output."""
        result = VisualizationResult(
            visualizer="test_viz",
            generated_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            charts=[],
            dashboard=None,
            streamlit_code="import streamlit",
        )
        context = result.to_context_string()

        assert "# Visualization Output: test_viz" in context
        assert "Charts: 0" in context
        assert "## Streamlit Code" in context
        assert "import streamlit" in context

    def test_to_context_string_with_charts(self):
        """to_context_string includes chart details."""
        charts = [
            ChartSpec(chart_type="bar", title="Bar Chart", data={}),
            ChartSpec(chart_type="line", title="Line Chart", data={}),
        ]
        result = VisualizationResult(
            visualizer="test",
            generated_at=datetime.now(UTC),
            charts=charts,
            dashboard=None,
            streamlit_code="",
        )
        context = result.to_context_string()

        assert "## Chart 1: Bar Chart" in context
        assert "Type: bar" in context
        assert "## Chart 2: Line Chart" in context
        assert "Type: line" in context

    def test_to_context_string_with_dashboard(self):
        """to_context_string includes dashboard details."""
        chart = ChartSpec(chart_type="bar", title="Test", data={})
        dashboard = DashboardSpec(title="My Dashboard", charts=[chart], layout="grid")

        result = VisualizationResult(
            visualizer="test",
            generated_at=datetime.now(UTC),
            charts=[chart],
            dashboard=dashboard,
            streamlit_code="",
        )
        context = result.to_context_string()

        assert "## Dashboard: My Dashboard" in context
        assert "Layout: grid" in context


class TestChartGenerator:
    """Tests for ChartGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create ChartGenerator instance."""
        return ChartGenerator()

    def test_generate_from_dict(self, generator):
        """Generate charts from dict data."""
        data = {
            "metrics": {"count": 100, "rate": 0.95},
            "severity": "info",
        }
        result = generator.generate(data, {})

        assert isinstance(result, VisualizationResult)
        assert result.visualizer == "chart_generator"
        assert len(result.charts) > 0

    def test_generate_creates_gauge_chart(self, generator):
        """Generates gauge chart for severity."""
        data = {"severity": "warning"}
        result = generator.generate(data, {})

        gauge_charts = [c for c in result.charts if c.chart_type == "gauge"]
        assert len(gauge_charts) == 1
        assert gauge_charts[0].title == "Operations Status"
        assert gauge_charts[0].data["label"] == "WARNING"

    def test_severity_values(self, generator):
        """Severity maps to correct gauge values."""
        for severity, expected_value in [
            ("info", 100),
            ("warning", 60),
            ("critical", 20),
        ]:
            data = {"severity": severity}
            result = generator.generate(data, {})
            gauge = [c for c in result.charts if c.chart_type == "gauge"][0]
            assert gauge.data["value"] == expected_value

    def test_generate_creates_bar_chart_for_metrics(self, generator):
        """Generates bar chart for numeric metrics."""
        data = {"metrics": {"requests": 100, "errors": 5, "latency_ms": 250}}
        result = generator.generate(data, {})

        bar_charts = [c for c in result.charts if c.chart_type == "bar"]
        assert len(bar_charts) == 1
        assert bar_charts[0].title == "Key Metrics"
        assert bar_charts[0].data["requests"] == 100

    def test_generate_filters_non_numeric_metrics(self, generator):
        """Only includes numeric values in bar chart."""
        data = {"metrics": {"count": 100, "status": "ok", "rate": 0.5}}
        result = generator.generate(data, {})

        bar_charts = [c for c in result.charts if c.chart_type == "bar"]
        assert len(bar_charts) == 1
        assert "count" in bar_charts[0].data
        assert "rate" in bar_charts[0].data
        assert "status" not in bar_charts[0].data

    def test_generate_creates_table_for_findings(self, generator):
        """Generates table chart for findings."""
        data = {
            "findings": [
                {"severity": "warning", "message": "High latency detected"},
                {"severity": "info", "message": "System running normally"},
            ]
        }
        result = generator.generate(data, {})

        table_charts = [c for c in result.charts if c.chart_type == "table"]
        assert len(table_charts) == 1
        assert table_charts[0].title == "Findings Summary"
        assert len(table_charts[0].data["rows"]) == 2

    def test_generate_includes_streamlit_code(self, generator):
        """Generates Streamlit code when include_code is True."""
        data = {"metrics": {"count": 100}}
        result = generator.generate(data, {"include_code": True})

        assert "import streamlit as st" in result.streamlit_code
        assert "import pandas as pd" in result.streamlit_code

    def test_generate_excludes_code_when_disabled(self, generator):
        """Does not generate code when include_code is False."""
        data = {"metrics": {"count": 100}}
        result = generator.generate(data, {"include_code": False})

        assert result.streamlit_code == ""

    def test_generate_from_analysis_result_object(self, generator):
        """Generate from object with findings/metrics/severity attributes."""
        mock_data = MagicMock()
        mock_data.findings = [{"severity": "warning", "message": "Test"}]
        mock_data.metrics = {"count": 50}
        mock_data.severity = "warning"

        result = generator.generate(mock_data, {})

        assert len(result.charts) >= 2  # gauge + bar or table

    def test_generate_from_collected_data_object(self, generator):
        """Generate from object with data attribute."""
        mock_data = MagicMock()
        mock_data.findings = None
        mock_data.data = {"metrics": {"value": 42}}
        del mock_data.findings  # Remove findings attr

        result = generator.generate(mock_data, {})
        assert len(result.charts) >= 1

    def test_severity_color_mapping(self, generator):
        """Severity maps to correct colors."""
        assert generator._severity_color("info") == "#00d4ff"
        assert generator._severity_color("warning") == "#ffaa00"
        assert generator._severity_color("critical") == "#ff4444"
        assert generator._severity_color("unknown") == "#888888"

    def test_streamlit_code_includes_metrics(self, generator):
        """Streamlit code includes metric display."""
        data = {"metrics": {"requests": 100, "errors": 5}}
        result = generator.generate(data, {"include_code": True})

        assert "st.columns" in result.streamlit_code
        assert "metric" in result.streamlit_code

    def test_streamlit_code_includes_findings_table(self, generator):
        """Streamlit code includes findings dataframe."""
        data = {"findings": [{"severity": "info", "message": "Test finding"}]}
        result = generator.generate(data, {"include_code": True})

        assert "findings_data" in result.streamlit_code
        assert "st.dataframe" in result.streamlit_code


class TestDashboardBuilder:
    """Tests for DashboardBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create DashboardBuilder instance."""
        return DashboardBuilder()

    def test_build_basic_dashboard(self, builder):
        """Build creates dashboard with default settings."""
        data = {"metrics": {"value": 100}}
        result = builder.build(data, {})

        assert isinstance(result, VisualizationResult)
        assert result.visualizer == "dashboard_builder"
        assert result.dashboard is not None
        assert result.dashboard.title == "Analytics Dashboard"

    def test_build_with_custom_title(self, builder):
        """Build uses custom title from config."""
        data = {"metrics": {"value": 100}}
        result = builder.build(data, {"title": "Custom Dashboard"})

        assert result.dashboard.title == "Custom Dashboard"

    def test_build_with_grid_layout(self, builder):
        """Build uses grid layout when specified."""
        data = {"metrics": {"value": 100}}
        result = builder.build(data, {"layout": "grid"})

        assert result.dashboard.layout == "grid"

    def test_build_with_refresh_interval(self, builder):
        """Build sets refresh interval from config."""
        data = {"metrics": {"value": 100}}
        result = builder.build(data, {"refresh_interval": 300})

        assert result.dashboard.refresh_interval == 300

    def test_build_generates_charts(self, builder):
        """Build generates charts from data."""
        data = {
            "metrics": {"requests": 100, "errors": 5},
            "severity": "info",
        }
        result = builder.build(data, {})

        assert len(result.charts) > 0
        assert len(result.dashboard.charts) > 0

    def test_build_generates_streamlit_code(self, builder):
        """Build generates complete Streamlit dashboard code."""
        data = {"metrics": {"count": 100}}
        result = builder.build(data, {"title": "Test Dashboard"})

        assert "import streamlit as st" in result.streamlit_code
        assert 'st.title("Test Dashboard")' in result.streamlit_code
        assert "st.set_page_config" in result.streamlit_code

    def test_build_includes_auto_refresh_meta(self, builder):
        """Dashboard code includes auto-refresh meta tag when configured."""
        data = {"metrics": {"count": 100}}
        result = builder.build(data, {"refresh_interval": 60})

        assert 'content="60"' in result.streamlit_code
        assert "http-equiv" in result.streamlit_code

    def test_build_no_refresh_when_zero(self, builder):
        """Dashboard code excludes refresh when interval is 0."""
        data = {"metrics": {"count": 100}}
        result = builder.build(data, {"refresh_interval": 0})

        assert "http-equiv" not in result.streamlit_code

    def test_build_with_findings(self, builder):
        """Build includes findings in dashboard."""
        data = {
            "findings": [{"severity": "warning", "message": "Test warning"}],
        }
        result = builder.build(data, {})

        # Should have table chart for findings
        table_charts = [c for c in result.charts if c.chart_type == "table"]
        assert len(table_charts) >= 1

    def test_dashboard_inherits_charts_from_generator(self, builder):
        """Dashboard charts come from ChartGenerator."""
        data = {
            "metrics": {"a": 1, "b": 2},
            "severity": "warning",
        }
        result = builder.build(data, {})

        # Should have gauge from severity
        gauge_charts = [c for c in result.dashboard.charts if c.chart_type == "gauge"]
        assert len(gauge_charts) == 1
