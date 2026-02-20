"""Visualization Components for Analytics Pipelines.

Provides chart generation and dashboard building capabilities.
Note: These generate visualization specifications/code, not actual images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ChartSpec:
    """Specification for a chart/visualization."""

    chart_type: str  # "bar", "line", "pie", "gauge", "table"
    title: str
    data: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)
    code: str = ""  # Generated code for the chart

    def to_dict(self) -> dict:
        return {
            "type": self.chart_type,
            "title": self.title,
            "data": self.data,
            "config": self.config,
        }


@dataclass
class DashboardSpec:
    """Specification for a dashboard layout."""

    title: str
    charts: list[ChartSpec]
    layout: str  # "grid", "vertical", "tabs"
    refresh_interval: int = 0  # seconds, 0 = no auto-refresh
    code: str = ""  # Generated dashboard code


@dataclass
class VisualizationResult:
    """Container for visualization outputs."""

    visualizer: str
    generated_at: datetime
    charts: list[ChartSpec]
    dashboard: DashboardSpec | None
    streamlit_code: str  # Ready-to-use Streamlit code
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Convert to string format for AI agent context."""
        lines = [
            f"# Visualization Output: {self.visualizer}",
            f"Generated: {self.generated_at.isoformat()}",
            f"Charts: {len(self.charts)}",
            "",
        ]

        for i, chart in enumerate(self.charts, 1):
            lines.append(f"## Chart {i}: {chart.title}")
            lines.append(f"Type: {chart.chart_type}")
            lines.append("")

        if self.dashboard:
            lines.append(f"## Dashboard: {self.dashboard.title}")
            lines.append(f"Layout: {self.dashboard.layout}")
            lines.append("")

        lines.append("## Streamlit Code")
        lines.append("```python")
        lines.append(self.streamlit_code)
        lines.append("```")

        return "\n".join(lines)


class ChartGenerator:
    """Generates chart specifications and code from analysis results."""

    def generate(self, data: Any, config: dict) -> VisualizationResult:
        """Generate charts from analysis data.

        Config options:
            chart_types: list[str] - Types of charts to generate
            include_code: bool - Whether to include Streamlit code
        """
        include_code = config.get("include_code", True)
        charts = []

        # Handle different input types
        if hasattr(data, "findings"):
            # AnalysisResult
            findings = data.findings
            metrics = data.metrics
            severity = data.severity
        elif hasattr(data, "data"):
            # CollectedData
            findings = []
            metrics = data.data.get("metrics", data.data)
            severity = "info"
        elif isinstance(data, dict):
            findings = data.get("findings", [])
            metrics = data.get("metrics", data)
            severity = data.get("severity", "info")
        else:
            findings = []
            metrics = {}
            severity = "info"

        # Generate status gauge
        if severity:
            severity_value = {"info": 100, "warning": 60, "critical": 20}.get(severity, 50)
            charts.append(
                ChartSpec(
                    chart_type="gauge",
                    title="Operations Status",
                    data={"value": severity_value, "label": severity.upper()},
                    config={"color": self._severity_color(severity)},
                )
            )

        # Generate metrics bar chart
        if metrics:
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            if numeric_metrics:
                charts.append(
                    ChartSpec(
                        chart_type="bar",
                        title="Key Metrics",
                        data=numeric_metrics,
                        config={"orientation": "horizontal"},
                    )
                )

        # Generate findings table
        if findings:
            charts.append(
                ChartSpec(
                    chart_type="table",
                    title="Findings Summary",
                    data={
                        "rows": [
                            {
                                "severity": f.get("severity", "info"),
                                "message": f.get("message", ""),
                            }
                            for f in findings
                        ]
                    },
                )
            )

        # Generate Streamlit code if requested
        streamlit_code = ""
        if include_code:
            streamlit_code = self._generate_streamlit_code(charts, metrics, findings)

        return VisualizationResult(
            visualizer="chart_generator",
            generated_at=datetime.now(UTC),
            charts=charts,
            dashboard=None,
            streamlit_code=streamlit_code,
        )

    def _severity_color(self, severity: str) -> str:
        return {
            "info": "#00d4ff",
            "warning": "#ffaa00",
            "critical": "#ff4444",
        }.get(severity, "#888888")

    def _generate_streamlit_code(
        self,
        charts: list[ChartSpec],
        metrics: dict,
        findings: list,
    ) -> str:
        """Generate Streamlit code for the visualizations."""
        lines = [
            "import streamlit as st",
            "import pandas as pd",
            "",
            "# Auto-generated visualization code",
            "",
        ]

        # Metrics display
        if metrics:
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            if numeric_metrics:
                lines.append("# Key Metrics")
                lines.append(f"cols = st.columns({min(len(numeric_metrics), 4)})")
                for i, (key, value) in enumerate(list(numeric_metrics.items())[:4]):
                    lines.append(f"cols[{i}].metric('{key.replace('_', ' ').title()}', '{value}')")
                lines.append("")

        # Findings table
        if findings:
            lines.append("# Findings")
            lines.append("findings_data = [")
            for f in findings[:10]:
                sev = f.get("severity", "info")
                msg = f.get("message", "").replace("'", "\\'")
                lines.append(f"    {{'Severity': '{sev}', 'Finding': '{msg}'}},")
            lines.append("]")
            lines.append("st.dataframe(pd.DataFrame(findings_data), use_container_width=True)")
            lines.append("")

        return "\n".join(lines)


class DashboardBuilder:
    """Builds complete dashboard specifications from multiple data sources."""

    def build(self, data: Any, config: dict) -> VisualizationResult:
        """Build a dashboard from analysis data.

        Config options:
            title: str - Dashboard title
            layout: str - "grid" or "vertical"
            refresh_interval: int - Auto-refresh in seconds
        """
        title = config.get("title", "Analytics Dashboard")
        layout = config.get("layout", "vertical")
        refresh_interval = config.get("refresh_interval", 600)

        # Use ChartGenerator for individual charts
        chart_gen = ChartGenerator()
        chart_result = chart_gen.generate(data, {"include_code": False})

        dashboard = DashboardSpec(
            title=title,
            charts=chart_result.charts,
            layout=layout,
            refresh_interval=refresh_interval,
        )

        # Generate full dashboard code
        streamlit_code = self._generate_dashboard_code(dashboard, data)

        return VisualizationResult(
            visualizer="dashboard_builder",
            generated_at=datetime.now(UTC),
            charts=chart_result.charts,
            dashboard=dashboard,
            streamlit_code=streamlit_code,
        )

    def _generate_dashboard_code(self, dashboard: DashboardSpec, data: Any) -> str:
        """Generate complete Streamlit dashboard code."""
        lines = [
            '"""',
            f"Auto-generated dashboard: {dashboard.title}",
            '"""',
            "",
            "import streamlit as st",
            "import pandas as pd",
            "from datetime import datetime",
            "",
            "st.set_page_config(",
            f'    page_title="{dashboard.title}",',
            '    layout="wide",',
            ")",
            "",
            f'st.title("{dashboard.title}")',
            "st.caption(f\"Last updated: {datetime.now().strftime('%I:%M %p')}\")",
            "",
        ]

        # Add auto-refresh if configured
        if dashboard.refresh_interval > 0:
            lines.append(f"# Auto-refresh every {dashboard.refresh_interval} seconds")
            lines.append(
                f'st.markdown(\'<meta http-equiv="refresh" content="{dashboard.refresh_interval}">\', unsafe_allow_html=True)'
            )
            lines.append("")

        # Add chart code
        chart_gen = ChartGenerator()
        chart_result = chart_gen.generate(data, {"include_code": True})
        lines.append(chart_result.streamlit_code)

        return "\n".join(lines)
