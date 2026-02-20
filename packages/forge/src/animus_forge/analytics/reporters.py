"""Report and Alert Generators for Analytics Pipelines.

Provides report generation and alerting capabilities for analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ReportSection:
    """A section within a report."""

    title: str
    content: str
    priority: int = 0  # Higher = more important


@dataclass
class GeneratedReport:
    """Container for generated report output."""

    reporter: str
    generated_at: datetime
    title: str
    sections: list[ReportSection]
    summary: str
    format: str  # "markdown", "html", "text"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Convert to string format for AI agent context."""
        lines = [
            f"# Report: {self.title}",
            f"Generated: {self.generated_at.isoformat()}",
            f"Format: {self.format}",
            "",
            "## Summary",
            self.summary,
            "",
        ]

        for section in sorted(self.sections, key=lambda s: -s.priority):
            lines.append(f"## {section.title}")
            lines.append(section.content)
            lines.append("")

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Export report as markdown."""
        lines = [
            f"# {self.title}",
            "",
            f"*Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M')}*",
            "",
            "## Executive Summary",
            "",
            self.summary,
            "",
        ]

        for section in sorted(self.sections, key=lambda s: -s.priority):
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")

        return "\n".join(lines)


@dataclass
class Alert:
    """Represents an operational alert."""

    alert_id: str
    severity: str  # "info", "warning", "critical"
    title: str
    message: str
    source: str
    timestamp: datetime
    data: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "acknowledged": self.acknowledged,
        }


@dataclass
class AlertBatch:
    """Container for multiple alerts."""

    generator: str
    generated_at: datetime
    alerts: list[Alert]
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Convert to string format for AI agent context."""
        lines = [
            f"# Alert Batch: {self.generator}",
            f"Generated: {self.generated_at.isoformat()}",
            f"Total Alerts: {len(self.alerts)}",
            "",
            "## Summary",
            self.summary,
            "",
            "## Alerts by Severity",
        ]

        for severity in ["critical", "warning", "info"]:
            severity_alerts = [a for a in self.alerts if a.severity == severity]
            if severity_alerts:
                lines.append(f"\n### {severity.upper()} ({len(severity_alerts)})")
                for alert in severity_alerts:
                    lines.append(f"- **{alert.title}**: {alert.message}")

        return "\n".join(lines)


class ReportGenerator:
    """Generates reports from analysis results."""

    def generate(self, data: Any, config: dict) -> GeneratedReport:
        """Generate a report from analysis data.

        Config options:
            title: str - Report title
            format: str - Output format ("markdown", "html", "text")
            include_recommendations: bool - Include action items
        """
        title = config.get("title", "Analytics Report")
        output_format = config.get("format", "markdown")
        include_recommendations = config.get("include_recommendations", True)

        sections = []
        summary_parts = []

        # Handle different input types
        if hasattr(data, "findings"):
            # AnalysisResult
            findings = data.findings
            metrics = data.metrics
            recommendations = data.recommendations
            severity = data.severity
        elif hasattr(data, "charts"):
            # VisualizationResult
            findings = []
            metrics = {}
            recommendations = []
            severity = "info"
        elif isinstance(data, dict):
            findings = data.get("findings", [])
            metrics = data.get("metrics", {})
            recommendations = data.get("recommendations", [])
            severity = data.get("severity", "info")
        else:
            findings = []
            metrics = {}
            recommendations = []
            severity = "info"

        # Status section
        status_content = self._format_status(severity, findings)
        sections.append(
            ReportSection(
                title="Operations Status",
                content=status_content,
                priority=100,
            )
        )
        summary_parts.append(f"Status: {severity.upper()}")

        # Findings section
        if findings:
            findings_content = self._format_findings(findings)
            sections.append(
                ReportSection(
                    title="Key Findings",
                    content=findings_content,
                    priority=90,
                )
            )

            critical_count = sum(1 for f in findings if f.get("severity") == "critical")
            warning_count = sum(1 for f in findings if f.get("severity") == "warning")
            if critical_count:
                summary_parts.append(f"{critical_count} critical issues")
            if warning_count:
                summary_parts.append(f"{warning_count} warnings")

        # Metrics section
        if metrics:
            metrics_content = self._format_metrics(metrics)
            sections.append(
                ReportSection(
                    title="Metrics Summary",
                    content=metrics_content,
                    priority=80,
                )
            )

        # Recommendations section
        if include_recommendations and recommendations:
            rec_content = self._format_recommendations(recommendations)
            sections.append(
                ReportSection(
                    title="Recommended Actions",
                    content=rec_content,
                    priority=70,
                )
            )
            summary_parts.append(f"{len(recommendations)} action items")

        summary = ". ".join(summary_parts) + "." if summary_parts else "No issues detected."

        return GeneratedReport(
            reporter="report_generator",
            generated_at=datetime.now(UTC),
            title=title,
            sections=sections,
            summary=summary,
            format=output_format,
        )

    def _format_status(self, severity: str, findings: list) -> str:
        """Format the status section."""
        status_emoji = {
            "info": "✅",
            "warning": "⚠️",
            "critical": "🚨",
        }.get(severity, "ℹ️")

        status_text = {
            "info": "Operations running normally",
            "warning": "Attention required - issues detected",
            "critical": "Critical issues require immediate action",
        }.get(severity, "Status unknown")

        lines = [
            f"{status_emoji} **{status_text}**",
            "",
        ]

        if findings:
            lines.append(f"Total findings: {len(findings)}")

        return "\n".join(lines)

    def _format_findings(self, findings: list) -> str:
        """Format findings as markdown list."""
        lines = []
        for finding in findings:
            severity = finding.get("severity", "info")
            message = finding.get("message", "")
            category = finding.get("category", "")

            severity_marker = {
                "critical": "🔴",
                "warning": "🟡",
                "info": "🟢",
            }.get(severity, "⚪")

            if category:
                lines.append(f"- {severity_marker} **[{category}]** {message}")
            else:
                lines.append(f"- {severity_marker} {message}")

        return "\n".join(lines)

    def _format_metrics(self, metrics: dict) -> str:
        """Format metrics as markdown table."""
        lines = ["| Metric | Value |", "|--------|-------|"]
        for key, value in metrics.items():
            formatted_key = key.replace("_", " ").title()
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            lines.append(f"| {formatted_key} | {formatted_value} |")
        return "\n".join(lines)

    def _format_recommendations(self, recommendations: list) -> str:
        """Format recommendations as numbered list."""
        lines = []
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)


class AlertGenerator:
    """Generates alerts from analysis results."""

    def __init__(self):
        self._alert_counter = 0

    def generate(self, data: Any, config: dict) -> AlertBatch:
        """Generate alerts from analysis data.

        Config options:
            min_severity: str - Minimum severity to alert on ("info", "warning", "critical")
            source: str - Alert source identifier
        """
        min_severity = config.get("min_severity", "warning")
        source = config.get("source", "analytics")

        severity_levels = {"info": 0, "warning": 1, "critical": 2}
        min_level = severity_levels.get(min_severity, 1)

        alerts = []

        # Handle different input types
        if hasattr(data, "findings"):
            findings = data.findings
        elif isinstance(data, dict):
            findings = data.get("findings", [])
        else:
            findings = []

        for finding in findings:
            finding_severity = finding.get("severity", "info")
            finding_level = severity_levels.get(finding_severity, 0)

            if finding_level >= min_level:
                self._alert_counter += 1
                alert = Alert(
                    alert_id=f"ALT-{self._alert_counter:06d}",
                    severity=finding_severity,
                    title=finding.get("category", "Alert").title(),
                    message=finding.get("message", ""),
                    source=source,
                    timestamp=datetime.now(UTC),
                    data=finding.get("data", {}),
                )
                alerts.append(alert)

        # Generate summary
        if not alerts:
            summary = "No alerts generated."
        else:
            critical_count = sum(1 for a in alerts if a.severity == "critical")
            warning_count = sum(1 for a in alerts if a.severity == "warning")

            parts = []
            if critical_count:
                parts.append(f"{critical_count} critical")
            if warning_count:
                parts.append(f"{warning_count} warning")

            summary = f"Generated {len(alerts)} alerts: {', '.join(parts)}."

        return AlertBatch(
            generator="alert_generator",
            generated_at=datetime.now(UTC),
            alerts=alerts,
            summary=summary,
            metadata={"min_severity": min_severity, "source": source},
        )
