"""Data Analyzers for Analytics Pipelines.

Provides modular analysis components for processing collected data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    analyzer: str
    analyzed_at: datetime
    findings: list[dict[str, Any]]
    metrics: dict[str, Any]
    recommendations: list[str]
    severity: str  # "info", "warning", "critical"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Convert to string format for AI agent context."""
        lines = [
            f"# Analysis Results: {self.analyzer}",
            f"Analyzed: {self.analyzed_at.isoformat()}",
            f"Overall Severity: {self.severity.upper()}",
            "",
            "## Key Findings",
        ]

        for finding in self.findings:
            severity = finding.get("severity", "info")
            message = finding.get("message", "")
            lines.append(f"- [{severity.upper()}] {message}")

        lines.append("")
        lines.append("## Metrics")
        for key, value in self.metrics.items():
            lines.append(f"- {key}: {value}")

        if self.recommendations:
            lines.append("")
            lines.append("## Recommendations")
            for rec in self.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)


class DataAnalyzer(ABC):
    """Abstract base class for data analyzers."""

    @abstractmethod
    def analyze(self, data: Any, config: dict) -> AnalysisResult:
        """Analyze the collected data.

        Args:
            data: CollectedData or previous stage output
            config: Analyzer configuration

        Returns:
            AnalysisResult with findings and recommendations
        """
        pass

    def _extract_source_data(self, data: Any) -> dict:
        """Extract source data from different input types."""
        if hasattr(data, "data"):
            return data.data
        if isinstance(data, dict):
            return data
        return {}


class TrendAnalyzer(DataAnalyzer):
    """Analyzer for identifying trends in metrics data."""

    def _analyze_timing_metric(
        self,
        metric_name: str,
        timing_data: dict,
        metrics: dict,
        findings: list,
        recommendations: list,
    ) -> str | None:
        """Analyze a single timing metric. Returns severity if elevated."""
        avg_ms = timing_data.get("avg_ms", 0)
        max_ms = timing_data.get("max_ms", 0)
        count = timing_data.get("count", 0)

        metrics[f"{metric_name}_avg"] = avg_ms
        metrics[f"{metric_name}_count"] = count

        elevated_severity = None

        if avg_ms > 1000:
            elevated_severity = "warning"
            findings.append(
                {
                    "severity": "warning",
                    "category": "performance",
                    "message": f"Slow operation: {metric_name} averaging {avg_ms:.0f}ms",
                    "data": timing_data,
                }
            )
            recommendations.append(f"Investigate performance of {metric_name}")

        if max_ms > avg_ms * 3 and count > 5:
            findings.append(
                {
                    "severity": "info",
                    "category": "variance",
                    "message": f"High variance in {metric_name}: max {max_ms:.0f}ms vs avg {avg_ms:.0f}ms",
                }
            )

        return elevated_severity

    def _analyze_error_counters(self, counters: dict, findings: list) -> str | None:
        """Analyze error counters. Returns severity if elevated."""
        elevated_severity = None
        for counter_name, value in counters.items():
            if "error" in counter_name.lower() and value > 0:
                findings.append(
                    {
                        "severity": "warning",
                        "category": "errors",
                        "message": f"Error counter {counter_name}: {value}",
                    }
                )
                elevated_severity = "warning"
        return elevated_severity

    def analyze(self, data: Any, config: dict) -> AnalysisResult:
        """Analyze metrics for trends."""
        findings = []
        metrics = {}
        recommendations = []
        max_severity = "info"

        source_data = self._extract_source_data(data)
        app_metrics = source_data.get("metrics", source_data.get("app_performance", {}))

        if isinstance(app_metrics, dict):
            for metric_name, timing_data in app_metrics.get("timing", {}).items():
                if isinstance(timing_data, dict):
                    severity = self._analyze_timing_metric(
                        metric_name, timing_data, metrics, findings, recommendations
                    )
                    if severity and max_severity == "info":
                        max_severity = severity

            severity = self._analyze_error_counters(app_metrics.get("counters", {}), findings)
            if severity and max_severity == "info":
                max_severity = severity

        if not findings:
            findings.append(
                {
                    "severity": "info",
                    "category": "trends",
                    "message": "No significant trends detected",
                }
            )

        return AnalysisResult(
            analyzer="trends",
            analyzed_at=datetime.now(UTC),
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            severity=max_severity,
        )


def _get_nested_value(d: dict, path: str) -> Any:
    """Get value from nested dict using dot notation.

    First checks for exact key match, then tries nested access.
    """
    # First try exact key match (handles keys with dots in them)
    if path in d:
        return d[path]

    # Then try nested access
    current = d
    for key in path.split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


class ThresholdAnalyzer(DataAnalyzer):
    """Analyzer that checks metrics against configurable thresholds."""

    def _check_threshold(
        self,
        metric_path: str,
        value: Any,
        threshold_config: dict,
        findings: list,
        recommendations: list,
    ) -> str | None:
        """Check a single metric against thresholds. Returns severity if elevated."""
        warning_threshold = threshold_config.get("warning")
        critical_threshold = threshold_config.get("critical")
        direction = threshold_config.get("direction", "above")

        is_above = direction == "above"
        exceeds_critical = critical_threshold is not None and (
            value >= critical_threshold if is_above else value <= critical_threshold
        )
        exceeds_warning = warning_threshold is not None and (
            value >= warning_threshold if is_above else value <= warning_threshold
        )

        verb = "exceeds" if is_above else "below"

        if exceeds_critical:
            findings.append(
                {
                    "severity": "critical",
                    "category": "threshold",
                    "message": f"{metric_path} = {value} {verb} critical threshold {critical_threshold}",
                }
            )
            recommendations.append(f"Investigate critical {metric_path}")
            return "critical"

        if exceeds_warning:
            findings.append(
                {
                    "severity": "warning",
                    "category": "threshold",
                    "message": f"{metric_path} = {value} {verb} warning threshold {warning_threshold}",
                }
            )
            return "warning"

        return None

    def analyze(self, data: Any, config: dict) -> AnalysisResult:
        """Analyze data against threshold rules."""
        thresholds = config.get("thresholds", {})
        findings = []
        metrics = {}
        recommendations = []
        max_severity = "info"

        source_data = self._extract_source_data(data)

        for metric_path, threshold_config in thresholds.items():
            value = _get_nested_value(source_data, metric_path)
            if value is None:
                continue

            metrics[metric_path] = value
            severity = self._check_threshold(
                metric_path, value, threshold_config, findings, recommendations
            )

            if severity == "critical":
                max_severity = "critical"
            elif severity == "warning" and max_severity != "critical":
                max_severity = "warning"

        if not findings:
            findings.append(
                {
                    "severity": "info",
                    "category": "thresholds",
                    "message": "All metrics within acceptable thresholds",
                }
            )

        return AnalysisResult(
            analyzer="threshold",
            analyzed_at=datetime.now(UTC),
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            severity=max_severity,
        )


class CompositeAnalyzer(DataAnalyzer):
    """Analyzer that combines results from multiple analyzers."""

    def __init__(self, analyzers: list[DataAnalyzer]):
        self.analyzers = analyzers

    def analyze(self, data: Any, config: dict) -> AnalysisResult:
        """Run all analyzers and combine results.

        Config options:
            analyzer_configs: dict[int, dict] - Config for each analyzer by index
        """
        analyzer_configs = config.get("analyzer_configs", {})

        all_findings = []
        all_metrics = {}
        all_recommendations = []
        max_severity = "info"

        severity_order = {"info": 0, "warning": 1, "critical": 2}

        for i, analyzer in enumerate(self.analyzers):
            analyzer_config = analyzer_configs.get(i, {})
            result = analyzer.analyze(data, analyzer_config)

            all_findings.extend(result.findings)
            all_metrics.update(result.metrics)
            all_recommendations.extend(result.recommendations)

            if severity_order.get(result.severity, 0) > severity_order.get(max_severity, 0):
                max_severity = result.severity

        # Deduplicate recommendations
        unique_recommendations = list(dict.fromkeys(all_recommendations))

        return AnalysisResult(
            analyzer="composite",
            analyzed_at=datetime.now(UTC),
            findings=all_findings,
            metrics=all_metrics,
            recommendations=unique_recommendations,
            severity=max_severity,
            metadata={"analyzer_count": len(self.analyzers)},
        )
