"""Metrics exporters for various formats."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from .collector import MetricsCollector, WorkflowMetrics


class MetricsExporter(ABC):
    """Base class for metrics exporters."""

    @abstractmethod
    def export(self, collector: MetricsCollector) -> str:
        """Export metrics from collector.

        Args:
            collector: MetricsCollector instance

        Returns:
            Exported metrics as string
        """
        pass

    @abstractmethod
    def export_workflow(self, metrics: WorkflowMetrics) -> str:
        """Export a single workflow's metrics.

        Args:
            metrics: WorkflowMetrics instance

        Returns:
            Exported metrics as string
        """
        pass


class PrometheusExporter(MetricsExporter):
    """Export metrics in Prometheus text format.

    Compatible with Prometheus scraping and OpenMetrics.
    """

    def __init__(self, prefix: str = "gorgon"):
        """Initialize Prometheus exporter.

        Args:
            prefix: Metric name prefix
        """
        self.prefix = prefix

    def export(self, collector: MetricsCollector) -> str:
        """Export all metrics in Prometheus format."""
        lines = []
        summary = collector.get_summary()

        # Counters
        for name, value in summary["counters"].items():
            metric_name = f"{self.prefix}_{name}_total"
            lines.append(f"# TYPE {metric_name} counter")
            lines.append(f"{metric_name} {value}")

        # Gauges
        lines.append(f"# TYPE {self.prefix}_active_workflows gauge")
        lines.append(f"{self.prefix}_active_workflows {summary['active_workflows']}")

        lines.append(f"# TYPE {self.prefix}_success_rate gauge")
        lines.append(f"{self.prefix}_success_rate {summary['success_rate']:.2f}")

        # Histograms
        self._export_histogram(lines, "workflow_duration_ms", summary["workflow_duration"])
        self._export_histogram(lines, "workflow_tokens", summary["workflow_tokens"])
        self._export_histogram(lines, "step_duration_ms", summary["step_duration"])

        return "\n".join(lines) + "\n"

    def _export_histogram(self, lines: list[str], name: str, stats: dict) -> None:
        """Export histogram stats as summary metrics."""
        metric_name = f"{self.prefix}_{name}"
        lines.append(f"# TYPE {metric_name} summary")
        if stats["count"] > 0:
            lines.append(f'{metric_name}{{quantile="0.5"}} {stats["p50"]}')
            lines.append(f'{metric_name}{{quantile="0.95"}} {stats["p95"]}')
            lines.append(f"{metric_name}_sum {stats['avg'] * stats['count']}")
            lines.append(f"{metric_name}_count {stats['count']}")

    def export_workflow(self, metrics: WorkflowMetrics) -> str:
        """Export single workflow as Prometheus metrics."""
        lines = []
        labels = f'workflow_id="{metrics.workflow_id}",execution_id="{metrics.execution_id}"'

        lines.append(f"{self.prefix}_workflow_duration_ms{{{labels}}} {metrics.duration_ms}")
        lines.append(f"{self.prefix}_workflow_tokens{{{labels}}} {metrics.total_tokens}")
        lines.append(
            f"{self.prefix}_workflow_success{{{labels}}} {1 if metrics.status == 'success' else 0}"
        )
        lines.append(f"{self.prefix}_workflow_steps{{{labels}}} {len(metrics.steps)}")

        return "\n".join(lines) + "\n"


class JsonExporter(MetricsExporter):
    """Export metrics as JSON."""

    def __init__(self, pretty: bool = True):
        """Initialize JSON exporter.

        Args:
            pretty: Whether to format with indentation
        """
        self.pretty = pretty

    def export(self, collector: MetricsCollector) -> str:
        """Export all metrics as JSON."""
        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": collector.get_summary(),
            "active": [w.to_dict() for w in collector.get_active()],
            "recent": [w.to_dict() for w in collector.get_history(limit=10)],
        }
        if self.pretty:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, default=str)

    def export_workflow(self, metrics: WorkflowMetrics) -> str:
        """Export single workflow as JSON."""
        if self.pretty:
            return json.dumps(metrics.to_dict(), indent=2, default=str)
        return json.dumps(metrics.to_dict(), default=str)


class LogExporter(MetricsExporter):
    """Export metrics via Python logging.

    Useful for integration with existing logging infrastructure.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
    ):
        """Initialize log exporter.

        Args:
            logger: Logger instance (default: gorgon.metrics)
            level: Log level for metrics
        """
        self.logger = logger or logging.getLogger("gorgon.metrics")
        self.level = level

    def export(self, collector: MetricsCollector) -> str:
        """Export summary metrics to log."""
        summary = collector.get_summary()

        self.logger.log(
            self.level,
            "Metrics summary",
            extra={
                "active_workflows": summary["active_workflows"],
                "total_executions": summary["total_executions"],
                "success_rate": f"{summary['success_rate']:.1f}%",
                "avg_duration_ms": summary["workflow_duration"]["avg"],
                "counters": summary["counters"],
            },
        )

        return json.dumps(summary, default=str)

    def export_workflow(self, metrics: WorkflowMetrics) -> str:
        """Export workflow metrics to log."""
        self.logger.log(
            self.level,
            f"Workflow {metrics.status}: {metrics.workflow_name}",
            extra={
                "workflow_id": metrics.workflow_id,
                "execution_id": metrics.execution_id,
                "status": metrics.status,
                "duration_ms": metrics.duration_ms,
                "total_tokens": metrics.total_tokens,
                "steps_count": len(metrics.steps),
                "success_rate": f"{metrics.success_rate:.1%}",
            },
        )

        return json.dumps(metrics.to_dict(), default=str)


class FileExporter(MetricsExporter):
    """Export metrics to files.

    Supports periodic snapshots and workflow-level exports.
    """

    def __init__(
        self,
        output_dir: str | Path = "metrics",
        format: str = "json",
    ):
        """Initialize file exporter.

        Args:
            output_dir: Directory for metric files
            format: Output format ('json' or 'prometheus')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format

        if format == "json":
            self._inner = JsonExporter(pretty=True)
        elif format == "prometheus":
            self._inner = PrometheusExporter()
        else:
            raise ValueError(f"Unknown format: {format}")

    def export(self, collector: MetricsCollector) -> str:
        """Export metrics to snapshot file."""
        content = self._inner.export(collector)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        ext = "json" if self.format == "json" else "prom"
        filepath = self.output_dir / f"metrics_{timestamp}.{ext}"

        with open(filepath, "w") as f:
            f.write(content)

        return str(filepath)

    def export_workflow(self, metrics: WorkflowMetrics) -> str:
        """Export workflow to individual file."""
        content = self._inner.export_workflow(metrics)
        ext = "json" if self.format == "json" else "prom"
        filepath = self.output_dir / f"workflow_{metrics.execution_id}.{ext}"

        with open(filepath, "w") as f:
            f.write(content)

        return str(filepath)


def create_exporter(format: str, **kwargs) -> MetricsExporter:
    """Create an exporter by format name.

    Args:
        format: Exporter format ('prometheus', 'json', 'log', 'file')
        **kwargs: Format-specific arguments

    Returns:
        MetricsExporter instance

    Raises:
        ValueError: Unknown format
    """
    if format == "prometheus":
        return PrometheusExporter(**kwargs)
    elif format == "json":
        return JsonExporter(**kwargs)
    elif format == "log":
        return LogExporter(**kwargs)
    elif format == "file":
        return FileExporter(**kwargs)
    else:
        raise ValueError(f"Unknown exporter format: {format}")
