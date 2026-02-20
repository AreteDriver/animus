"""Prometheus metrics HTTP server and push gateway client.

Provides:
- HTTP endpoint for Prometheus scraping
- Push gateway client for pushing metrics
- OpenMetrics format support
"""

from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import URLError
from urllib.request import Request, urlopen

from .collector import MetricsCollector
from .exporters import PrometheusExporter

logger = logging.getLogger(__name__)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""

    collector: MetricsCollector | None = None
    exporter: PrometheusExporter | None = None

    def do_GET(self):
        """Handle GET request for /metrics."""
        if self.path == "/metrics":
            self._serve_metrics()
        elif self.path == "/health":
            self._serve_health()
        else:
            self.send_error(404, "Not Found")

    def _serve_metrics(self):
        """Serve Prometheus metrics."""
        if self.collector is None or self.exporter is None:
            self.send_error(500, "Metrics not configured")
            return

        try:
            content = self.exporter.export(self.collector)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
        except Exception as e:
            logger.error(f"Error serving metrics: {e}")
            self.send_error(500, str(e))

    def _serve_health(self):
        """Serve health check."""
        content = "ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class PrometheusMetricsServer:
    """HTTP server exposing Prometheus metrics endpoint.

    Usage:
        collector = MetricsCollector()
        server = PrometheusMetricsServer(collector, port=9090)
        server.start()

        # Prometheus can now scrape http://localhost:9090/metrics

        server.stop()
    """

    def __init__(
        self,
        collector: MetricsCollector,
        host: str = "0.0.0.0",
        port: int = 9090,
        prefix: str = "gorgon",
    ):
        """Initialize metrics server.

        Args:
            collector: MetricsCollector instance
            host: Host to bind to
            port: Port to bind to
            prefix: Prometheus metric prefix
        """
        self.collector = collector
        self.host = host
        self.port = port
        self.exporter = PrometheusExporter(prefix=prefix)
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the metrics server in a background thread."""
        if self._server is not None:
            logger.warning("Metrics server already running")
            return

        # Configure handler with collector and exporter
        MetricsHandler.collector = self.collector
        MetricsHandler.exporter = self.exporter

        self._server = HTTPServer((self.host, self.port), MetricsHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        logger.info(f"Prometheus metrics server started on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the metrics server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
            self._thread = None
            logger.info("Prometheus metrics server stopped")

    @property
    def url(self) -> str:
        """Get metrics endpoint URL."""
        return f"http://{self.host}:{self.port}/metrics"


class PrometheusPushGateway:
    """Client for pushing metrics to Prometheus Push Gateway.

    Use this when Prometheus cannot scrape your application directly
    (e.g., short-lived jobs, batch processes, serverless).

    Usage:
        collector = MetricsCollector()
        gateway = PrometheusPushGateway(
            url="http://pushgateway:9091",
            job="gorgon-workflows",
        )

        # Push metrics periodically or at workflow completion
        gateway.push(collector)

        # Delete metrics when done
        gateway.delete()
    """

    def __init__(
        self,
        url: str,
        job: str,
        instance: str | None = None,
        prefix: str = "gorgon",
        grouping_key: dict[str, str] | None = None,
    ):
        """Initialize push gateway client.

        Args:
            url: Push gateway URL (e.g., http://pushgateway:9091)
            job: Job name label
            instance: Instance label (optional)
            prefix: Prometheus metric prefix
            grouping_key: Additional grouping labels
        """
        self.url = url.rstrip("/")
        self.job = job
        self.instance = instance
        self.prefix = prefix
        self.grouping_key = grouping_key or {}
        self.exporter = PrometheusExporter(prefix=prefix)

    def _build_url(self) -> str:
        """Build push gateway URL with job and grouping key."""
        path = f"/metrics/job/{self.job}"

        if self.instance:
            path += f"/instance/{self.instance}"

        for key, value in self.grouping_key.items():
            path += f"/{key}/{value}"

        return self.url + path

    def push(self, collector: MetricsCollector) -> bool:
        """Push metrics to the gateway.

        Args:
            collector: MetricsCollector instance

        Returns:
            True if push succeeded
        """
        try:
            content = self.exporter.export(collector)
            url = self._build_url()

            req = Request(
                url,
                data=content.encode("utf-8"),
                method="POST",
                headers={"Content-Type": "text/plain"},
            )

            with urlopen(req, timeout=10) as response:
                if response.status in (200, 202):
                    logger.debug(f"Pushed metrics to {url}")
                    return True
                else:
                    logger.warning(f"Push gateway returned {response.status}")
                    return False

        except URLError as e:
            logger.error(f"Failed to push metrics: {e}")
            return False

    def delete(self) -> bool:
        """Delete metrics from the gateway.

        Returns:
            True if delete succeeded
        """
        try:
            url = self._build_url()
            req = Request(url, method="DELETE")

            with urlopen(req, timeout=10) as response:
                if response.status in (200, 202, 204):
                    logger.debug(f"Deleted metrics from {url}")
                    return True
                else:
                    logger.warning(f"Push gateway delete returned {response.status}")
                    return False

        except URLError as e:
            logger.error(f"Failed to delete metrics: {e}")
            return False


class MetricsPusher:
    """Background thread that periodically pushes metrics.

    Usage:
        collector = MetricsCollector()
        pusher = MetricsPusher(
            collector=collector,
            gateway_url="http://pushgateway:9091",
            job="gorgon",
            interval=60,  # Push every 60 seconds
        )
        pusher.start()

        # ... run workflows ...

        pusher.stop()
    """

    def __init__(
        self,
        collector: MetricsCollector,
        gateway_url: str,
        job: str,
        interval: float = 60.0,
        instance: str | None = None,
        prefix: str = "gorgon",
    ):
        """Initialize metrics pusher.

        Args:
            collector: MetricsCollector instance
            gateway_url: Push gateway URL
            job: Job name
            interval: Push interval in seconds
            instance: Instance label
            prefix: Metric prefix
        """
        self.collector = collector
        self.gateway = PrometheusPushGateway(
            url=gateway_url,
            job=job,
            instance=instance,
            prefix=prefix,
        )
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background pusher."""
        if self._thread is not None:
            logger.warning("Metrics pusher already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Started metrics pusher (interval={self.interval}s)")

    def stop(self) -> None:
        """Stop the background pusher."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            self._thread = None

            # Push final metrics and clean up
            self.gateway.push(self.collector)
            logger.info("Stopped metrics pusher")

    def _run(self) -> None:
        """Background push loop."""
        while not self._stop_event.is_set():
            self.gateway.push(self.collector)
            self._stop_event.wait(self.interval)


# Grafana dashboard configuration
GRAFANA_DASHBOARD = {
    "title": "Gorgon Workflow Metrics",
    "uid": "gorgon-workflows",
    "tags": ["gorgon", "workflows", "ai"],
    "timezone": "browser",
    "schemaVersion": 38,
    "panels": [
        {
            "title": "Active Workflows",
            "type": "stat",
            "gridPos": {"h": 4, "w": 4, "x": 0, "y": 0},
            "targets": [{"expr": "gorgon_active_workflows", "legendFormat": "Active"}],
        },
        {
            "title": "Success Rate",
            "type": "gauge",
            "gridPos": {"h": 4, "w": 4, "x": 4, "y": 0},
            "targets": [{"expr": "gorgon_success_rate * 100", "legendFormat": "Success %"}],
            "options": {
                "minValue": 0,
                "maxValue": 100,
                "thresholds": {
                    "steps": [
                        {"value": 0, "color": "red"},
                        {"value": 80, "color": "yellow"},
                        {"value": 95, "color": "green"},
                    ]
                },
            },
        },
        {
            "title": "Total Executions",
            "type": "stat",
            "gridPos": {"h": 4, "w": 4, "x": 8, "y": 0},
            "targets": [{"expr": "gorgon_workflows_total", "legendFormat": "Total"}],
        },
        {
            "title": "Total Tokens Used",
            "type": "stat",
            "gridPos": {"h": 4, "w": 4, "x": 12, "y": 0},
            "targets": [{"expr": "gorgon_tokens_used_total", "legendFormat": "Tokens"}],
        },
        {
            "title": "Workflow Duration (p95)",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
            "targets": [
                {
                    "expr": 'gorgon_workflow_duration_ms{quantile="0.95"}',
                    "legendFormat": "p95 duration (ms)",
                }
            ],
        },
        {
            "title": "Tokens per Workflow",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
            "targets": [
                {
                    "expr": 'gorgon_workflow_tokens{quantile="0.5"}',
                    "legendFormat": "p50 tokens",
                },
                {
                    "expr": 'gorgon_workflow_tokens{quantile="0.95"}',
                    "legendFormat": "p95 tokens",
                },
            ],
        },
        {
            "title": "Workflow Throughput",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12},
            "targets": [
                {
                    "expr": "rate(gorgon_workflows_total[5m]) * 60",
                    "legendFormat": "workflows/min",
                }
            ],
        },
        {
            "title": "Error Rate",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12},
            "targets": [
                {
                    "expr": "rate(gorgon_workflow_errors_total[5m]) / rate(gorgon_workflows_total[5m]) * 100",
                    "legendFormat": "error %",
                }
            ],
        },
    ],
}


def get_grafana_dashboard() -> dict:
    """Get Grafana dashboard configuration.

    Returns:
        Dashboard JSON that can be imported into Grafana
    """
    return GRAFANA_DASHBOARD
