"""Tests for Prometheus metrics server and push gateway."""

import sys
import time
from unittest.mock import MagicMock, Mock, patch
from urllib.request import urlopen

sys.path.insert(0, "src")

from animus_forge.metrics import (
    MetricsCollector,
    MetricsPusher,
    PrometheusMetricsServer,
    PrometheusPushGateway,
    get_grafana_dashboard,
)


class TestPrometheusMetricsServer:
    """Tests for PrometheusMetricsServer."""

    def test_server_starts_and_stops(self):
        """Server can start and stop."""
        collector = MetricsCollector()
        server = PrometheusMetricsServer(collector, port=19090)

        server.start()
        assert server._server is not None
        assert server._thread is not None

        server.stop()
        assert server._server is None

    def test_server_url_property(self):
        """URL property returns correct endpoint."""
        collector = MetricsCollector()
        server = PrometheusMetricsServer(collector, host="localhost", port=19091)
        assert server.url == "http://localhost:19091/metrics"

    def test_server_serves_metrics(self):
        """Server serves metrics at /metrics endpoint."""
        collector = MetricsCollector()

        # Add some test data
        collector.start_workflow("wf-1", "test-workflow", "exec-1")
        collector.complete_workflow("exec-1")

        server = PrometheusMetricsServer(collector, port=19092)
        server.start()

        try:
            # Give server time to start
            time.sleep(0.1)

            # Fetch metrics
            response = urlopen("http://localhost:19092/metrics", timeout=5)
            content = response.read().decode("utf-8")

            assert response.status == 200
            assert "gorgon_" in content
            assert "workflows_total" in content or "success_rate" in content
        finally:
            server.stop()

    def test_server_serves_health(self):
        """Server serves health at /health endpoint."""
        collector = MetricsCollector()
        server = PrometheusMetricsServer(collector, port=19093)
        server.start()

        try:
            time.sleep(0.1)
            response = urlopen("http://localhost:19093/health", timeout=5)
            content = response.read().decode("utf-8")

            assert response.status == 200
            assert content == "ok"
        finally:
            server.stop()

    def test_server_custom_prefix(self):
        """Server uses custom metric prefix."""
        collector = MetricsCollector()
        server = PrometheusMetricsServer(collector, port=19094, prefix="myapp")
        server.start()

        try:
            time.sleep(0.1)
            response = urlopen("http://localhost:19094/metrics", timeout=5)
            content = response.read().decode("utf-8")

            assert "myapp_" in content
        finally:
            server.stop()


class TestPrometheusPushGateway:
    """Tests for PrometheusPushGateway."""

    def test_build_url_basic(self):
        """URL is built correctly with just job."""
        gateway = PrometheusPushGateway(
            url="http://pushgateway:9091",
            job="test-job",
        )
        url = gateway._build_url()
        assert url == "http://pushgateway:9091/metrics/job/test-job"

    def test_build_url_with_instance(self):
        """URL includes instance when provided."""
        gateway = PrometheusPushGateway(
            url="http://pushgateway:9091",
            job="test-job",
            instance="worker-1",
        )
        url = gateway._build_url()
        assert url == "http://pushgateway:9091/metrics/job/test-job/instance/worker-1"

    def test_build_url_with_grouping_key(self):
        """URL includes grouping key labels."""
        gateway = PrometheusPushGateway(
            url="http://pushgateway:9091",
            job="test-job",
            grouping_key={"env": "prod", "region": "us-east"},
        )
        url = gateway._build_url()
        assert "/env/prod" in url
        assert "/region/us-east" in url

    def test_push_success(self):
        """Push returns True on success."""
        gateway = PrometheusPushGateway(
            url="http://pushgateway:9091",
            job="test-job",
        )
        collector = MetricsCollector()

        # Mock urlopen to return success
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("animus_forge.metrics.prometheus_server.urlopen", return_value=mock_response):
            result = gateway.push(collector)
            assert result is True

    def test_push_failure(self):
        """Push returns False on network error."""
        gateway = PrometheusPushGateway(
            url="http://nonexistent:9091",
            job="test-job",
        )
        collector = MetricsCollector()

        # This will fail because the server doesn't exist
        result = gateway.push(collector)
        assert result is False

    def test_delete_success(self):
        """Delete returns True on success."""
        gateway = PrometheusPushGateway(
            url="http://pushgateway:9091",
            job="test-job",
        )

        mock_response = MagicMock()
        mock_response.status = 202
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("animus_forge.metrics.prometheus_server.urlopen", return_value=mock_response):
            result = gateway.delete()
            assert result is True


class TestMetricsPusher:
    """Tests for MetricsPusher background thread."""

    def test_pusher_starts_and_stops(self):
        """Pusher can start and stop."""
        collector = MetricsCollector()
        pusher = MetricsPusher(
            collector=collector,
            gateway_url="http://localhost:9091",
            job="test",
            interval=1.0,
        )

        # Mock the gateway to avoid network calls
        pusher.gateway.push = Mock(return_value=True)

        pusher.start()
        assert pusher._thread is not None

        time.sleep(0.1)  # Let it run briefly
        pusher.stop()
        assert pusher._thread is None

    def test_pusher_calls_gateway(self):
        """Pusher calls gateway.push periodically."""
        collector = MetricsCollector()
        pusher = MetricsPusher(
            collector=collector,
            gateway_url="http://localhost:9091",
            job="test",
            interval=0.1,  # Short interval for testing
        )

        pusher.gateway.push = Mock(return_value=True)

        pusher.start()
        time.sleep(0.35)  # Should trigger ~3 pushes
        pusher.stop()

        # Should have been called multiple times
        assert pusher.gateway.push.call_count >= 2


class TestGrafanaDashboard:
    """Tests for Grafana dashboard configuration."""

    def test_dashboard_has_required_fields(self):
        """Dashboard has required Grafana fields."""
        dashboard = get_grafana_dashboard()

        assert "title" in dashboard
        assert "uid" in dashboard
        assert "panels" in dashboard
        assert isinstance(dashboard["panels"], list)

    def test_dashboard_has_panels(self):
        """Dashboard has multiple panels."""
        dashboard = get_grafana_dashboard()
        assert len(dashboard["panels"]) >= 4

    def test_panels_have_targets(self):
        """Each panel has Prometheus targets."""
        dashboard = get_grafana_dashboard()

        for panel in dashboard["panels"]:
            assert "title" in panel
            assert "type" in panel
            assert "targets" in panel
            assert len(panel["targets"]) >= 1

    def test_panels_use_gorgon_prefix(self):
        """Panel queries use gorgon_ metric prefix."""
        dashboard = get_grafana_dashboard()

        for panel in dashboard["panels"]:
            for target in panel["targets"]:
                expr = target.get("expr", "")
                assert "gorgon_" in expr, f"Panel '{panel['title']}' missing gorgon_ prefix"
