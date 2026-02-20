"""Tests for trace export functionality."""

import sys
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")

from animus_forge.tracing.export import (
    BatchExporter,
    ConsoleExporter,
    ExportConfig,
    OTLPHTTPExporter,
    export_trace,
    get_batch_exporter,
    shutdown_exporter,
)


class TestExportConfig:
    """Tests for ExportConfig."""

    def test_default_values(self):
        """Has sensible defaults."""
        config = ExportConfig()
        assert config.service_name == "gorgon"
        assert config.environment == "development"
        assert config.batch_size == 100
        assert config.export_interval == 5.0

    def test_custom_values(self):
        """Accepts custom configuration."""
        config = ExportConfig(
            otlp_endpoint="http://localhost:4318/v1/traces",
            service_name="my-service",
            environment="production",
            batch_size=50,
        )
        assert config.otlp_endpoint == "http://localhost:4318/v1/traces"
        assert config.service_name == "my-service"
        assert config.environment == "production"
        assert config.batch_size == 50


class TestConsoleExporter:
    """Tests for ConsoleExporter."""

    def test_exports_to_console(self, capsys):
        """Prints traces to console."""
        exporter = ConsoleExporter(pretty=False)
        trace = {
            "trace_id": "abc123",
            "spans": [{"name": "test"}],
        }

        result = exporter.export([trace])

        assert result is True
        captured = capsys.readouterr()
        assert "abc123" in captured.out
        assert "test" in captured.out

    def test_pretty_print(self, capsys):
        """Pretty prints when enabled."""
        exporter = ConsoleExporter(pretty=True)
        trace = {"trace_id": "abc123"}

        exporter.export([trace])

        captured = capsys.readouterr()
        # Pretty print adds newlines and indentation
        assert "\n" in captured.out

    def test_shutdown_noop(self):
        """Shutdown does nothing for console exporter."""
        exporter = ConsoleExporter()
        exporter.shutdown()  # Should not raise


class TestOTLPHTTPExporter:
    """Tests for OTLPHTTPExporter."""

    def test_skips_export_without_endpoint(self):
        """Skips export when no endpoint configured."""
        config = ExportConfig(otlp_endpoint="")
        exporter = OTLPHTTPExporter(config)

        result = exporter.export([{"trace_id": "abc"}])

        assert result is True  # Returns True (no-op success)

    def test_converts_to_otlp_format(self):
        """Converts internal trace format to OTLP."""
        config = ExportConfig(
            otlp_endpoint="http://localhost:4318/v1/traces",
            service_name="test-service",
        )
        exporter = OTLPHTTPExporter(config)

        trace = {
            "trace_id": "0af7651916cd43dd8448eb211c80319c",
            "spans": [
                {
                    "trace_id": "0af7651916cd43dd8448eb211c80319c",
                    "span_id": "b7ad6b7169203331",
                    "name": "test-span",
                    "start_time": "2024-01-01T12:00:00+00:00",
                    "end_time": "2024-01-01T12:00:01+00:00",
                    "status": "ok",
                    "attributes": {"key": "value"},
                },
            ],
        }

        otlp_data = exporter._convert_to_otlp([trace])

        assert "resourceSpans" in otlp_data
        assert len(otlp_data["resourceSpans"]) == 1

        resource = otlp_data["resourceSpans"][0]
        assert "resource" in resource
        assert "scopeSpans" in resource

    def test_converts_attributes_correctly(self):
        """Converts different attribute types."""
        config = ExportConfig()
        exporter = OTLPHTTPExporter(config)

        attrs = {
            "string_val": "hello",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
        }

        converted = exporter._convert_attributes(attrs)

        # Find each attribute
        string_attr = next(a for a in converted if a["key"] == "string_val")
        int_attr = next(a for a in converted if a["key"] == "int_val")
        float_attr = next(a for a in converted if a["key"] == "float_val")
        bool_attr = next(a for a in converted if a["key"] == "bool_val")

        assert string_attr["value"]["stringValue"] == "hello"
        assert int_attr["value"]["intValue"] == "42"
        assert float_attr["value"]["doubleValue"] == 3.14
        assert bool_attr["value"]["boolValue"] is True

    @patch("urllib.request.urlopen")
    def test_exports_to_endpoint(self, mock_urlopen):
        """Exports traces to OTLP endpoint."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = ExportConfig(
            otlp_endpoint="http://localhost:4318/v1/traces",
        )
        exporter = OTLPHTTPExporter(config)

        trace = {
            "trace_id": "abc123",
            "spans": [
                {
                    "trace_id": "abc123",
                    "span_id": "def456",
                    "name": "test",
                    "status": "ok",
                },
            ],
        }

        result = exporter.export([trace])

        assert result is True
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_handles_export_failure(self, mock_urlopen):
        """Handles HTTP errors gracefully."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        config = ExportConfig(
            otlp_endpoint="http://localhost:4318/v1/traces",
        )
        exporter = OTLPHTTPExporter(config)

        result = exporter.export([{"trace_id": "abc"}])

        assert result is False


class TestBatchExporter:
    """Tests for BatchExporter."""

    def test_queues_traces(self):
        """Queues traces for export."""
        mock_exporter = MagicMock(spec=ConsoleExporter)
        mock_exporter.export.return_value = True

        batch = BatchExporter(mock_exporter, ExportConfig(batch_size=10))

        result = batch.add_trace({"trace_id": "abc"})

        assert result is True
        assert not batch._queue.empty()

    def test_exports_when_batch_full(self):
        """Exports when batch size reached."""
        mock_exporter = MagicMock(spec=ConsoleExporter)
        mock_exporter.export.return_value = True

        config = ExportConfig(batch_size=2, export_interval=60.0)
        batch = BatchExporter(mock_exporter, config)
        batch.start()

        try:
            batch.add_trace({"trace_id": "1"})
            batch.add_trace({"trace_id": "2"})

            # Give export thread time to process
            time.sleep(0.3)

            assert mock_exporter.export.called
        finally:
            batch.stop()

    def test_exports_on_interval(self):
        """Exports on time interval even with small batch."""
        mock_exporter = MagicMock(spec=ConsoleExporter)
        mock_exporter.export.return_value = True

        config = ExportConfig(batch_size=100, export_interval=0.1)
        batch = BatchExporter(mock_exporter, config)
        batch.start()

        try:
            batch.add_trace({"trace_id": "1"})

            # Wait for interval
            time.sleep(0.3)

            assert mock_exporter.export.called
        finally:
            batch.stop()

    def test_drops_when_queue_full(self):
        """Drops traces when queue is full."""
        mock_exporter = MagicMock(spec=ConsoleExporter)
        config = ExportConfig(max_queue_size=2)
        batch = BatchExporter(mock_exporter, config)

        # Don't start the thread so queue fills up
        batch.add_trace({"trace_id": "1"})
        batch.add_trace({"trace_id": "2"})
        result = batch.add_trace({"trace_id": "3"})

        assert result is False
        assert batch._stats["dropped"] == 1

    def test_stats_tracking(self):
        """Tracks export statistics."""
        mock_exporter = MagicMock(spec=ConsoleExporter)
        mock_exporter.export.return_value = True

        config = ExportConfig(batch_size=1, export_interval=0.1)
        batch = BatchExporter(mock_exporter, config)
        batch.start()

        try:
            batch.add_trace({"trace_id": "1"})
            time.sleep(0.3)

            stats = batch.get_stats()
            assert stats["exported"] >= 1
        finally:
            batch.stop()

    def test_shutdown_exports_remaining(self):
        """Exports remaining traces on shutdown."""
        mock_exporter = MagicMock(spec=ConsoleExporter)
        mock_exporter.export.return_value = True

        config = ExportConfig(batch_size=100, export_interval=60.0)
        batch = BatchExporter(mock_exporter, config)
        batch.start()

        batch.add_trace({"trace_id": "1"})
        batch.stop()

        # Should have exported the remaining trace
        assert mock_exporter.export.called


class TestGlobalExporter:
    """Tests for global exporter functions."""

    def test_get_batch_exporter_creates_singleton(self):
        """Creates singleton batch exporter."""
        # Reset global state
        shutdown_exporter()

        exporter1 = get_batch_exporter()
        exporter2 = get_batch_exporter()

        try:
            assert exporter1 is exporter2
        finally:
            shutdown_exporter()

    def test_export_trace_queues_trace(self):
        """export_trace queues trace for export."""
        shutdown_exporter()

        result = export_trace({"trace_id": "test"})

        try:
            assert result is True
        finally:
            shutdown_exporter()

    def test_shutdown_stops_exporter(self):
        """shutdown_exporter stops the global exporter."""
        # Get exporter first
        get_batch_exporter()

        # Should not raise
        shutdown_exporter()
        shutdown_exporter()  # Double shutdown should be safe
