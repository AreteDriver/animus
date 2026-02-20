"""Trace exporters for OpenTelemetry-compatible backends.

Provides exporters for:
- OTLP (gRPC and HTTP)
- Console (for debugging)
- Custom backends

OpenTelemetry SDK integration is optional - if not installed,
traces can still be exported via HTTP to OTLP endpoints.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for trace export."""

    # OTLP endpoint (e.g., "http://localhost:4318/v1/traces")
    otlp_endpoint: str = ""
    # Service name for trace attribution
    service_name: str = "gorgon"
    # Environment (e.g., "production", "staging")
    environment: str = "development"
    # Export batch size
    batch_size: int = 100
    # Export interval in seconds
    export_interval: float = 5.0
    # Maximum queue size
    max_queue_size: int = 10000
    # Headers for OTLP endpoint
    headers: dict[str, str] | None = None
    # Timeout for HTTP requests
    timeout_seconds: float = 10.0


class TraceExporter(ABC):
    """Abstract base for trace exporters."""

    @abstractmethod
    def export(self, traces: list[dict[str, Any]]) -> bool:
        """Export trace data.

        Args:
            traces: List of trace dictionaries

        Returns:
            True if export succeeded
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class ConsoleExporter(TraceExporter):
    """Exporter that prints traces to console (for debugging)."""

    def __init__(self, pretty: bool = True):
        """Initialize console exporter.

        Args:
            pretty: Use pretty-printed JSON
        """
        self.pretty = pretty

    def export(self, traces: list[dict[str, Any]]) -> bool:
        """Print traces to console."""
        for trace in traces:
            if self.pretty:
                print(json.dumps(trace, indent=2, default=str))
            else:
                print(json.dumps(trace, default=str))
        return True

    def shutdown(self) -> None:
        """No-op for console exporter."""
        pass


class OTLPHTTPExporter(TraceExporter):
    """Exporter that sends traces to OTLP HTTP endpoint.

    Compatible with:
    - Jaeger (with OTLP receiver)
    - Honeycomb
    - Grafana Tempo
    - OpenTelemetry Collector
    """

    def __init__(self, config: ExportConfig):
        """Initialize OTLP HTTP exporter.

        Args:
            config: Export configuration
        """
        self.config = config
        self._session = None

    def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            pass
        return self._session

    def _convert_to_otlp(self, traces: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert internal trace format to OTLP format."""
        resource_spans = []

        for trace in traces:
            scope_spans = []

            for span in trace.get("spans", []):
                # Convert timestamps to nanoseconds
                start_ns = self._iso_to_nanos(span.get("start_time", ""))
                end_ns = (
                    self._iso_to_nanos(span.get("end_time", ""))
                    if span.get("end_time")
                    else start_ns
                )

                otlp_span = {
                    "traceId": self._hex_to_bytes(span.get("trace_id", "")),
                    "spanId": self._hex_to_bytes(span.get("span_id", "")),
                    "name": span.get("name", "unknown"),
                    "kind": 1,  # SPAN_KIND_INTERNAL
                    "startTimeUnixNano": str(start_ns),
                    "endTimeUnixNano": str(end_ns),
                    "attributes": self._convert_attributes(span.get("attributes", {})),
                    "status": {
                        "code": 1 if span.get("status") == "ok" else 2,
                    },
                }

                if span.get("parent_span_id"):
                    otlp_span["parentSpanId"] = self._hex_to_bytes(span["parent_span_id"])

                # Add events
                if span.get("events"):
                    otlp_span["events"] = [
                        {
                            "name": event.get("name", ""),
                            "timeUnixNano": str(self._iso_to_nanos(event.get("timestamp", ""))),
                            "attributes": self._convert_attributes(event.get("attributes", {})),
                        }
                        for event in span["events"]
                    ]

                scope_spans.append(otlp_span)

            if scope_spans:
                resource_spans.append(
                    {
                        "resource": {
                            "attributes": [
                                {
                                    "key": "service.name",
                                    "value": {"stringValue": self.config.service_name},
                                },
                                {
                                    "key": "deployment.environment",
                                    "value": {"stringValue": self.config.environment},
                                },
                            ],
                        },
                        "scopeSpans": [
                            {
                                "scope": {"name": "gorgon"},
                                "spans": scope_spans,
                            },
                        ],
                    }
                )

        return {"resourceSpans": resource_spans}

    def _hex_to_bytes(self, hex_str: str) -> str:
        """Convert hex string to base64 for OTLP JSON format."""
        import base64

        try:
            return base64.b64encode(bytes.fromhex(hex_str)).decode("ascii")
        except ValueError:
            return ""

    def _iso_to_nanos(self, iso_str: str) -> int:
        """Convert ISO timestamp to nanoseconds."""
        if not iso_str:
            return int(time.time() * 1e9)
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1e9)
        except ValueError:
            return int(time.time() * 1e9)

    def _convert_attributes(self, attrs: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert attributes to OTLP format."""
        result = []
        for key, value in attrs.items():
            if isinstance(value, bool):
                result.append({"key": key, "value": {"boolValue": value}})
            elif isinstance(value, int):
                result.append({"key": key, "value": {"intValue": str(value)}})
            elif isinstance(value, float):
                result.append({"key": key, "value": {"doubleValue": value}})
            else:
                result.append({"key": key, "value": {"stringValue": str(value)}})
        return result

    def export(self, traces: list[dict[str, Any]]) -> bool:
        """Export traces to OTLP endpoint via HTTP."""
        if not self.config.otlp_endpoint:
            logger.debug("No OTLP endpoint configured, skipping export")
            return True

        try:
            import urllib.error
            import urllib.request

            otlp_data = self._convert_to_otlp(traces)
            json_data = json.dumps(otlp_data).encode("utf-8")

            headers = {
                "Content-Type": "application/json",
            }
            if self.config.headers:
                headers.update(self.config.headers)

            request = urllib.request.Request(
                self.config.otlp_endpoint,
                data=json_data,
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                if response.status >= 300:
                    logger.warning(f"OTLP export failed: {response.status}")
                    return False

            logger.debug(f"Exported {len(traces)} traces to OTLP endpoint")
            return True

        except urllib.error.URLError as e:
            logger.warning(f"OTLP export failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"OTLP export error: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown HTTP session."""
        pass


class BatchExporter:
    """Batches traces and exports them periodically.

    Uses a background thread to batch and export traces
    without blocking the main application.
    """

    def __init__(
        self,
        exporter: TraceExporter,
        config: ExportConfig | None = None,
    ):
        """Initialize batch exporter.

        Args:
            exporter: Underlying trace exporter
            config: Export configuration
        """
        self.exporter = exporter
        self.config = config or ExportConfig()
        self._queue: queue.Queue = queue.Queue(maxsize=self.config.max_queue_size)
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats = {
            "exported": 0,
            "dropped": 0,
            "errors": 0,
        }

    def start(self) -> None:
        """Start the background export thread."""
        if self._thread and self._thread.is_alive():
            return

        self._shutdown.clear()
        self._thread = threading.Thread(target=self._export_loop, daemon=True)
        self._thread.start()
        logger.info("Started trace batch exporter")

    def stop(self) -> None:
        """Stop the background export thread."""
        self._shutdown.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.exporter.shutdown()
        logger.info("Stopped trace batch exporter")

    def add_trace(self, trace: dict[str, Any]) -> bool:
        """Add a trace to the export queue.

        Args:
            trace: Trace dictionary

        Returns:
            True if added, False if queue is full
        """
        try:
            self._queue.put_nowait(trace)
            return True
        except queue.Full:
            self._stats["dropped"] += 1
            logger.warning("Trace export queue is full, dropping trace")
            return False

    def _export_loop(self) -> None:
        """Background thread that batches and exports traces."""
        batch: list[dict[str, Any]] = []
        last_export = time.monotonic()

        while not self._shutdown.is_set():
            # Collect traces from queue
            try:
                trace = self._queue.get(timeout=0.1)
                batch.append(trace)
            except queue.Empty:
                pass  # Graceful degradation: queue poll timeout is normal control flow

            # Export when batch is full or interval elapsed
            should_export = len(batch) >= self.config.batch_size or (
                batch and time.monotonic() - last_export >= self.config.export_interval
            )

            if should_export and batch:
                success = self.exporter.export(batch)
                if success:
                    self._stats["exported"] += len(batch)
                else:
                    self._stats["errors"] += 1
                batch = []
                last_export = time.monotonic()

        # Export remaining traces on shutdown
        if batch:
            self.exporter.export(batch)
            self._stats["exported"] += len(batch)

    def get_stats(self) -> dict[str, int]:
        """Get export statistics."""
        return self._stats.copy()


# Global batch exporter instance
_batch_exporter: BatchExporter | None = None
_exporter_lock = threading.Lock()


def get_batch_exporter(config: ExportConfig | None = None) -> BatchExporter:
    """Get or create global batch exporter."""
    global _batch_exporter

    with _exporter_lock:
        if _batch_exporter is None:
            config = config or ExportConfig()
            exporter = OTLPHTTPExporter(config)
            _batch_exporter = BatchExporter(exporter, config)
            _batch_exporter.start()
        return _batch_exporter


def export_trace(trace: dict[str, Any]) -> bool:
    """Export a trace using the global batch exporter.

    Args:
        trace: Trace dictionary to export

    Returns:
        True if queued for export
    """
    exporter = get_batch_exporter()
    return exporter.add_trace(trace)


def shutdown_exporter() -> None:
    """Shutdown the global batch exporter."""
    global _batch_exporter

    with _exporter_lock:
        if _batch_exporter:
            _batch_exporter.stop()
            _batch_exporter = None
