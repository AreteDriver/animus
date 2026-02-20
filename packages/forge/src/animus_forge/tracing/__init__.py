"""Distributed tracing for Gorgon workflows.

Provides W3C Trace Context compatible tracing with:
- Trace and span ID generation
- Context propagation across functions and async calls
- Integration with structured logging
- HTTP header propagation (traceparent, tracestate)
- OTLP export for OpenTelemetry-compatible backends
"""

from animus_forge.tracing.context import (
    Span,
    TraceContext,
    get_current_span,
    get_current_trace,
    span_context,
    start_span,
    start_trace,
    trace_context,
)
from animus_forge.tracing.export import (
    BatchExporter,
    ConsoleExporter,
    ExportConfig,
    OTLPHTTPExporter,
    TraceExporter,
    export_trace,
    get_batch_exporter,
    shutdown_exporter,
)
from animus_forge.tracing.middleware import TracingMiddleware
from animus_forge.tracing.propagation import (
    TRACEPARENT_HEADER,
    TRACESTATE_HEADER,
    extract_trace_context,
    inject_trace_headers,
)

__all__ = [
    # Context
    "TraceContext",
    "Span",
    "get_current_trace",
    "get_current_span",
    "start_trace",
    "start_span",
    "trace_context",
    "span_context",
    # Propagation
    "extract_trace_context",
    "inject_trace_headers",
    "TRACEPARENT_HEADER",
    "TRACESTATE_HEADER",
    # Middleware
    "TracingMiddleware",
    # Export
    "ExportConfig",
    "TraceExporter",
    "ConsoleExporter",
    "OTLPHTTPExporter",
    "BatchExporter",
    "get_batch_exporter",
    "export_trace",
    "shutdown_exporter",
]
