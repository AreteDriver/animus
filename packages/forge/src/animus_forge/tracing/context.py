"""Trace context management for distributed tracing.

Implements W3C Trace Context compatible tracing with:
- 128-bit trace IDs (32 hex chars)
- 64-bit span IDs (16 hex chars)
- Context propagation via contextvars
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def _generate_trace_id() -> str:
    """Generate a W3C compliant 128-bit trace ID (32 hex chars)."""
    return secrets.token_hex(16)


def _generate_span_id() -> str:
    """Generate a W3C compliant 64-bit span ID (16 hex chars)."""
    return secrets.token_hex(8)


@dataclass
class Span:
    """A single span in a trace.

    Represents a unit of work within a distributed trace.
    """

    span_id: str
    trace_id: str
    name: str
    parent_span_id: str | None = None
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    duration_ms: float = 0
    status: str = "running"  # running, ok, error
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def end(self, status: str = "ok", error: str | None = None):
        """Mark span as completed."""
        self.end_time = datetime.now(UTC)
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
        if error:
            self.attributes["error"] = error

    def add_event(self, name: str, attributes: dict[str, Any] | None = None):
        """Add an event to the span."""
        self.events.append(
            {
                "name": name,
                "timestamp": datetime.now(UTC).isoformat(),
                "attributes": attributes or {},
            }
        )

    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for logging/export."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


@dataclass
class TraceContext:
    """Context for a distributed trace.

    Contains the trace ID and manages the span stack.
    """

    trace_id: str
    root_span: Span | None = None
    spans: list[Span] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    _span_stack: list[Span] = field(default_factory=list)

    @classmethod
    def new(cls, name: str = "root", attributes: dict[str, Any] | None = None) -> TraceContext:
        """Create a new trace context with a root span."""
        trace_id = _generate_trace_id()
        root_span = Span(
            span_id=_generate_span_id(),
            trace_id=trace_id,
            name=name,
            attributes=attributes or {},
        )
        ctx = cls(
            trace_id=trace_id,
            root_span=root_span,
            spans=[root_span],
            attributes=attributes or {},
        )
        ctx._span_stack.append(root_span)
        return ctx

    @classmethod
    def from_parent(
        cls,
        trace_id: str,
        parent_span_id: str,
        name: str = "child",
        attributes: dict[str, Any] | None = None,
    ) -> TraceContext:
        """Create a trace context continuing from a parent trace."""
        root_span = Span(
            span_id=_generate_span_id(),
            trace_id=trace_id,
            name=name,
            parent_span_id=parent_span_id,
            attributes=attributes or {},
        )
        ctx = cls(
            trace_id=trace_id,
            root_span=root_span,
            spans=[root_span],
            attributes=attributes or {},
        )
        ctx._span_stack.append(root_span)
        return ctx

    @property
    def current_span(self) -> Span | None:
        """Get the current active span."""
        return self._span_stack[-1] if self._span_stack else None

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Span:
        """Start a new child span."""
        parent = self.current_span
        span = Span(
            span_id=_generate_span_id(),
            trace_id=self.trace_id,
            name=name,
            parent_span_id=parent.span_id if parent else None,
            attributes=attributes or {},
        )
        self.spans.append(span)
        self._span_stack.append(span)
        return span

    def end_span(self, status: str = "ok", error: str | None = None):
        """End the current span."""
        if self._span_stack:
            span = self._span_stack.pop()
            span.end(status, error)

    def end(self, status: str = "ok", error: str | None = None):
        """End the trace (ends root span)."""
        # End all remaining spans
        while self._span_stack:
            span = self._span_stack.pop()
            span.end(
                status if span == self.root_span else "ok",
                error if span == self.root_span else None,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "attributes": self.attributes,
            "spans": [s.to_dict() for s in self.spans],
        }

    def get_traceparent(self) -> str:
        """Generate W3C traceparent header value.

        Format: {version}-{trace_id}-{span_id}-{flags}
        Example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
        """
        span = self.current_span
        if not span:
            return f"00-{self.trace_id}-{_generate_span_id()}-01"
        return f"00-{self.trace_id}-{span.span_id}-01"


# Context variables for trace propagation
_trace_context: ContextVar[TraceContext | None] = ContextVar("trace_context", default=None)
_current_span: ContextVar[Span | None] = ContextVar("current_span", default=None)


def get_current_trace() -> TraceContext | None:
    """Get the current trace context."""
    return _trace_context.get()


def get_current_span() -> Span | None:
    """Get the current span."""
    trace = get_current_trace()
    if trace:
        return trace.current_span
    return None


def start_trace(
    name: str = "root",
    trace_id: str | None = None,
    parent_span_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> TraceContext:
    """Start a new trace or continue from parent.

    Args:
        name: Name for the root span
        trace_id: Existing trace ID to continue (for propagation)
        parent_span_id: Parent span ID (for propagation)
        attributes: Initial attributes

    Returns:
        New TraceContext
    """
    if trace_id and parent_span_id:
        ctx = TraceContext.from_parent(trace_id, parent_span_id, name, attributes)
    else:
        ctx = TraceContext.new(name, attributes)

    _trace_context.set(ctx)
    return ctx


def start_span(name: str, attributes: dict[str, Any] | None = None) -> Span | None:
    """Start a new span in the current trace.

    Returns None if no active trace.
    """
    trace = get_current_trace()
    if trace:
        return trace.start_span(name, attributes)
    return None


def end_span(status: str = "ok", error: str | None = None):
    """End the current span."""
    trace = get_current_trace()
    if trace:
        trace.end_span(status, error)


def end_trace(status: str = "ok", error: str | None = None):
    """End the current trace."""
    trace = get_current_trace()
    if trace:
        trace.end(status, error)
        _trace_context.set(None)


@contextmanager
def trace_context(
    name: str = "root",
    trace_id: str | None = None,
    parent_span_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Generator[TraceContext, None, None]:
    """Context manager for traces.

    Usage:
        with trace_context("my-operation") as trace:
            # ... do work ...
            trace.current_span.set_attribute("key", "value")
    """
    trace = start_trace(name, trace_id, parent_span_id, attributes)
    try:
        yield trace
    except Exception as e:
        trace.end("error", str(e))
        _trace_context.set(None)
        raise
    else:
        trace.end("ok")
        _trace_context.set(None)


@contextmanager
def span_context(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Span | None, None, None]:
    """Context manager for spans within a trace.

    Usage:
        with span_context("database-query") as span:
            # ... do work ...
            if span:
                span.set_attribute("rows", 42)
    """
    span = start_span(name, attributes)
    try:
        yield span
    except Exception as e:
        if span:
            end_span("error", str(e))
        raise
    else:
        if span:
            end_span("ok")


def get_trace_logging_context() -> dict[str, Any]:
    """Get trace context for logging.

    Returns dict with trace_id and span_id if available.
    """
    trace = get_current_trace()
    if not trace:
        return {}

    span = trace.current_span
    context = {"trace_id": trace.trace_id}
    if span:
        context["span_id"] = span.span_id
        if span.parent_span_id:
            context["parent_span_id"] = span.parent_span_id
    return context
