"""FastAPI middleware for distributed tracing.

Automatically:
- Extracts trace context from incoming requests
- Creates new traces for requests without context
- Adds trace headers to responses
- Logs trace information
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from animus_forge.tracing.context import (
    end_trace,
    get_current_trace,
    get_trace_logging_context,
    start_trace,
)
from animus_forge.tracing.propagation import (
    TRACEPARENT_HEADER,
    extract_trace_context,
)

logger = logging.getLogger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware that adds distributed tracing to all requests.

    Features:
    - Extracts incoming trace context from headers
    - Creates new trace if none provided
    - Adds trace/span IDs to response headers
    - Logs request with trace context

    Usage:
        app.add_middleware(TracingMiddleware)
    """

    def __init__(
        self,
        app,
        service_name: str = "gorgon",
        exclude_paths: list[str] | None = None,
    ):
        """Initialize tracing middleware.

        Args:
            app: FastAPI application
            service_name: Name of this service for tracing
            exclude_paths: Paths to exclude from tracing (e.g., health checks)
        """
        super().__init__(app)
        self.service_name = service_name
        self.exclude_paths = exclude_paths or [
            "/health",
            "/health/live",
            "/health/ready",
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with tracing."""
        path = request.url.path

        # Skip tracing for excluded paths
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)

        # Extract trace context from headers
        headers = dict(request.headers)
        propagated = extract_trace_context(headers)

        # Start trace (continue from parent or create new)
        if propagated:
            trace = start_trace(
                name=f"{request.method} {path}",
                trace_id=propagated.trace_id,
                parent_span_id=propagated.parent_span_id,
                attributes={
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.route": path,
                    "service.name": self.service_name,
                },
            )
        else:
            trace = start_trace(
                name=f"{request.method} {path}",
                attributes={
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.route": path,
                    "service.name": self.service_name,
                },
            )

        # Add client info
        if request.client:
            trace.root_span.set_attribute("http.client_ip", request.client.host)

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as e:
            # Record error and re-raise
            duration_ms = (time.perf_counter() - start_time) * 1000
            trace.root_span.set_attribute("http.status_code", 500)
            trace.root_span.set_attribute("http.duration_ms", duration_ms)
            end_trace("error", str(e))

            logger.error(
                f"Request failed: {request.method} {path}",
                extra={
                    **get_trace_logging_context(),
                    "http.method": request.method,
                    "http.path": path,
                    "http.status_code": 500,
                    "duration_ms": round(duration_ms, 2),
                    "error": str(e),
                },
            )
            raise

        # Record success
        duration_ms = (time.perf_counter() - start_time) * 1000
        trace.root_span.set_attribute("http.status_code", response.status_code)
        trace.root_span.set_attribute("http.duration_ms", duration_ms)

        status = "ok" if response.status_code < 400 else "error"
        end_trace(status)

        # Add trace headers to response
        response.headers["X-Trace-ID"] = trace.trace_id
        if trace.root_span:
            response.headers["X-Span-ID"] = trace.root_span.span_id
            response.headers[TRACEPARENT_HEADER] = trace.get_traceparent()

        # Log request completion with trace context
        log_level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            log_level,
            f"{request.method} {path} - {response.status_code} in {duration_ms:.1f}ms",
            extra={
                **get_trace_logging_context(),
                "http.method": request.method,
                "http.path": path,
                "http.status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        return response


def trace_workflow_step(step_id: str, step_type: str, action: str):
    """Decorator to trace workflow step execution.

    Usage:
        @trace_workflow_step("step_1", "openai", "generate")
        def execute_step(...):
            ...
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            trace = get_current_trace()
            if trace:
                trace.start_span(
                    f"{step_type}:{action}",
                    attributes={
                        "step.id": step_id,
                        "step.type": step_type,
                        "step.action": action,
                    },
                )

            try:
                result = func(*args, **kwargs)
                if trace:
                    trace.end_span("ok")
                return result
            except Exception as e:
                if trace:
                    trace.end_span("error", str(e))
                raise

        return wrapper

    return decorator


async def trace_async_workflow_step(step_id: str, step_type: str, action: str):
    """Async decorator to trace workflow step execution.

    Usage:
        @trace_async_workflow_step("step_1", "openai", "generate")
        async def execute_step(...):
            ...
    """

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            trace = get_current_trace()
            if trace:
                trace.start_span(
                    f"{step_type}:{action}",
                    attributes={
                        "step.id": step_id,
                        "step.type": step_type,
                        "step.action": action,
                    },
                )

            try:
                result = await func(*args, **kwargs)
                if trace:
                    trace.end_span("ok")
                return result
            except Exception as e:
                if trace:
                    trace.end_span("error", str(e))
                raise

        return wrapper

    return decorator
