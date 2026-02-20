"""Tests for distributed tracing module."""

import pytest

from animus_forge.tracing.context import (
    Span,
    TraceContext,
    _generate_span_id,
    _generate_trace_id,
    end_span,
    end_trace,
    get_current_span,
    get_current_trace,
    get_trace_logging_context,
    span_context,
    start_span,
    start_trace,
    trace_context,
)
from animus_forge.tracing.propagation import (
    TRACEPARENT_HEADER,
    TRACESTATE_HEADER,
    add_gorgon_tracestate,
    extract_trace_context,
    format_traceparent,
    inject_trace_headers,
    parse_gorgon_tracestate,
    parse_traceparent,
)

# =============================================================================
# ID Generation Tests
# =============================================================================


class TestIdGeneration:
    """Tests for trace and span ID generation."""

    def test_generate_trace_id_length(self):
        """Test trace ID is 32 hex characters."""
        trace_id = _generate_trace_id()
        assert len(trace_id) == 32
        assert all(c in "0123456789abcdef" for c in trace_id)

    def test_generate_span_id_length(self):
        """Test span ID is 16 hex characters."""
        span_id = _generate_span_id()
        assert len(span_id) == 16
        assert all(c in "0123456789abcdef" for c in span_id)

    def test_ids_are_unique(self):
        """Test that generated IDs are unique."""
        trace_ids = {_generate_trace_id() for _ in range(100)}
        span_ids = {_generate_span_id() for _ in range(100)}
        assert len(trace_ids) == 100
        assert len(span_ids) == 100


# =============================================================================
# Span Tests
# =============================================================================


class TestSpan:
    """Tests for Span class."""

    def test_create_span(self):
        """Test creating a span."""
        span = Span(
            span_id="abc123",
            trace_id="def456",
            name="test-span",
        )
        assert span.span_id == "abc123"
        assert span.trace_id == "def456"
        assert span.name == "test-span"
        assert span.status == "running"
        assert span.parent_span_id is None

    def test_span_with_parent(self):
        """Test creating span with parent."""
        span = Span(
            span_id="child",
            trace_id="trace1",
            name="child-span",
            parent_span_id="parent",
        )
        assert span.parent_span_id == "parent"

    def test_span_end(self):
        """Test ending a span."""
        span = Span(
            span_id="test",
            trace_id="trace1",
            name="test",
        )
        span.end()

        assert span.status == "ok"
        assert span.end_time is not None
        assert span.duration_ms >= 0

    def test_span_end_with_error(self):
        """Test ending a span with error."""
        span = Span(span_id="test", trace_id="trace1", name="test")
        span.end(status="error", error="Something failed")

        assert span.status == "error"
        assert span.attributes["error"] == "Something failed"

    def test_span_add_event(self):
        """Test adding events to span."""
        span = Span(span_id="test", trace_id="trace1", name="test")
        span.add_event("query_start", {"sql": "SELECT *"})

        assert len(span.events) == 1
        assert span.events[0]["name"] == "query_start"
        assert span.events[0]["attributes"]["sql"] == "SELECT *"

    def test_span_set_attribute(self):
        """Test setting span attributes."""
        span = Span(span_id="test", trace_id="trace1", name="test")
        span.set_attribute("user_id", "123")
        span.set_attribute("count", 42)

        assert span.attributes["user_id"] == "123"
        assert span.attributes["count"] == 42

    def test_span_to_dict(self):
        """Test span serialization."""
        span = Span(span_id="test", trace_id="trace1", name="test")
        span.set_attribute("key", "value")
        span.end()

        data = span.to_dict()

        assert data["span_id"] == "test"
        assert data["trace_id"] == "trace1"
        assert data["name"] == "test"
        assert data["status"] == "ok"
        assert data["attributes"]["key"] == "value"
        assert data["end_time"] is not None


# =============================================================================
# TraceContext Tests
# =============================================================================


class TestTraceContext:
    """Tests for TraceContext class."""

    def test_create_new_trace(self):
        """Test creating a new trace."""
        ctx = TraceContext.new("root-operation")

        assert len(ctx.trace_id) == 32
        assert ctx.root_span is not None
        assert ctx.root_span.name == "root-operation"
        assert len(ctx.spans) == 1

    def test_create_from_parent(self):
        """Test creating trace from parent context."""
        ctx = TraceContext.from_parent(
            trace_id="parent_trace_id_32_chars_long__",
            parent_span_id="parent_span_16ch",
            name="child-operation",
        )

        assert ctx.trace_id == "parent_trace_id_32_chars_long__"
        assert ctx.root_span.parent_span_id == "parent_span_16ch"

    def test_current_span(self):
        """Test getting current span."""
        ctx = TraceContext.new("root")
        assert ctx.current_span == ctx.root_span

    def test_start_child_span(self):
        """Test starting child spans."""
        ctx = TraceContext.new("root")
        child = ctx.start_span("child-operation")

        assert child.parent_span_id == ctx.root_span.span_id
        assert ctx.current_span == child
        assert len(ctx.spans) == 2

    def test_end_span(self):
        """Test ending spans."""
        ctx = TraceContext.new("root")
        ctx.start_span("child")
        ctx.end_span()

        # Should be back to root
        assert ctx.current_span == ctx.root_span

    def test_nested_spans(self):
        """Test nested span hierarchy."""
        ctx = TraceContext.new("root")
        span1 = ctx.start_span("level1")
        span2 = ctx.start_span("level2")

        assert span2.parent_span_id == span1.span_id
        assert ctx.current_span == span2

        ctx.end_span()
        assert ctx.current_span == span1

        ctx.end_span()
        assert ctx.current_span == ctx.root_span

    def test_end_trace(self):
        """Test ending entire trace."""
        ctx = TraceContext.new("root")
        ctx.start_span("child")
        ctx.end()

        # All spans should be ended
        for span in ctx.spans:
            assert span.status != "running"

    def test_to_dict(self):
        """Test trace serialization."""
        ctx = TraceContext.new("root")
        ctx.start_span("child")
        ctx.end()

        data = ctx.to_dict()

        assert data["trace_id"] == ctx.trace_id
        assert len(data["spans"]) == 2

    def test_get_traceparent(self):
        """Test generating traceparent header."""
        ctx = TraceContext.new("root")
        traceparent = ctx.get_traceparent()

        parts = traceparent.split("-")
        assert len(parts) == 4
        assert parts[0] == "00"  # version
        assert parts[1] == ctx.trace_id
        assert len(parts[2]) == 16  # span_id
        assert parts[3] == "01"  # sampled


# =============================================================================
# Context Variable Tests
# =============================================================================


class TestContextVariables:
    """Tests for context variable management."""

    def setup_method(self):
        """Reset context before each test."""
        end_trace()

    def teardown_method(self):
        """Clean up context after each test."""
        end_trace()

    def test_start_and_get_trace(self):
        """Test starting and getting trace."""
        trace = start_trace("test")
        current = get_current_trace()

        assert current is trace
        assert current.root_span.name == "test"

    def test_get_current_span(self):
        """Test getting current span."""
        start_trace("test")
        span = get_current_span()

        assert span is not None
        assert span.name == "test"

    def test_start_span_in_trace(self):
        """Test starting span within trace."""
        start_trace("root")
        span = start_span("child")

        assert span is not None
        assert span.name == "child"
        assert get_current_span() == span

    def test_start_span_no_trace(self):
        """Test starting span without trace returns None."""
        span = start_span("orphan")
        assert span is None

    def test_end_span_in_trace(self):
        """Test ending span."""
        start_trace("root")
        start_span("child")
        end_span()

        # Should be back at root
        current = get_current_span()
        assert current.name == "root"

    def test_end_trace_clears_context(self):
        """Test ending trace clears context."""
        start_trace("test")
        end_trace()

        assert get_current_trace() is None

    def test_trace_context_manager(self):
        """Test trace context manager."""
        with trace_context("test-operation") as trace:
            assert get_current_trace() is trace
            trace.root_span.set_attribute("key", "value")

        # Context should be cleared
        assert get_current_trace() is None

    def test_trace_context_manager_with_error(self):
        """Test trace context manager handles errors."""
        with pytest.raises(ValueError):
            with trace_context("test"):
                raise ValueError("Test error")

        # Trace should still be ended
        assert get_current_trace() is None

    def test_span_context_manager(self):
        """Test span context manager."""
        with trace_context("root"):
            with span_context("child") as span:
                assert span is not None
                span.set_attribute("key", "value")

            # Back to root
            assert get_current_span().name == "root"

    def test_get_trace_logging_context(self):
        """Test getting logging context."""
        with trace_context("test") as trace:
            ctx = get_trace_logging_context()

            assert ctx["trace_id"] == trace.trace_id
            assert "span_id" in ctx

    def test_get_trace_logging_context_empty(self):
        """Test logging context when no trace."""
        ctx = get_trace_logging_context()
        assert ctx == {}


# =============================================================================
# Propagation Tests
# =============================================================================


class TestTraceparentParsing:
    """Tests for traceparent header parsing."""

    def test_parse_valid_traceparent(self):
        """Test parsing valid traceparent."""
        result = parse_traceparent("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")

        assert result is not None
        version, trace_id, span_id, flags = result
        assert version == "00"
        assert trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert span_id == "b7ad6b7169203331"
        assert flags == "01"

    def test_parse_invalid_format(self):
        """Test parsing invalid format."""
        assert parse_traceparent("invalid") is None
        assert parse_traceparent("00-abc-def-01") is None

    def test_parse_all_zero_trace_id(self):
        """Test that all-zero trace ID is rejected."""
        result = parse_traceparent("00-00000000000000000000000000000000-b7ad6b7169203331-01")
        assert result is None

    def test_parse_all_zero_span_id(self):
        """Test that all-zero span ID is rejected."""
        result = parse_traceparent("00-0af7651916cd43dd8448eb211c80319c-0000000000000000-01")
        assert result is None

    def test_parse_case_insensitive(self):
        """Test parsing is case insensitive."""
        result = parse_traceparent("00-0AF7651916CD43DD8448EB211C80319C-B7AD6B7169203331-01")
        assert result is not None


class TestExtractTraceContext:
    """Tests for extracting trace context from headers."""

    def test_extract_from_traceparent(self):
        """Test extracting from traceparent header."""
        headers = {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}
        result = extract_trace_context(headers)

        assert result is not None
        assert result.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert result.parent_span_id == "b7ad6b7169203331"
        assert result.sampled is True

    def test_extract_with_tracestate(self):
        """Test extracting with tracestate."""
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "tracestate": "gorgon=wf:abc",
        }
        result = extract_trace_context(headers)

        assert result is not None
        assert result.tracestate == "gorgon=wf:abc"

    def test_extract_not_sampled(self):
        """Test extracting with sampled=false."""
        headers = {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00"}
        result = extract_trace_context(headers)

        assert result is not None
        assert result.sampled is False

    def test_extract_missing_header(self):
        """Test extracting with no traceparent."""
        headers = {"content-type": "application/json"}
        result = extract_trace_context(headers)

        assert result is None

    def test_extract_case_insensitive(self):
        """Test case-insensitive header matching."""
        headers = {"TraceParent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}
        result = extract_trace_context(headers)

        assert result is not None

    def test_extract_from_x_trace_id(self):
        """Test extracting from x-trace-id alias."""
        headers = {"x-trace-id": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}
        result = extract_trace_context(headers)

        assert result is not None


class TestFormatTraceparent:
    """Tests for formatting traceparent header."""

    def test_format_sampled(self):
        """Test formatting with sampled=true."""
        result = format_traceparent(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            sampled=True,
        )

        assert result == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

    def test_format_not_sampled(self):
        """Test formatting with sampled=false."""
        result = format_traceparent(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            sampled=False,
        )

        assert result == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00"


class TestInjectTraceHeaders:
    """Tests for injecting trace headers."""

    def test_inject_basic(self):
        """Test basic header injection."""
        headers = inject_trace_headers(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
        )

        assert TRACEPARENT_HEADER in headers
        assert "0af7651916cd43dd8448eb211c80319c" in headers[TRACEPARENT_HEADER]

    def test_inject_with_tracestate(self):
        """Test injection with tracestate."""
        headers = inject_trace_headers(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            tracestate="gorgon=wf:test",
        )

        assert TRACESTATE_HEADER in headers
        assert headers[TRACESTATE_HEADER] == "gorgon=wf:test"


class TestGorgonTracestate:
    """Tests for Gorgon-specific tracestate handling."""

    def test_add_gorgon_tracestate_new(self):
        """Test adding Gorgon tracestate to empty."""
        result = add_gorgon_tracestate(
            existing=None,
            workflow_id="wf123",
            step_id="step1",
        )

        assert result == "gorgon=wf:wf123;st:step1"

    def test_add_gorgon_tracestate_existing(self):
        """Test adding to existing tracestate."""
        result = add_gorgon_tracestate(
            existing="other=value",
            workflow_id="wf123",
        )

        assert result.startswith("gorgon=wf:wf123")
        assert "other=value" in result

    def test_add_gorgon_tracestate_replaces_existing(self):
        """Test that existing gorgon entry is replaced."""
        result = add_gorgon_tracestate(
            existing="gorgon=old,other=value",
            workflow_id="new",
        )

        # Should have new gorgon entry, not old
        assert "gorgon=wf:new" in result
        assert "gorgon=old" not in result

    def test_parse_gorgon_tracestate(self):
        """Test parsing Gorgon tracestate."""
        result = parse_gorgon_tracestate("gorgon=wf:workflow1;st:step2,other=x")

        assert result["workflow_id"] == "workflow1"
        assert result["step_id"] == "step2"

    def test_parse_gorgon_tracestate_partial(self):
        """Test parsing partial Gorgon tracestate."""
        result = parse_gorgon_tracestate("gorgon=wf:workflow1")

        assert result["workflow_id"] == "workflow1"
        assert "step_id" not in result

    def test_parse_gorgon_tracestate_missing(self):
        """Test parsing when Gorgon entry not present."""
        result = parse_gorgon_tracestate("other=value")

        assert result == {}


# =============================================================================
# Integration Tests
# =============================================================================


class TestTracingIntegration:
    """Integration tests for tracing."""

    def setup_method(self):
        """Reset context before each test."""
        end_trace()

    def teardown_method(self):
        """Clean up after test."""
        end_trace()

    def test_full_trace_flow(self):
        """Test complete trace flow."""
        # Start trace
        with trace_context("api-request", attributes={"user": "test"}) as trace:
            # First span
            with span_context("database-query") as db_span:
                db_span.set_attribute("sql", "SELECT *")
                db_span.add_event("query_sent")

            # Second span
            with span_context("external-api") as api_span:
                api_span.set_attribute("url", "https://api.example.com")

        # Verify trace structure
        assert len(trace.spans) == 3  # root + 2 children
        assert trace.root_span.name == "api-request"

        # Verify spans ended
        for span in trace.spans:
            assert span.status == "ok"
            assert span.end_time is not None

    def test_trace_propagation_simulation(self):
        """Test simulating trace propagation across services."""
        # Service A starts trace
        with trace_context("service-a") as trace_a:
            traceparent = trace_a.get_traceparent()

        # Simulate extracting in Service B
        headers = {"traceparent": traceparent}
        propagated = extract_trace_context(headers)

        # Service B continues trace
        with trace_context(
            "service-b",
            trace_id=propagated.trace_id,
            parent_span_id=propagated.parent_span_id,
        ) as trace_b:
            pass

        # Same trace ID
        assert trace_b.trace_id == trace_a.trace_id

        # Service B's root span has parent from Service A
        assert trace_b.root_span.parent_span_id == trace_a.root_span.span_id

    def test_error_handling_in_spans(self):
        """Test that errors are properly recorded."""
        with pytest.raises(RuntimeError):
            with trace_context("error-test"):
                with span_context("failing-operation"):
                    raise RuntimeError("Test error")

        # The outer trace should have captured the error
        # (Note: in real usage, the trace would be logged/exported before this)


# =============================================================================
# Middleware Tests
# =============================================================================


class TestTracingMiddleware:
    """Tests for TracingMiddleware."""

    def setup_method(self):
        """Reset context before each test."""
        end_trace()

    def teardown_method(self):
        """Clean up after test."""
        end_trace()

    @pytest.fixture
    def app(self):
        """Create a test FastAPI app with tracing middleware."""
        from fastapi import FastAPI

        from animus_forge.tracing.middleware import TracingMiddleware

        app = FastAPI()
        app.add_middleware(TracingMiddleware, service_name="test-service")

        @app.get("/")
        async def root():
            return {"message": "ok"}

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error")

        return app

    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        from fastapi.testclient import TestClient

        return TestClient(app, raise_server_exceptions=False)

    def test_middleware_adds_trace_headers(self, client):
        """Test that middleware adds trace headers to response."""
        response = client.get("/")

        assert response.status_code == 200
        assert "X-Trace-ID" in response.headers
        assert "X-Span-ID" in response.headers
        assert "traceparent" in response.headers

    def test_middleware_excludes_health_paths(self, client):
        """Test that health paths are excluded from tracing."""
        response = client.get("/health")

        assert response.status_code == 200
        # Health paths should not have trace headers
        assert "X-Trace-ID" not in response.headers

    def test_middleware_propagates_incoming_trace(self, client):
        """Test that middleware continues trace from incoming traceparent header."""
        incoming_trace_id = "0af7651916cd43dd8448eb211c80319c"
        incoming_span_id = "b7ad6b7169203331"
        traceparent = f"00-{incoming_trace_id}-{incoming_span_id}-01"

        response = client.get("/", headers={"traceparent": traceparent})

        assert response.status_code == 200
        # Response should have the same trace ID (propagated)
        assert response.headers["X-Trace-ID"] == incoming_trace_id

    def test_middleware_handles_exception(self, client):
        """Test that middleware handles exceptions in request handler."""
        response = client.get("/error")

        # Should return 500 error
        assert response.status_code == 500

    def test_middleware_logs_error_on_exception(self, client, caplog):
        """Test that middleware logs errors when exception occurs."""
        import logging

        with caplog.at_level(logging.ERROR):
            client.get("/error")

        # Should have logged the error
        assert any("Request failed" in record.message for record in caplog.records)


# =============================================================================
# Workflow Step Decorator Tests
# =============================================================================


class TestTraceWorkflowStepDecorator:
    """Tests for trace_workflow_step decorator."""

    def setup_method(self):
        """Reset context before each test."""
        end_trace()

    def teardown_method(self):
        """Clean up after test."""
        end_trace()

    def test_decorator_traces_successful_call(self):
        """Test decorator traces successful function call."""
        from animus_forge.tracing.middleware import trace_workflow_step

        @trace_workflow_step("step_1", "openai", "generate")
        def my_step():
            return "result"

        with trace_context("test-workflow") as trace:
            result = my_step()

        assert result == "result"
        # Should have root span + step span
        assert len(trace.spans) == 2
        step_span = trace.spans[1]
        assert step_span.name == "openai:generate"
        assert step_span.attributes["step.id"] == "step_1"
        assert step_span.attributes["step.type"] == "openai"
        assert step_span.attributes["step.action"] == "generate"
        assert step_span.status == "ok"

    def test_decorator_traces_error(self):
        """Test decorator traces function that raises error."""
        from animus_forge.tracing.middleware import trace_workflow_step

        @trace_workflow_step("step_err", "claude", "analyze")
        def failing_step():
            raise ValueError("Step failed")

        with pytest.raises(ValueError, match="Step failed"):
            with trace_context("test-workflow") as trace:
                failing_step()

        # Span should be marked as error
        step_span = trace.spans[1]
        assert step_span.status == "error"

    def test_decorator_without_active_trace(self):
        """Test decorator works without active trace (no-op)."""
        from animus_forge.tracing.middleware import trace_workflow_step

        @trace_workflow_step("step_no_trace", "test", "action")
        def standalone_step():
            return "standalone"

        # Should work fine without trace context
        result = standalone_step()
        assert result == "standalone"


class TestTraceAsyncWorkflowStepDecorator:
    """Tests for trace_async_workflow_step decorator."""

    def setup_method(self):
        """Reset context before each test."""
        end_trace()

    def teardown_method(self):
        """Clean up after test."""
        end_trace()

    @pytest.mark.asyncio
    async def test_async_decorator_traces_successful_call(self):
        """Test async decorator traces successful function call."""
        from animus_forge.tracing.middleware import trace_async_workflow_step

        decorator = await trace_async_workflow_step("async_step", "anthropic", "chat")

        @decorator
        async def my_async_step():
            return "async_result"

        with trace_context("async-workflow") as trace:
            result = await my_async_step()

        assert result == "async_result"
        # Should have root span + step span
        assert len(trace.spans) == 2
        step_span = trace.spans[1]
        assert step_span.name == "anthropic:chat"
        assert step_span.attributes["step.id"] == "async_step"
        assert step_span.status == "ok"

    @pytest.mark.asyncio
    async def test_async_decorator_traces_error(self):
        """Test async decorator traces function that raises error."""
        from animus_forge.tracing.middleware import trace_async_workflow_step

        decorator = await trace_async_workflow_step("async_err", "github", "create_pr")

        @decorator
        async def failing_async_step():
            raise RuntimeError("Async step failed")

        with pytest.raises(RuntimeError, match="Async step failed"):
            with trace_context("async-workflow") as trace:
                await failing_async_step()

        # Span should be marked as error
        step_span = trace.spans[1]
        assert step_span.status == "error"

    @pytest.mark.asyncio
    async def test_async_decorator_without_active_trace(self):
        """Test async decorator works without active trace (no-op)."""
        from animus_forge.tracing.middleware import trace_async_workflow_step

        decorator = await trace_async_workflow_step("async_no_trace", "test", "action")

        @decorator
        async def standalone_async_step():
            return "standalone_async"

        # Should work fine without trace context
        result = await standalone_async_step()
        assert result == "standalone_async"
