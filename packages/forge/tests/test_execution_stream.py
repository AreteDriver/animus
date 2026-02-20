"""Tests for SSE execution streaming endpoint."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from animus_forge.executions.models import (
    Execution,
    ExecutionLog,
    ExecutionMetrics,
    ExecutionStatus,
    LogLevel,
)


def _mock_execution(
    execution_id="exec-stream",
    status=ExecutionStatus.RUNNING,
):
    """Create a mock Execution model."""
    return Execution(
        id=execution_id,
        workflow_id="wf-1",
        workflow_name="Test Workflow",
        status=status,
        created_at=datetime.now(),
        variables={},
    )


def _mock_metrics(execution_id="exec-stream"):
    return ExecutionMetrics(
        execution_id=execution_id,
        total_tokens=100,
        total_cost_cents=5,
        duration_ms=1200,
        steps_completed=2,
        steps_failed=0,
    )


@pytest.fixture(autouse=True)
def _skip_auth():
    """Bypass verify_auth for all SSE tests."""
    with patch("animus_forge.api_routes.executions.verify_auth"):
        yield


class TestSSEEndpointUnit:
    """Test SSE streaming endpoint behavior via direct calls."""

    def test_stream_returns_404_for_nonexistent(self):
        """GET /stream returns 404 for nonexistent execution."""
        mock_em = MagicMock()
        mock_em.get_execution.return_value = None

        with patch("animus_forge.api_routes.executions.state") as mock_state:
            mock_state.execution_manager = mock_em
            from animus_forge.api_routes.executions import stream_execution

            with pytest.raises(Exception) as exc_info:
                import asyncio

                asyncio.run(stream_execution("nonexistent", MagicMock()))

            # Should raise not_found (HTTPException with 404)
            assert "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()

    def test_completed_execution_sends_done_immediately(self):
        """Completed execution sends snapshot then done event."""
        mock_em = MagicMock()
        execution = _mock_execution(status=ExecutionStatus.COMPLETED)
        mock_em.get_execution.return_value = execution
        mock_em.get_logs.return_value = []
        mock_em.get_metrics.return_value = _mock_metrics()

        with patch("animus_forge.api_routes.executions.state") as mock_state:
            mock_state.execution_manager = mock_em
            import asyncio

            from animus_forge.api_routes.executions import stream_execution

            async def _run():
                mock_request = MagicMock()
                response = await stream_execution("exec-stream", mock_request, authorization=None)
                events = []
                async for chunk in response.body_iterator:
                    events.append(chunk)
                return events

            events = asyncio.run(_run())

        event_text = "".join(events)
        assert "event: snapshot" in event_text
        assert "event: done" in event_text

    def test_snapshot_includes_execution_data(self):
        """Initial snapshot contains execution state, logs, metrics."""
        mock_em = MagicMock()
        execution = _mock_execution(status=ExecutionStatus.COMPLETED)
        mock_em.get_execution.return_value = execution

        log = ExecutionLog(
            id=1,
            execution_id="exec-stream",
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test log",
        )
        mock_em.get_logs.return_value = [log]
        mock_em.get_metrics.return_value = _mock_metrics()

        with patch("animus_forge.api_routes.executions.state") as mock_state:
            mock_state.execution_manager = mock_em
            import asyncio

            from animus_forge.api_routes.executions import stream_execution

            async def _run():
                mock_request = MagicMock()
                response = await stream_execution("exec-stream", mock_request, authorization=None)
                events = []
                async for chunk in response.body_iterator:
                    events.append(chunk)
                return events

            events = asyncio.run(_run())

        event_text = "".join(events)
        snapshot_line = [
            line
            for line in event_text.split("\n")
            if line.startswith("data:") and "Test Workflow" in line
        ]
        assert len(snapshot_line) >= 1
        snapshot_data = json.loads(snapshot_line[0].replace("data: ", ""))
        assert snapshot_data["workflow_name"] == "Test Workflow"
        assert len(snapshot_data["logs"]) == 1
        assert snapshot_data["metrics"]["total_tokens"] == 100

    def test_stream_response_headers(self):
        """Response has correct SSE headers."""
        mock_em = MagicMock()
        execution = _mock_execution(status=ExecutionStatus.COMPLETED)
        mock_em.get_execution.return_value = execution
        mock_em.get_logs.return_value = []
        mock_em.get_metrics.return_value = None

        with patch("animus_forge.api_routes.executions.state") as mock_state:
            mock_state.execution_manager = mock_em
            import asyncio

            from animus_forge.api_routes.executions import stream_execution

            async def _run():
                mock_request = MagicMock()
                response = await stream_execution("exec-stream", mock_request, authorization=None)
                return response

            response = asyncio.run(_run())

        assert response.media_type == "text/event-stream"
        assert response.headers.get("Cache-Control") == "no-cache"

    def test_callback_registered_and_unregistered(self):
        """Callback is registered at start and unregistered after completion."""
        mock_em = MagicMock()
        execution = _mock_execution(status=ExecutionStatus.COMPLETED)
        mock_em.get_execution.return_value = execution
        mock_em.get_logs.return_value = []
        mock_em.get_metrics.return_value = None

        with patch("animus_forge.api_routes.executions.state") as mock_state:
            mock_state.execution_manager = mock_em
            import asyncio

            from animus_forge.api_routes.executions import stream_execution

            async def _run():
                mock_request = MagicMock()
                response = await stream_execution("exec-stream", mock_request, authorization=None)
                async for _ in response.body_iterator:
                    pass

            asyncio.run(_run())

        mock_em.register_callback.assert_called_once()
        mock_em.unregister_callback.assert_called_once()

    def test_failed_execution_sends_done(self):
        """Failed execution sends snapshot + done immediately."""
        mock_em = MagicMock()
        execution = _mock_execution(status=ExecutionStatus.FAILED)
        mock_em.get_execution.return_value = execution
        mock_em.get_logs.return_value = []
        mock_em.get_metrics.return_value = None

        with patch("animus_forge.api_routes.executions.state") as mock_state:
            mock_state.execution_manager = mock_em
            import asyncio

            from animus_forge.api_routes.executions import stream_execution

            async def _run():
                mock_request = MagicMock()
                response = await stream_execution("exec-stream", mock_request, authorization=None)
                events = []
                async for chunk in response.body_iterator:
                    events.append(chunk)
                return events

            events = asyncio.run(_run())

        event_text = "".join(events)
        assert "event: done" in event_text


class TestExecutionManagerUnregister:
    """Verify unregister_callback works correctly."""

    def test_unregister_removes_callback(self):
        """unregister_callback removes the callback from the list."""
        from animus_forge.executions.manager import ExecutionManager

        mock_backend = MagicMock()
        em = ExecutionManager(backend=mock_backend)

        cb = MagicMock()
        em.register_callback(cb)
        assert cb in em._callbacks

        em.unregister_callback(cb)
        assert cb not in em._callbacks

    def test_unregister_nonexistent_is_noop(self):
        """unregister_callback for non-registered callback is a no-op."""
        from animus_forge.executions.manager import ExecutionManager

        mock_backend = MagicMock()
        em = ExecutionManager(backend=mock_backend)

        cb = MagicMock()
        em.unregister_callback(cb)  # Should not raise
        assert len(em._callbacks) == 0
