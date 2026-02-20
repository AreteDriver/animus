"""Tests for WebSocket real-time execution updates."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocket

from animus_forge.websocket import (
    Broadcaster,
    ConnectedMessage,
    Connection,
    ConnectionManager,
    ErrorMessage,
    ExecutionLogMessage,
    ExecutionMetricsMessage,
    ExecutionStatusMessage,
    MessageType,
    PongMessage,
)

# =============================================================================
# Message Model Tests
# =============================================================================


class TestMessageModels:
    """Tests for WebSocket message models."""

    def test_message_type_values(self) -> None:
        """Test MessageType enum values."""
        assert MessageType.SUBSCRIBE == "subscribe"
        assert MessageType.UNSUBSCRIBE == "unsubscribe"
        assert MessageType.PING == "ping"
        assert MessageType.CONNECTED == "connected"
        assert MessageType.EXECUTION_STATUS == "execution_status"
        assert MessageType.EXECUTION_LOG == "execution_log"
        assert MessageType.EXECUTION_METRICS == "execution_metrics"
        assert MessageType.PONG == "pong"
        assert MessageType.ERROR == "error"

    def test_connected_message(self) -> None:
        """Test ConnectedMessage creation."""
        msg = ConnectedMessage(connection_id="abc123")
        assert msg.type == "connected"
        assert msg.connection_id == "abc123"
        assert msg.server_time is not None

    def test_execution_status_message(self) -> None:
        """Test ExecutionStatusMessage creation."""
        msg = ExecutionStatusMessage(
            execution_id="exec-1",
            status="running",
            progress=50,
            current_step="step-2",
        )
        assert msg.type == "execution_status"
        assert msg.execution_id == "exec-1"
        assert msg.status == "running"
        assert msg.progress == 50
        assert msg.current_step == "step-2"

    def test_execution_log_message(self) -> None:
        """Test ExecutionLogMessage creation."""
        msg = ExecutionLogMessage(
            execution_id="exec-1",
            log={"level": "info", "message": "Test log"},
        )
        assert msg.type == "execution_log"
        assert msg.execution_id == "exec-1"
        assert msg.log["level"] == "info"

    def test_execution_metrics_message(self) -> None:
        """Test ExecutionMetricsMessage creation."""
        msg = ExecutionMetricsMessage(
            execution_id="exec-1",
            metrics={"total_tokens": 100, "total_cost_cents": 5},
        )
        assert msg.type == "execution_metrics"
        assert msg.execution_id == "exec-1"
        assert msg.metrics["total_tokens"] == 100

    def test_pong_message(self) -> None:
        """Test PongMessage creation."""
        msg = PongMessage(timestamp=1234567890)
        assert msg.type == "pong"
        assert msg.timestamp == 1234567890

    def test_error_message(self) -> None:
        """Test ErrorMessage creation."""
        msg = ErrorMessage(
            code="INVALID_JSON",
            message="Could not parse message",
        )
        assert msg.type == "error"
        assert msg.code == "INVALID_JSON"
        assert msg.message == "Could not parse message"


# =============================================================================
# Connection Tests
# =============================================================================


class TestConnection:
    """Tests for WebSocket Connection class."""

    @pytest.mark.asyncio
    async def test_connection_creation(self) -> None:
        """Test Connection creation with defaults."""
        mock_ws = AsyncMock(spec=WebSocket)
        conn = Connection(id="conn-1", websocket=mock_ws)

        assert conn.id == "conn-1"
        assert conn.websocket == mock_ws
        assert len(conn.subscriptions) == 0
        assert conn.connected_at > 0

    @pytest.mark.asyncio
    async def test_connection_send_success(self) -> None:
        """Test successful message send."""
        mock_ws = AsyncMock(spec=WebSocket)
        conn = Connection(id="conn-1", websocket=mock_ws)

        msg = PongMessage(timestamp=123)
        result = await conn.send(msg)

        assert result is True
        mock_ws.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_send_failure(self) -> None:
        """Test message send failure."""
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.send_json.side_effect = Exception("Connection closed")
        conn = Connection(id="conn-1", websocket=mock_ws)

        msg = PongMessage(timestamp=123)
        result = await conn.send(msg)

        assert result is False


# =============================================================================
# ConnectionManager Tests
# =============================================================================


class TestConnectionManager:
    """Tests for WebSocket ConnectionManager."""

    @pytest.mark.asyncio
    async def test_connect(self) -> None:
        """Test WebSocket connection."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        conn = await manager.connect(mock_ws)

        assert conn.id is not None
        assert manager.connection_count == 1
        mock_ws.accept.assert_called_once()
        mock_ws.send_json.assert_called_once()  # Connected message

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """Test WebSocket disconnection."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        conn = await manager.connect(mock_ws)
        await manager.disconnect(conn.id)

        assert manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_subscribe(self) -> None:
        """Test subscribing to execution updates."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        conn = await manager.connect(mock_ws)
        subscribed = await manager.subscribe(conn.id, ["exec-1", "exec-2"])

        assert subscribed == ["exec-1", "exec-2"]
        assert "exec-1" in conn.subscriptions
        assert "exec-2" in conn.subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        """Test unsubscribing from execution updates."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        conn = await manager.connect(mock_ws)
        await manager.subscribe(conn.id, ["exec-1", "exec-2"])
        unsubscribed = await manager.unsubscribe(conn.id, ["exec-1"])

        assert unsubscribed == ["exec-1"]
        assert "exec-1" not in conn.subscriptions
        assert "exec-2" in conn.subscriptions

    @pytest.mark.asyncio
    async def test_broadcast_to_execution(self) -> None:
        """Test broadcasting to execution subscribers."""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)

        conn1 = await manager.connect(mock_ws1)
        conn2 = await manager.connect(mock_ws2)

        await manager.subscribe(conn1.id, ["exec-1"])
        await manager.subscribe(conn2.id, ["exec-1"])

        msg = ExecutionStatusMessage(
            execution_id="exec-1",
            status="running",
            progress=50,
        )
        sent = await manager.broadcast_to_execution("exec-1", msg)

        assert sent == 2

    @pytest.mark.asyncio
    async def test_handle_ping_message(self) -> None:
        """Test handling ping message."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        conn = await manager.connect(mock_ws)
        mock_ws.send_json.reset_mock()

        await manager.handle_client_message(conn, json.dumps({"type": "ping", "timestamp": 123}))

        # Should send pong
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "pong"
        assert call_args["timestamp"] == 123

    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self) -> None:
        """Test handling subscribe message."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        conn = await manager.connect(mock_ws)

        await manager.handle_client_message(
            conn, json.dumps({"type": "subscribe", "execution_ids": ["exec-1"]})
        )

        assert "exec-1" in conn.subscriptions

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self) -> None:
        """Test handling invalid JSON message."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        conn = await manager.connect(mock_ws)
        mock_ws.send_json.reset_mock()

        await manager.handle_client_message(conn, "not valid json")

        # Should send error
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "error"
        assert call_args["code"] == "INVALID_JSON"

    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(self) -> None:
        """Test handling unknown message type."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        conn = await manager.connect(mock_ws)
        mock_ws.send_json.reset_mock()

        await manager.handle_client_message(conn, json.dumps({"type": "unknown_type"}))

        # Should send error
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "error"
        assert call_args["code"] == "UNKNOWN_MESSAGE_TYPE"

    @pytest.mark.asyncio
    async def test_get_stats(self) -> None:
        """Test getting connection statistics."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        await manager.connect(mock_ws)
        stats = manager.get_stats()

        assert stats["active_connections"] == 1
        assert len(stats["connections"]) == 1

    @pytest.mark.asyncio
    async def test_cleanup_on_disconnect(self) -> None:
        """Test subscription cleanup on disconnect."""
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        conn = await manager.connect(mock_ws)
        await manager.subscribe(conn.id, ["exec-1"])

        # Verify subscription exists
        subs = await manager.get_subscriptions("exec-1")
        assert conn.id in subs

        # Disconnect
        await manager.disconnect(conn.id)

        # Subscription should be cleaned up
        subs = await manager.get_subscriptions("exec-1")
        assert conn.id not in subs


# =============================================================================
# Broadcaster Tests
# =============================================================================


class TestBroadcaster:
    """Tests for WebSocket Broadcaster."""

    @pytest.mark.asyncio
    async def test_broadcaster_start_stop(self) -> None:
        """Test broadcaster start and stop."""
        manager = ConnectionManager()
        broadcaster = Broadcaster(manager)

        loop = asyncio.get_running_loop()
        broadcaster.start(loop)

        assert broadcaster._running is True

        await broadcaster.stop()

        assert broadcaster._running is False

    @pytest.mark.asyncio
    async def test_on_status_change(self) -> None:
        """Test status change callback."""
        manager = ConnectionManager()
        broadcaster = Broadcaster(manager)

        loop = asyncio.get_running_loop()
        broadcaster.start(loop)

        mock_ws = AsyncMock(spec=WebSocket)
        conn = await manager.connect(mock_ws)
        await manager.subscribe(conn.id, ["exec-1"])
        mock_ws.send_json.reset_mock()

        # Trigger status change
        broadcaster.on_status_change(
            execution_id="exec-1",
            status="running",
            progress=50,
        )

        # Allow queue processing
        await asyncio.sleep(0.1)

        # Should have broadcast the status
        assert mock_ws.send_json.called
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "execution_status"
        assert call_args["execution_id"] == "exec-1"
        assert call_args["status"] == "running"

        await broadcaster.stop()

    @pytest.mark.asyncio
    async def test_on_log(self) -> None:
        """Test log callback."""
        manager = ConnectionManager()
        broadcaster = Broadcaster(manager)

        loop = asyncio.get_running_loop()
        broadcaster.start(loop)

        mock_ws = AsyncMock(spec=WebSocket)
        conn = await manager.connect(mock_ws)
        await manager.subscribe(conn.id, ["exec-1"])
        mock_ws.send_json.reset_mock()

        # Trigger log
        broadcaster.on_log(
            execution_id="exec-1",
            level="info",
            message="Test log",
        )

        # Allow queue processing
        await asyncio.sleep(0.1)

        # Should have broadcast the log
        assert mock_ws.send_json.called
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "execution_log"
        assert call_args["log"]["message"] == "Test log"

        await broadcaster.stop()

    @pytest.mark.asyncio
    async def test_on_metrics(self) -> None:
        """Test metrics callback."""
        manager = ConnectionManager()
        broadcaster = Broadcaster(manager)

        loop = asyncio.get_running_loop()
        broadcaster.start(loop)

        mock_ws = AsyncMock(spec=WebSocket)
        conn = await manager.connect(mock_ws)
        await manager.subscribe(conn.id, ["exec-1"])
        mock_ws.send_json.reset_mock()

        # Trigger metrics update
        broadcaster.on_metrics(
            execution_id="exec-1",
            total_tokens=100,
            total_cost_cents=5,
        )

        # Allow queue processing
        await asyncio.sleep(0.1)

        # Should have broadcast the metrics
        assert mock_ws.send_json.called
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "execution_metrics"
        assert call_args["metrics"]["total_tokens"] == 100

        await broadcaster.stop()

    @pytest.mark.asyncio
    async def test_create_execution_callback(self) -> None:
        """Test creating callback for ExecutionManager."""
        manager = ConnectionManager()
        broadcaster = Broadcaster(manager)

        loop = asyncio.get_running_loop()
        broadcaster.start(loop)

        mock_ws = AsyncMock(spec=WebSocket)
        conn = await manager.connect(mock_ws)
        await manager.subscribe(conn.id, ["exec-1"])
        mock_ws.send_json.reset_mock()

        # Get callback and call it
        callback = broadcaster.create_execution_callback()
        callback("status", "exec-1", status="completed", progress=100)

        # Allow queue processing
        await asyncio.sleep(0.1)

        # Should have broadcast
        assert mock_ws.send_json.called
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "execution_status"
        assert call_args["status"] == "completed"

        await broadcaster.stop()

    @pytest.mark.asyncio
    async def test_no_broadcast_without_subscribers(self) -> None:
        """Test no broadcast when no subscribers."""
        manager = ConnectionManager()
        broadcaster = Broadcaster(manager)

        loop = asyncio.get_running_loop()
        broadcaster.start(loop)

        mock_ws = AsyncMock(spec=WebSocket)
        await manager.connect(mock_ws)
        # Don't subscribe to exec-1
        mock_ws.send_json.reset_mock()

        # Trigger status change for unsubscribed execution
        broadcaster.on_status_change(
            execution_id="exec-1",
            status="running",
            progress=50,
        )

        # Allow queue processing
        await asyncio.sleep(0.1)

        # Should not have broadcast
        assert not mock_ws.send_json.called

        await broadcaster.stop()


# =============================================================================
# ExecutionManager Callback Integration Tests
# =============================================================================


class TestExecutionManagerCallbacks:
    """Tests for ExecutionManager callback system."""

    def test_register_callback(self) -> None:
        """Test registering a callback."""
        from animus_forge.executions import ExecutionManager

        # Mock backend
        mock_backend = MagicMock()
        mock_backend.transaction.return_value.__enter__ = MagicMock()
        mock_backend.transaction.return_value.__exit__ = MagicMock()
        mock_backend.execute.return_value = MagicMock(rowcount=1)
        mock_backend.fetchone.return_value = None

        manager = ExecutionManager(backend=mock_backend)

        callback = MagicMock()
        manager.register_callback(callback)

        assert callback in manager._callbacks

    def test_unregister_callback(self) -> None:
        """Test unregistering a callback."""
        from animus_forge.executions import ExecutionManager

        mock_backend = MagicMock()
        manager = ExecutionManager(backend=mock_backend)

        callback = MagicMock()
        manager.register_callback(callback)
        manager.unregister_callback(callback)

        assert callback not in manager._callbacks

    def test_callback_on_start_execution(self) -> None:
        """Test callback is called on start_execution."""
        from animus_forge.executions import ExecutionManager

        mock_backend = MagicMock()
        mock_backend.transaction.return_value.__enter__ = MagicMock()
        mock_backend.transaction.return_value.__exit__ = MagicMock()
        mock_backend.execute.return_value = MagicMock(rowcount=1)
        mock_backend.fetchone.return_value = None

        manager = ExecutionManager(backend=mock_backend)

        callback = MagicMock()
        manager.register_callback(callback)

        manager.start_execution("exec-1")

        # Should have been called for status change and log
        assert callback.call_count >= 1
        # Check for status callback
        calls = [c for c in callback.call_args_list if c[0][0] == "status"]
        assert len(calls) >= 1

    def test_callback_on_update_progress(self) -> None:
        """Test callback is called on update_progress."""
        from animus_forge.executions import ExecutionManager

        mock_backend = MagicMock()
        mock_backend.transaction.return_value.__enter__ = MagicMock()
        mock_backend.transaction.return_value.__exit__ = MagicMock()

        manager = ExecutionManager(backend=mock_backend)

        callback = MagicMock()
        manager.register_callback(callback)

        manager.update_progress("exec-1", 50, "step-2")

        # Should have been called with status event
        callback.assert_called()
        args = callback.call_args[0]
        assert args[0] == "status"
        assert args[1] == "exec-1"

    def test_callback_on_add_log(self) -> None:
        """Test callback is called on add_log."""
        from animus_forge.executions import ExecutionManager, LogLevel

        mock_backend = MagicMock()
        mock_backend.transaction.return_value.__enter__ = MagicMock()
        mock_backend.transaction.return_value.__exit__ = MagicMock()

        manager = ExecutionManager(backend=mock_backend)

        callback = MagicMock()
        manager.register_callback(callback)

        manager.add_log("exec-1", LogLevel.INFO, "Test message")

        # Should have been called with log event
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "log"
        assert args[1] == "exec-1"
        kwargs = callback.call_args[1]
        assert kwargs["message"] == "Test message"

    def test_callback_on_update_metrics(self) -> None:
        """Test callback is called on update_metrics."""
        from animus_forge.executions import ExecutionManager

        mock_backend = MagicMock()
        mock_backend.transaction.return_value.__enter__ = MagicMock()
        mock_backend.transaction.return_value.__exit__ = MagicMock()
        mock_backend.fetchone.return_value = {
            "execution_id": "exec-1",
            "total_tokens": 100,
            "total_cost_cents": 5,
            "duration_ms": 1000,
            "steps_completed": 2,
            "steps_failed": 0,
        }

        manager = ExecutionManager(backend=mock_backend)

        callback = MagicMock()
        manager.register_callback(callback)

        manager.update_metrics("exec-1", tokens=50)

        # Should have been called with metrics event
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "metrics"
        assert args[1] == "exec-1"
