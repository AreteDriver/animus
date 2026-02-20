"""WebSocket connection manager for handling client connections."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

from .messages import (
    ConnectedMessage,
    ErrorMessage,
    PongMessage,
)

if TYPE_CHECKING:
    from .messages import OutboundMessage

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    """Represents a WebSocket client connection."""

    id: str
    websocket: WebSocket
    subscriptions: set[str] = field(default_factory=set)
    connected_at: float = field(default_factory=time.time)
    last_ping: float = field(default_factory=time.time)

    async def send(self, message: OutboundMessage) -> bool:
        """Send a message to this connection.

        Returns:
            True if sent successfully, False otherwise.
        """
        try:
            await self.websocket.send_json(message.model_dump())
            return True
        except Exception as e:
            logger.debug(f"Failed to send to {self.id}: {e}")
            return False


class ConnectionManager:
    """Manages WebSocket connections and subscriptions."""

    def __init__(self):
        """Initialize the connection manager."""
        self._connections: dict[str, Connection] = {}
        self._lock = asyncio.Lock()
        # Map execution_id -> set of connection_ids subscribed to it
        self._execution_subscribers: dict[str, set[str]] = {}

    @property
    def connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self._connections)

    async def connect(self, websocket: WebSocket) -> Connection:
        """Accept a WebSocket connection and register it.

        Args:
            websocket: The WebSocket to accept.

        Returns:
            The created Connection object.
        """
        await websocket.accept()
        connection_id = str(uuid.uuid4())[:8]

        connection = Connection(id=connection_id, websocket=websocket)

        async with self._lock:
            self._connections[connection_id] = connection

        # Send connected message
        await connection.send(ConnectedMessage(connection_id=connection_id))

        logger.info(f"WebSocket connected: {connection_id}")
        return connection

    async def disconnect(self, connection_id: str) -> None:
        """Remove a connection and clean up subscriptions.

        Args:
            connection_id: The connection ID to remove.
        """
        async with self._lock:
            connection = self._connections.pop(connection_id, None)
            if connection:
                # Clean up subscriptions
                for execution_id in connection.subscriptions:
                    subs = self._execution_subscribers.get(execution_id)
                    if subs:
                        subs.discard(connection_id)
                        if not subs:
                            del self._execution_subscribers[execution_id]

        logger.info(f"WebSocket disconnected: {connection_id}")

    async def subscribe(self, connection_id: str, execution_ids: list[str]) -> list[str]:
        """Subscribe a connection to execution updates.

        Args:
            connection_id: The connection ID.
            execution_ids: List of execution IDs to subscribe to.

        Returns:
            List of successfully subscribed execution IDs.
        """
        subscribed = []

        async with self._lock:
            connection = self._connections.get(connection_id)
            if not connection:
                return subscribed

            for execution_id in execution_ids:
                connection.subscriptions.add(execution_id)
                if execution_id not in self._execution_subscribers:
                    self._execution_subscribers[execution_id] = set()
                self._execution_subscribers[execution_id].add(connection_id)
                subscribed.append(execution_id)

        logger.debug(f"Connection {connection_id} subscribed to: {subscribed}")
        return subscribed

    async def unsubscribe(self, connection_id: str, execution_ids: list[str]) -> list[str]:
        """Unsubscribe a connection from execution updates.

        Args:
            connection_id: The connection ID.
            execution_ids: List of execution IDs to unsubscribe from.

        Returns:
            List of successfully unsubscribed execution IDs.
        """
        unsubscribed = []

        async with self._lock:
            connection = self._connections.get(connection_id)
            if not connection:
                return unsubscribed

            for execution_id in execution_ids:
                if execution_id in connection.subscriptions:
                    connection.subscriptions.discard(execution_id)
                    subs = self._execution_subscribers.get(execution_id)
                    if subs:
                        subs.discard(connection_id)
                        if not subs:
                            del self._execution_subscribers[execution_id]
                    unsubscribed.append(execution_id)

        logger.debug(f"Connection {connection_id} unsubscribed from: {unsubscribed}")
        return unsubscribed

    async def broadcast_to_execution(self, execution_id: str, message: OutboundMessage) -> int:
        """Broadcast a message to all subscribers of an execution.

        Args:
            execution_id: The execution ID to broadcast to.
            message: The message to send.

        Returns:
            Number of connections the message was sent to.
        """
        sent_count = 0
        failed_connections = []

        async with self._lock:
            subscriber_ids = self._execution_subscribers.get(execution_id, set()).copy()

        for connection_id in subscriber_ids:
            connection = self._connections.get(connection_id)
            if connection:
                if await connection.send(message):
                    sent_count += 1
                else:
                    failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)

        return sent_count

    async def get_subscriptions(self, execution_id: str) -> set[str]:
        """Get all connection IDs subscribed to an execution.

        Args:
            execution_id: The execution ID.

        Returns:
            Set of connection IDs.
        """
        async with self._lock:
            return self._execution_subscribers.get(execution_id, set()).copy()

    async def handle_client_message(self, connection: Connection, raw_data: str) -> None:
        """Handle an incoming message from a client.

        Args:
            connection: The connection that sent the message.
            raw_data: The raw JSON message data.
        """
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            await connection.send(
                ErrorMessage(
                    code="INVALID_JSON",
                    message="Invalid JSON in message",
                )
            )
            return

        msg_type = data.get("type")

        if msg_type == "subscribe":
            execution_ids = data.get("execution_ids", [])
            if isinstance(execution_ids, list):
                await self.subscribe(connection.id, execution_ids)

        elif msg_type == "unsubscribe":
            execution_ids = data.get("execution_ids", [])
            if isinstance(execution_ids, list):
                await self.unsubscribe(connection.id, execution_ids)

        elif msg_type == "ping":
            timestamp = data.get("timestamp", int(time.time() * 1000))
            connection.last_ping = time.time()
            await connection.send(PongMessage(timestamp=timestamp))

        else:
            await connection.send(
                ErrorMessage(
                    code="UNKNOWN_MESSAGE_TYPE",
                    message=f"Unknown message type: {msg_type}",
                )
            )

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Handle a WebSocket connection lifecycle.

        Args:
            websocket: The WebSocket connection.
        """
        connection = await self.connect(websocket)

        try:
            while True:
                data = await websocket.receive_text()
                await self.handle_client_message(connection, data)
        except WebSocketDisconnect:
            pass  # Graceful degradation: client disconnect is normal lifecycle event
        except Exception as e:
            logger.error(f"WebSocket error for {connection.id}: {e}")
        finally:
            await self.disconnect(connection.id)

    def get_stats(self) -> dict:
        """Get connection manager statistics.

        Returns:
            Dict with connection stats.
        """
        return {
            "active_connections": len(self._connections),
            "subscribed_executions": len(self._execution_subscribers),
            "connections": [
                {
                    "id": conn.id,
                    "subscriptions": len(conn.subscriptions),
                    "connected_at": conn.connected_at,
                }
                for conn in self._connections.values()
            ],
        }
