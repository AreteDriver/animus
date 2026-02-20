"""
Sync Client

WebSocket client for connecting to other Animus devices.
"""

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from animus.logging import get_logger
from animus.sync.protocol import (
    MessageType,
    SyncMessage,
    create_delta_push,
    create_ping,
    create_snapshot_request,
)
from animus.sync.state import StateDelta, StateSnapshot, SyncableState

logger = get_logger("sync.client")


@dataclass
class SyncResult:
    """Result of a sync operation."""

    success: bool
    changes_sent: int = 0
    changes_received: int = 0
    error: str | None = None
    duration_ms: int = 0


class SyncClient:
    """
    WebSocket client for connecting to and syncing with other Animus devices.
    """

    def __init__(
        self,
        state: SyncableState,
        shared_secret: str,
    ):
        self.state = state
        self.shared_secret = shared_secret

        self._websocket = None
        self._connected = False
        self._peer_device_id: str | None = None
        self._peer_device_name: str | None = None
        self._peer_version: int = 0

        # Callbacks
        self._on_delta_received: list[Callable[[StateDelta], None]] = []

        logger.info("SyncClient initialized")

    def add_delta_callback(self, callback: Callable[[StateDelta], None]) -> None:
        """Add callback for received deltas."""
        self._on_delta_received.append(callback)

    async def connect(self, address: str) -> bool:
        """
        Connect to a sync server.

        Args:
            address: WebSocket address (e.g., "ws://192.168.1.100:8422")

        Returns:
            True if connected and authenticated
        """
        try:
            import websockets
        except ImportError:
            logger.error("websockets not installed. Install with: pip install websockets")
            return False

        if self._connected:
            logger.warning("Already connected")
            return True

        try:
            self._websocket = await websockets.connect(address)

            # Authenticate
            auth_hash = hashlib.sha256(self.shared_secret.encode()).hexdigest()
            auth_message = SyncMessage(
                type=MessageType.AUTH,
                device_id=self.state.device_id,
                payload={
                    "auth_hash": auth_hash,
                    "device_name": f"animus-{self.state.device_id[:8]}",
                },
            )
            await self._websocket.send(auth_message.to_json())

            # Wait for response
            response = await asyncio.wait_for(self._websocket.recv(), timeout=10.0)
            message = SyncMessage.from_json(response)

            if message.type == MessageType.AUTH_OK:
                self._connected = True
                self._peer_device_id = message.device_id
                self._peer_device_name = message.payload.get("device_name", "unknown")
                self._peer_version = message.payload.get("version", 0)
                logger.info(f"Connected to {self._peer_device_name}")
                return True

            elif message.type == MessageType.AUTH_FAIL:
                reason = message.payload.get("reason", "Unknown")
                logger.error(f"Authentication failed: {reason}")
                await self._websocket.close()
                return False

            else:
                logger.error(f"Unexpected response: {message.type}")
                await self._websocket.close()
                return False

        except asyncio.TimeoutError:
            logger.error("Connection timeout")
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the sync server."""
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass  # Best-effort cleanup: websocket may already be closed

        self._websocket = None
        self._connected = False
        self._peer_device_id = None
        self._peer_device_name = None
        logger.info("Disconnected")

    async def sync(self) -> SyncResult:
        """
        Perform a full sync with the connected peer.

        Returns:
            SyncResult with sync statistics
        """
        if not self._connected:
            return SyncResult(success=False, error="Not connected")

        start_time = datetime.now()

        try:
            # Request peer's snapshot
            await self._websocket.send(
                create_snapshot_request(
                    self.state.device_id,
                    self.state.get_peer_version(self._peer_device_id or ""),
                ).to_json()
            )

            # Wait for snapshot response
            response = await asyncio.wait_for(self._websocket.recv(), timeout=30.0)
            message = SyncMessage.from_json(response)

            if message.type != MessageType.SNAPSHOT_RESPONSE:
                return SyncResult(
                    success=False,
                    error=f"Unexpected response: {message.type}",
                )

            # Process received snapshot
            peer_snapshot_data = message.payload.get("snapshot", {})
            peer_snapshot = StateSnapshot.from_dict(peer_snapshot_data)
            peer_version = message.payload.get("version", 0)

            # Compute what we need from peer
            incoming_delta = StateDelta.compute(
                source_device=peer_snapshot.device_id,
                target_device=self.state.device_id,
                old_data=self.state.collect_state(),
                new_data=peer_snapshot.data,
                base_version=self.state.version,
            )

            changes_received = 0
            if not incoming_delta.is_empty():
                self.state.apply_delta(incoming_delta)
                changes_received = (
                    len(incoming_delta.changes.get("added", {}))
                    + len(incoming_delta.changes.get("modified", {}))
                    + len(incoming_delta.changes.get("deleted", []))
                )

                # Notify callbacks
                for callback in self._on_delta_received:
                    try:
                        callback(incoming_delta)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            # Compute what peer needs from us
            outgoing_delta = StateDelta.compute(
                source_device=self.state.device_id,
                target_device=peer_snapshot.device_id,
                old_data=peer_snapshot.data,
                new_data=self.state.collect_state(),
                base_version=peer_version,
            )

            changes_sent = 0
            if not outgoing_delta.is_empty():
                await self._websocket.send(
                    create_delta_push(
                        self.state.device_id,
                        outgoing_delta.to_dict(),
                    ).to_json()
                )

                # Wait for acknowledgment
                ack = await asyncio.wait_for(self._websocket.recv(), timeout=10.0)
                ack_message = SyncMessage.from_json(ack)

                if ack_message.type == MessageType.DELTA_ACK:
                    if ack_message.payload.get("success"):
                        changes_sent = (
                            len(outgoing_delta.changes.get("added", {}))
                            + len(outgoing_delta.changes.get("modified", {}))
                            + len(outgoing_delta.changes.get("deleted", []))
                        )
                        self.state.increment_version()

            # Update peer version tracking
            self.state.set_peer_version(self._peer_device_id or "", peer_version)

            duration = int((datetime.now() - start_time).total_seconds() * 1000)

            return SyncResult(
                success=True,
                changes_sent=changes_sent,
                changes_received=changes_received,
                duration_ms=duration,
            )

        except asyncio.TimeoutError:
            return SyncResult(success=False, error="Sync timeout")
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return SyncResult(success=False, error=str(e))

    async def push_changes(self, delta: StateDelta) -> bool:
        """
        Push a delta to the connected peer.

        Args:
            delta: Changes to push

        Returns:
            True if acknowledged successfully
        """
        if not self._connected:
            logger.error("Not connected")
            return False

        try:
            await self._websocket.send(
                create_delta_push(self.state.device_id, delta.to_dict()).to_json()
            )

            # Wait for acknowledgment
            response = await asyncio.wait_for(self._websocket.recv(), timeout=10.0)
            message = SyncMessage.from_json(response)

            if message.type == MessageType.DELTA_ACK:
                return message.payload.get("success", False)

            return False

        except Exception as e:
            logger.error(f"Push failed: {e}")
            return False

    async def ping(self) -> int | None:
        """
        Send a ping and measure round-trip time.

        Returns:
            Round-trip time in milliseconds, or None if failed
        """
        if not self._connected:
            return None

        try:
            start = datetime.now()
            await self._websocket.send(create_ping(self.state.device_id).to_json())

            response = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
            message = SyncMessage.from_json(response)

            if message.type == MessageType.PONG:
                return int((datetime.now() - start).total_seconds() * 1000)

            return None

        except Exception:
            return None

    async def listen(self) -> None:
        """
        Listen for incoming messages from the server.

        This is a blocking call that should be run in a task.
        """
        if not self._connected:
            return

        try:
            async for raw_message in self._websocket:
                try:
                    message = SyncMessage.from_json(raw_message)
                    await self._handle_message(message)
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        except Exception as e:
            logger.error(f"Listen error: {e}")
            self._connected = False

    async def _handle_message(self, message: SyncMessage) -> None:
        """Handle an incoming message."""
        if message.type == MessageType.DELTA_PUSH:
            # Incoming delta from server
            delta_data = message.payload.get("delta", {})
            delta = StateDelta.from_dict(delta_data)

            success = self.state.apply_delta(delta)

            # Notify callbacks
            if success:
                for callback in self._on_delta_received:
                    try:
                        callback(delta)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

        elif message.type == MessageType.STATUS:
            status = message.payload.get("status", "unknown")
            logger.debug(f"Server status: {status}")

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    @property
    def peer_device_id(self) -> str | None:
        """Get connected peer's device ID."""
        return self._peer_device_id

    @property
    def peer_device_name(self) -> str | None:
        """Get connected peer's name."""
        return self._peer_device_name
