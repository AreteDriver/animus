"""
Sync Server

WebSocket server for handling sync connections from other devices.
"""

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from animus.logging import get_logger
from animus.sync.protocol import (
    MessageType,
    SyncMessage,
    create_auth_fail_message,
    create_auth_ok_message,
    create_delta_ack,
    create_error_message,
    create_pong,
    create_snapshot_response,
)
from animus.sync.state import StateDelta, SyncableState

logger = get_logger("sync.server")


@dataclass
class ConnectedPeer:
    """Information about a connected peer."""

    device_id: str
    device_name: str
    websocket: Any  # WebSocket connection
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    version: int = 0
    authenticated: bool = False


class SyncServer:
    """
    WebSocket server for cross-device synchronization.

    Handles incoming connections, authentication, and sync operations.
    """

    def __init__(
        self,
        state: SyncableState,
        port: int = 8422,
        shared_secret: str | None = None,
    ):
        self.state = state
        self.port = port
        self.shared_secret = shared_secret or self._generate_secret()

        self._peers: dict[str, ConnectedPeer] = {}
        self._server = None
        self._running = False

        # Callbacks
        self._on_peer_connected: list[Callable[[ConnectedPeer], None]] = []
        self._on_peer_disconnected: list[Callable[[ConnectedPeer], None]] = []
        self._on_sync: list[Callable[[str, StateDelta], None]] = []

        logger.info(f"SyncServer initialized on port {port}")

    def _generate_secret(self) -> str:
        """Generate a random shared secret."""
        import secrets

        return secrets.token_hex(16)

    def add_peer_connected_callback(self, callback: Callable[[ConnectedPeer], None]) -> None:
        """Add callback for peer connected events."""
        self._on_peer_connected.append(callback)

    def add_peer_disconnected_callback(self, callback: Callable[[ConnectedPeer], None]) -> None:
        """Add callback for peer disconnected events."""
        self._on_peer_disconnected.append(callback)

    def add_sync_callback(self, callback: Callable[[str, StateDelta], None]) -> None:
        """Add callback for sync events."""
        self._on_sync.append(callback)

    async def start(self) -> bool:
        """
        Start the sync server.

        Returns:
            True if started successfully
        """
        try:
            import websockets
        except ImportError:
            logger.error("websockets not installed. Install with: pip install websockets")
            return False

        if self._running:
            logger.warning("Server already running")
            return True

        try:
            self._server = await websockets.serve(
                self._handle_connection,
                "0.0.0.0",
                self.port,
            )
            self._running = True
            logger.info(f"Sync server started on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False

    async def stop(self) -> None:
        """Stop the sync server."""
        if not self._running:
            return

        self._running = False

        # Close all peer connections
        for peer in list(self._peers.values()):
            try:
                await peer.websocket.close()
            except Exception:
                pass

        self._peers.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        logger.info("Sync server stopped")

    async def _handle_connection(self, websocket, path) -> None:
        """Handle a new WebSocket connection."""
        peer = None
        try:
            # Wait for authentication
            peer = await self._authenticate(websocket)
            if not peer:
                return

            self._peers[peer.device_id] = peer
            logger.info(f"Peer connected: {peer.device_name} ({peer.device_id[:8]}...)")

            # Notify callbacks
            for callback in self._on_peer_connected:
                try:
                    callback(peer)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            # Handle messages
            async for message in websocket:
                await self._handle_message(peer, message)

        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            if peer and peer.device_id in self._peers:
                del self._peers[peer.device_id]
                logger.info(f"Peer disconnected: {peer.device_name}")

                for callback in self._on_peer_disconnected:
                    try:
                        callback(peer)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

    async def _authenticate(self, websocket) -> ConnectedPeer | None:
        """
        Authenticate an incoming connection.

        Returns:
            ConnectedPeer if authenticated, None otherwise
        """
        try:
            # Wait for auth message with timeout
            raw_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            message = SyncMessage.from_json(raw_message)

            if message.type != MessageType.AUTH:
                await websocket.send(
                    create_error_message(
                        self.state.device_id,
                        "Expected AUTH message",
                        "auth_expected",
                    ).to_json()
                )
                return None

            # Verify auth hash
            auth_hash = message.payload.get("auth_hash", "")
            expected_hash = hashlib.sha256(self.shared_secret.encode()).hexdigest()

            if auth_hash != expected_hash:
                await websocket.send(
                    create_auth_fail_message(
                        self.state.device_id,
                        "Invalid authentication",
                    ).to_json()
                )
                logger.warning(f"Authentication failed for device {message.device_id[:8]}...")
                return None

            # Send auth OK
            await websocket.send(
                create_auth_ok_message(
                    self.state.device_id,
                    f"animus-{self.state.device_id[:8]}",
                    self.state.version,
                ).to_json()
            )

            return ConnectedPeer(
                device_id=message.device_id,
                device_name=message.payload.get("device_name", "unknown"),
                websocket=websocket,
                authenticated=True,
            )

        except asyncio.TimeoutError:
            logger.warning("Authentication timeout")
            return None
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None

    async def _handle_message(self, peer: ConnectedPeer, raw_message: str) -> None:
        """Handle an incoming message from a peer."""
        try:
            message = SyncMessage.from_json(raw_message)
            peer.last_activity = datetime.now()

            handlers = {
                MessageType.PING: self._handle_ping,
                MessageType.SNAPSHOT_REQUEST: self._handle_snapshot_request,
                MessageType.DELTA_PUSH: self._handle_delta_push,
                MessageType.STATUS: self._handle_status,
            }

            handler = handlers.get(message.type)
            if handler:
                await handler(peer, message)
            else:
                logger.warning(f"Unknown message type: {message.type}")

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            try:
                await peer.websocket.send(
                    create_error_message(
                        self.state.device_id,
                        str(e),
                        "message_error",
                    ).to_json()
                )
            except Exception:
                pass

    async def _handle_ping(self, peer: ConnectedPeer, message: SyncMessage) -> None:
        """Handle ping message."""
        await peer.websocket.send(create_pong(self.state.device_id).to_json())

    async def _handle_snapshot_request(
        self,
        peer: ConnectedPeer,
        message: SyncMessage,
    ) -> None:
        """Handle snapshot request, using incremental sync when possible."""
        since_version = message.payload.get("since_version", 0)

        if since_version > 0:
            # Try incremental sync first
            incremental = self.state.collect_state_since(since_version)
            if incremental is not None:
                await peer.websocket.send(
                    create_snapshot_response(
                        self.state.device_id,
                        {"incremental": True, "changes": incremental},
                        self.state.version,
                    ).to_json()
                )
                logger.debug(
                    f"Sent incremental sync to {peer.device_name} "
                    f"(since v{since_version})"
                )
                return

        # Fall back to full snapshot
        snapshot = self.state.create_snapshot()

        await peer.websocket.send(
            create_snapshot_response(
                self.state.device_id,
                snapshot.to_dict(),
                self.state.version,
            ).to_json()
        )

        logger.debug(f"Sent full snapshot to {peer.device_name}")

    async def _handle_delta_push(self, peer: ConnectedPeer, message: SyncMessage) -> None:
        """Handle incoming delta from peer."""
        delta_data = message.payload.get("delta", {})

        try:
            delta = StateDelta.from_dict(delta_data)

            # Apply delta
            success = self.state.apply_delta(delta)

            # Send acknowledgment
            await peer.websocket.send(
                create_delta_ack(
                    self.state.device_id,
                    delta.id,
                    success,
                ).to_json()
            )

            if success:
                # Record delta for future incremental sync
                self.state.record_delta(delta)

                # Notify callbacks
                for callback in self._on_sync:
                    try:
                        callback(peer.device_id, delta)
                    except Exception as e:
                        logger.error(f"Sync callback error: {e}")

                logger.info(f"Applied delta from {peer.device_name}")

        except Exception as e:
            logger.error(f"Failed to apply delta: {e}")
            await peer.websocket.send(
                create_delta_ack(
                    self.state.device_id,
                    delta_data.get("id", ""),
                    False,
                ).to_json()
            )

    async def _handle_status(self, peer: ConnectedPeer, message: SyncMessage) -> None:
        """Handle status message."""
        status = message.payload.get("status", "unknown")
        logger.debug(f"Peer {peer.device_name} status: {status}")

    async def broadcast_delta(self, delta: StateDelta) -> int:
        """
        Broadcast a delta to all connected peers.

        Returns:
            Number of peers notified
        """
        if not self._peers:
            return 0

        message = SyncMessage(
            type=MessageType.DELTA_PUSH,
            device_id=self.state.device_id,
            payload={"delta": delta.to_dict()},
        )

        count = 0
        for peer in list(self._peers.values()):
            try:
                await peer.websocket.send(message.to_json())
                count += 1
            except Exception as e:
                logger.error(f"Failed to send to {peer.device_name}: {e}")

        return count

    def get_peers(self) -> list[ConnectedPeer]:
        """Get list of connected peers."""
        return list(self._peers.values())

    def get_peer(self, device_id: str) -> ConnectedPeer | None:
        """Get a specific peer by device ID."""
        return self._peers.get(device_id)

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    @property
    def peer_count(self) -> int:
        """Get number of connected peers."""
        return len(self._peers)
