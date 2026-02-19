"""Tests for logging and sync subsystems.

Covers: logging.py, sync/client.py, sync/server.py, sync/discovery.py, sync/state.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from animus.logging import get_logger, setup_logging
from animus.sync.discovery import DeviceDiscovery, DiscoveredDevice, MockDeviceDiscovery
from animus.sync.protocol import MessageType, SyncMessage
from animus.sync.state import StateDelta, StateSnapshot, SyncableState

# ===================================================================
# Logging
# ===================================================================


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_setup_with_file(self, tmp_path: Path):
        log_file = tmp_path / "test.log"
        logger = setup_logging(log_file=log_file, level="DEBUG")
        assert logger.name == "animus"
        assert logger.level == logging.DEBUG
        # Should have file + console handlers
        assert len(logger.handlers) == 2
        logger.handlers.clear()

    def test_setup_without_file(self):
        logger = setup_logging(log_file=None, level="INFO")
        assert logger.name == "animus"
        # Only console handler
        assert len(logger.handlers) == 1
        logger.handlers.clear()

    def test_setup_log_to_file_false(self, tmp_path: Path):
        logger = setup_logging(log_file=tmp_path / "test.log", level="WARNING", log_to_file=False)
        assert logger.level == logging.WARNING
        assert len(logger.handlers) == 1  # Only console
        logger.handlers.clear()

    def test_setup_creates_parent_dirs(self, tmp_path: Path):
        log_file = tmp_path / "deep" / "nested" / "test.log"
        logger = setup_logging(log_file=log_file, level="INFO")
        assert log_file.parent.exists()
        logger.handlers.clear()

    def test_get_logger(self):
        logger = get_logger("test_module")
        assert logger.name == "animus.test_module"

    def test_setup_clears_existing_handlers(self, tmp_path: Path):
        logger = setup_logging(log_file=tmp_path / "a.log", level="INFO")
        count1 = len(logger.handlers)
        # Setup again — should clear and rebuild
        logger = setup_logging(log_file=tmp_path / "b.log", level="INFO")
        count2 = len(logger.handlers)
        assert count1 == count2
        logger.handlers.clear()

    def test_invalid_level_defaults_to_info(self):
        logger = setup_logging(log_file=None, level="INVALID_LEVEL")
        assert logger.level == logging.INFO
        logger.handlers.clear()


# ===================================================================
# Sync Client
# ===================================================================


class TestSyncClient:
    """Tests for SyncClient."""

    def _make_state(self, tmp_path: Path) -> SyncableState:
        return SyncableState(tmp_path, device_id="client-device-123")

    def test_init(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")
        assert client.is_connected is False
        assert client.peer_device_id is None
        assert client.peer_device_name is None

    def test_add_delta_callback(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")
        cb = MagicMock()
        client.add_delta_callback(cb)
        assert cb in client._on_delta_received

    def test_connect_websockets_import_error(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")

        with patch.dict("sys.modules", {"websockets": None}):
            result = asyncio.run(client.connect("ws://localhost:8422"))
            assert result is False

    def test_connect_already_connected(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")
        client._connected = True
        with patch.dict("sys.modules", {"websockets": MagicMock()}):
            result = asyncio.run(client.connect("ws://localhost:8422"))
            assert result is True

    def test_connect_auth_ok(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")

        mock_ws = AsyncMock()
        auth_response = SyncMessage(
            type=MessageType.AUTH_OK,
            device_id="server-device",
            payload={"device_name": "server", "version": 5},
        )
        mock_ws.recv = AsyncMock(return_value=auth_response.to_json())

        mock_websockets = MagicMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)
        with patch.dict("sys.modules", {"websockets": mock_websockets}):
            result = asyncio.run(client.connect("ws://localhost:8422"))

        assert result is True
        assert client.is_connected is True
        assert client.peer_device_id == "server-device"

    def test_connect_auth_fail(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")

        mock_ws = AsyncMock()
        fail_response = SyncMessage(
            type=MessageType.AUTH_FAIL,
            device_id="server-device",
            payload={"reason": "bad secret"},
        )
        mock_ws.recv = AsyncMock(return_value=fail_response.to_json())

        mock_websockets = MagicMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)
        with patch.dict("sys.modules", {"websockets": mock_websockets}):
            result = asyncio.run(client.connect("ws://localhost:8422"))

        assert result is False

    def test_connect_unexpected_response(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")

        mock_ws = AsyncMock()
        unexpected = SyncMessage(
            type=MessageType.PONG,
            device_id="server-device",
            payload={},
        )
        mock_ws.recv = AsyncMock(return_value=unexpected.to_json())

        mock_websockets = MagicMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)
        with patch.dict("sys.modules", {"websockets": mock_websockets}):
            result = asyncio.run(client.connect("ws://localhost:8422"))

        assert result is False

    def test_connect_timeout(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.TimeoutError)

        mock_websockets = MagicMock()
        mock_websockets.connect = AsyncMock(return_value=mock_ws)
        with patch.dict("sys.modules", {"websockets": mock_websockets}):
            result = asyncio.run(client.connect("ws://localhost:8422"))

        assert result is False

    def test_connect_exception(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")

        mock_websockets = MagicMock()
        mock_websockets.connect = AsyncMock(side_effect=ConnectionRefusedError)
        with patch.dict("sys.modules", {"websockets": mock_websockets}):
            result = asyncio.run(client.connect("ws://localhost:8422"))

        assert result is False

    def test_disconnect(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")
        client._connected = True
        client._websocket = AsyncMock()
        client._peer_device_id = "server"
        client._peer_device_name = "srv"

        asyncio.run(client.disconnect())
        assert client.is_connected is False
        assert client._websocket is None
        assert client._peer_device_id is None

    def test_disconnect_websocket_error(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")
        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock(side_effect=Exception("close error"))
        client._websocket = mock_ws
        client._connected = True

        asyncio.run(client.disconnect())
        assert client.is_connected is False

    def test_sync_not_connected(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")
        result = asyncio.run(client.sync())
        assert result.success is False
        assert "Not connected" in result.error

    def test_push_not_connected(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")
        delta = StateDelta.compute("a", "b", {}, {"key": "val"}, 0)
        result = asyncio.run(client.push_changes(delta))
        assert result is False

    def test_ping_not_connected(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")
        result = asyncio.run(client.ping())
        assert result is None

    def test_ping_success(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")
        client._connected = True
        mock_ws = AsyncMock()
        pong = SyncMessage(type=MessageType.PONG, device_id="server", payload={})
        mock_ws.recv = AsyncMock(return_value=pong.to_json())
        client._websocket = mock_ws

        result = asyncio.run(client.ping())
        assert result is not None
        assert isinstance(result, int)

    def test_ping_exception(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")
        client._connected = True
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=Exception("network"))
        client._websocket = mock_ws

        result = asyncio.run(client.ping())
        assert result is None

    def test_listen_not_connected(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")
        asyncio.run(client.listen())  # Should return immediately

    def test_handle_message_delta_push(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")
        callback = MagicMock()
        client.add_delta_callback(callback)

        delta = StateDelta.compute("peer", "client", {}, {"memories": [{"id": "1"}]}, 0)
        msg = SyncMessage(
            type=MessageType.DELTA_PUSH,
            device_id="peer",
            payload={"delta": delta.to_dict()},
        )
        asyncio.run(client._handle_message(msg))

    def test_handle_message_status(self, tmp_path: Path):
        from animus.sync.client import SyncClient

        state = self._make_state(tmp_path)
        client = SyncClient(state, "secret123")
        msg = SyncMessage(
            type=MessageType.STATUS,
            device_id="peer",
            payload={"status": "ok"},
        )
        asyncio.run(client._handle_message(msg))  # No error


# ===================================================================
# Sync Server
# ===================================================================


class TestSyncServer:
    """Tests for SyncServer."""

    def _make_state(self, tmp_path: Path) -> SyncableState:
        return SyncableState(tmp_path, device_id="server-device-456")

    def test_init(self, tmp_path: Path):
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state, port=8422)
        assert server.is_running is False
        assert server.peer_count == 0
        assert server.shared_secret  # Auto-generated

    def test_init_with_secret(self, tmp_path: Path):
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state, shared_secret="my-secret")
        assert server.shared_secret == "my-secret"

    def test_callbacks(self, tmp_path: Path):
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        cb1 = MagicMock()
        cb2 = MagicMock()
        cb3 = MagicMock()
        server.add_peer_connected_callback(cb1)
        server.add_peer_disconnected_callback(cb2)
        server.add_sync_callback(cb3)

        assert cb1 in server._on_peer_connected
        assert cb2 in server._on_peer_disconnected
        assert cb3 in server._on_sync

    def test_start_import_error(self, tmp_path: Path):
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        with patch.dict("sys.modules", {"websockets": None}):
            result = asyncio.run(server.start())
            assert result is False

    def test_start_already_running(self, tmp_path: Path):
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)
        server._running = True
        with patch.dict("sys.modules", {"websockets": MagicMock()}):
            result = asyncio.run(server.start())
            assert result is True

    def test_stop_not_running(self, tmp_path: Path):
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)
        asyncio.run(server.stop())  # No error

    def test_stop_with_peers(self, tmp_path: Path):
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)
        server._running = True
        mock_ws = AsyncMock()
        peer = ConnectedPeer(
            device_id="peer1",
            device_name="test-peer",
            websocket=mock_ws,
        )
        server._peers["peer1"] = peer
        mock_server = MagicMock()
        mock_server.close = MagicMock()
        mock_server.wait_closed = AsyncMock()
        server._server = mock_server

        asyncio.run(server.stop())
        assert server.is_running is False
        assert server.peer_count == 0

    def test_get_peers(self, tmp_path: Path):
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)
        assert server.get_peers() == []
        assert server.get_peer("x") is None

    def test_authenticate_success(self, tmp_path: Path):
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state, shared_secret="test-secret")

        auth_hash = hashlib.sha256(b"test-secret").hexdigest()
        auth_msg = SyncMessage(
            type=MessageType.AUTH,
            device_id="client-device",
            payload={"auth_hash": auth_hash, "device_name": "test-client"},
        )

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=auth_msg.to_json())

        peer = asyncio.run(server._authenticate(mock_ws))
        assert peer is not None
        assert peer.device_id == "client-device"
        assert peer.authenticated is True

    def test_authenticate_wrong_hash(self, tmp_path: Path):
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state, shared_secret="test-secret")

        auth_msg = SyncMessage(
            type=MessageType.AUTH,
            device_id="client-device",
            payload={"auth_hash": "wrong_hash", "device_name": "bad-client"},
        )

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=auth_msg.to_json())

        peer = asyncio.run(server._authenticate(mock_ws))
        assert peer is None

    def test_authenticate_wrong_message_type(self, tmp_path: Path):
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        ping_msg = SyncMessage(type=MessageType.PING, device_id="client", payload={})
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=ping_msg.to_json())

        peer = asyncio.run(server._authenticate(mock_ws))
        assert peer is None

    def test_authenticate_timeout(self, tmp_path: Path):
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.TimeoutError)

        peer = asyncio.run(server._authenticate(mock_ws))
        assert peer is None

    def test_authenticate_exception(self, tmp_path: Path):
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=Exception("parse error"))

        peer = asyncio.run(server._authenticate(mock_ws))
        assert peer is None

    def test_handle_ping(self, tmp_path: Path):
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)
        mock_ws = AsyncMock()
        peer = ConnectedPeer(device_id="p1", device_name="test", websocket=mock_ws)
        msg = SyncMessage(type=MessageType.PING, device_id="p1", payload={})

        asyncio.run(server._handle_ping(peer, msg))
        mock_ws.send.assert_called_once()

    def test_handle_snapshot_request(self, tmp_path: Path):
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)
        mock_ws = AsyncMock()
        peer = ConnectedPeer(device_id="p1", device_name="test", websocket=mock_ws)
        msg = SyncMessage(
            type=MessageType.SNAPSHOT_REQUEST,
            device_id="p1",
            payload={"since_version": 0},
        )

        asyncio.run(server._handle_snapshot_request(peer, msg))
        mock_ws.send.assert_called_once()

    def test_handle_delta_push(self, tmp_path: Path):
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)
        mock_ws = AsyncMock()
        peer = ConnectedPeer(device_id="p1", device_name="test", websocket=mock_ws)

        delta = StateDelta.compute("p1", "server", {}, {"memories": [{"id": "1"}]}, 0)
        msg = SyncMessage(
            type=MessageType.DELTA_PUSH,
            device_id="p1",
            payload={"delta": delta.to_dict()},
        )

        asyncio.run(server._handle_delta_push(peer, msg))
        assert mock_ws.send.called

    def test_handle_status(self, tmp_path: Path):
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)
        mock_ws = AsyncMock()
        peer = ConnectedPeer(device_id="p1", device_name="test", websocket=mock_ws)
        msg = SyncMessage(type=MessageType.STATUS, device_id="p1", payload={"status": "idle"})

        asyncio.run(server._handle_status(peer, msg))

    def test_handle_message_unknown_type(self, tmp_path: Path):
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)
        mock_ws = AsyncMock()
        peer = ConnectedPeer(device_id="p1", device_name="test", websocket=mock_ws)
        raw_msg = SyncMessage(type=MessageType.AUTH, device_id="p1", payload={}).to_json()

        asyncio.run(server._handle_message(peer, raw_msg))

    def test_handle_message_parse_error(self, tmp_path: Path):
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)
        mock_ws = AsyncMock()
        peer = ConnectedPeer(device_id="p1", device_name="test", websocket=mock_ws)

        asyncio.run(server._handle_message(peer, "not valid json"))

    def test_broadcast_delta_no_peers(self, tmp_path: Path):
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)
        delta = StateDelta.compute("s", "t", {}, {"k": "v"}, 0)
        count = asyncio.run(server.broadcast_delta(delta))
        assert count == 0

    def test_broadcast_delta_with_peers(self, tmp_path: Path):
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        server._peers["p1"] = ConnectedPeer(device_id="p1", device_name="a", websocket=mock_ws1)
        server._peers["p2"] = ConnectedPeer(device_id="p2", device_name="b", websocket=mock_ws2)
        delta = StateDelta.compute("s", "t", {}, {"k": "v"}, 0)
        count = asyncio.run(server.broadcast_delta(delta))
        assert count == 2

    def test_broadcast_delta_peer_error(self, tmp_path: Path):
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock(side_effect=Exception("send fail"))
        server._peers["p1"] = ConnectedPeer(device_id="p1", device_name="a", websocket=mock_ws)
        delta = StateDelta.compute("s", "t", {}, {"k": "v"}, 0)
        count = asyncio.run(server.broadcast_delta(delta))
        assert count == 0


# ===================================================================
# Sync Discovery
# ===================================================================


class TestDeviceDiscovery:
    """Tests for DeviceDiscovery."""

    def test_init(self):
        dd = DeviceDiscovery("dev-123", "my-device", 8422)
        assert dd.device_id == "dev-123"
        assert dd.is_running is False

    def test_add_callback(self):
        dd = DeviceDiscovery("dev-123", "my-device", 8422)
        cb = MagicMock()
        dd.add_callback(cb)
        assert cb in dd._callbacks

    def test_start_without_zeroconf(self):
        dd = DeviceDiscovery("dev-123", "my-device", 8422)
        with patch.dict("sys.modules", {"zeroconf": None}):
            result = dd.start()
            assert result is False

    def test_start_already_running(self):
        dd = DeviceDiscovery("dev-123", "my-device", 8422)
        dd._running = True
        mock_zeroconf_mod = MagicMock()
        with patch.dict("sys.modules", {"zeroconf": mock_zeroconf_mod}):
            result = dd.start()
        assert result is True

    def test_stop_not_running(self):
        dd = DeviceDiscovery("dev-123", "my-device", 8422)
        dd.stop()  # No error

    def test_stop_with_zeroconf(self):
        dd = DeviceDiscovery("dev-123", "my-device", 8422)
        dd._running = True
        dd._zeroconf = MagicMock()
        dd._service_info = MagicMock()
        dd.stop()
        assert dd.is_running is False
        assert dd._zeroconf is None

    def test_get_devices_empty(self):
        dd = DeviceDiscovery("dev-123", "my-device", 8422)
        assert dd.get_devices() == []
        assert dd.get_device("x") is None

    def test_notify_callbacks(self):
        dd = DeviceDiscovery("dev-123", "my-device", 8422)
        cb = MagicMock()
        dd.add_callback(cb)
        device = DiscoveredDevice(
            device_id="other", name="other-dev", host="192.168.1.2", port=8422, version="0.5"
        )
        dd._notify_callbacks(device, "added")
        cb.assert_called_once_with(device, "added")

    def test_notify_callbacks_error(self):
        dd = DeviceDiscovery("dev-123", "my-device", 8422)
        bad_cb = MagicMock(side_effect=Exception("cb error"))
        dd.add_callback(bad_cb)
        device = DiscoveredDevice(
            device_id="other", name="other-dev", host="192.168.1.2", port=8422, version="0.5"
        )
        dd._notify_callbacks(device, "added")  # Should not raise

    def test_get_local_ip_failure(self):
        dd = DeviceDiscovery("dev-123", "my-device", 8422)
        with patch("socket.socket") as mock_sock:
            mock_sock.return_value.connect.side_effect = OSError("no network")
            ip = dd._get_local_ip()
            assert ip == "127.0.0.1"

    def test_handle_service_removed(self):
        dd = DeviceDiscovery("dev-123", "my-device", 8422)
        from animus.sync.discovery import SERVICE_TYPE

        device = DiscoveredDevice(
            device_id="rem-id", name="removable", host="1.2.3.4", port=8422, version="0.5"
        )
        dd._discovered["rem-id"] = device
        dd._handle_service_removed(f"removable.{SERVICE_TYPE}")
        assert "rem-id" not in dd._discovered

    def test_handle_service_removed_not_found(self):
        dd = DeviceDiscovery("dev-123", "my-device", 8422)
        dd._handle_service_removed("nonexistent._animus-sync._tcp.local.")


class TestMockDeviceDiscovery:
    """Tests for MockDeviceDiscovery."""

    def test_start_stop(self):
        md = MockDeviceDiscovery("dev-123", "my-device", 8422)
        assert md.start() is True
        assert md.is_running is True
        md.stop()
        assert md.is_running is False

    def test_add_mock_device(self):
        md = MockDeviceDiscovery("dev-123", "my-device", 8422)
        md.start()
        cb = MagicMock()
        md.add_callback(cb)

        device = DiscoveredDevice(
            device_id="mock-1", name="mock", host="1.2.3.4", port=8422, version="0.5"
        )
        md.add_mock_device(device)

        assert md.get_device("mock-1") is device
        cb.assert_called_once_with(device, "added")

    def test_remove_mock_device(self):
        md = MockDeviceDiscovery("dev-123", "my-device", 8422)
        md.start()
        device = DiscoveredDevice(
            device_id="mock-1", name="mock", host="1.2.3.4", port=8422, version="0.5"
        )
        md.add_mock_device(device)
        md.remove_mock_device("mock-1")
        assert md.get_device("mock-1") is None


# ===================================================================
# Sync State - Additional Coverage
# ===================================================================


class TestSyncableStateAdditional:
    """Additional tests for SyncableState uncovered paths."""

    def test_collect_state_with_files(self, tmp_path: Path):
        state = SyncableState(tmp_path, device_id="dev-1")

        # Create memories file
        memories = [{"id": "m1", "content": "test memory"}]
        (tmp_path / "memories.json").write_text(json.dumps(memories))

        # Create learnings
        learning_dir = tmp_path / "learning"
        learning_dir.mkdir()
        learnings = [{"id": "l1", "pattern": "test"}]
        (learning_dir / "learned_items.json").write_text(json.dumps(learnings))

        # Create guardrails
        guardrails = [{"id": "g1", "rule": "no spam"}]
        (learning_dir / "user_guardrails.json").write_text(json.dumps(guardrails))

        collected = state.collect_state()
        assert len(collected["memories"]) == 1
        assert len(collected["learnings"]) == 1
        assert len(collected["guardrails"]) == 1

    def test_collect_config_yaml(self, tmp_path: Path):
        state = SyncableState(tmp_path, device_id="dev-1")
        config = {"model": "llama3", "api_key": "SECRET", "some_setting": True}
        import yaml

        (tmp_path / "config.yaml").write_text(yaml.dump(config))
        collected = state.collect_state()
        assert "api_key" not in collected["config"]
        assert collected["config"]["some_setting"] is True

    def test_apply_delta_memory_changes(self, tmp_path: Path):
        state = SyncableState(tmp_path, device_id="dev-1")
        changes = {
            "added": {"memories": [{"id": "m1", "content": "new", "updated_at": "2025-01-01"}]},
            "modified": {},
            "deleted": [],
        }
        delta = StateDelta(
            id="d1",
            source_device="peer",
            target_device="dev-1",
            timestamp=datetime.now(),
            base_version=0,
            new_version=1,
            changes=changes,
        )
        result = state.apply_delta(delta)
        assert result is True
        assert state.version == 1

    def test_apply_delta_learning_changes(self, tmp_path: Path):
        state = SyncableState(tmp_path, device_id="dev-1")
        changes = {
            "added": {"learnings": [{"id": "l1", "pattern": "test"}]},
            "modified": {},
            "deleted": [],
        }
        delta = StateDelta(
            id="d1",
            source_device="peer",
            target_device="dev-1",
            timestamp=datetime.now(),
            base_version=0,
            new_version=1,
            changes=changes,
        )
        result = state.apply_delta(delta)
        assert result is True

    def test_apply_delta_guardrail_changes(self, tmp_path: Path):
        state = SyncableState(tmp_path, device_id="dev-1")
        changes = {
            "added": {"guardrails": [{"id": "g1", "rule": "no spam"}]},
            "modified": {},
            "deleted": [],
        }
        delta = StateDelta(
            id="d1",
            source_device="peer",
            target_device="dev-1",
            timestamp=datetime.now(),
            base_version=0,
            new_version=1,
            changes=changes,
        )
        result = state.apply_delta(delta)
        assert result is True

    def test_apply_delta_future_version_warning(self, tmp_path: Path):
        state = SyncableState(tmp_path, device_id="dev-1")
        delta = StateDelta(
            id="d1",
            source_device="peer",
            target_device="dev-1",
            timestamp=datetime.now(),
            base_version=99,
            new_version=100,
            changes={
                "added": {"memories": [{"id": "m1", "content": "test"}]},
                "modified": {},
                "deleted": [],
            },
        )
        result = state.apply_delta(delta)
        assert result is True

    def test_apply_memory_merge_existing(self, tmp_path: Path):
        state = SyncableState(tmp_path, device_id="dev-1")

        # Pre-existing memory
        (tmp_path / "memories.json").write_text(
            json.dumps([{"id": "m1", "content": "old", "updated_at": "2024-01-01"}])
        )

        changes = {
            "added": {},
            "modified": {"memories": [{"id": "m1", "content": "new", "updated_at": "2025-01-01"}]},
            "deleted": [],
        }
        state._apply_memory_changes(changes)

        memories = json.loads((tmp_path / "memories.json").read_text())
        assert memories[0]["content"] == "new"

    def test_apply_learning_merge(self, tmp_path: Path):
        state = SyncableState(tmp_path, device_id="dev-1")
        learning_dir = tmp_path / "learning"
        learning_dir.mkdir()
        (learning_dir / "learned_items.json").write_text(
            json.dumps([{"id": "l1", "val": "old", "updated_at": "2024-01-01"}])
        )

        changes = {
            "added": {},
            "modified": {"learnings": [{"id": "l1", "val": "new", "updated_at": "2025-01-01"}]},
            "deleted": [],
        }
        state._apply_learning_changes(changes)

        items = json.loads((learning_dir / "learned_items.json").read_text())
        assert items[0]["val"] == "new"

    def test_export_import_snapshot(self, tmp_path: Path):
        state = SyncableState(tmp_path, device_id="dev-1")
        (tmp_path / "memories.json").write_text(json.dumps([{"id": "m1", "content": "test"}]))

        export_path = tmp_path / "export.json"
        state.export_snapshot(export_path)
        assert export_path.exists()

        # Create new state to import into
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        other = SyncableState(other_dir, device_id="dev-2")
        other.import_snapshot(export_path)
        # May or may not apply depending on if there are actual changes

    def test_import_snapshot_no_changes(self, tmp_path: Path):
        state = SyncableState(tmp_path, device_id="dev-1")
        snapshot = state.create_snapshot()
        export_path = tmp_path / "export.json"
        export_path.write_text(json.dumps(snapshot.to_dict(), default=str))

        result = state.import_snapshot(export_path)
        assert result is False  # No changes to import

    def test_compute_delta(self, tmp_path: Path):
        state = SyncableState(tmp_path, device_id="dev-1")
        peer_snapshot = StateSnapshot.create("peer", {"key": "val"}, 1)
        delta = state.compute_delta(peer_snapshot)
        assert delta.source_device == "dev-1"
        assert delta.target_device == "peer"

    def test_peer_version_persistence(self, tmp_path: Path):
        state = SyncableState(tmp_path, device_id="dev-1")
        state.set_peer_version("peer-x", 5)
        assert state.get_peer_version("peer-x") == 5

        # Reload from disk
        state2 = SyncableState(tmp_path, device_id="dev-1")
        assert state2.get_peer_version("peer-x") == 5

    def test_version_persistence(self, tmp_path: Path):
        state = SyncableState(tmp_path, device_id="dev-1")
        state.increment_version()
        state.increment_version()
        assert state.version == 2

        state2 = SyncableState(tmp_path, device_id="dev-1")
        assert state2.version == 2

    def test_device_id_persistence(self, tmp_path: Path):
        state = SyncableState(tmp_path)
        device_id = state.device_id
        state2 = SyncableState(tmp_path)
        assert state2.device_id == device_id
