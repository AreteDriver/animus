"""
Coverage push round 2.

Targets: sync/discovery.py, sync/server.py, learning/transparency.py,
         learning/approval.py, learning/rollback.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ===================================================================
# sync/discovery.py — 66% → 85%+
# ===================================================================


class TestDiscoveryCoveragePush:
    """Cover uncovered lines in sync/discovery.py."""

    def test_discovered_device_hash_and_eq(self):
        """Lines 32-38: __hash__ and __eq__."""
        from animus.sync.discovery import DiscoveredDevice

        d1 = DiscoveredDevice(device_id="a", name="a", host="1.2.3.4", port=8422, version="0.5")
        d2 = DiscoveredDevice(device_id="a", name="b", host="5.6.7.8", port=9999, version="0.6")
        d3 = DiscoveredDevice(device_id="c", name="c", host="1.2.3.4", port=8422, version="0.5")
        # Same device_id → equal, same hash
        assert d1 == d2
        assert hash(d1) == hash(d2)
        # Different device_id → not equal
        assert d1 != d3
        # Not a DiscoveredDevice → False
        assert d1 != "not-a-device"

    def test_discovered_device_address(self):
        """Line 43: address property."""
        from animus.sync.discovery import DiscoveredDevice

        d = DiscoveredDevice(device_id="a", name="a", host="10.0.0.1", port=9000, version="1.0")
        assert d.address == "ws://10.0.0.1:9000"

    def test_discovered_device_to_dict(self):
        """Lines 45-55: to_dict serialization."""
        from animus.sync.discovery import DiscoveredDevice

        d = DiscoveredDevice(
            device_id="id1",
            name="dev1",
            host="host1",
            port=8422,
            version="0.5",
        )
        data = d.to_dict()
        assert data["device_id"] == "id1"
        assert data["name"] == "dev1"
        assert "discovered_at" in data
        assert "last_seen" in data

    def test_start_success(self):
        """Lines 119-157: start() full success path with mocked zeroconf."""
        from animus.sync.discovery import DeviceDiscovery

        dd = DeviceDiscovery("dev-1", "my-dev", 8422)

        mock_zeroconf_cls = MagicMock()
        mock_service_info_cls = MagicMock()
        mock_browser_cls = MagicMock()

        mock_zeroconf_mod = MagicMock()
        mock_zeroconf_mod.Zeroconf = mock_zeroconf_cls
        mock_zeroconf_mod.ServiceInfo = mock_service_info_cls
        mock_zeroconf_mod.ServiceBrowser = mock_browser_cls

        with (
            patch.dict("sys.modules", {"zeroconf": mock_zeroconf_mod}),
            patch("socket.socket") as mock_sock,
        ):
            mock_sock.return_value.getsockname.return_value = ("192.168.1.10", 0)
            result = dd.start()

        assert result is True
        assert dd.is_running is True
        mock_zeroconf_cls.return_value.register_service.assert_called_once()
        mock_browser_cls.assert_called_once()

    def test_start_exception(self):
        """Lines 154-157: start() catches exception and calls stop."""
        from animus.sync.discovery import DeviceDiscovery

        dd = DeviceDiscovery("dev-1", "my-dev", 8422)

        mock_zeroconf_mod = MagicMock()
        mock_zeroconf_mod.Zeroconf.side_effect = RuntimeError("fail")

        with (
            patch.dict("sys.modules", {"zeroconf": mock_zeroconf_mod}),
            patch.object(dd, "stop") as mock_stop,
        ):
            result = dd.start()
        assert result is False
        mock_stop.assert_called_once()

    def test_stop_with_unregister_exception(self):
        """Lines 166-176: stop() handles exceptions during unregister/close."""
        from animus.sync.discovery import DeviceDiscovery

        dd = DeviceDiscovery("dev-1", "my-dev", 8422)
        dd._running = True
        mock_zc = MagicMock()
        mock_zc.unregister_service.side_effect = RuntimeError("unregister fail")
        mock_zc.close.side_effect = RuntimeError("close fail")
        dd._zeroconf = mock_zc
        dd._service_info = MagicMock()

        dd.stop()  # Should not raise
        assert dd.is_running is False
        assert dd._zeroconf is None

    def test_get_local_ip_success(self):
        """Lines 189-193: _get_local_ip successful."""
        from animus.sync.discovery import DeviceDiscovery

        dd = DeviceDiscovery("dev-1", "my-dev", 8422)
        with patch("socket.socket") as mock_sock:
            mock_sock.return_value.getsockname.return_value = ("10.0.0.5", 0)
            ip = dd._get_local_ip()
        assert ip == "10.0.0.5"

    def test_on_service_state_change_added(self):
        """Lines 197-214: _on_service_state_change with Added."""
        from animus.sync.discovery import DeviceDiscovery

        dd = DeviceDiscovery("dev-1", "my-dev", 8422)

        # Mock zeroconf module for ServiceStateChange import
        mock_state_change = MagicMock()
        mock_state_change.Added = "added"
        mock_state_change.Removed = "removed"
        mock_zeroconf_mod = MagicMock()
        mock_zeroconf_mod.ServiceStateChange = mock_state_change

        mock_zc = MagicMock()
        mock_info = MagicMock()
        mock_zc.get_service_info.return_value = mock_info

        with (
            patch.dict("sys.modules", {"zeroconf": mock_zeroconf_mod}),
            patch.object(dd, "_handle_service_added") as mock_add,
        ):
            dd._on_service_state_change(mock_zc, "_animus._tcp.", "svc", "added")
        mock_add.assert_called_once_with(mock_info, "svc")

    def test_on_service_state_change_removed(self):
        """Lines 213-214: _on_service_state_change with Removed."""
        from animus.sync.discovery import DeviceDiscovery

        mock_state_change = MagicMock()
        mock_state_change.Added = "added"
        mock_state_change.Removed = "removed"
        mock_zeroconf_mod = MagicMock()
        mock_zeroconf_mod.ServiceStateChange = mock_state_change

        dd = DeviceDiscovery("dev-1", "my-dev", 8422)
        with (
            patch.dict("sys.modules", {"zeroconf": mock_zeroconf_mod}),
            patch.object(dd, "_handle_service_removed") as mock_rem,
        ):
            dd._on_service_state_change(MagicMock(), "_animus._tcp.", "svc", "removed")
        mock_rem.assert_called_once_with("svc")

    def test_on_service_state_change_no_info(self):
        """_on_service_state_change Added but get_service_info returns None."""
        from animus.sync.discovery import DeviceDiscovery

        mock_state_change = MagicMock()
        mock_state_change.Added = "added"
        mock_zeroconf_mod = MagicMock()
        mock_zeroconf_mod.ServiceStateChange = mock_state_change

        dd = DeviceDiscovery("dev-1", "my-dev", 8422)
        mock_zc = MagicMock()
        mock_zc.get_service_info.return_value = None

        with (
            patch.dict("sys.modules", {"zeroconf": mock_zeroconf_mod}),
            patch.object(dd, "_handle_service_added") as mock_add,
        ):
            dd._on_service_state_change(mock_zc, "_animus._tcp.", "svc", "added")
        mock_add.assert_not_called()

    def test_handle_service_added_success(self):
        """Lines 216-248: _handle_service_added full success path."""
        from animus.sync.discovery import SERVICE_TYPE, DeviceDiscovery

        dd = DeviceDiscovery("dev-1", "my-dev", 8422)
        cb = MagicMock()
        dd.add_callback(cb)

        mock_info = MagicMock()
        mock_info.properties = {
            b"device_id": b"other-dev",
            b"version": b"1.0",
        }
        mock_info.parsed_addresses.return_value = ["192.168.1.50"]
        mock_info.port = 9000

        dd._handle_service_added(mock_info, f"other-device.{SERVICE_TYPE}")

        assert "other-dev" in dd._discovered
        device = dd._discovered["other-dev"]
        assert device.host == "192.168.1.50"
        assert device.port == 9000
        assert device.name == "other-device"
        cb.assert_called_once()

    def test_handle_service_added_skip_self(self):
        """Lines 224-226: _handle_service_added skips own device_id."""
        from animus.sync.discovery import DeviceDiscovery

        dd = DeviceDiscovery("dev-1", "my-dev", 8422)
        mock_info = MagicMock()
        mock_info.properties = {b"device_id": b"dev-1", b"version": b"1.0"}

        dd._handle_service_added(mock_info, "svc")
        assert len(dd._discovered) == 0

    def test_handle_service_added_no_addresses(self):
        """Lines 230-231: _handle_service_added returns when no addresses."""
        from animus.sync.discovery import DeviceDiscovery

        dd = DeviceDiscovery("dev-1", "my-dev", 8422)
        mock_info = MagicMock()
        mock_info.properties = {b"device_id": b"other", b"version": b"1.0"}
        mock_info.parsed_addresses.return_value = []

        dd._handle_service_added(mock_info, "svc")
        assert len(dd._discovered) == 0

    def test_handle_service_added_exception(self):
        """Lines 250-251: _handle_service_added catches exception."""
        from animus.sync.discovery import DeviceDiscovery

        dd = DeviceDiscovery("dev-1", "my-dev", 8422)
        mock_info = MagicMock()
        mock_info.properties = {b"device_id": b"other", b"version": b"1.0"}
        mock_info.parsed_addresses.side_effect = RuntimeError("boom")

        dd._handle_service_added(mock_info, "svc")  # Should not raise

    def test_remove_mock_device_not_found(self):
        """MockDeviceDiscovery.remove_mock_device with unknown ID."""
        from animus.sync.discovery import MockDeviceDiscovery

        md = MockDeviceDiscovery("dev-1", "my-dev", 8422)
        md.remove_mock_device("nonexistent")  # Should not raise


# ===================================================================
# sync/server.py — 70% → 85%+
# ===================================================================


class TestSyncServerCoveragePush:
    """Cover uncovered lines in sync/server.py."""

    def _make_state(self, tmp_path: Path):
        from animus.sync.state import SyncableState

        return SyncableState(tmp_path, device_id="srv-dev")

    def test_start_success(self, tmp_path: Path):
        """Lines 106-117: start() success with mocked websockets."""
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state, port=9999)

        mock_ws_mod = MagicMock()
        mock_serve = AsyncMock()
        mock_ws_mod.serve = mock_serve

        with patch.dict("sys.modules", {"websockets": mock_ws_mod}):
            result = asyncio.run(server.start())
        assert result is True
        assert server.is_running is True

    def test_start_exception(self, tmp_path: Path):
        """Lines 115-117: start() catches exception."""
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state, port=9999)

        mock_ws_mod = MagicMock()
        mock_ws_mod.serve = AsyncMock(side_effect=RuntimeError("bind fail"))

        with patch.dict("sys.modules", {"websockets": mock_ws_mod}):
            result = asyncio.run(server.start())
        assert result is False

    def test_stop_peer_close_error(self, tmp_path: Path):
        """Lines 130-131: stop() handles peer close exception."""
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)
        server._running = True

        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock(side_effect=RuntimeError("close fail"))
        peer = ConnectedPeer(device_id="p1", device_name="peer1", websocket=mock_ws)
        server._peers["p1"] = peer
        mock_srv = MagicMock()
        mock_srv.close = MagicMock()
        mock_srv.wait_closed = AsyncMock()
        server._server = mock_srv

        asyncio.run(server.stop())
        assert server.peer_count == 0

    def test_handle_connection_full_flow(self, tmp_path: Path):
        """Lines 144-176: _handle_connection with auth, messages, disconnect."""
        from animus.sync.protocol import MessageType, SyncMessage
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state, shared_secret="secret")

        # Prepare callbacks
        connected_cb = MagicMock()
        disconnected_cb = MagicMock()
        server.add_peer_connected_callback(connected_cb)
        server.add_peer_disconnected_callback(disconnected_cb)

        import hashlib

        auth_hash = hashlib.sha256(b"secret").hexdigest()
        auth_msg = SyncMessage(
            type=MessageType.AUTH,
            device_id="client-1",
            payload={"auth_hash": auth_hash, "device_name": "test-client"},
        )

        ping_msg = SyncMessage(
            type=MessageType.PING,
            device_id="client-1",
            payload={},
        )

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=auth_msg.to_json())
        # Simulate one message then disconnect
        mock_ws.__aiter__ = MagicMock(return_value=iter([ping_msg.to_json()]))

        asyncio.run(server._handle_connection(mock_ws, "/sync"))

        connected_cb.assert_called_once()
        disconnected_cb.assert_called_once()

    def test_handle_connection_auth_fails(self, tmp_path: Path):
        """_handle_connection when authentication fails."""
        from animus.sync.protocol import MessageType, SyncMessage
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state, shared_secret="secret")

        bad_auth = SyncMessage(
            type=MessageType.AUTH,
            device_id="bad",
            payload={"auth_hash": "wrong"},
        )

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=bad_auth.to_json())

        asyncio.run(server._handle_connection(mock_ws, "/sync"))
        # No peers added
        assert server.peer_count == 0

    def test_handle_connection_callback_error(self, tmp_path: Path):
        """Lines 158-159: connected callback raises exception."""
        import hashlib

        from animus.sync.protocol import MessageType, SyncMessage
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state, shared_secret="secret")

        bad_cb = MagicMock(side_effect=RuntimeError("callback boom"))
        server.add_peer_connected_callback(bad_cb)

        auth_hash = hashlib.sha256(b"secret").hexdigest()
        auth_msg = SyncMessage(
            type=MessageType.AUTH,
            device_id="client-1",
            payload={"auth_hash": auth_hash, "device_name": "test"},
        )

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=auth_msg.to_json())
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        asyncio.run(server._handle_connection(mock_ws, "/sync"))
        # Should not raise despite callback error

    def test_handle_message_unknown_type(self, tmp_path: Path):
        """Line 254: unknown message type logs warning."""
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        peer = ConnectedPeer(device_id="p1", device_name="peer1", websocket=AsyncMock())

        # Build raw JSON with unknown type (can't use SyncMessage which validates enum)
        import json as _json

        raw = _json.dumps(
            {
                "type": "unknown_type",
                "device_id": "p1",
                "timestamp": datetime.now().isoformat(),
                "payload": {},
                "message_id": "test",
            }
        )

        asyncio.run(server._handle_message(peer, raw))

    def test_handle_message_parse_error(self, tmp_path: Path):
        """Lines 256-267: _handle_message exception sends error to peer."""
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        mock_ws = AsyncMock()
        peer = ConnectedPeer(device_id="p1", device_name="peer1", websocket=mock_ws)

        asyncio.run(server._handle_message(peer, "not valid json"))
        mock_ws.send.assert_called()  # Error message sent

    def test_handle_message_error_send_fails(self, tmp_path: Path):
        """Lines 266-267: error send also fails — silently."""
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock(side_effect=RuntimeError("send fail"))
        peer = ConnectedPeer(device_id="p1", device_name="peer1", websocket=mock_ws)

        asyncio.run(server._handle_message(peer, "invalid"))
        # Should not raise

    def test_handle_ping(self, tmp_path: Path):
        """Line 271: _handle_ping sends pong."""
        from animus.sync.protocol import MessageType, SyncMessage
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        mock_ws = AsyncMock()
        peer = ConnectedPeer(device_id="p1", device_name="peer1", websocket=mock_ws)
        msg = SyncMessage(type=MessageType.PING, device_id="p1", payload={})

        asyncio.run(server._handle_ping(peer, msg))
        mock_ws.send.assert_called_once()

    def test_handle_snapshot_request_full(self, tmp_path: Path):
        """Lines 284-316: snapshot request with since_version=0."""
        from animus.sync.protocol import MessageType, SyncMessage
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        mock_ws = AsyncMock()
        peer = ConnectedPeer(device_id="p1", device_name="peer1", websocket=mock_ws)
        msg = SyncMessage(
            type=MessageType.SNAPSHOT_REQUEST,
            device_id="p1",
            payload={"since_version": 0},
        )

        asyncio.run(server._handle_snapshot_request(peer, msg))
        mock_ws.send.assert_called_once()
        assert peer.version == state.version

    def test_handle_snapshot_request_incremental(self, tmp_path: Path):
        """Lines 281-300: snapshot request with since_version > 0."""
        from animus.sync.protocol import MessageType, SyncMessage
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        state._version = 5
        state._peer_versions["p1"] = 3
        server = SyncServer(state)

        mock_ws = AsyncMock()
        peer = ConnectedPeer(device_id="p1", device_name="peer1", websocket=mock_ws)
        msg = SyncMessage(
            type=MessageType.SNAPSHOT_REQUEST,
            device_id="p1",
            payload={"since_version": 3},
        )

        asyncio.run(server._handle_snapshot_request(peer, msg))
        mock_ws.send.assert_called_once()

    def test_handle_delta_push_success(self, tmp_path: Path):
        """Lines 318-345: delta push success."""
        from animus.sync.protocol import MessageType, SyncMessage
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        sync_cb = MagicMock()
        server.add_sync_callback(sync_cb)

        delta_dict = {
            "id": "delta-1",
            "source_device": "p1",
            "target_device": "srv-dev",
            "timestamp": datetime.now().isoformat(),
            "base_version": 0,
            "new_version": 1,
            "changes": {
                "added": {
                    "memories": [{"id": "mem-1", "content": "test"}],
                },
            },
        }

        mock_ws = AsyncMock()
        peer = ConnectedPeer(device_id="p1", device_name="peer1", websocket=mock_ws)
        msg = SyncMessage(
            type=MessageType.DELTA_PUSH,
            device_id="p1",
            payload={"delta": delta_dict},
        )

        asyncio.run(server._handle_delta_push(peer, msg))
        mock_ws.send.assert_called_once()  # ack sent
        sync_cb.assert_called_once()

    def test_handle_delta_push_callback_error(self, tmp_path: Path):
        """Lines 342-343: sync callback error during delta push."""
        from animus.sync.protocol import MessageType, SyncMessage
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        bad_cb = MagicMock(side_effect=RuntimeError("cb error"))
        server.add_sync_callback(bad_cb)

        delta_dict = {
            "id": "delta-2",
            "source_device": "p1",
            "target_device": "srv-dev",
            "timestamp": datetime.now().isoformat(),
            "base_version": 0,
            "new_version": 1,
            "changes": {"k": "v"},
        }

        mock_ws = AsyncMock()
        peer = ConnectedPeer(device_id="p1", device_name="peer1", websocket=mock_ws)
        msg = SyncMessage(
            type=MessageType.DELTA_PUSH,
            device_id="p1",
            payload={"delta": delta_dict},
        )

        asyncio.run(server._handle_delta_push(peer, msg))
        # Should not raise

    def test_handle_delta_push_exception(self, tmp_path: Path):
        """Lines 347-355: delta push with invalid data."""
        from animus.sync.protocol import MessageType, SyncMessage
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        mock_ws = AsyncMock()
        peer = ConnectedPeer(device_id="p1", device_name="peer1", websocket=mock_ws)
        msg = SyncMessage(
            type=MessageType.DELTA_PUSH,
            device_id="p1",
            payload={"delta": {}},  # Missing required fields
        )

        asyncio.run(server._handle_delta_push(peer, msg))
        # Should send failure ack
        mock_ws.send.assert_called()

    def test_handle_status(self, tmp_path: Path):
        """Lines 357-360: _handle_status."""
        from animus.sync.protocol import MessageType, SyncMessage
        from animus.sync.server import ConnectedPeer, SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        peer = ConnectedPeer(device_id="p1", device_name="peer1", websocket=AsyncMock())
        msg = SyncMessage(
            type=MessageType.STATUS,
            device_id="p1",
            payload={"status": "synced"},
        )

        asyncio.run(server._handle_status(peer, msg))  # Just logs

    def test_broadcast_delta_no_peers(self, tmp_path: Path):
        """Lines 369-370: broadcast with no peers returns 0."""
        from animus.sync.server import SyncServer
        from animus.sync.state import StateDelta

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        delta = StateDelta(
            id="d1",
            source_device="srv",
            target_device="*",
            timestamp=datetime.now(),
            base_version=0,
            new_version=1,
            changes={},
        )
        count = asyncio.run(server.broadcast_delta(delta))
        assert count == 0

    def test_broadcast_delta_with_peers(self, tmp_path: Path):
        """Lines 378-386: broadcast delta to peers."""
        from animus.sync.server import ConnectedPeer, SyncServer
        from animus.sync.state import StateDelta

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws2.send = AsyncMock(side_effect=RuntimeError("send fail"))

        server._peers["p1"] = ConnectedPeer(device_id="p1", device_name="peer1", websocket=mock_ws1)
        server._peers["p2"] = ConnectedPeer(device_id="p2", device_name="peer2", websocket=mock_ws2)

        delta = StateDelta(
            id="d1",
            source_device="srv",
            target_device="*",
            timestamp=datetime.now(),
            base_version=0,
            new_version=1,
            changes={},
        )
        count = asyncio.run(server.broadcast_delta(delta))
        assert count == 1  # One succeeded, one failed

    def test_authenticate_exception(self, tmp_path: Path):
        """Lines 233-235: _authenticate catches generic exception."""
        from animus.sync.server import SyncServer

        state = self._make_state(tmp_path)
        server = SyncServer(state)

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=RuntimeError("recv error"))

        peer = asyncio.run(server._authenticate(mock_ws))
        assert peer is None


# ===================================================================
# learning/transparency.py — 70% → 85%+
# ===================================================================


class TestTransparencyCoveragePush:
    """Cover uncovered lines in learning/transparency.py."""

    def test_learning_event_from_dict(self, tmp_path: Path):
        """Lines 62-72: LearningEvent.from_dict."""
        from animus.learning.transparency import LearningEvent

        data = {
            "id": "evt-1",
            "event_type": "detected",
            "learned_item_id": "item-1",
            "timestamp": "2025-01-01T10:00:00",
            "details": {"key": "val"},
            "user_action": "approve",
        }
        evt = LearningEvent.from_dict(data)
        assert evt.id == "evt-1"
        assert evt.event_type == "detected"
        assert evt.user_action == "approve"

    def test_load_events_from_file(self, tmp_path: Path):
        """Lines 118-129: _load_events from disk."""
        from animus.learning.transparency import LearningTransparency

        events_file = tmp_path / "learning_events.json"
        events_data = [
            {
                "id": "e1",
                "event_type": "detected",
                "learned_item_id": "item-1",
                "timestamp": "2025-01-01T12:00:00",
                "details": {},
                "user_action": None,
            }
        ]
        events_file.write_text(json.dumps(events_data))

        t = LearningTransparency(tmp_path)
        assert len(t._events) == 1
        assert t._events[0].id == "e1"

    def test_load_events_corrupt_file(self, tmp_path: Path):
        """Lines 128-129: _load_events with corrupt file."""
        events_file = tmp_path / "learning_events.json"
        events_file.write_text("not valid json")

        from animus.learning.transparency import LearningTransparency

        t = LearningTransparency(tmp_path)
        assert len(t._events) == 0

    def test_dashboard_data_to_dict(self, tmp_path: Path):
        """Lines 88-99: LearningDashboardData.to_dict."""
        from animus.learning.transparency import LearningDashboardData

        data = LearningDashboardData(
            total_learned=5,
            pending_approval=2,
            recently_applied=[],
            recently_rejected=["rej-1"],
            by_category={"style": 3},
            confidence_distribution={"low": 1, "medium": 2, "high": 2},
            events_today=10,
            guardrail_violations=0,
        )
        d = data.to_dict()
        assert d["total_learned"] == 5
        assert d["recently_rejected"] == ["rej-1"]

    def test_get_history_with_filters(self, tmp_path: Path):
        """Lines 251-257: get_history with event_type and since filters."""
        from animus.learning.transparency import LearningTransparency

        t = LearningTransparency(tmp_path)
        t.log_event("detected", "item-1")
        t.log_event("approved", "item-1")
        t.log_event("detected", "item-2")

        # Filter by type
        detected = t.get_history(event_type="detected")
        assert len(detected) == 2

        # Filter by since
        future = datetime.now() + timedelta(hours=1)
        empty = t.get_history(since=future)
        assert len(empty) == 0

        # Both filters
        past = datetime.now() - timedelta(hours=1)
        recent = t.get_history(event_type="approved", since=past)
        assert len(recent) == 1

    def test_export_log_json(self, tmp_path: Path):
        """Lines 269-273: export_log JSON format."""
        from animus.learning.transparency import LearningTransparency

        t = LearningTransparency(tmp_path)
        t.log_event("detected", "item-1")
        t.log_event("approved", "item-1")

        result = t.export_log(format="json")
        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_export_log_jsonl(self, tmp_path: Path):
        """Lines 269-271: export_log JSONL format."""
        from animus.learning.transparency import LearningTransparency

        t = LearningTransparency(tmp_path)
        t.log_event("detected", "item-1")

        result = t.export_log(format="jsonl")
        lines = result.strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["event_type"] == "detected"

    def test_explain_learning(self, tmp_path: Path):
        """Lines 275-305: explain_learning."""
        from animus.learning.categories import LearnedItem, LearningCategory
        from animus.learning.transparency import LearningTransparency

        t = LearningTransparency(tmp_path)
        item = LearnedItem.create(
            category=LearningCategory.STYLE,
            content="Prefers dark mode",
            confidence=0.85,
            evidence=["e1", "e2"],
        )

        # Log some events for this item
        t.log_event("detected", item.id)
        t.log_event("approved", item.id, user_action="approve")

        explanation = t.explain_learning(item)
        assert "Prefers dark mode" in explanation
        assert "85%" in explanation
        assert "Timeline:" in explanation
        assert "detected" in explanation
        assert "User action: approve" in explanation

    def test_get_statistics(self, tmp_path: Path):
        """Lines 307-324: get_statistics."""
        from animus.learning.transparency import LearningTransparency

        t = LearningTransparency(tmp_path)
        t.log_event("detected", "item-1")
        t.log_event("detected", "item-2")
        t.log_event("approved", "item-1")

        stats = t.get_statistics()
        assert stats["total_events"] == 3
        assert stats["by_type"]["detected"] == 2
        assert stats["by_type"]["approved"] == 1
        assert len(stats["daily_counts"]) == 7


# ===================================================================
# learning/approval.py — 71% → 85%+
# ===================================================================


class TestApprovalCoveragePush:
    """Cover uncovered lines in learning/approval.py."""

    def test_approval_request_is_expired_no_expires(self):
        """Lines 75-77: is_expired with no expires_at."""
        from animus.learning.approval import ApprovalRequest
        from animus.learning.categories import LearningCategory

        req = ApprovalRequest(
            id="r1",
            learned_item_id="i1",
            category=LearningCategory.STYLE,
            description="test",
            evidence_summary="test",
            created_at=datetime.now(),
            expires_at=None,
        )
        assert req.is_expired() is False

    def test_approval_request_from_dict(self):
        """Lines 107-126: ApprovalRequest.from_dict."""
        from animus.learning.approval import ApprovalRequest

        data = {
            "id": "r1",
            "learned_item_id": "i1",
            "category": "style",
            "description": "test desc",
            "evidence_summary": "3 obs",
            "created_at": "2025-01-01T10:00:00",
            "expires_at": "2025-01-08T10:00:00",
            "status": "approved",
            "user_response": "LGTM",
            "responded_at": "2025-01-02T10:00:00",
            "metadata": {"key": "val"},
        }
        req = ApprovalRequest.from_dict(data)
        assert req.id == "r1"
        assert req.user_response == "LGTM"
        assert req.responded_at is not None

    def test_approval_request_from_dict_no_optional(self):
        """from_dict with missing optional fields."""
        from animus.learning.approval import ApprovalRequest, ApprovalStatus

        data = {
            "id": "r2",
            "learned_item_id": "i2",
            "category": "preference",
            "description": "test",
            "evidence_summary": "obs",
            "created_at": "2025-01-01T10:00:00",
        }
        req = ApprovalRequest.from_dict(data)
        assert req.expires_at is None
        assert req.responded_at is None
        assert req.status == ApprovalStatus.PENDING

    def test_load_requests_from_file_with_expired(self, tmp_path: Path):
        """Lines 151-169: _load_requests with expired requests."""
        from animus.learning.approval import ApprovalManager

        past = (datetime.now() - timedelta(days=30)).isoformat()
        future = (datetime.now() + timedelta(days=7)).isoformat()

        data = {
            "pending": [
                {
                    "id": "expired-1",
                    "learned_item_id": "i1",
                    "category": "style",
                    "description": "old request",
                    "evidence_summary": "obs",
                    "created_at": past,
                    "expires_at": past,
                    "status": "pending",
                },
                {
                    "id": "active-1",
                    "learned_item_id": "i2",
                    "category": "style",
                    "description": "active request",
                    "evidence_summary": "obs",
                    "created_at": datetime.now().isoformat(),
                    "expires_at": future,
                    "status": "pending",
                },
            ],
            "history": [
                {
                    "id": "hist-1",
                    "learned_item_id": "i3",
                    "category": "preference",
                    "description": "old",
                    "evidence_summary": "obs",
                    "created_at": past,
                    "status": "approved",
                },
            ],
        }

        requests_file = tmp_path / "approval_requests.json"
        requests_file.write_text(json.dumps(data))

        mgr = ApprovalManager(tmp_path)
        # expired-1 should be in history, not pending
        assert "expired-1" not in mgr._pending_requests
        assert "active-1" in mgr._pending_requests
        assert len(mgr._history) == 2  # hist-1 + expired-1

    def test_load_requests_corrupt_file(self, tmp_path: Path):
        """Lines 168-169: _load_requests with corrupt file."""
        requests_file = tmp_path / "approval_requests.json"
        requests_file.write_text("not json")

        from animus.learning.approval import ApprovalManager

        mgr = ApprovalManager(tmp_path)
        assert len(mgr._pending_requests) == 0

    def test_needs_approval(self, tmp_path: Path):
        """Lines 186-189: needs_approval for different categories."""
        from animus.learning.approval import ApprovalManager
        from animus.learning.categories import LearnedItem, LearningCategory

        mgr = ApprovalManager(tmp_path)

        # CAPABILITY requires APPROVE
        cap = LearnedItem.create(
            category=LearningCategory.CAPABILITY,
            content="test",
            confidence=0.9,
            evidence=[],
        )
        assert mgr.needs_approval(cap) is True

        # STYLE is AUTO — doesn't need approval
        style = LearnedItem.create(
            category=LearningCategory.STYLE,
            content="test",
            confidence=0.9,
            evidence=[],
        )
        assert mgr.needs_approval(style) is False

    def test_should_notify(self, tmp_path: Path):
        """Lines 191-194: should_notify."""
        from animus.learning.approval import ApprovalManager
        from animus.learning.categories import LearnedItem, LearningCategory

        mgr = ApprovalManager(tmp_path)

        # WORKFLOW is NOTIFY
        wf = LearnedItem.create(
            category=LearningCategory.WORKFLOW,
            content="test",
            confidence=0.9,
            evidence=[],
        )
        assert mgr.should_notify(wf) is True

        # STYLE is AUTO — not NOTIFY
        style = LearnedItem.create(
            category=LearningCategory.STYLE,
            content="test",
            confidence=0.9,
            evidence=[],
        )
        assert mgr.should_notify(style) is False

    def test_approve_not_found(self, tmp_path: Path):
        """Line 236: approve with unknown request_id."""
        from animus.learning.approval import ApprovalManager

        mgr = ApprovalManager(tmp_path)
        assert mgr.approve("nonexistent") is False

    def test_reject_not_found(self, tmp_path: Path):
        """Line 259: reject with unknown request_id."""
        from animus.learning.approval import ApprovalManager

        mgr = ApprovalManager(tmp_path)
        assert mgr.reject("nonexistent") is False

    def test_get_pending_expires_stale(self, tmp_path: Path):
        """Lines 269-284: get_pending removes expired requests."""
        from animus.learning.approval import ApprovalManager, ApprovalRequest
        from animus.learning.categories import LearningCategory

        mgr = ApprovalManager(tmp_path)

        # Manually insert an expired request
        expired_req = ApprovalRequest(
            id="exp-1",
            learned_item_id="i1",
            category=LearningCategory.STYLE,
            description="old",
            evidence_summary="obs",
            created_at=datetime.now() - timedelta(days=30),
            expires_at=datetime.now() - timedelta(days=1),
        )
        mgr._pending_requests["exp-1"] = expired_req

        pending = mgr.get_pending()
        assert "exp-1" not in [r.id for r in pending]

    def test_get_request(self, tmp_path: Path):
        """Line 289: get_request."""
        from animus.learning.approval import ApprovalManager
        from animus.learning.categories import LearnedItem, LearningCategory

        mgr = ApprovalManager(tmp_path)
        item = LearnedItem.create(
            category=LearningCategory.CAPABILITY,
            content="test",
            confidence=0.9,
            evidence=[],
        )
        req = mgr.request_approval(item)
        found = mgr.get_request(req.id)
        assert found is not None
        assert found.id == req.id
        assert mgr.get_request("nonexistent") is None

    def test_get_history(self, tmp_path: Path):
        """Lines 291-293: get_history."""
        from animus.learning.approval import ApprovalManager
        from animus.learning.categories import LearnedItem, LearningCategory

        mgr = ApprovalManager(tmp_path)
        item = LearnedItem.create(
            category=LearningCategory.CAPABILITY,
            content="test",
            confidence=0.9,
            evidence=[],
        )
        req = mgr.request_approval(item)
        mgr.approve(req.id)

        history = mgr.get_history()
        assert len(history) == 1

    def test_notify_user_with_callback(self, tmp_path: Path):
        """Lines 302-304: notify_user with callback."""
        cb = MagicMock()
        from animus.learning.approval import ApprovalManager

        mgr = ApprovalManager(tmp_path, notification_callback=cb)
        mgr.notify_user("Hello!")
        cb.assert_called_once_with("Hello!")

    def test_notify_user_no_callback(self, tmp_path: Path):
        """notify_user without callback."""
        from animus.learning.approval import ApprovalManager

        mgr = ApprovalManager(tmp_path)
        mgr.notify_user("Hello!")  # Should not raise

    def test_get_statistics(self, tmp_path: Path):
        """Lines 306-319: get_statistics."""
        from animus.learning.approval import ApprovalManager
        from animus.learning.categories import LearnedItem, LearningCategory

        mgr = ApprovalManager(tmp_path)
        item1 = LearnedItem.create(
            category=LearningCategory.CAPABILITY,
            content="t1",
            confidence=0.9,
            evidence=[],
        )
        item2 = LearnedItem.create(
            category=LearningCategory.CAPABILITY,
            content="t2",
            confidence=0.9,
            evidence=[],
        )
        req1 = mgr.request_approval(item1)
        req2 = mgr.request_approval(item2)
        mgr.approve(req1.id)
        mgr.reject(req2.id, "no")

        stats = mgr.get_statistics()
        assert stats["approved"] == 1
        assert stats["rejected"] == 1
        assert stats["total_processed"] == 2


# ===================================================================
# learning/rollback.py — 73% → 85%+
# ===================================================================


class TestRollbackCoveragePush:
    """Cover uncovered lines in learning/rollback.py."""

    def test_rollback_point_verify_integrity_pass(self):
        """Lines 56-71: verify_integrity passes."""
        from animus.learning.rollback import RollbackPoint

        rp = RollbackPoint(
            id="rp1",
            timestamp=datetime.now(),
            description="test",
            learned_item_ids=["a", "b"],
            checksum="abc",
        )
        assert rp.verify_integrity(["a", "b", "c"]) is True

    def test_rollback_point_verify_integrity_fail(self):
        """Lines 68-70: verify_integrity fails when item missing."""
        from animus.learning.rollback import RollbackPoint

        rp = RollbackPoint(
            id="rp1",
            timestamp=datetime.now(),
            description="test",
            learned_item_ids=["a", "b"],
            checksum="abc",
        )
        assert rp.verify_integrity(["a"]) is False

    def test_rollback_point_from_dict(self):
        """Lines 84-94: RollbackPoint.from_dict."""
        from animus.learning.rollback import RollbackPoint

        data = {
            "id": "rp1",
            "timestamp": "2025-01-01T10:00:00",
            "description": "checkpoint",
            "learned_item_ids": ["i1", "i2"],
            "checksum": "abc123",
            "metadata": {"key": "val"},
        }
        rp = RollbackPoint.from_dict(data)
        assert rp.id == "rp1"
        assert rp.metadata == {"key": "val"}

    def test_unlearn_record_to_dict(self):
        """Lines 107-115: UnlearnRecord.to_dict."""
        from animus.learning.rollback import UnlearnRecord

        rec = UnlearnRecord(
            id="ur1",
            learned_item_id="i1",
            learned_item_content="content",
            unlearned_at=datetime.now(),
            reason="test",
        )
        d = rec.to_dict()
        assert d["id"] == "ur1"
        assert d["reason"] == "test"

    def test_load_data_from_file(self, tmp_path: Path):
        """Lines 135-156: _load_data from disk."""
        from animus.learning.rollback import RollbackManager

        data = {
            "rollback_points": [
                {
                    "id": "rp1",
                    "timestamp": "2025-01-01T10:00:00",
                    "description": "cp1",
                    "learned_item_ids": ["a"],
                    "checksum": "abc",
                    "metadata": {},
                }
            ],
            "unlearn_history": [
                {
                    "id": "ur1",
                    "learned_item_id": "i1",
                    "learned_item_content": "stuff",
                    "unlearned_at": "2025-01-01T10:00:00",
                    "reason": "user asked",
                }
            ],
        }
        data_file = tmp_path / "rollback_data.json"
        data_file.write_text(json.dumps(data))

        mgr = RollbackManager(tmp_path)
        assert len(mgr._rollback_points) == 1
        assert len(mgr._unlearn_history) == 1
        assert mgr._unlearn_history[0].reason == "user asked"

    def test_load_data_corrupt_file(self, tmp_path: Path):
        """Lines 155-156: _load_data with corrupt file."""
        data_file = tmp_path / "rollback_data.json"
        data_file.write_text("not json")

        from animus.learning.rollback import RollbackManager

        mgr = RollbackManager(tmp_path)
        assert len(mgr._rollback_points) == 0

    def test_get_point_by_time_found(self, tmp_path: Path):
        """Lines 263-276: get_point_by_time finds closest point."""
        from animus.learning.rollback import RollbackManager

        mgr = RollbackManager(tmp_path)
        mgr.create_checkpoint("early", [])
        mgr.create_checkpoint("later", [])

        # Target far in the future — should get the latest
        result = mgr.get_point_by_time(datetime.now() + timedelta(hours=1))
        assert result is not None
        assert result.description == "later"

    def test_get_point_by_time_none(self, tmp_path: Path):
        """Lines 273-275: get_point_by_time returns None."""
        from animus.learning.rollback import RollbackManager

        mgr = RollbackManager(tmp_path)
        mgr.create_checkpoint("now", [])

        # Target in the past — no candidates
        result = mgr.get_point_by_time(datetime(2000, 1, 1))
        assert result is None

    def test_get_unlearn_history(self, tmp_path: Path):
        """Lines 278-280: get_unlearn_history."""
        from animus.learning.categories import LearnedItem, LearningCategory
        from animus.learning.rollback import RollbackManager

        mgr = RollbackManager(tmp_path)
        item = LearnedItem.create(
            category=LearningCategory.STYLE,
            content="test",
            confidence=0.8,
            evidence=[],
        )
        mgr.record_unlearn(item, "test reason")

        history = mgr.get_unlearn_history()
        assert len(history) == 1
        assert history[0].reason == "test reason"

    def test_get_statistics(self, tmp_path: Path):
        """Lines 282-293: get_statistics."""
        from animus.learning.rollback import RollbackManager

        mgr = RollbackManager(tmp_path)
        mgr.create_checkpoint("cp1", [])

        stats = mgr.get_statistics()
        assert stats["checkpoint_count"] == 1
        assert stats["unlearn_count"] == 0
        assert stats["oldest_checkpoint"] is not None
        assert stats["newest_checkpoint"] is not None

    def test_get_statistics_empty(self, tmp_path: Path):
        """get_statistics with no data."""
        from animus.learning.rollback import RollbackManager

        mgr = RollbackManager(tmp_path)
        stats = mgr.get_statistics()
        assert stats["checkpoint_count"] == 0
        assert stats["oldest_checkpoint"] is None
        assert stats["newest_checkpoint"] is None

    def test_get_items_to_unlearn_not_found(self, tmp_path: Path):
        """Lines 240-242: get_items_to_unlearn with unknown point ID."""
        from animus.learning.rollback import RollbackManager

        mgr = RollbackManager(tmp_path)
        result = mgr.get_items_to_unlearn("nonexistent", [])
        assert result == []
