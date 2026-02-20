"""Tests for cross-device sync module."""

import json
import tempfile
from pathlib import Path

from animus.sync.discovery import DiscoveredDevice, MockDeviceDiscovery
from animus.sync.protocol import (
    MessageType,
    SyncMessage,
    create_auth_message,
    create_auth_ok_message,
    create_delta_ack,
    create_delta_push,
    create_ping,
    create_pong,
    create_snapshot_request,
    create_snapshot_response,
)
from animus.sync.state import StateDelta, StateSnapshot, SyncableState


class TestStateSnapshot:
    """Tests for StateSnapshot class."""

    def test_create_snapshot(self):
        """Test creating a snapshot from data."""
        data = {"memories": [{"id": "1", "content": "test"}], "config": {"key": "value"}}
        snapshot = StateSnapshot.create("device-123", data, version=5)

        assert snapshot.device_id == "device-123"
        assert snapshot.version == 5
        assert snapshot.data == data
        assert snapshot.checksum  # Should have computed checksum
        assert snapshot.id  # Should have generated ID

    def test_snapshot_serialization(self):
        """Test snapshot to_dict and from_dict."""
        data = {"memories": [], "learnings": []}
        snapshot = StateSnapshot.create("device-abc", data, version=1)

        serialized = snapshot.to_dict()
        restored = StateSnapshot.from_dict(serialized)

        assert restored.device_id == snapshot.device_id
        assert restored.version == snapshot.version
        assert restored.data == snapshot.data
        assert restored.checksum == snapshot.checksum

    def test_checksum_consistency(self):
        """Test that same data produces same checksum."""
        data = {"key": "value", "list": [1, 2, 3]}

        snapshot1 = StateSnapshot.create("device-1", data, version=1)
        snapshot2 = StateSnapshot.create("device-2", data, version=1)

        assert snapshot1.checksum == snapshot2.checksum


class TestStateDelta:
    """Tests for StateDelta class."""

    def test_compute_delta_added(self):
        """Test computing delta with added items."""
        old_data = {"existing": "value"}
        new_data = {"existing": "value", "new_key": "new_value"}

        delta = StateDelta.compute(
            source_device="device-a",
            target_device="device-b",
            old_data=old_data,
            new_data=new_data,
            base_version=1,
        )

        assert "new_key" in delta.changes["added"]
        assert delta.changes["added"]["new_key"] == "new_value"
        assert not delta.changes["modified"]
        assert not delta.changes["deleted"]

    def test_compute_delta_modified(self):
        """Test computing delta with modified items."""
        old_data = {"key": "old_value"}
        new_data = {"key": "new_value"}

        delta = StateDelta.compute(
            source_device="device-a",
            target_device="device-b",
            old_data=old_data,
            new_data=new_data,
            base_version=1,
        )

        assert "key" in delta.changes["modified"]
        assert delta.changes["modified"]["key"] == "new_value"
        assert not delta.changes["added"]
        assert not delta.changes["deleted"]

    def test_compute_delta_deleted(self):
        """Test computing delta with deleted items."""
        old_data = {"key1": "value1", "key2": "value2"}
        new_data = {"key1": "value1"}

        delta = StateDelta.compute(
            source_device="device-a",
            target_device="device-b",
            old_data=old_data,
            new_data=new_data,
            base_version=1,
        )

        assert "key2" in delta.changes["deleted"]
        assert not delta.changes["added"]
        assert not delta.changes["modified"]

    def test_delta_is_empty(self):
        """Test is_empty method."""
        data = {"key": "value"}

        delta = StateDelta.compute(
            source_device="a",
            target_device="b",
            old_data=data,
            new_data=data,
            base_version=1,
        )

        assert delta.is_empty()

        delta2 = StateDelta.compute(
            source_device="a",
            target_device="b",
            old_data={},
            new_data={"new": "item"},
            base_version=1,
        )

        assert not delta2.is_empty()

    def test_delta_serialization(self):
        """Test delta to_dict and from_dict."""
        delta = StateDelta.compute(
            source_device="src",
            target_device="dst",
            old_data={"old": "data"},
            new_data={"new": "data"},
            base_version=5,
        )

        serialized = delta.to_dict()
        restored = StateDelta.from_dict(serialized)

        assert restored.source_device == delta.source_device
        assert restored.target_device == delta.target_device
        assert restored.base_version == delta.base_version
        assert restored.new_version == delta.new_version
        assert restored.changes == delta.changes


class TestSyncableState:
    """Tests for SyncableState class."""

    def test_device_id_persistence(self):
        """Test that device ID is persisted and reloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # First instance creates device ID
            state1 = SyncableState(data_dir)
            device_id = state1.device_id

            # Second instance should load same ID
            state2 = SyncableState(data_dir)
            assert state2.device_id == device_id

    def test_version_tracking(self):
        """Test version increment and persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            state = SyncableState(data_dir)
            assert state.version == 0

            state.increment_version()
            assert state.version == 1

            # Reload and check persistence
            state2 = SyncableState(data_dir)
            assert state2.version == 1

    def test_peer_version_tracking(self):
        """Test tracking peer versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            state = SyncableState(data_dir)
            assert state.get_peer_version("peer-1") == 0

            state.set_peer_version("peer-1", 5)
            assert state.get_peer_version("peer-1") == 5

            # Reload and check persistence
            state2 = SyncableState(data_dir)
            assert state2.get_peer_version("peer-1") == 5

    def test_collect_state(self):
        """Test collecting state from data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create some test data
            (data_dir / "memories.json").write_text(
                json.dumps([{"id": "m1", "content": "test memory"}])
            )

            state = SyncableState(data_dir)
            collected = state.collect_state()

            assert "memories" in collected
            assert "learnings" in collected
            assert "guardrails" in collected
            assert "config" in collected

    def test_create_snapshot(self):
        """Test creating snapshot from current state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            state = SyncableState(data_dir)
            state.increment_version()

            snapshot = state.create_snapshot()

            assert snapshot.device_id == state.device_id
            assert snapshot.version == state.version
            assert "memories" in snapshot.data


class TestSyncMessage:
    """Tests for sync protocol messages."""

    def test_message_serialization(self):
        """Test message to_json and from_json."""
        message = SyncMessage(
            type=MessageType.PING,
            device_id="test-device",
            payload={"data": "value"},
        )

        json_str = message.to_json()
        restored = SyncMessage.from_json(json_str)

        assert restored.type == message.type
        assert restored.device_id == message.device_id
        assert restored.payload == message.payload

    def test_message_id_generation(self):
        """Test that message IDs are auto-generated."""
        message = SyncMessage(
            type=MessageType.PING,
            device_id="device",
        )

        assert message.message_id
        assert len(message.message_id) == 8


class TestMessageFactories:
    """Tests for message factory functions."""

    def test_create_auth_message(self):
        """Test auth message creation."""
        msg = create_auth_message("device-123", "secret-password")

        assert msg.type == MessageType.AUTH
        assert msg.device_id == "device-123"
        assert "auth_hash" in msg.payload
        # Should be SHA-256 hash, not plain text
        assert msg.payload["auth_hash"] != "secret-password"

    def test_create_auth_ok_message(self):
        """Test auth OK message creation."""
        msg = create_auth_ok_message("device-1", "Device Name", 5)

        assert msg.type == MessageType.AUTH_OK
        assert msg.payload["device_name"] == "Device Name"
        assert msg.payload["version"] == 5

    def test_create_snapshot_request(self):
        """Test snapshot request creation."""
        msg = create_snapshot_request("device-1", since_version=10)

        assert msg.type == MessageType.SNAPSHOT_REQUEST
        assert msg.payload["since_version"] == 10

    def test_create_snapshot_response(self):
        """Test snapshot response creation."""
        snapshot_data = {"memories": [], "config": {}}
        msg = create_snapshot_response("device-1", snapshot_data, version=15)

        assert msg.type == MessageType.SNAPSHOT_RESPONSE
        assert msg.payload["snapshot"] == snapshot_data
        assert msg.payload["version"] == 15

    def test_create_delta_push(self):
        """Test delta push creation."""
        delta_data = {"added": {"key": "value"}, "modified": {}, "deleted": []}
        msg = create_delta_push("device-1", delta_data)

        assert msg.type == MessageType.DELTA_PUSH
        assert msg.payload["delta"] == delta_data

    def test_create_delta_ack(self):
        """Test delta ack creation."""
        msg = create_delta_ack("device-1", "delta-123", success=True)

        assert msg.type == MessageType.DELTA_ACK
        assert msg.payload["delta_id"] == "delta-123"
        assert msg.payload["success"] is True

    def test_create_ping_pong(self):
        """Test ping/pong creation."""
        ping = create_ping("device-1")
        pong = create_pong("device-2")

        assert ping.type == MessageType.PING
        assert pong.type == MessageType.PONG


class TestMockDeviceDiscovery:
    """Tests for mock device discovery."""

    def test_mock_discovery_start_stop(self):
        """Test starting and stopping mock discovery."""
        discovery = MockDeviceDiscovery(
            device_id="test-device",
            device_name="Test Device",
            port=8422,
        )

        assert not discovery.is_running
        assert discovery.start()
        assert discovery.is_running
        discovery.stop()
        assert not discovery.is_running

    def test_add_remove_mock_device(self):
        """Test adding and removing mock devices."""
        discovery = MockDeviceDiscovery(
            device_id="local-device",
            device_name="Local",
            port=8422,
        )
        discovery.start()

        # Track callbacks
        events = []
        discovery.add_callback(lambda d, e: events.append((d.device_id, e)))

        # Add device
        device = DiscoveredDevice(
            device_id="remote-device",
            name="Remote",
            host="192.168.1.100",
            port=8422,
            version="0.5.0",
        )
        discovery.add_mock_device(device)

        assert len(discovery.get_devices()) == 1
        assert discovery.get_device("remote-device") == device
        assert events[-1] == ("remote-device", "added")

        # Remove device
        discovery.remove_mock_device("remote-device")

        assert len(discovery.get_devices()) == 0
        assert discovery.get_device("remote-device") is None
        assert events[-1] == ("remote-device", "removed")

    def test_discovered_device_address(self):
        """Test DiscoveredDevice address property."""
        device = DiscoveredDevice(
            device_id="test",
            name="Test",
            host="10.0.0.5",
            port=8422,
            version="1.0.0",
        )

        assert device.address == "ws://10.0.0.5:8422"

    def test_discovered_device_serialization(self):
        """Test DiscoveredDevice to_dict."""
        device = DiscoveredDevice(
            device_id="device-abc",
            name="My Device",
            host="192.168.1.50",
            port=8422,
            version="0.5.0",
        )

        data = device.to_dict()

        assert data["device_id"] == "device-abc"
        assert data["name"] == "My Device"
        assert data["host"] == "192.168.1.50"
        assert data["port"] == 8422
        assert data["version"] == "0.5.0"
