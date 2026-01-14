"""
Animus Cross-Device Sync

Provides peer-to-peer synchronization of Animus state across devices
on the same local network using mDNS discovery and WebSocket communication.
"""

from animus.sync.client import SyncClient
from animus.sync.discovery import DeviceDiscovery, DiscoveredDevice
from animus.sync.protocol import MessageType, SyncMessage
from animus.sync.server import SyncServer
from animus.sync.state import StateDelta, StateSnapshot, SyncableState

__all__ = [
    "SyncableState",
    "StateSnapshot",
    "StateDelta",
    "DeviceDiscovery",
    "DiscoveredDevice",
    "SyncServer",
    "SyncClient",
    "SyncMessage",
    "MessageType",
]
