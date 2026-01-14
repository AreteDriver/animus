"""
mDNS Service Discovery for Animus Sync

Uses Zeroconf to discover other Animus instances on the local network.
"""

import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from animus.logging import get_logger

logger = get_logger("sync.discovery")

# Service type for Animus sync
SERVICE_TYPE = "_animus-sync._tcp.local."


@dataclass
class DiscoveredDevice:
    """Information about a discovered Animus device."""

    device_id: str
    name: str
    host: str
    port: int
    version: str
    discovered_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

    def __hash__(self):
        return hash(self.device_id)

    def __eq__(self, other):
        if isinstance(other, DiscoveredDevice):
            return self.device_id == other.device_id
        return False

    @property
    def address(self) -> str:
        """Get full address for connection."""
        return f"ws://{self.host}:{self.port}"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "version": self.version,
            "discovered_at": self.discovered_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }


class DeviceDiscovery:
    """
    Handles discovery of Animus devices on the local network.

    Uses mDNS/Zeroconf for service advertisement and discovery.
    """

    def __init__(
        self,
        device_id: str,
        device_name: str,
        port: int,
        version: str = "0.5.0",
    ):
        self.device_id = device_id
        self.device_name = device_name
        self.port = port
        self.version = version

        self._discovered: dict[str, DiscoveredDevice] = {}
        self._callbacks: list[Callable[[DiscoveredDevice, str], None]] = []
        self._zeroconf = None
        self._service_info = None
        self._browser = None
        self._running = False

        logger.info(f"DeviceDiscovery initialized for {device_name} ({device_id[:8]}...)")

    def add_callback(self, callback: Callable[[DiscoveredDevice, str], None]) -> None:
        """
        Add callback for device events.

        Callback receives (device, event_type) where event_type is "added" or "removed".
        """
        self._callbacks.append(callback)

    def _notify_callbacks(self, device: DiscoveredDevice, event: str) -> None:
        """Notify all callbacks of device event."""
        for callback in self._callbacks:
            try:
                callback(device, event)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def start(self) -> bool:
        """
        Start service advertisement and discovery.

        Returns:
            True if started successfully
        """
        try:
            from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
        except ImportError:
            logger.warning("Zeroconf not installed. Install with: pip install zeroconf")
            return False

        if self._running:
            logger.warning("Discovery already running")
            return True

        try:
            self._zeroconf = Zeroconf()

            # Get local IP
            local_ip = self._get_local_ip()

            # Create service info for advertisement
            properties = {
                b"device_id": self.device_id.encode(),
                b"version": self.version.encode(),
            }

            self._service_info = ServiceInfo(
                SERVICE_TYPE,
                f"{self.device_name}.{SERVICE_TYPE}",
                addresses=[socket.inet_aton(local_ip)],
                port=self.port,
                properties=properties,
            )

            # Register service
            self._zeroconf.register_service(self._service_info)
            logger.info(f"Registered service: {self.device_name} on {local_ip}:{self.port}")

            # Start browser for discovering other devices
            self._browser = ServiceBrowser(
                self._zeroconf,
                SERVICE_TYPE,
                handlers=[self._on_service_state_change],
            )

            self._running = True
            logger.info("Discovery started")
            return True

        except Exception as e:
            logger.error(f"Failed to start discovery: {e}")
            self.stop()
            return False

    def stop(self) -> None:
        """Stop service advertisement and discovery."""
        if not self._running:
            return

        self._running = False

        if self._zeroconf:
            if self._service_info:
                try:
                    self._zeroconf.unregister_service(self._service_info)
                except Exception:
                    pass

            try:
                self._zeroconf.close()
            except Exception:
                pass

        self._zeroconf = None
        self._service_info = None
        self._browser = None
        self._discovered.clear()

        logger.info("Discovery stopped")

    def _get_local_ip(self) -> str:
        """Get the local IP address."""
        try:
            # Create a socket to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _on_service_state_change(
        self,
        zeroconf,
        service_type: str,
        name: str,
        state_change,
    ) -> None:
        """Handle service state changes from browser."""
        from zeroconf import ServiceStateChange

        if state_change == ServiceStateChange.Added:
            # Get service info
            info = zeroconf.get_service_info(service_type, name)
            if info:
                self._handle_service_added(info, name)

        elif state_change == ServiceStateChange.Removed:
            self._handle_service_removed(name)

    def _handle_service_added(self, info, name: str) -> None:
        """Handle a newly discovered service."""
        try:
            # Extract device info from properties
            props = info.properties
            device_id = props.get(b"device_id", b"").decode()
            version = props.get(b"version", b"0.0.0").decode()

            # Skip self
            if device_id == self.device_id:
                return

            # Get address
            addresses = info.parsed_addresses()
            if not addresses:
                return
            host = addresses[0]
            port = info.port

            # Clean service name
            display_name = name.replace(f".{SERVICE_TYPE}", "")

            device = DiscoveredDevice(
                device_id=device_id,
                name=display_name,
                host=host,
                port=port,
                version=version,
            )

            self._discovered[device_id] = device
            logger.info(f"Discovered device: {display_name} at {host}:{port}")
            self._notify_callbacks(device, "added")

        except Exception as e:
            logger.error(f"Error handling service: {e}")

    def _handle_service_removed(self, name: str) -> None:
        """Handle a removed service."""
        # Find device by name
        display_name = name.replace(f".{SERVICE_TYPE}", "")

        for device_id, device in list(self._discovered.items()):
            if device.name == display_name:
                del self._discovered[device_id]
                logger.info(f"Device removed: {display_name}")
                self._notify_callbacks(device, "removed")
                break

    def get_devices(self) -> list[DiscoveredDevice]:
        """Get list of discovered devices."""
        return list(self._discovered.values())

    def get_device(self, device_id: str) -> DiscoveredDevice | None:
        """Get a specific device by ID."""
        return self._discovered.get(device_id)

    @property
    def is_running(self) -> bool:
        """Check if discovery is running."""
        return self._running


class MockDeviceDiscovery(DeviceDiscovery):
    """Mock discovery for testing without Zeroconf."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mock_devices: list[DiscoveredDevice] = []

    def start(self) -> bool:
        """Start mock discovery."""
        self._running = True
        logger.info("Mock discovery started")
        return True

    def stop(self) -> None:
        """Stop mock discovery."""
        self._running = False
        self._mock_devices.clear()
        logger.info("Mock discovery stopped")

    def add_mock_device(self, device: DiscoveredDevice) -> None:
        """Add a mock device for testing."""
        self._mock_devices.append(device)
        self._discovered[device.device_id] = device
        self._notify_callbacks(device, "added")

    def remove_mock_device(self, device_id: str) -> None:
        """Remove a mock device."""
        device = self._discovered.pop(device_id, None)
        if device:
            self._mock_devices = [d for d in self._mock_devices if d.device_id != device_id]
            self._notify_callbacks(device, "removed")
