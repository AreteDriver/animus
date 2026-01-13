"""
Integration Manager

Orchestrates all integrations, manages connections, and aggregates tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from animus.integrations.base import BaseIntegration, IntegrationInfo
from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.tools import Tool

logger = get_logger("integrations")


class IntegrationManager:
    """
    Central manager for all integrations.

    Handles:
    - Registration of integrations
    - Credential storage and retrieval
    - Connection lifecycle management
    - Tool aggregation from connected integrations
    """

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize integration manager.

        Args:
            data_dir: Directory for storing integration credentials.
                      Defaults to ~/.animus/integrations/
        """
        self._integrations: dict[str, BaseIntegration] = {}
        self._data_dir = data_dir or Path.home() / ".animus" / "integrations"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"IntegrationManager initialized, data_dir: {self._data_dir}")

    def register(self, integration: BaseIntegration) -> None:
        """
        Register an integration.

        Args:
            integration: Integration instance to register
        """
        self._integrations[integration.name] = integration
        logger.info(f"Registered integration: {integration.name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister an integration.

        Args:
            name: Integration name

        Returns:
            True if integration was registered and removed
        """
        if name in self._integrations:
            del self._integrations[name]
            logger.info(f"Unregistered integration: {name}")
            return True
        return False

    def get(self, name: str) -> BaseIntegration | None:
        """
        Get an integration by name.

        Args:
            name: Integration name

        Returns:
            Integration instance or None
        """
        return self._integrations.get(name)

    def list_all(self) -> list[IntegrationInfo]:
        """
        List all registered integrations.

        Returns:
            List of integration info
        """
        return [integration.get_info() for integration in self._integrations.values()]

    def list_connected(self) -> list[IntegrationInfo]:
        """
        List only connected integrations.

        Returns:
            List of connected integration info
        """
        return [
            integration.get_info()
            for integration in self._integrations.values()
            if integration.is_connected
        ]

    async def connect(self, name: str, credentials: dict[str, Any]) -> bool:
        """
        Connect an integration.

        Args:
            name: Integration name
            credentials: Authentication credentials

        Returns:
            True if connection successful
        """
        integration = self._integrations.get(name)
        if not integration:
            logger.error(f"Integration not found: {name}")
            return False

        logger.info(f"Connecting integration: {name}")
        success = await integration.connect(credentials)

        if success:
            self._save_credentials(name, credentials)
            logger.info(f"Integration connected: {name}")
        else:
            logger.error(f"Failed to connect integration: {name}")

        return success

    async def disconnect(self, name: str) -> bool:
        """
        Disconnect an integration.

        Args:
            name: Integration name

        Returns:
            True if disconnection successful
        """
        integration = self._integrations.get(name)
        if not integration:
            logger.error(f"Integration not found: {name}")
            return False

        logger.info(f"Disconnecting integration: {name}")
        success = await integration.disconnect()

        if success:
            self._clear_credentials(name)
            logger.info(f"Integration disconnected: {name}")

        return success

    async def verify(self, name: str) -> bool:
        """
        Verify an integration's connection.

        Args:
            name: Integration name

        Returns:
            True if connection is valid
        """
        integration = self._integrations.get(name)
        if not integration:
            return False

        return await integration.verify()

    async def verify_all(self) -> dict[str, bool]:
        """
        Verify all connected integrations.

        Returns:
            Dict mapping integration name to verification result
        """
        results = {}
        for name, integration in self._integrations.items():
            if integration.is_connected:
                results[name] = await integration.verify()
        return results

    async def reconnect_from_stored(self) -> dict[str, bool]:
        """
        Reconnect integrations using stored credentials.

        Returns:
            Dict mapping integration name to connection result
        """
        results = {}
        for name in self._integrations:
            credentials = self._load_credentials(name)
            if credentials:
                logger.debug(f"Found stored credentials for: {name}")
                results[name] = await self.connect(name, credentials)
        return results

    def get_all_tools(self) -> list[Tool]:
        """
        Get tools from all connected integrations.

        Returns:
            List of all tools from connected integrations
        """
        tools = []
        for integration in self._integrations.values():
            if integration.is_connected:
                tools.extend(integration.get_tools())
        return tools

    def get_tools_by_integration(self) -> dict[str, list[Tool]]:
        """
        Get tools grouped by integration.

        Returns:
            Dict mapping integration name to its tools
        """
        return {
            name: integration.get_tools()
            for name, integration in self._integrations.items()
            if integration.is_connected
        }

    def _credentials_path(self, name: str) -> Path:
        """Get path for integration credentials file."""
        return self._data_dir / f"{name}.json"

    def _save_credentials(self, name: str, credentials: dict[str, Any]) -> None:
        """Save credentials to disk (encrypted in future)."""
        path = self._credentials_path(name)
        # TODO: Encrypt credentials before saving
        with open(path, "w") as f:
            json.dump(credentials, f)
        path.chmod(0o600)  # Owner read/write only
        logger.debug(f"Saved credentials for: {name}")

    def _load_credentials(self, name: str) -> dict[str, Any] | None:
        """Load credentials from disk."""
        path = self._credentials_path(name)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load credentials for {name}: {e}")
            return None

    def _clear_credentials(self, name: str) -> None:
        """Remove stored credentials."""
        path = self._credentials_path(name)
        if path.exists():
            path.unlink()
            logger.debug(f"Cleared credentials for: {name}")

    def get_status_summary(self) -> dict[str, Any]:
        """
        Get summary of all integration statuses.

        Returns:
            Dict with integration statuses and counts
        """
        statuses = {}
        connected_count = 0
        for name, integration in self._integrations.items():
            info = integration.get_info()
            statuses[name] = info.status.value
            if integration.is_connected:
                connected_count += 1

        return {
            "total": len(self._integrations),
            "connected": connected_count,
            "statuses": statuses,
        }
