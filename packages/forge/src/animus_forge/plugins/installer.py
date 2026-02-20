"""Plugin installer for managing plugin installations."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from urllib.request import urlretrieve

from .loader import load_plugin_from_file
from .models import (
    PluginInstallation,
    PluginInstallRequest,
    PluginSource,
    PluginUpdateRequest,
)
from .registry import PluginRegistry, get_registry

if TYPE_CHECKING:
    from animus_forge.state.backends import DatabaseBackend

    from .marketplace import PluginMarketplace

logger = logging.getLogger(__name__)


class PluginInstaller:
    """Manages plugin installation, updates, and removal.

    Handles downloading, verifying, installing, and managing
    plugin lifecycle on the local system.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS plugin_installations (
        id TEXT PRIMARY KEY,
        plugin_name TEXT UNIQUE NOT NULL,
        version TEXT NOT NULL,
        installed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP,
        enabled BOOLEAN DEFAULT TRUE,
        config TEXT,  -- JSON object
        local_path TEXT NOT NULL,
        source TEXT DEFAULT 'local',
        source_url TEXT,
        checksum TEXT,
        auto_update BOOLEAN DEFAULT FALSE
    );

    CREATE INDEX IF NOT EXISTS idx_plugin_installations_enabled
        ON plugin_installations(enabled);
    """

    def __init__(
        self,
        backend: DatabaseBackend,
        plugins_dir: Path | str,
        marketplace: PluginMarketplace | None = None,
        registry: PluginRegistry | None = None,
    ):
        """Initialize installer.

        Args:
            backend: Database backend for persistence.
            plugins_dir: Directory for installed plugins.
            marketplace: Optional marketplace for remote plugins.
            registry: Plugin registry (uses global if not provided).
        """
        self.backend = backend
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.marketplace = marketplace
        self.registry = registry or get_registry()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure installation tables exist."""
        try:
            with self.backend.transaction() as conn:
                for statement in self.SCHEMA.split(";"):
                    statement = statement.strip()
                    if statement:
                        conn.execute(statement)
            logger.info("Plugin installer schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize installer schema: {e}")

    def install(self, request: PluginInstallRequest) -> PluginInstallation | None:
        """Install a plugin.

        Args:
            request: Installation request with plugin details.

        Returns:
            Installation record or None if failed.
        """
        # Check if already installed
        existing = self.get_installation(request.name)
        if existing:
            logger.warning(f"Plugin {request.name} already installed")
            return None

        # Get plugin source
        plugin_path = None
        checksum = None
        version = request.version

        if request.source == PluginSource.MARKETPLACE:
            plugin_path, checksum, version = self._download_from_marketplace(
                request.name, request.version
            )
        elif request.source == PluginSource.GITHUB:
            plugin_path, checksum = self._download_from_github(request.source_url, request.name)
        elif request.source == PluginSource.URL:
            plugin_path, checksum = self._download_from_url(request.source_url, request.name)
        elif request.source == PluginSource.LOCAL:
            if request.source_url:
                plugin_path = self._copy_local_plugin(request.source_url, request.name)
                checksum = self._compute_checksum(plugin_path)

        if not plugin_path or not plugin_path.exists():
            logger.error(f"Failed to obtain plugin {request.name}")
            return None

        # Try loading the plugin to validate
        try:
            plugin = load_plugin_from_file(
                plugin_path,
                registry=None,  # Don't register yet
                trusted_dir=self.plugins_dir,
            )
            if not plugin:
                logger.error(f"Failed to load plugin from {plugin_path}")
                self._cleanup_plugin_dir(request.name)
                return None

            # Use plugin's version if not specified
            if not version:
                version = plugin.version
        except Exception as e:
            logger.error(f"Plugin validation failed: {e}")
            self._cleanup_plugin_dir(request.name)
            return None

        # Create installation record
        installation = PluginInstallation(
            id=str(uuid.uuid4()),
            plugin_name=request.name,
            version=version or "1.0.0",
            installed_at=datetime.now(),
            enabled=request.enable,
            config=request.config,
            local_path=str(plugin_path),
            source=request.source,
            source_url=request.source_url,
            checksum=checksum,
            auto_update=request.auto_update,
        )

        # Save to database
        if not self._save_installation(installation):
            self._cleanup_plugin_dir(request.name)
            return None

        # Register plugin if enabled
        if request.enable:
            try:
                plugin = load_plugin_from_file(
                    plugin_path,
                    registry=self.registry,
                    config=request.config,
                    trusted_dir=self.plugins_dir,
                )
                if plugin:
                    logger.info(f"Plugin {request.name} installed and enabled")
            except Exception as e:
                logger.warning(f"Plugin installed but failed to enable: {e}")

        # Increment marketplace download count
        if self.marketplace and request.source == PluginSource.MARKETPLACE:
            self.marketplace.increment_downloads(request.name)

        return installation

    def uninstall(self, name: str) -> bool:
        """Uninstall a plugin.

        Args:
            name: Plugin name.

        Returns:
            True if uninstalled successfully.
        """
        installation = self.get_installation(name)
        if not installation:
            logger.warning(f"Plugin {name} not installed")
            return False

        # Unregister from registry
        try:
            self.registry.unregister(name)
        except Exception as e:
            logger.warning(f"Failed to unregister plugin {name}: {e}")

        # Remove files
        self._cleanup_plugin_dir(name)

        # Remove from database
        sql = "DELETE FROM plugin_installations WHERE plugin_name = ?"
        try:
            self.backend.execute(sql, [name])
            logger.info(f"Uninstalled plugin {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove installation record: {e}")
            return False

    def update(self, request: PluginUpdateRequest) -> PluginInstallation | None:
        """Update a plugin to a new version.

        Args:
            request: Update request.

        Returns:
            Updated installation or None if failed.
        """
        installation = self.get_installation(request.name)
        if not installation:
            logger.warning(f"Plugin {request.name} not installed")
            return None

        # Get new version info
        new_version = request.version
        if not new_version and self.marketplace:
            listing = self.marketplace.get_plugin(request.name)
            if listing:
                new_version = listing.latest_version

        if new_version == installation.version:
            logger.info(f"Plugin {request.name} already at version {new_version}")
            return installation

        # Backup current installation
        backup_path = self._backup_plugin(request.name)

        # Unregister current version
        try:
            self.registry.unregister(request.name)
        except Exception:
            pass  # Graceful degradation: plugin may not be registered yet

        # Download and install new version
        plugin_path = None
        checksum = None

        source = PluginSource(installation.source)
        if source == PluginSource.MARKETPLACE and self.marketplace:
            plugin_path, checksum, new_version = self._download_from_marketplace(
                request.name, new_version
            )
        elif source == PluginSource.GITHUB and installation.source_url:
            plugin_path, checksum = self._download_from_github(
                installation.source_url, request.name
            )
        elif source == PluginSource.URL and installation.source_url:
            plugin_path, checksum = self._download_from_url(installation.source_url, request.name)

        if not plugin_path or not plugin_path.exists():
            logger.error(f"Failed to download update for {request.name}")
            # Restore backup
            self._restore_backup(request.name, backup_path)
            return None

        # Update config if provided
        config = request.config if request.config is not None else installation.config

        # Validate new version
        try:
            plugin = load_plugin_from_file(
                plugin_path,
                registry=None,
                trusted_dir=self.plugins_dir,
            )
            if not plugin:
                raise ValueError("Failed to load plugin")
        except Exception as e:
            logger.error(f"New version validation failed: {e}")
            self._restore_backup(request.name, backup_path)
            return None

        # Update installation record
        installation.version = new_version or plugin.version
        installation.updated_at = datetime.now()
        installation.checksum = checksum
        installation.config = config
        installation.local_path = str(plugin_path)

        if not self._update_installation(installation):
            self._restore_backup(request.name, backup_path)
            return None

        # Re-register if was enabled
        if installation.enabled:
            try:
                load_plugin_from_file(
                    plugin_path,
                    registry=self.registry,
                    config=config,
                    trusted_dir=self.plugins_dir,
                )
            except Exception as e:
                logger.warning(f"Failed to re-enable plugin: {e}")

        # Cleanup backup
        if backup_path and backup_path.exists():
            shutil.rmtree(backup_path, ignore_errors=True)

        logger.info(f"Updated plugin {request.name} to version {installation.version}")
        return installation

    def enable(self, name: str, config: dict | None = None) -> bool:
        """Enable an installed plugin.

        Args:
            name: Plugin name.
            config: Optional configuration override.

        Returns:
            True if enabled successfully.
        """
        installation = self.get_installation(name)
        if not installation:
            logger.warning(f"Plugin {name} not installed")
            return False

        if installation.enabled:
            return True

        plugin_path = Path(installation.local_path)
        effective_config = config if config is not None else installation.config

        try:
            plugin = load_plugin_from_file(
                plugin_path,
                registry=self.registry,
                config=effective_config,
                trusted_dir=self.plugins_dir,
            )
            if not plugin:
                return False

            # Update database
            sql = "UPDATE plugin_installations SET enabled = 1, config = ? WHERE plugin_name = ?"
            self.backend.execute(sql, [json.dumps(effective_config), name])
            logger.info(f"Enabled plugin {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to enable plugin {name}: {e}")
            return False

    def disable(self, name: str) -> bool:
        """Disable an installed plugin.

        Args:
            name: Plugin name.

        Returns:
            True if disabled successfully.
        """
        installation = self.get_installation(name)
        if not installation:
            logger.warning(f"Plugin {name} not installed")
            return False

        if not installation.enabled:
            return True

        try:
            self.registry.unregister(name)
        except Exception as e:
            logger.warning(f"Failed to unregister plugin: {e}")

        sql = "UPDATE plugin_installations SET enabled = 0 WHERE plugin_name = ?"
        try:
            self.backend.execute(sql, [name])
            logger.info(f"Disabled plugin {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to disable plugin {name}: {e}")
            return False

    def get_installation(self, name: str) -> PluginInstallation | None:
        """Get installation info for a plugin.

        Args:
            name: Plugin name.

        Returns:
            Installation record or None.
        """
        sql = """
            SELECT id, plugin_name, version, installed_at, updated_at, enabled,
                   config, local_path, source, source_url, checksum, auto_update
            FROM plugin_installations
            WHERE plugin_name = ?
        """
        try:
            rows = self.backend.execute(sql, [name])
            if rows:
                return self._row_to_installation(rows[0])
        except Exception as e:
            logger.error(f"Failed to get installation for {name}: {e}")
        return None

    def list_installations(
        self,
        enabled_only: bool = False,
    ) -> list[PluginInstallation]:
        """List all installed plugins.

        Args:
            enabled_only: Only return enabled plugins.

        Returns:
            List of installation records.
        """
        sql = """
            SELECT id, plugin_name, version, installed_at, updated_at, enabled,
                   config, local_path, source, source_url, checksum, auto_update
            FROM plugin_installations
        """
        if enabled_only:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY plugin_name"

        results = []
        try:
            rows = self.backend.execute(sql, [])
            for row in rows:
                results.append(self._row_to_installation(row))
        except Exception as e:
            logger.error(f"Failed to list installations: {e}")
        return results

    def load_enabled_plugins(self) -> int:
        """Load all enabled plugins into the registry.

        Returns:
            Number of plugins loaded.
        """
        installations = self.list_installations(enabled_only=True)
        loaded = 0
        for installation in installations:
            try:
                plugin = load_plugin_from_file(
                    installation.local_path,
                    registry=self.registry,
                    config=installation.config,
                    trusted_dir=self.plugins_dir,
                )
                if plugin:
                    loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load plugin {installation.plugin_name}: {e}")
        logger.info(f"Loaded {loaded} plugins")
        return loaded

    def _download_from_marketplace(
        self, name: str, version: str | None
    ) -> tuple[Path | None, str | None, str | None]:
        """Download plugin from marketplace."""
        if not self.marketplace:
            logger.error("No marketplace configured")
            return None, None, None

        listing = self.marketplace.get_plugin(name)
        if not listing:
            logger.error(f"Plugin {name} not found in marketplace")
            return None, None, None

        target_version = version or listing.latest_version
        release = self.marketplace.get_release(name, target_version)
        if not release:
            logger.error(f"Release {name}@{target_version} not found")
            return None, None, None

        return self._download_and_verify(release.download_url, name, release.checksum) + (
            target_version,
        )

    def _download_from_github(
        self, repo_url: str | None, name: str
    ) -> tuple[Path | None, str | None]:
        """Download plugin from GitHub."""
        if not repo_url:
            return None, None

        # Validate GitHub URL using proper URL parsing
        parsed = urlparse(repo_url)
        if parsed.hostname not in ("github.com", "raw.githubusercontent.com"):
            logger.warning(f"Invalid GitHub URL host: {parsed.hostname}")
            return None, None

        # Convert to raw URL if needed
        if parsed.hostname == "github.com" and "/blob/" not in repo_url:
            # Assume main branch, plugin.py
            raw_url = (
                repo_url.replace("github.com", "raw.githubusercontent.com") + "/main/plugin.py"
            )
        else:
            raw_url = repo_url.replace("/blob/", "/raw/")

        return self._download_and_verify(raw_url, name, None)

    def _download_from_url(self, url: str | None, name: str) -> tuple[Path | None, str | None]:
        """Download plugin from URL."""
        if not url:
            return None, None
        return self._download_and_verify(url, name, None)

    def _download_and_verify(
        self, url: str, name: str, expected_checksum: str | None
    ) -> tuple[Path | None, str | None]:
        """Download file and verify checksum."""
        plugin_dir = self.plugins_dir / name
        plugin_dir.mkdir(parents=True, exist_ok=True)
        target_path = plugin_dir / "plugin.py"

        try:
            # Download to temp file first
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                urlretrieve(url, tmp.name)
                actual_checksum = self._compute_checksum(Path(tmp.name))

                # Verify checksum if provided
                if expected_checksum and actual_checksum != expected_checksum:
                    logger.error(f"Checksum mismatch for {name}")
                    Path(tmp.name).unlink()
                    return None, None

                # Move to final location
                shutil.move(tmp.name, target_path)
                return target_path, actual_checksum

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None, None

    def _copy_local_plugin(self, source: str, name: str) -> Path | None:
        """Copy local plugin file to plugins directory."""
        source_path = Path(source)
        if not source_path.exists():
            return None

        plugin_dir = self.plugins_dir / name
        plugin_dir.mkdir(parents=True, exist_ok=True)

        if source_path.is_file():
            target = plugin_dir / "plugin.py"
            shutil.copy2(source_path, target)
            return target
        elif source_path.is_dir():
            shutil.copytree(source_path, plugin_dir, dirs_exist_ok=True)
            return plugin_dir / "plugin.py"

        return None

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _cleanup_plugin_dir(self, name: str) -> None:
        """Remove plugin directory."""
        plugin_dir = self.plugins_dir / name
        if plugin_dir.exists():
            shutil.rmtree(plugin_dir, ignore_errors=True)

    def _backup_plugin(self, name: str) -> Path | None:
        """Backup plugin directory before update."""
        plugin_dir = self.plugins_dir / name
        if not plugin_dir.exists():
            return None

        backup_dir = self.plugins_dir / f"{name}.backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(plugin_dir, backup_dir)
        return backup_dir

    def _restore_backup(self, name: str, backup_path: Path | None) -> None:
        """Restore plugin from backup."""
        if not backup_path or not backup_path.exists():
            return

        plugin_dir = self.plugins_dir / name
        if plugin_dir.exists():
            shutil.rmtree(plugin_dir)
        shutil.move(str(backup_path), str(plugin_dir))

    def _save_installation(self, installation: PluginInstallation) -> bool:
        """Save installation to database."""
        sql = """
            INSERT INTO plugin_installations (
                id, plugin_name, version, installed_at, updated_at, enabled,
                config, local_path, source, source_url, checksum, auto_update
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            self.backend.execute(
                sql,
                [
                    installation.id,
                    installation.plugin_name,
                    installation.version,
                    installation.installed_at.isoformat(),
                    installation.updated_at.isoformat() if installation.updated_at else None,
                    installation.enabled,
                    json.dumps(installation.config),
                    installation.local_path,
                    installation.source.value,
                    installation.source_url,
                    installation.checksum,
                    installation.auto_update,
                ],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save installation: {e}")
            return False

    def _update_installation(self, installation: PluginInstallation) -> bool:
        """Update installation in database."""
        sql = """
            UPDATE plugin_installations SET
                version = ?, updated_at = ?, enabled = ?, config = ?,
                local_path = ?, checksum = ?, auto_update = ?
            WHERE plugin_name = ?
        """
        try:
            self.backend.execute(
                sql,
                [
                    installation.version,
                    installation.updated_at.isoformat() if installation.updated_at else None,
                    installation.enabled,
                    json.dumps(installation.config),
                    installation.local_path,
                    installation.checksum,
                    installation.auto_update,
                    installation.plugin_name,
                ],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update installation: {e}")
            return False

    def _row_to_installation(self, row: tuple) -> PluginInstallation:
        """Convert database row to PluginInstallation."""
        return PluginInstallation(
            id=row[0],
            plugin_name=row[1],
            version=row[2],
            installed_at=datetime.fromisoformat(row[3]) if row[3] else datetime.now(),
            updated_at=datetime.fromisoformat(row[4]) if row[4] else None,
            enabled=bool(row[5]),
            config=json.loads(row[6]) if row[6] else {},
            local_path=row[7],
            source=PluginSource(row[8]) if row[8] else PluginSource.LOCAL,
            source_url=row[9],
            checksum=row[10],
            auto_update=bool(row[11]),
        )
