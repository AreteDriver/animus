"""Sandbox execution for self-improvement proposals — safe config/prompt changes with rollback."""

from __future__ import annotations

import logging
import shutil
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import tomli_w

logger = logging.getLogger(__name__)


class ImprovementSandbox:
    """Execute improvement proposals safely with backup and rollback.

    Supports two types of safe changes:
    1. Config changes — modify animus TOML config values
    2. Identity file updates — modify LEARNED.md, system prompts, persona configs

    All changes create backups before applying. Rollback restores from backup.
    """

    def __init__(
        self,
        data_dir: Path,
        config_path: Path | None = None,
        on_config_changed: Callable[[], None] | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._backup_dir = self._data_dir / "improvement_backups"
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._config_path = config_path
        self._on_config_changed = on_config_changed

    def apply_config_change(
        self,
        proposal_id: int,
        config_path: str,
        value: object,
    ) -> dict:
        """Apply a TOML config change with backup.

        Args:
            proposal_id: ID of the proposal being applied.
            config_path: Dot-separated path (e.g. "intelligence.tool_timeout_seconds").
            value: New value to set.

        Returns:
            Dict with status, old_value, new_value, backup_path.
        """
        if self._config_path is None or not self._config_path.exists():
            return {"status": "error", "reason": "No config file found"}

        # Read current config (TOML)
        config_bytes = self._config_path.read_bytes()
        config = tomllib.loads(config_bytes.decode())

        # Navigate to the target key
        keys = config_path.split(".")
        old_value = _get_nested(config, keys)

        # Create backup
        backup_path = self._backup_file(self._config_path, proposal_id)

        # Set the new value
        _set_nested(config, keys, value)

        # Write updated config (TOML)
        self._config_path.write_bytes(tomli_w.dumps(config).encode())

        logger.info(
            "Proposal #%d: config %s changed from %r to %r",
            proposal_id,
            config_path,
            old_value,
            value,
        )

        # Notify runtime to reload config
        if self._on_config_changed is not None:
            try:
                self._on_config_changed()
                logger.info("Config reloaded after proposal #%d", proposal_id)
            except Exception:
                logger.exception("Config reload failed after proposal #%d", proposal_id)

        return {
            "status": "applied",
            "config_path": config_path,
            "old_value": old_value,
            "new_value": value,
            "backup_path": str(backup_path),
        }

    def apply_identity_append(
        self,
        proposal_id: int,
        filename: str,
        section: str,
        content: str,
        identity_manager: object | None = None,
    ) -> dict:
        """Append content to an identity file (e.g. LEARNED.md) with backup.

        Args:
            proposal_id: ID of the proposal being applied.
            filename: Identity file name (e.g. "LEARNED.md").
            section: Section heading for the entry.
            content: Content to append.
            identity_manager: IdentityFileManager instance.

        Returns:
            Dict with status, filename, content_added.
        """
        if identity_manager is None:
            return {"status": "error", "reason": "No identity manager available"}

        # Backup current file
        current = identity_manager.read(filename)
        if current:
            backup_path = self._backup_dir / f"proposal_{proposal_id}_{filename}"
            backup_path.write_text(current)

        # Append via identity manager
        identity_manager.append_to_learned(section, content)

        logger.info("Proposal #%d: appended to %s section '%s'", proposal_id, filename, section)

        return {
            "status": "applied",
            "filename": filename,
            "section": section,
            "content_added": content,
        }

    def rollback(self, proposal_id: int, file_path: Path) -> dict:
        """Rollback a change by restoring from backup.

        Args:
            proposal_id: ID of the proposal to rollback.
            file_path: Path of the file to restore.

        Returns:
            Dict with status and details.
        """
        backup_name = f"proposal_{proposal_id}_{file_path.name}"
        backup_path = self._backup_dir / backup_name

        if not backup_path.exists():
            return {"status": "error", "reason": f"No backup found for proposal #{proposal_id}"}

        shutil.copy2(backup_path, file_path)
        logger.info("Proposal #%d: rolled back %s from backup", proposal_id, file_path)

        # Trigger reload if config was rolled back
        if self._on_config_changed is not None and file_path == self._config_path:
            try:
                self._on_config_changed()
            except Exception:
                logger.exception("Config reload failed after rollback of proposal #%d", proposal_id)

        return {
            "status": "rolled_back",
            "restored_from": str(backup_path),
            "restored_to": str(file_path),
        }

    def list_backups(self) -> list[dict]:
        """List all backup files."""
        backups = []
        for f in sorted(self._backup_dir.iterdir()):
            if f.is_file() and f.name.startswith("proposal_"):
                parts = f.name.split("_", 2)
                proposal_id = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else None
                backups.append(
                    {
                        "proposal_id": proposal_id,
                        "filename": f.name,
                        "size": f.stat().st_size,
                        "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=UTC).isoformat(),
                    }
                )
        return backups

    def _backup_file(self, file_path: Path, proposal_id: int) -> Path:
        """Create a timestamped backup of a file."""
        backup_name = f"proposal_{proposal_id}_{file_path.name}"
        backup_path = self._backup_dir / backup_name
        shutil.copy2(file_path, backup_path)
        return backup_path


def _get_nested(d: dict, keys: list[str]) -> object:
    """Get a value from a nested dict using a list of keys."""
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _set_nested(d: dict, keys: list[str], value: object) -> None:
    """Set a value in a nested dict using a list of keys."""
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
