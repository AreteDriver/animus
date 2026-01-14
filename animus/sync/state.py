"""
Syncable State Management

Handles serialization and delta computation for cross-device sync.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from animus.logging import get_logger

logger = get_logger("sync.state")


@dataclass
class StateSnapshot:
    """A point-in-time snapshot of syncable state."""

    id: str
    device_id: str
    timestamp: datetime
    version: int
    data: dict[str, Any]
    checksum: str

    @classmethod
    def create(cls, device_id: str, data: dict[str, Any], version: int = 1) -> "StateSnapshot":
        """Create a new snapshot from data."""
        snapshot_id = str(uuid.uuid4())
        timestamp = datetime.now()
        checksum = cls._compute_checksum(data)

        return cls(
            id=snapshot_id,
            device_id=device_id,
            timestamp=timestamp,
            version=version,
            data=data,
            checksum=checksum,
        )

    @staticmethod
    def _compute_checksum(data: dict[str, Any]) -> str:
        """Compute SHA-256 checksum of data."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "device_id": self.device_id,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "data": self.data,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateSnapshot":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            device_id=data["device_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data["version"],
            data=data["data"],
            checksum=data["checksum"],
        )


@dataclass
class StateDelta:
    """Changes between two state snapshots."""

    id: str
    source_device: str
    target_device: str
    timestamp: datetime
    base_version: int
    new_version: int
    changes: dict[str, Any]  # {"added": {...}, "modified": {...}, "deleted": [...]}

    @classmethod
    def compute(
        cls,
        source_device: str,
        target_device: str,
        old_data: dict[str, Any],
        new_data: dict[str, Any],
        base_version: int,
    ) -> "StateDelta":
        """Compute delta between two data states."""
        added = {}
        modified = {}
        deleted = []

        # Find added and modified
        for key, value in new_data.items():
            if key not in old_data:
                added[key] = value
            elif old_data[key] != value:
                modified[key] = value

        # Find deleted
        for key in old_data:
            if key not in new_data:
                deleted.append(key)

        return cls(
            id=str(uuid.uuid4()),
            source_device=source_device,
            target_device=target_device,
            timestamp=datetime.now(),
            base_version=base_version,
            new_version=base_version + 1,
            changes={"added": added, "modified": modified, "deleted": deleted},
        )

    def is_empty(self) -> bool:
        """Check if delta has no changes."""
        changes = self.changes
        return (
            not changes.get("added")
            and not changes.get("modified")
            and not changes.get("deleted")
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "source_device": self.source_device,
            "target_device": self.target_device,
            "timestamp": self.timestamp.isoformat(),
            "base_version": self.base_version,
            "new_version": self.new_version,
            "changes": self.changes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateDelta":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            source_device=data["source_device"],
            target_device=data["target_device"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            base_version=data["base_version"],
            new_version=data["new_version"],
            changes=data["changes"],
        )


class SyncableState:
    """
    Manages syncable state for cross-device synchronization.

    Coordinates between memory, learnings, and config to provide
    a unified sync interface.
    """

    def __init__(self, data_dir: Path, device_id: str | None = None):
        self.data_dir = data_dir
        self.sync_dir = data_dir / "sync"
        self.sync_dir.mkdir(parents=True, exist_ok=True)

        # Device identity
        self.device_id = device_id or self._load_or_create_device_id()
        self._version = self._load_version()

        # Track last sync per peer
        self._peer_versions: dict[str, int] = self._load_peer_versions()

        logger.info(f"SyncableState initialized for device {self.device_id[:8]}...")

    def _load_or_create_device_id(self) -> str:
        """Load existing device ID or create new one."""
        device_file = self.sync_dir / "device_id"
        if device_file.exists():
            return device_file.read_text().strip()

        device_id = str(uuid.uuid4())
        device_file.write_text(device_id)
        logger.info(f"Created new device ID: {device_id[:8]}...")
        return device_id

    def _load_version(self) -> int:
        """Load current state version."""
        version_file = self.sync_dir / "version"
        if version_file.exists():
            return int(version_file.read_text().strip())
        return 0

    def _save_version(self) -> None:
        """Save current state version."""
        version_file = self.sync_dir / "version"
        version_file.write_text(str(self._version))

    def _load_peer_versions(self) -> dict[str, int]:
        """Load last known versions for each peer."""
        peer_file = self.sync_dir / "peer_versions.json"
        if peer_file.exists():
            return json.loads(peer_file.read_text())
        return {}

    def _save_peer_versions(self) -> None:
        """Save peer version tracking."""
        peer_file = self.sync_dir / "peer_versions.json"
        peer_file.write_text(json.dumps(self._peer_versions, indent=2))

    @property
    def version(self) -> int:
        """Current state version."""
        return self._version

    def increment_version(self) -> int:
        """Increment and return new version."""
        self._version += 1
        self._save_version()
        return self._version

    def get_peer_version(self, peer_id: str) -> int:
        """Get last known version for a peer."""
        return self._peer_versions.get(peer_id, 0)

    def set_peer_version(self, peer_id: str, version: int) -> None:
        """Update last known version for a peer."""
        self._peer_versions[peer_id] = version
        self._save_peer_versions()

    def collect_state(self) -> dict[str, Any]:
        """
        Collect all syncable state from the data directory.

        Returns:
            Dictionary of all syncable data organized by category.
        """
        state = {
            "memories": self._collect_memories(),
            "learnings": self._collect_learnings(),
            "guardrails": self._collect_guardrails(),
            "config": self._collect_config(),
        }
        return state

    def _collect_memories(self) -> list[dict]:
        """Collect memories for sync."""
        memories_file = self.data_dir / "memories.json"
        if memories_file.exists():
            return json.loads(memories_file.read_text())
        return []

    def _collect_learnings(self) -> list[dict]:
        """Collect learned items for sync."""
        learnings_file = self.data_dir / "learning" / "learned_items.json"
        if learnings_file.exists():
            return json.loads(learnings_file.read_text())
        return []

    def _collect_guardrails(self) -> list[dict]:
        """Collect user-defined guardrails for sync."""
        guardrails_file = self.data_dir / "learning" / "user_guardrails.json"
        if guardrails_file.exists():
            return json.loads(guardrails_file.read_text())
        return []

    def _collect_config(self) -> dict:
        """Collect user config for sync (excluding sensitive data)."""
        config_file = self.data_dir / "config.yaml"
        if config_file.exists():
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f) or {}
            # Remove sensitive fields
            for key in ["api_key", "anthropic_api_key", "openai_api_key"]:
                config.pop(key, None)
            return config
        return {}

    def create_snapshot(self) -> StateSnapshot:
        """Create a snapshot of current state."""
        data = self.collect_state()
        return StateSnapshot.create(self.device_id, data, self._version)

    def compute_delta(self, peer_snapshot: StateSnapshot) -> StateDelta:
        """Compute delta between local state and peer snapshot."""
        local_data = self.collect_state()
        return StateDelta.compute(
            source_device=self.device_id,
            target_device=peer_snapshot.device_id,
            old_data=peer_snapshot.data,
            new_data=local_data,
            base_version=peer_snapshot.version,
        )

    def apply_delta(self, delta: StateDelta) -> bool:
        """
        Apply a delta from a peer to local state.

        Args:
            delta: Delta to apply

        Returns:
            True if applied successfully
        """
        if delta.base_version > self._version:
            logger.warning(
                f"Delta base version {delta.base_version} > local {self._version}, "
                "may have missed updates"
            )

        changes = delta.changes
        applied = False

        # Apply memory changes
        if "memories" in changes.get("added", {}) or "memories" in changes.get("modified", {}):
            self._apply_memory_changes(changes)
            applied = True

        # Apply learning changes
        if "learnings" in changes.get("added", {}) or "learnings" in changes.get("modified", {}):
            self._apply_learning_changes(changes)
            applied = True

        # Apply guardrail changes
        if "guardrails" in changes.get("added", {}) or "guardrails" in changes.get("modified", {}):
            self._apply_guardrail_changes(changes)
            applied = True

        if applied:
            self._version = max(self._version, delta.new_version)
            self._save_version()
            self.set_peer_version(delta.source_device, delta.new_version)
            logger.info(f"Applied delta from {delta.source_device[:8]}, now at version {self._version}")

        return applied

    def _apply_memory_changes(self, changes: dict) -> None:
        """Apply memory changes from delta."""
        memories_file = self.data_dir / "memories.json"
        existing = []
        if memories_file.exists():
            existing = json.loads(memories_file.read_text())

        # Index by ID for merging
        by_id = {m.get("id"): m for m in existing}

        # Add new memories
        for mem in changes.get("added", {}).get("memories", []):
            if mem.get("id") not in by_id:
                by_id[mem["id"]] = mem

        # Update modified memories (last-write-wins)
        for mem in changes.get("modified", {}).get("memories", []):
            existing_mem = by_id.get(mem.get("id"))
            if existing_mem:
                # Compare timestamps, keep newer
                if mem.get("updated_at", "") > existing_mem.get("updated_at", ""):
                    by_id[mem["id"]] = mem
            else:
                by_id[mem["id"]] = mem

        # Write back
        memories_file.write_text(json.dumps(list(by_id.values()), indent=2, default=str))

    def _apply_learning_changes(self, changes: dict) -> None:
        """Apply learning changes from delta."""
        learnings_dir = self.data_dir / "learning"
        learnings_dir.mkdir(exist_ok=True)
        learnings_file = learnings_dir / "learned_items.json"

        existing = []
        if learnings_file.exists():
            existing = json.loads(learnings_file.read_text())

        by_id = {learning.get("id"): learning for learning in existing}

        for item in changes.get("added", {}).get("learnings", []):
            if item.get("id") not in by_id:
                by_id[item["id"]] = item

        for item in changes.get("modified", {}).get("learnings", []):
            existing_item = by_id.get(item.get("id"))
            if existing_item:
                if item.get("updated_at", "") > existing_item.get("updated_at", ""):
                    by_id[item["id"]] = item
            else:
                by_id[item["id"]] = item

        learnings_file.write_text(json.dumps(list(by_id.values()), indent=2, default=str))

    def _apply_guardrail_changes(self, changes: dict) -> None:
        """Apply guardrail changes from delta."""
        learnings_dir = self.data_dir / "learning"
        learnings_dir.mkdir(exist_ok=True)
        guardrails_file = learnings_dir / "user_guardrails.json"

        existing = []
        if guardrails_file.exists():
            existing = json.loads(guardrails_file.read_text())

        by_id = {g.get("id"): g for g in existing}

        for item in changes.get("added", {}).get("guardrails", []):
            if item.get("id") not in by_id:
                by_id[item["id"]] = item

        for item in changes.get("modified", {}).get("guardrails", []):
            by_id[item["id"]] = item

        guardrails_file.write_text(json.dumps(list(by_id.values()), indent=2, default=str))

    def export_snapshot(self, filepath: Path) -> None:
        """Export current state to a file."""
        snapshot = self.create_snapshot()
        filepath.write_text(json.dumps(snapshot.to_dict(), indent=2, default=str))
        logger.info(f"Exported snapshot to {filepath}")

    def import_snapshot(self, filepath: Path) -> bool:
        """Import state from a snapshot file."""
        data = json.loads(filepath.read_text())
        snapshot = StateSnapshot.from_dict(data)

        # Compute and apply delta
        delta = StateDelta.compute(
            source_device=snapshot.device_id,
            target_device=self.device_id,
            old_data=self.collect_state(),
            new_data=snapshot.data,
            base_version=self._version,
        )

        if delta.is_empty():
            logger.info("No changes to import")
            return False

        return self.apply_delta(delta)
