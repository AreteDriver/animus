"""Rollback management for self-improvement operations."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    """A snapshot of file state before changes."""

    id: str
    created_at: datetime
    description: str
    files: dict[str, str]  # path -> content
    metadata: dict[str, Any] = field(default_factory=dict)


class RollbackManager:
    """Manages snapshots and rollback for self-improvement.

    Creates snapshots before changes are applied and allows
    rolling back if something goes wrong.
    """

    def __init__(
        self,
        storage_path: Path | str | None = None,
        max_snapshots: int = 10,
    ):
        """Initialize rollback manager.

        Args:
            storage_path: Where to store snapshots. Defaults to .gorgon/snapshots
            max_snapshots: Maximum snapshots to keep.
        """
        if storage_path is None:
            storage_path = Path(".gorgon/snapshots")
        self.storage_path = Path(storage_path)
        self.max_snapshots = max_snapshots
        self._snapshots: list[Snapshot] = []
        self._ensure_storage()
        self._load_snapshots()

    def _ensure_storage(self) -> None:
        """Ensure storage directory exists."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _load_snapshots(self) -> None:
        """Load existing snapshots from storage."""
        index_file = self.storage_path / "index.json"
        if index_file.exists():
            try:
                with open(index_file) as f:
                    data = json.load(f)
                    for item in data.get("snapshots", []):
                        self._snapshots.append(
                            Snapshot(
                                id=item["id"],
                                created_at=datetime.fromisoformat(item["created_at"]),
                                description=item["description"],
                                files={},  # Files loaded on demand
                                metadata=item.get("metadata", {}),
                            )
                        )
            except Exception as e:
                logger.warning(f"Could not load snapshot index: {e}")

    def _save_index(self) -> None:
        """Save snapshot index to storage."""
        index_file = self.storage_path / "index.json"
        data = {
            "snapshots": [
                {
                    "id": s.id,
                    "created_at": s.created_at.isoformat(),
                    "description": s.description,
                    "metadata": s.metadata,
                }
                for s in self._snapshots
            ]
        }
        with open(index_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_snapshot(
        self,
        files: list[str],
        description: str,
        codebase_path: Path | str = ".",
        metadata: dict[str, Any] | None = None,
    ) -> Snapshot:
        """Create a snapshot of specified files.

        Args:
            files: List of file paths to snapshot.
            description: Description of what this snapshot is for.
            codebase_path: Base path for files.
            metadata: Optional metadata.

        Returns:
            Created snapshot.
        """
        import uuid

        snapshot_id = str(uuid.uuid4())[:8]
        codebase_path = Path(codebase_path)

        # Read file contents
        file_contents = {}
        for file_path in files:
            full_path = codebase_path / file_path
            if full_path.exists():
                try:
                    file_contents[file_path] = full_path.read_text()
                except Exception as e:
                    logger.warning(f"Could not read {file_path} for snapshot: {e}")

        snapshot = Snapshot(
            id=snapshot_id,
            created_at=datetime.now(),
            description=description,
            files=file_contents,
            metadata=metadata or {},
        )

        # Save snapshot files
        snapshot_dir = self.storage_path / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        for file_path, content in file_contents.items():
            safe_name = file_path.replace("/", "_").replace("\\", "_")
            (snapshot_dir / safe_name).write_text(content)

        # Save file mapping
        with open(snapshot_dir / "files.json", "w") as f:
            json.dump(list(file_contents.keys()), f)

        # Add to list and save index
        self._snapshots.append(snapshot)
        self._save_index()

        # Cleanup old snapshots
        self._cleanup_old_snapshots()

        logger.info(f"Created snapshot {snapshot_id}: {description}")
        return snapshot

    def get_snapshot(self, snapshot_id: str) -> Snapshot | None:
        """Get a snapshot by ID.

        Args:
            snapshot_id: ID to find.

        Returns:
            Snapshot if found, None otherwise.
        """
        for snapshot in self._snapshots:
            if snapshot.id == snapshot_id:
                # Load files if not already loaded
                if not snapshot.files:
                    self._load_snapshot_files(snapshot)
                return snapshot
        return None

    def _load_snapshot_files(self, snapshot: Snapshot) -> None:
        """Load files for a snapshot from storage.

        Args:
            snapshot: Snapshot to load files for.
        """
        snapshot_dir = self.storage_path / snapshot.id
        if not snapshot_dir.exists():
            return

        files_json = snapshot_dir / "files.json"
        if not files_json.exists():
            return

        with open(files_json) as f:
            file_paths = json.load(f)

        for file_path in file_paths:
            safe_name = file_path.replace("/", "_").replace("\\", "_")
            file_content_path = snapshot_dir / safe_name
            if file_content_path.exists():
                snapshot.files[file_path] = file_content_path.read_text()

    def rollback(
        self,
        snapshot_id: str,
        codebase_path: Path | str = ".",
    ) -> bool:
        """Rollback to a snapshot.

        Args:
            snapshot_id: ID of snapshot to rollback to.
            codebase_path: Base path for files.

        Returns:
            True if rollback succeeded.
        """
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            logger.error(f"Snapshot {snapshot_id} not found")
            return False

        codebase_path = Path(codebase_path)

        try:
            for file_path, content in snapshot.files.items():
                full_path = codebase_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                logger.debug(f"Restored {file_path}")

            logger.info(f"Rolled back to snapshot {snapshot_id}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def list_snapshots(self, limit: int = 10) -> list[Snapshot]:
        """List recent snapshots.

        Args:
            limit: Max snapshots to return.

        Returns:
            List of snapshots, most recent first.
        """
        return list(reversed(self._snapshots[-limit:]))

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot.

        Args:
            snapshot_id: ID to delete.

        Returns:
            True if deleted.
        """
        for i, snapshot in enumerate(self._snapshots):
            if snapshot.id == snapshot_id:
                # Delete files
                snapshot_dir = self.storage_path / snapshot_id
                if snapshot_dir.exists():
                    shutil.rmtree(snapshot_dir)

                # Remove from list
                self._snapshots.pop(i)
                self._save_index()

                logger.info(f"Deleted snapshot {snapshot_id}")
                return True

        return False

    def _cleanup_old_snapshots(self) -> None:
        """Remove old snapshots beyond max_snapshots."""
        while len(self._snapshots) > self.max_snapshots:
            oldest = self._snapshots.pop(0)
            snapshot_dir = self.storage_path / oldest.id
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            logger.debug(f"Cleaned up old snapshot {oldest.id}")

        self._save_index()
