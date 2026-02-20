"""Skill version history management backed by SQLite."""

from __future__ import annotations

import difflib
import logging
import threading
from datetime import UTC, datetime

from animus_forge.state.backends import DatabaseBackend

logger = logging.getLogger(__name__)


class SkillVersioner:
    """Tracks the version history of skills with full YAML snapshots.

    Args:
        backend: A ``DatabaseBackend`` instance.
    """

    def __init__(self, backend: DatabaseBackend) -> None:
        self._backend = backend
        self._lock = threading.Lock()

    def record_version(
        self,
        skill_name: str,
        version: str,
        previous_version: str | None,
        change_type: str,
        description: str,
        yaml_snapshot: str,
        approval_id: str = "",
        diff_summary: str = "",
    ) -> None:
        """Record a new version of a skill.

        Args:
            skill_name: Name of the skill.
            version: New version string.
            previous_version: Previous version (None for first version).
            change_type: Type of change (tune, generate, deprecate, manual).
            description: Human-readable change description.
            yaml_snapshot: Full YAML content at this version.
            approval_id: Approval gate request ID, if applicable.
            diff_summary: Summary of changes from previous version.
        """
        query = (
            "INSERT OR REPLACE INTO skill_versions "
            "(skill_name, version, previous_version, change_type, "
            "change_description, schema_snapshot, diff_summary, approval_id, "
            "created_at, created_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            skill_name,
            version,
            previous_version,
            change_type,
            description,
            yaml_snapshot,
            diff_summary,
            approval_id,
            datetime.now(UTC).isoformat(),
            "skill_evolver",
        )

        with self._lock:
            with self._backend.transaction():
                self._backend.execute(query, params)

    def get_version_history(self, skill_name: str) -> list[dict]:
        """Get all version records for a skill, newest first.

        Args:
            skill_name: Name of the skill.

        Returns:
            List of version record dicts.
        """
        query = "SELECT * FROM skill_versions WHERE skill_name = ? ORDER BY created_at DESC"
        with self._lock:
            return self._backend.fetchall(query, (skill_name,))

    def get_version_snapshot(self, skill_name: str, version: str) -> str:
        """Get the YAML snapshot for a specific version.

        Args:
            skill_name: Name of the skill.
            version: Version string to retrieve.

        Returns:
            YAML string, or empty string if not found.
        """
        query = "SELECT schema_snapshot FROM skill_versions WHERE skill_name = ? AND version = ?"
        with self._lock:
            row = self._backend.fetchone(query, (skill_name, version))
        if row:
            return str(row["schema_snapshot"])
        return ""

    def compute_diff(self, skill_name: str, version_a: str, version_b: str) -> str:
        """Compute a unified diff between two versions of a skill.

        Args:
            skill_name: Name of the skill.
            version_a: Base version.
            version_b: Target version.

        Returns:
            Unified diff string.
        """
        snap_a = self.get_version_snapshot(skill_name, version_a)
        snap_b = self.get_version_snapshot(skill_name, version_b)

        if not snap_a and not snap_b:
            return ""

        diff_lines = difflib.unified_diff(
            snap_a.splitlines(keepends=True),
            snap_b.splitlines(keepends=True),
            fromfile=f"{skill_name} v{version_a}",
            tofile=f"{skill_name} v{version_b}",
        )
        return "".join(diff_lines)

    def get_latest_version(self, skill_name: str) -> str:
        """Get the most recent version string for a skill.

        Args:
            skill_name: Name of the skill.

        Returns:
            Version string, or empty string if no versions recorded.
        """
        query = (
            "SELECT version FROM skill_versions "
            "WHERE skill_name = ? "
            "ORDER BY created_at DESC LIMIT 1"
        )
        with self._lock:
            row = self._backend.fetchone(query, (skill_name,))
        if row:
            return str(row["version"])
        return ""
