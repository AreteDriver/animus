"""Workflow version management.

Provides version history tracking, rollback, comparison, and import/export
for workflow version control.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml

from animus_forge.state.backends import DatabaseBackend

from .versioning import (
    SemanticVersion,
    VersionDiff,
    WorkflowVersion,
    compare_versions,
    compute_content_hash,
    deserialize_metadata,
    serialize_metadata,
)

logger = logging.getLogger(__name__)


class WorkflowVersionManager:
    """Manages workflow version history and operations.

    Provides CRUD operations for workflow versions with support for
    semantic versioning, rollback, and diff comparison.
    """

    def __init__(self, backend: DatabaseBackend):
        """Initialize version manager.

        Args:
            backend: Database backend for persistence
        """
        self.backend = backend

    def save_version(
        self,
        workflow_name: str,
        content: str,
        version: str | None = None,
        description: str | None = None,
        author: str | None = None,
        auto_bump: Literal["major", "minor", "patch"] = "patch",
        activate: bool = True,
        metadata: dict | None = None,
    ) -> WorkflowVersion:
        """Save a new workflow version.

        If version is not specified, automatically bumps from the latest version.
        Detects duplicate content via hash and skips save if unchanged.

        Args:
            workflow_name: Name of the workflow
            content: YAML content of the workflow
            version: Explicit version string (optional)
            description: Version description
            author: Author identifier
            auto_bump: Bump type when version is auto-generated
            activate: Whether to make this the active version
            metadata: Additional metadata to store

        Returns:
            Saved WorkflowVersion

        Raises:
            ValueError: If version already exists or content is invalid
        """
        content_hash = compute_content_hash(content)

        # Check for duplicate content
        existing = self.backend.fetchone(
            "SELECT id, version FROM workflow_versions "
            "WHERE workflow_name = ? AND content_hash = ?",
            (workflow_name, content_hash),
        )
        if existing:
            logger.info(
                f"Content unchanged for {workflow_name}, matching version {existing['version']}"
            )
            return self.get_version(workflow_name, existing["version"])

        # Determine version
        if version is None:
            latest = self.get_latest_version(workflow_name)
            if latest:
                sem_ver = latest.get_semantic_version().bump(auto_bump)
            else:
                sem_ver = SemanticVersion(1, 0, 0)
            version = str(sem_ver)
        else:
            sem_ver = SemanticVersion.parse(version)

        # Check if version exists
        if self._version_exists(workflow_name, version):
            raise ValueError(f"Version {version} already exists for workflow {workflow_name}")

        # Validate YAML content
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML content: {e}")

        # Create version record
        workflow_version = WorkflowVersion.create(
            workflow_name=workflow_name,
            version=version,
            content=content,
            description=description,
            author=author,
            is_active=False,
            metadata=metadata,
        )

        with self.backend.transaction():
            # Insert new version
            self.backend.execute(
                """
                INSERT INTO workflow_versions
                (workflow_name, version, version_major, version_minor, version_patch,
                 content, content_hash, description, author, is_active, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workflow_version.workflow_name,
                    workflow_version.version,
                    workflow_version.version_major,
                    workflow_version.version_minor,
                    workflow_version.version_patch,
                    workflow_version.content,
                    workflow_version.content_hash,
                    workflow_version.description,
                    workflow_version.author,
                    workflow_version.is_active,
                    serialize_metadata(workflow_version.metadata),
                ),
            )

            if activate:
                self._set_active_internal(workflow_name, version)

        logger.info(f"Saved version {version} for workflow {workflow_name}")
        return self.get_version(workflow_name, version)

    def get_version(self, workflow_name: str, version: str) -> WorkflowVersion | None:
        """Get a specific workflow version.

        Args:
            workflow_name: Name of the workflow
            version: Version string

        Returns:
            WorkflowVersion or None if not found
        """
        row = self.backend.fetchone(
            "SELECT * FROM workflow_versions WHERE workflow_name = ? AND version = ?",
            (workflow_name, version),
        )
        return self._row_to_version(row) if row else None

    def get_active_version(self, workflow_name: str) -> WorkflowVersion | None:
        """Get the currently active version of a workflow.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Active WorkflowVersion or None if none active
        """
        row = self.backend.fetchone(
            "SELECT * FROM workflow_versions WHERE workflow_name = ? AND is_active = TRUE",
            (workflow_name,),
        )
        return self._row_to_version(row) if row else None

    def get_latest_version(self, workflow_name: str) -> WorkflowVersion | None:
        """Get the latest (highest) version of a workflow.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Latest WorkflowVersion or None if no versions exist
        """
        row = self.backend.fetchone(
            """
            SELECT * FROM workflow_versions
            WHERE workflow_name = ?
            ORDER BY version_major DESC, version_minor DESC, version_patch DESC
            LIMIT 1
            """,
            (workflow_name,),
        )
        return self._row_to_version(row) if row else None

    def list_versions(
        self,
        workflow_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[WorkflowVersion]:
        """List all versions of a workflow.

        Args:
            workflow_name: Name of the workflow
            limit: Maximum versions to return
            offset: Number of versions to skip

        Returns:
            List of WorkflowVersions, newest first
        """
        rows = self.backend.fetchall(
            """
            SELECT * FROM workflow_versions
            WHERE workflow_name = ?
            ORDER BY version_major DESC, version_minor DESC, version_patch DESC
            LIMIT ? OFFSET ?
            """,
            (workflow_name, limit, offset),
        )
        return [self._row_to_version(row) for row in rows]

    def list_workflows(self) -> list[dict]:
        """List all workflows with version info.

        Returns:
            List of workflow summaries with name, latest version, and active version
        """
        rows = self.backend.fetchall(
            """
            SELECT DISTINCT workflow_name FROM workflow_versions
            ORDER BY workflow_name
            """
        )

        workflows = []
        for row in rows:
            name = row["workflow_name"]
            latest = self.get_latest_version(name)
            active = self.get_active_version(name)

            workflows.append(
                {
                    "workflow_name": name,
                    "latest_version": latest.version if latest else None,
                    "active_version": active.version if active else None,
                    "total_versions": self._count_versions(name),
                }
            )

        return workflows

    def set_active(self, workflow_name: str, version: str) -> bool:
        """Set a specific version as active.

        Args:
            workflow_name: Name of the workflow
            version: Version to activate

        Returns:
            True if successful

        Raises:
            ValueError: If version doesn't exist
        """
        if not self._version_exists(workflow_name, version):
            raise ValueError(f"Version {version} doesn't exist for workflow {workflow_name}")

        with self.backend.transaction():
            self._set_active_internal(workflow_name, version)

        logger.info(f"Activated version {version} for workflow {workflow_name}")
        return True

    def rollback(self, workflow_name: str) -> WorkflowVersion | None:
        """Rollback to the previous version.

        Activates the version immediately before the current active version.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Newly active WorkflowVersion, or None if no previous version
        """
        active = self.get_active_version(workflow_name)
        if not active:
            # No active version, try to activate latest
            latest = self.get_latest_version(workflow_name)
            if latest:
                self.set_active(workflow_name, latest.version)
                return latest
            return None

        # Find previous version
        rows = self.backend.fetchall(
            """
            SELECT * FROM workflow_versions
            WHERE workflow_name = ?
            ORDER BY version_major DESC, version_minor DESC, version_patch DESC
            """,
            (workflow_name,),
        )

        versions = [self._row_to_version(row) for row in rows]
        active_sem = active.get_semantic_version()

        # Find version just before active
        for v in versions:
            if v.get_semantic_version() < active_sem:
                self.set_active(workflow_name, v.version)
                logger.info(f"Rolled back {workflow_name} from {active.version} to {v.version}")
                return v

        logger.warning(f"No previous version to rollback to for {workflow_name}")
        return None

    def rollback_to(self, workflow_name: str, version: str) -> WorkflowVersion:
        """Rollback to a specific version.

        Args:
            workflow_name: Name of the workflow
            version: Version to rollback to

        Returns:
            Activated WorkflowVersion

        Raises:
            ValueError: If version doesn't exist
        """
        self.set_active(workflow_name, version)
        return self.get_version(workflow_name, version)

    def compare_versions(
        self,
        workflow_name: str,
        from_version: str,
        to_version: str,
    ) -> VersionDiff:
        """Compare two workflow versions.

        Args:
            workflow_name: Name of the workflow
            from_version: Base version for comparison
            to_version: Target version for comparison

        Returns:
            VersionDiff with comparison results

        Raises:
            ValueError: If either version doesn't exist
        """
        from_v = self.get_version(workflow_name, from_version)
        to_v = self.get_version(workflow_name, to_version)

        if not from_v:
            raise ValueError(f"Version {from_version} not found for {workflow_name}")
        if not to_v:
            raise ValueError(f"Version {to_version} not found for {workflow_name}")

        diff = compare_versions(from_v.content, to_v.content)
        diff.from_version = from_version
        diff.to_version = to_version

        return diff

    def get_unified_diff(
        self,
        workflow_name: str,
        from_version: str,
        to_version: str,
    ) -> str:
        """Get git-style unified diff between versions.

        Args:
            workflow_name: Name of the workflow
            from_version: Base version
            to_version: Target version

        Returns:
            Unified diff string
        """
        diff = self.compare_versions(workflow_name, from_version, to_version)
        return diff.unified_diff

    def import_from_file(
        self,
        file_path: str | Path,
        description: str | None = None,
        author: str | None = None,
        activate: bool = True,
    ) -> WorkflowVersion:
        """Import a workflow version from a YAML file.

        Extracts workflow name and version from file content.

        Args:
            file_path: Path to workflow YAML file
            description: Version description (optional)
            author: Author identifier (optional)
            activate: Whether to make this the active version

        Returns:
            Imported WorkflowVersion

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file content is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {file_path}")

        content = file_path.read_text()

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")

        if not isinstance(data, dict):
            raise ValueError(f"Workflow file must contain a YAML mapping: {file_path}")

        workflow_name = data.get("name")
        if not workflow_name:
            raise ValueError(f"Workflow must have a 'name' field: {file_path}")

        # Use version from file or default to 1.0.0
        version = data.get("version", "1.0.0")

        return self.save_version(
            workflow_name=workflow_name,
            content=content,
            version=version,
            description=description or f"Imported from {file_path.name}",
            author=author,
            activate=activate,
        )

    def export_to_file(
        self,
        workflow_name: str,
        version: str | None = None,
        file_path: str | Path | None = None,
    ) -> Path:
        """Export a workflow version to a YAML file.

        Args:
            workflow_name: Name of the workflow
            version: Version to export (default: active version)
            file_path: Output file path (default: {workflow_name}.yaml)

        Returns:
            Path to exported file

        Raises:
            ValueError: If version doesn't exist
        """
        if version:
            wv = self.get_version(workflow_name, version)
        else:
            wv = self.get_active_version(workflow_name)
            if not wv:
                wv = self.get_latest_version(workflow_name)

        if not wv:
            raise ValueError(f"No versions found for workflow {workflow_name}")

        if file_path is None:
            file_path = Path(f"{workflow_name}.yaml")
        else:
            file_path = Path(file_path)

        file_path.write_text(wv.content)
        logger.info(f"Exported {workflow_name} v{wv.version} to {file_path}")

        return file_path

    def migrate_existing_workflows(
        self,
        workflows_dir: str | Path,
        author: str | None = None,
    ) -> list[WorkflowVersion]:
        """One-time import of existing workflow files.

        Imports all YAML files from a directory that don't already have versions.

        Args:
            workflows_dir: Directory containing workflow YAML files
            author: Author identifier for imported versions

        Returns:
            List of imported WorkflowVersions
        """
        workflows_dir = Path(workflows_dir)
        if not workflows_dir.exists():
            logger.warning(f"Workflows directory not found: {workflows_dir}")
            return []

        imported = []
        for yaml_file in workflows_dir.glob("*.yaml"):
            try:
                content = yaml_file.read_text()
                data = yaml.safe_load(content)

                if not isinstance(data, dict) or "name" not in data:
                    logger.warning(f"Skipping invalid workflow file: {yaml_file}")
                    continue

                workflow_name = data["name"]

                # Skip if already has versions
                if self.get_latest_version(workflow_name):
                    logger.debug(f"Workflow {workflow_name} already has versions, skipping")
                    continue

                version = self.import_from_file(
                    yaml_file,
                    description="Initial import from filesystem",
                    author=author,
                    activate=True,
                )
                imported.append(version)
                logger.info(f"Imported workflow {workflow_name} from {yaml_file}")

            except Exception as e:
                logger.error(f"Failed to import {yaml_file}: {e}")

        if imported:
            logger.info(f"Migrated {len(imported)} existing workflows")

        return imported

    def delete_version(self, workflow_name: str, version: str) -> bool:
        """Delete a specific workflow version.

        Cannot delete the active version.

        Args:
            workflow_name: Name of the workflow
            version: Version to delete

        Returns:
            True if deleted

        Raises:
            ValueError: If version is active or doesn't exist
        """
        wv = self.get_version(workflow_name, version)
        if not wv:
            raise ValueError(f"Version {version} doesn't exist for workflow {workflow_name}")

        if wv.is_active:
            raise ValueError(
                f"Cannot delete active version {version}. Activate a different version first."
            )

        self.backend.execute(
            "DELETE FROM workflow_versions WHERE workflow_name = ? AND version = ?",
            (workflow_name, version),
        )
        logger.info(f"Deleted version {version} of workflow {workflow_name}")
        return True

    def _version_exists(self, workflow_name: str, version: str) -> bool:
        """Check if a version exists."""
        row = self.backend.fetchone(
            "SELECT 1 FROM workflow_versions WHERE workflow_name = ? AND version = ?",
            (workflow_name, version),
        )
        return row is not None

    def _count_versions(self, workflow_name: str) -> int:
        """Count total versions for a workflow."""
        row = self.backend.fetchone(
            "SELECT COUNT(*) as count FROM workflow_versions WHERE workflow_name = ?",
            (workflow_name,),
        )
        return row["count"] if row else 0

    def _set_active_internal(self, workflow_name: str, version: str) -> None:
        """Set active version (must be called within transaction)."""
        # Deactivate all versions
        self.backend.execute(
            "UPDATE workflow_versions SET is_active = FALSE WHERE workflow_name = ?",
            (workflow_name,),
        )
        # Activate specified version
        self.backend.execute(
            "UPDATE workflow_versions SET is_active = TRUE WHERE workflow_name = ? AND version = ?",
            (workflow_name, version),
        )

    def _row_to_version(self, row: dict) -> WorkflowVersion:
        """Convert database row to WorkflowVersion."""
        return WorkflowVersion(
            id=row["id"],
            workflow_name=row["workflow_name"],
            version=row["version"],
            version_major=row["version_major"],
            version_minor=row["version_minor"],
            version_patch=row["version_patch"],
            content=row["content"],
            content_hash=row["content_hash"],
            description=row.get("description"),
            author=row.get("author"),
            created_at=row.get("created_at"),
            is_active=bool(row.get("is_active", False)),
            metadata=deserialize_metadata(row.get("metadata")),
        )
