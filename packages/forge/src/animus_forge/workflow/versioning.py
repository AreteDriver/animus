"""Workflow versioning models and utilities.

Provides semantic versioning support, version comparison, and diff generation
for workflow version history tracking.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from difflib import unified_diff
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class SemanticVersion:
    """Semantic version representation (major.minor.patch).

    Supports parsing, comparison, and version bumping.
    """

    major: int
    minor: int
    patch: int

    VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

    @classmethod
    def parse(cls, version_str: str) -> SemanticVersion:
        """Parse version string into SemanticVersion.

        Args:
            version_str: Version string like "1.2.3"

        Returns:
            SemanticVersion instance

        Raises:
            ValueError: If version string is invalid
        """
        match = cls.VERSION_PATTERN.match(version_str.strip())
        if not match:
            raise ValueError(
                f"Invalid version format: '{version_str}'. Expected 'major.minor.patch'"
            )
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
        )

    def __str__(self) -> str:
        """Return version as string."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (
            other.major,
            other.minor,
            other.patch,
        )

    def __lt__(self, other: SemanticVersion) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )

    def __le__(self, other: SemanticVersion) -> bool:
        return self == other or self < other

    def __gt__(self, other: SemanticVersion) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) > (
            other.major,
            other.minor,
            other.patch,
        )

    def __ge__(self, other: SemanticVersion) -> bool:
        return self == other or self > other

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    def bump_major(self) -> SemanticVersion:
        """Return new version with major incremented, minor/patch reset to 0."""
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> SemanticVersion:
        """Return new version with minor incremented, patch reset to 0."""
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> SemanticVersion:
        """Return new version with patch incremented."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)

    def bump(self, bump_type: str = "patch") -> SemanticVersion:
        """Bump version by type.

        Args:
            bump_type: One of 'major', 'minor', 'patch'

        Returns:
            New bumped version

        Raises:
            ValueError: If bump_type is invalid
        """
        if bump_type == "major":
            return self.bump_major()
        elif bump_type == "minor":
            return self.bump_minor()
        elif bump_type == "patch":
            return self.bump_patch()
        else:
            raise ValueError(f"Invalid bump type: {bump_type}. Use 'major', 'minor', or 'patch'")


class WorkflowVersion(BaseModel):
    """Workflow version database record representation."""

    id: int | None = None
    workflow_name: str
    version: str
    version_major: int
    version_minor: int
    version_patch: int
    content: str
    content_hash: str
    description: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    is_active: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        workflow_name: str,
        version: str | SemanticVersion,
        content: str,
        description: str | None = None,
        author: str | None = None,
        is_active: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowVersion:
        """Create a new WorkflowVersion instance.

        Args:
            workflow_name: Name of the workflow
            version: Version string or SemanticVersion
            content: YAML content of the workflow
            description: Optional version description
            author: Optional author identifier
            is_active: Whether this is the active version
            metadata: Optional additional metadata

        Returns:
            New WorkflowVersion instance
        """
        if isinstance(version, SemanticVersion):
            sem_ver = version
            version_str = str(version)
        else:
            sem_ver = SemanticVersion.parse(version)
            version_str = version

        return cls(
            workflow_name=workflow_name,
            version=version_str,
            version_major=sem_ver.major,
            version_minor=sem_ver.minor,
            version_patch=sem_ver.patch,
            content=content,
            content_hash=compute_content_hash(content),
            description=description,
            author=author,
            is_active=is_active,
            metadata=metadata or {},
        )

    model_config = ConfigDict(from_attributes=True)

    def get_semantic_version(self) -> SemanticVersion:
        """Get version as SemanticVersion object."""
        return SemanticVersion(self.version_major, self.version_minor, self.version_patch)


@dataclass
class VersionDiff:
    """Result of comparing two workflow versions."""

    from_version: str
    to_version: str
    added_lines: int = 0
    removed_lines: int = 0
    changed_sections: list[str] = field(default_factory=list)
    unified_diff: str = ""

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return self.added_lines > 0 or self.removed_lines > 0


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of workflow content.

    Normalizes content before hashing for consistent deduplication.

    Args:
        content: Workflow YAML content

    Returns:
        SHA256 hex digest
    """
    # Normalize whitespace for consistent hashing
    normalized = content.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compare_versions(old_content: str, new_content: str) -> VersionDiff:
    """Compare two workflow versions and generate diff.

    Args:
        old_content: Previous version content
        new_content: New version content

    Returns:
        VersionDiff with comparison results
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff_lines = list(unified_diff(old_lines, new_lines, fromfile="old", tofile="new", lineterm=""))

    added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

    # Identify changed sections (YAML top-level keys)
    changed_sections = set()
    for line in diff_lines:
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            # Extract section name if it's a top-level key
            content = line[1:].strip()
            if content and not content.startswith(" ") and ":" in content:
                section = content.split(":")[0].strip()
                changed_sections.add(section)

    return VersionDiff(
        from_version="old",
        to_version="new",
        added_lines=added,
        removed_lines=removed,
        changed_sections=sorted(changed_sections),
        unified_diff="".join(diff_lines),
    )


def serialize_metadata(metadata: dict[str, Any] | None) -> str | None:
    """Serialize metadata dict to JSON string for database storage."""
    if metadata is None:
        return None
    return json.dumps(metadata)


def deserialize_metadata(metadata_str: str | None) -> dict[str, Any]:
    """Deserialize metadata JSON string from database."""
    if metadata_str is None:
        return {}
    try:
        return json.loads(metadata_str)
    except json.JSONDecodeError:
        return {}
