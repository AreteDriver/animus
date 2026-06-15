"""Path validation and security checks for filesystem tools.

Security-first design:
- All paths must resolve within allowed project directory
- No symlink traversal outside project bounds
- Excluded patterns for sensitive directories
- File size limits to prevent memory issues
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class SecurityError(Exception):
    """Raised when a filesystem operation fails security validation."""

    pass


# Patterns to exclude from file operations
DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    r"^\.git(/|$)",
    r"^\.git$",
    r"node_modules(/|$)",
    r"__pycache__(/|$)",
    r"\.pyc$",
    r"^\.venv(/|$)",
    r"^venv(/|$)",
    r"^\.env$",
    r"\.env\.local$",
    r"\.env\.\w+$",
    r"^\.DS_Store$",
    r"^\.idea(/|$)",
    r"^\.vscode(/|$)",
    r"\.egg-info(/|$)",
    r"^dist(/|$)",
    r"^build(/|$)",
    r"^\.coverage$",
    r"\.sqlite3?$",
    r"^\.pytest_cache(/|$)",
    r"^\.mypy_cache(/|$)",
    r"^\.ruff_cache(/|$)",
    r"^htmlcov(/|$)",
    r"\.pem$",
    r"\.key$",
    r"^secrets\.",
    r"^credentials\.",
]

# Maximum file size for read operations (1MB default)
DEFAULT_MAX_FILE_SIZE: int = 1 * 1024 * 1024


class PathValidator:
    """Validates paths against security constraints.

    All filesystem operations should use this validator before accessing files.
    """

    def __init__(
        self,
        project_path: str | Path,
        allowed_paths: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    ):
        """Initialize the path validator.

        Args:
            project_path: Root directory for the project. All operations are
                          constrained to this directory.
            allowed_paths: Optional list of additional allowed paths outside
                           the project root.
            exclude_patterns: Regex patterns for paths to exclude. Defaults to
                              DEFAULT_EXCLUDE_PATTERNS.
            max_file_size: Maximum file size in bytes for read operations.
        """
        self.project_path = Path(project_path).resolve()
        self.allowed_paths = [Path(p).resolve() for p in (allowed_paths or [])]
        self.exclude_patterns = [
            re.compile(p) for p in (exclude_patterns or DEFAULT_EXCLUDE_PATTERNS)
        ]
        self.max_file_size = max_file_size

        # Validate project path exists
        if not self.project_path.is_dir():
            raise SecurityError(
                f"Project path does not exist or is not a directory: {project_path}"
            )

    def validate_path(self, path: str | Path) -> Path:
        """Validate a path and return its resolved form.

        Args:
            path: Path to validate (relative or absolute).

        Returns:
            Resolved absolute path if valid.

        Raises:
            SecurityError: If path fails validation.
        """
        # Convert to Path object
        target = Path(path)

        # If relative, make it relative to project root
        if not target.is_absolute():
            target = self.project_path / target

        # Resolve to absolute path (handles .., symlinks, etc.)
        resolved = target.resolve()

        # Check if within project bounds
        if not self._is_within_allowed_paths(resolved):
            raise SecurityError(f"Path is outside allowed directories: {path}")

        # Check against exclude patterns
        rel_path = self._get_relative_path(resolved)
        if rel_path and self._matches_exclude_pattern(rel_path):
            raise SecurityError(f"Path matches excluded pattern: {path}")

        return resolved

    def validate_file_for_read(self, path: str | Path) -> Path:
        """Validate a file path for reading.

        Args:
            path: Path to validate.

        Returns:
            Resolved path if valid for reading.

        Raises:
            SecurityError: If path fails validation or file is too large.
        """
        resolved = self.validate_path(path)

        if not resolved.is_file():
            raise SecurityError(f"Path is not a file: {path}")

        # Check file size
        size = resolved.stat().st_size
        if size > self.max_file_size:
            raise SecurityError(f"File exceeds size limit ({size} > {self.max_file_size}): {path}")

        return resolved

    def validate_directory(self, path: str | Path) -> Path:
        """Validate a directory path.

        Args:
            path: Path to validate.

        Returns:
            Resolved path if valid directory.

        Raises:
            SecurityError: If path fails validation or is not a directory.
        """
        resolved = self.validate_path(path)

        if not resolved.is_dir():
            raise SecurityError(f"Path is not a directory: {path}")

        return resolved

    def validate_file_for_write(self, path: str | Path) -> Path:
        """Validate a file path for writing.

        Args:
            path: Path to validate.

        Returns:
            Resolved path if valid for writing.

        Raises:
            SecurityError: If path fails validation or parent doesn't exist.
        """
        # Convert to Path object
        target = Path(path)

        # If relative, make it relative to project root
        if not target.is_absolute():
            target = self.project_path / target

        # Resolve parent directory
        parent = target.parent.resolve()

        # Check parent is within allowed paths
        if not self._is_within_allowed_paths(parent):
            raise SecurityError(f"Path is outside allowed directories: {path}")

        # Check parent exists
        if not parent.is_dir():
            raise SecurityError(f"Parent directory does not exist: {path}")

        # Resolve the full path (may not exist yet)
        resolved = target.resolve()

        # Check against exclude patterns
        rel_path = self._get_relative_path(resolved)
        if rel_path and self._matches_exclude_pattern(rel_path):
            raise SecurityError(f"Path matches excluded pattern: {path}")

        return resolved

    def is_excluded(self, path: str | Path) -> bool:
        """Check if a path matches exclusion patterns without raising.

        Args:
            path: Path to check.

        Returns:
            True if path should be excluded.
        """
        rel_path = str(path)
        return self._matches_exclude_pattern(rel_path)

    def _is_within_allowed_paths(self, resolved: Path) -> bool:
        """Check if resolved path is within allowed directories."""
        # Check project path
        try:
            resolved.relative_to(self.project_path)
            return True
        except ValueError:
            pass  # Graceful degradation: path not under project root, check allowed paths next

        # Check additional allowed paths
        for allowed in self.allowed_paths:
            try:
                resolved.relative_to(allowed)
                return True
            except ValueError:
                pass  # Graceful degradation: path not under this allowed dir, try next

        return False

    def _get_relative_path(self, resolved: Path) -> str | None:
        """Get path relative to project root, or None if outside."""
        try:
            return str(resolved.relative_to(self.project_path))
        except ValueError:
            return None

    def _matches_exclude_pattern(self, rel_path: str) -> bool:
        """Check if relative path matches any exclude pattern."""
        # Normalize path separators
        normalized = rel_path.replace(os.sep, "/")

        for pattern in self.exclude_patterns:
            if pattern.search(normalized):
                return True
        return False

    def get_project_root(self) -> Path:
        """Return the validated project root path."""
        return self.project_path
