"""Core filesystem operations for chat agents.

All operations use PathValidator for security.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from animus_forge.tools.models import (
    DirectoryListing,
    FileContent,
    FileEntry,
    ProjectStructure,
    SearchMatch,
    SearchResult,
)
from animus_forge.tools.safety import PathValidator, SecurityError

if TYPE_CHECKING:
    pass


# Maximum results for listings and searches
DEFAULT_MAX_RESULTS = 100
DEFAULT_MAX_TREE_DEPTH = 4


class FilesystemTools:
    """Filesystem operations for local project access.

    All operations are constrained to the project directory via PathValidator.
    """

    def __init__(
        self,
        validator: PathValidator,
        max_results: int = DEFAULT_MAX_RESULTS,
    ):
        """Initialize filesystem tools.

        Args:
            validator: Path validator with project constraints.
            max_results: Maximum results for listings and searches.
        """
        self.validator = validator
        self.max_results = max_results
        self.project_root = validator.get_project_root()

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> FileContent:
        """Read a file's content with optional line range.

        Args:
            path: Path to file (relative or absolute).
            start_line: Start line (1-indexed, inclusive). None for start.
            end_line: End line (1-indexed, inclusive). None for end.

        Returns:
            FileContent with file content and metadata.

        Raises:
            SecurityError: If path fails validation.
            FileNotFoundError: If file doesn't exist.
        """
        resolved = self.validator.validate_file_for_read(path)
        rel_path = str(resolved.relative_to(self.project_root))

        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try latin-1 as fallback
            content = resolved.read_text(encoding="latin-1")

        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        # Apply line range
        truncated = False
        if start_line is not None or end_line is not None:
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else total_lines
            lines = lines[start_idx:end_idx]
            truncated = (start_idx > 0) or (end_idx < total_lines)

        # Add line numbers
        numbered_lines = []
        line_offset = (start_line - 1) if start_line else 0
        for i, line in enumerate(lines, start=line_offset + 1):
            numbered_lines.append(f"{i:>6}\t{line.rstrip()}")

        return FileContent(
            path=rel_path,
            content="\n".join(numbered_lines),
            line_count=total_lines,
            size_bytes=resolved.stat().st_size,
            truncated=truncated,
        )

    def list_files(
        self,
        path: str = ".",
        pattern: str | None = None,
        recursive: bool = False,
    ) -> DirectoryListing:
        """List files in a directory.

        Args:
            path: Directory path (relative or absolute).
            pattern: Optional glob pattern to filter files.
            recursive: Whether to list recursively.

        Returns:
            DirectoryListing with directory contents.

        Raises:
            SecurityError: If path fails validation.
        """
        resolved = self.validator.validate_directory(path)
        rel_path = str(resolved.relative_to(self.project_root))

        entries: list[FileEntry] = []
        total_files = 0
        total_dirs = 0
        truncated = False

        if recursive:
            iterator = resolved.rglob(pattern or "*")
        else:
            iterator = resolved.glob(pattern or "*")

        for item in iterator:
            # Skip excluded paths
            try:
                item_rel = str(item.relative_to(self.project_root))
                if self.validator.is_excluded(item_rel):
                    continue
            except ValueError:
                continue

            if item.is_dir():
                total_dirs += 1
            else:
                total_files += 1

            if len(entries) >= self.max_results:
                truncated = True
                continue  # Keep counting but don't add more

            try:
                entries.append(
                    FileEntry(
                        name=item.name,
                        path=item_rel,
                        is_dir=item.is_dir(),
                        size_bytes=item.stat().st_size if item.is_file() else None,
                    )
                )
            except OSError:
                pass  # Best-effort cleanup: skip files we can't stat

        # Sort: directories first, then by name
        entries.sort(key=lambda e: (not e.is_dir, e.name.lower()))

        return DirectoryListing(
            path=rel_path,
            entries=entries,
            total_files=total_files,
            total_dirs=total_dirs,
            truncated=truncated,
        )

    def search_code(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str | None = None,
        case_sensitive: bool = True,
        max_results: int | None = None,
    ) -> SearchResult:
        """Search for a pattern in files.

        Args:
            pattern: Regex pattern to search for.
            path: Directory to search in.
            file_pattern: Optional glob pattern to filter files.
            case_sensitive: Whether search is case-sensitive.
            max_results: Maximum number of matches to return.

        Returns:
            SearchResult with matches.

        Raises:
            SecurityError: If path fails validation.
        """
        resolved = self.validator.validate_directory(path)
        max_res = max_results or self.max_results

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            raise SecurityError(f"Invalid regex pattern: {e}")

        matches: list[SearchMatch] = []
        files_searched = 0
        truncated = False

        # Walk through files
        for root, _, files in os.walk(resolved):
            root_path = Path(root)

            for filename in files:
                file_path = root_path / filename

                # Apply file pattern filter
                if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                    continue

                # Check exclusions
                try:
                    rel_path = str(file_path.relative_to(self.project_root))
                    if self.validator.is_excluded(rel_path):
                        continue
                except ValueError:
                    continue

                # Check file is readable
                try:
                    self.validator.validate_file_for_read(file_path)
                except SecurityError:
                    continue

                files_searched += 1

                # Search file content
                try:
                    content = file_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue

                for line_num, line in enumerate(content.splitlines(), start=1):
                    for match in regex.finditer(line):
                        if len(matches) >= max_res:
                            truncated = True
                            break

                        matches.append(
                            SearchMatch(
                                path=rel_path,
                                line_number=line_num,
                                line_content=line.strip(),
                                match_start=match.start(),
                                match_end=match.end(),
                            )
                        )

                    if truncated:
                        break

                if truncated:
                    break

            if truncated:
                break

        return SearchResult(
            pattern=pattern,
            matches=matches,
            total_matches=len(matches),
            files_searched=files_searched,
            truncated=truncated,
        )

    def get_structure(
        self,
        max_depth: int = DEFAULT_MAX_TREE_DEPTH,
    ) -> ProjectStructure:
        """Get an overview of the project structure.

        Args:
            max_depth: Maximum depth for tree representation.

        Returns:
            ProjectStructure with tree and file stats.
        """
        tree_lines: list[str] = []
        total_files = 0
        total_dirs = 0
        file_types: dict[str, int] = {}

        def build_tree(path: Path, prefix: str = "", depth: int = 0) -> None:
            nonlocal total_files, total_dirs

            if depth > max_depth:
                return

            try:
                entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except PermissionError:
                return

            # Filter out excluded entries
            visible_entries = []
            for entry in entries:
                try:
                    rel_path = str(entry.relative_to(self.project_root))
                    if not self.validator.is_excluded(rel_path):
                        visible_entries.append(entry)
                except ValueError:
                    pass  # Graceful degradation: path not relative to root, skip entry

            for i, entry in enumerate(visible_entries):
                is_last = i == len(visible_entries) - 1
                connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
                extension = "    " if is_last else "\u2502   "

                if entry.is_dir():
                    total_dirs += 1
                    tree_lines.append(f"{prefix}{connector}{entry.name}/")
                    if depth < max_depth:
                        build_tree(entry, prefix + extension, depth + 1)
                else:
                    total_files += 1
                    tree_lines.append(f"{prefix}{connector}{entry.name}")

                    # Track file types
                    ext = entry.suffix.lower() or "(no extension)"
                    file_types[ext] = file_types.get(ext, 0) + 1

        # Build tree
        tree_lines.append(f"{self.project_root.name}/")
        build_tree(self.project_root)

        return ProjectStructure(
            root_path=str(self.project_root),
            total_files=total_files,
            total_dirs=total_dirs,
            tree="\n".join(tree_lines),
            file_types=file_types,
        )

    def glob_files(
        self,
        pattern: str,
    ) -> list[str]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py").

        Returns:
            List of relative paths matching the pattern.
        """
        results: list[str] = []

        for path in self.project_root.glob(pattern):
            try:
                rel_path = str(path.relative_to(self.project_root))
                if not self.validator.is_excluded(rel_path) and path.is_file():
                    results.append(rel_path)
            except ValueError:
                pass  # Graceful degradation: path not relative to root, skip

            if len(results) >= self.max_results:
                break

        return sorted(results)
