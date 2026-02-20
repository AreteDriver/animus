"""
Filesystem Integration

Index and search local files.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from animus.integrations.base import AuthType, BaseIntegration
from animus.logging import get_logger
from animus.tools import Tool, ToolResult

logger = get_logger("integrations.filesystem")


@dataclass
class FileEntry:
    """Indexed file entry."""

    path: str
    name: str
    extension: str
    size: int
    modified: datetime
    is_dir: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "extension": self.extension,
            "size": self.size,
            "modified": self.modified.isoformat(),
            "is_dir": self.is_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileEntry:
        return cls(
            path=data["path"],
            name=data["name"],
            extension=data["extension"],
            size=data["size"],
            modified=datetime.fromisoformat(data["modified"]),
            is_dir=data["is_dir"],
        )


class FilesystemIntegration(BaseIntegration):
    """
    Filesystem integration for indexing and searching local files.

    Provides tools for:
    - Indexing directories
    - Searching files by name/pattern
    - Searching file contents
    - Reading files
    """

    name = "filesystem"
    display_name = "Local Filesystem"
    auth_type = AuthType.NONE

    def __init__(self, data_dir: Path | None = None):
        super().__init__()
        self._data_dir = data_dir or Path.home() / ".animus" / "integrations"
        self._index: dict[str, FileEntry] = {}
        self._indexed_paths: list[str] = []
        self._exclude_patterns: list[str] = [
            "*.pyc",
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
        ]

    async def connect(self, credentials: dict[str, Any]) -> bool:
        """
        Connect filesystem integration.

        Credentials:
            paths: List of paths to index
            exclude_patterns: Patterns to exclude (optional)
        """
        paths = credentials.get("paths", [])
        if exclude := credentials.get("exclude_patterns"):
            self._exclude_patterns = exclude

        self._indexed_paths = paths
        self._load_index()
        self._set_connected()
        logger.info(
            f"Filesystem integration connected, {len(self._indexed_paths)} paths configured"
        )
        return True

    async def disconnect(self) -> bool:
        """Disconnect and clear index."""
        self._index.clear()
        self._indexed_paths.clear()
        self._set_disconnected()
        return True

    async def verify(self) -> bool:
        """Verify connection - always valid for filesystem."""
        return self.is_connected

    def get_tools(self) -> list[Tool]:
        """Get filesystem tools."""
        return [
            Tool(
                name="fs_index",
                description="Index a directory to make files searchable",
                parameters={
                    "path": {
                        "type": "string",
                        "description": "Directory path to index",
                        "required": True,
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Index subdirectories (default: true)",
                        "required": False,
                    },
                },
                handler=self._tool_index,
            ),
            Tool(
                name="fs_search",
                description="Search indexed files by name or glob pattern",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "Search query or glob pattern (e.g., '*.py', 'config*')",
                        "required": True,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 20)",
                        "required": False,
                    },
                },
                handler=self._tool_search,
            ),
            Tool(
                name="fs_search_content",
                description="Search file contents using regex pattern",
                parameters={
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                        "required": True,
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g., '*.py')",
                        "required": False,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 20)",
                        "required": False,
                    },
                },
                handler=self._tool_search_content,
            ),
            Tool(
                name="fs_read",
                description="Read a file's contents",
                parameters={
                    "path": {
                        "type": "string",
                        "description": "File path to read",
                        "required": True,
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum lines to return (default: 100)",
                        "required": False,
                    },
                },
                handler=self._tool_read,
            ),
        ]

    def _should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded."""
        for pattern in self._exclude_patterns:
            if fnmatch.fnmatch(path.name, pattern):
                return True
            if fnmatch.fnmatch(str(path), f"*/{pattern}/*"):
                return True
        return False

    def _index_directory(self, path: Path, recursive: bool = True) -> int:
        """Index a directory and return count of files indexed."""
        count = 0
        path = Path(path).expanduser().resolve()

        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            return 0

        try:
            for entry in path.iterdir():
                if self._should_exclude(entry):
                    continue

                try:
                    stat = entry.stat()
                    file_entry = FileEntry(
                        path=str(entry),
                        name=entry.name,
                        extension=entry.suffix.lower(),
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime),
                        is_dir=entry.is_dir(),
                    )
                    self._index[str(entry)] = file_entry
                    count += 1

                    if entry.is_dir() and recursive:
                        count += self._index_directory(entry, recursive)

                except (PermissionError, OSError) as e:
                    logger.debug(f"Cannot access {entry}: {e}")

        except PermissionError as e:
            logger.warning(f"Cannot access directory {path}: {e}")

        return count

    def _save_index(self) -> None:
        """Save index to disk."""
        index_path = self._data_dir / "filesystem_index.json"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "indexed_paths": self._indexed_paths,
            "exclude_patterns": self._exclude_patterns,
            "entries": {k: v.to_dict() for k, v in self._index.items()},
        }
        with open(index_path, "w") as f:
            json.dump(data, f)
        logger.debug(f"Saved index with {len(self._index)} entries")

    def _load_index(self) -> None:
        """Load index from disk."""
        index_path = self._data_dir / "filesystem_index.json"
        if not index_path.exists():
            return

        try:
            with open(index_path) as f:
                data = json.load(f)

            self._indexed_paths = data.get("indexed_paths", [])
            self._exclude_patterns = data.get(
                "exclude_patterns",
                ["*.pyc", "__pycache__", ".git", "node_modules", ".venv"],
            )
            self._index = {k: FileEntry.from_dict(v) for k, v in data.get("entries", {}).items()}
            logger.debug(f"Loaded index with {len(self._index)} entries")
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error(f"Failed to load index: {e}")

    async def _tool_index(self, path: str, recursive: bool = True) -> ToolResult:
        """Index a directory."""
        expanded = Path(path).expanduser().resolve()
        if not expanded.exists():
            return ToolResult(
                tool_name="fs_tool",
                success=False,
                output=None,
                error=f"Path does not exist: {path}",
            )
        if not expanded.is_dir():
            return ToolResult(
                tool_name="fs_tool",
                success=False,
                output=None,
                error=f"Path is not a directory: {path}",
            )

        count = self._index_directory(expanded, recursive)

        if str(expanded) not in self._indexed_paths:
            self._indexed_paths.append(str(expanded))

        self._save_index()

        return ToolResult(
            tool_name="fs_tool",
            success=True,
            output={
                "indexed_path": str(expanded),
                "files_indexed": count,
                "total_indexed": len(self._index),
            },
        )

    async def _tool_search(self, query: str, limit: int = 20) -> ToolResult:
        """Search files by name or pattern."""
        results = []

        for entry in self._index.values():
            if entry.is_dir:
                continue

            # Match by glob pattern or substring
            if (
                fnmatch.fnmatch(entry.name.lower(), query.lower())
                or query.lower() in entry.name.lower()
            ):
                results.append(entry.to_dict())
                if len(results) >= limit:
                    break

        return ToolResult(
            tool_name="fs_tool",
            success=True,
            output={
                "query": query,
                "count": len(results),
                "results": results,
            },
        )

    async def _tool_search_content(
        self, pattern: str, file_pattern: str | None = None, limit: int = 20
    ) -> ToolResult:
        """Search file contents."""
        results = []

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult(
                tool_name="fs_tool", success=False, output=None, error=f"Invalid regex pattern: {e}"
            )

        for entry in self._index.values():
            if entry.is_dir:
                continue

            # Filter by file pattern if specified
            if file_pattern and not fnmatch.fnmatch(entry.name, file_pattern):
                continue

            # Skip binary files and large files
            if entry.extension in [".exe", ".dll", ".so", ".bin", ".zip", ".tar", ".gz"]:
                continue
            if entry.size > 1_000_000:  # Skip files > 1MB
                continue

            try:
                with open(entry.path, errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append(
                                {
                                    "file": entry.path,
                                    "line": line_num,
                                    "content": line.strip()[:200],
                                }
                            )
                            if len(results) >= limit:
                                break
                if len(results) >= limit:
                    break
            except (OSError, UnicodeDecodeError):
                continue

        return ToolResult(
            tool_name="fs_tool",
            success=True,
            output={
                "pattern": pattern,
                "count": len(results),
                "results": results,
            },
        )

    async def _tool_read(self, path: str, max_lines: int = 100) -> ToolResult:
        """Read a file."""
        file_path = Path(path).expanduser().resolve()

        if not file_path.exists():
            return ToolResult(
                tool_name="fs_tool", success=False, output=None, error=f"File not found: {path}"
            )
        if file_path.is_dir():
            return ToolResult(
                tool_name="fs_tool",
                success=False,
                output=None,
                error=f"Path is a directory: {path}",
            )

        try:
            with open(file_path, errors="ignore") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip())

            return ToolResult(
                tool_name="fs_read",
                success=True,
                output={
                    "path": str(file_path),
                    "lines": len(lines),
                    "truncated": len(lines) >= max_lines,
                    "content": "\n".join(lines),
                },
            )
        except OSError as e:
            return ToolResult(
                tool_name="fs_tool", success=False, output=None, error=f"Cannot read file: {e}"
            )
