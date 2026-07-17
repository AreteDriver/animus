"""CodeIndexReindexer — file-watched incremental codebase indexing for the daemon.

Watches arbitrary codebase directories via polling-based FileWatchHandler and
triggers ``ingest_codebase()`` with ``skip_existing=True`` on change.

Usage::

    from animus.daemon.code_watch import CodeIndexReindexer
    from animus.memory import MemoryLayer

    memory = MemoryLayer(Path("~/.animus"))
    watcher = CodeIndexReindexer(memory)
    watcher.add_codebase(Path("~/projects/myapp"), tags=["myapp"])

    # In daemon tick loop:
    for handler in watcher.handlers:
        for event in handler.scan():
            watcher.on_file_event(event)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from animus.daemon.events import FileWatchEvent, FileWatchHandler
from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.memory import MemoryLayer

logger = get_logger("daemon.code_watch")

# Minimum seconds between reindexes for the same root (debounce).
_DEBOUNCE_SECONDS = 2.0


@dataclass
class _WatchConfig:
    """Internal record of a watched codebase."""

    root: Path
    tags: list[str]
    globs: list[str]
    exclude: list[str]
    handler: FileWatchHandler = field(init=False)
    last_reindex: float = 0.0

    def __post_init__(self) -> None:
        self.handler = FileWatchHandler(
            watch_path=self.root,
            patterns=self.globs,
        )


class CodeIndexReindexer:
    """Manages watched codebases and triggers incremental reindex on change."""

    def __init__(self, memory: MemoryLayer | None = None) -> None:
        self.memory = memory
        self._watches: dict[str, _WatchConfig] = {}

    @property
    def handlers(self) -> list[FileWatchHandler]:
        """Return all active file-watch handlers for scanning."""
        return [w.handler for w in self._watches.values()]

    def add_codebase(
        self,
        root: Path,
        *,
        tags: list[str] | None = None,
        globs: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        """Register a codebase for file-watched incremental reindex.

        Args:
            root: Absolute path to the codebase root. Must be a directory.
            tags: Tags applied to every chunk (e.g. ``["projectname"]``).
            globs: Filename patterns to watch (default: ``["*.py", "*.md"]``).
            exclude: Patterns to skip (default: common test/cache dirs).
        """
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Not a directory: {root}")

        self._watches[str(resolved)] = _WatchConfig(
            root=resolved,
            tags=tags or [],
            globs=globs or ["*.py", "*.md"],
            exclude=exclude or ["*/test*", "*/__pycache__/*", "*/node_modules/*"],
        )
        logger.info("Watching codebase for reindex: %s", resolved)

    def remove_codebase(self, root: Path) -> bool:
        """Remove a watched codebase. Returns True if it existed."""
        key = str(Path(root).expanduser().resolve())
        existed = key in self._watches
        if existed:
            del self._watches[key]
            logger.info("Stopped watching codebase: %s", key)
        return existed

    def list_watched(self) -> list[dict]:
        """Return summary of all watched codebases."""
        return [
            {
                "root": str(w.root),
                "tags": w.tags,
                "globs": w.globs,
                "exclude": w.exclude,
            }
            for w in self._watches.values()
        ]

    def on_file_event(self, event: FileWatchEvent) -> dict[str, str]:
        """Handle a single file watch event.

        Maps the changed file to its watched codebase root and triggers
        incremental reindex if the debounce window has elapsed.

        Returns a result dict with ``action`` and optional ``error`` keys.
        """
        path = Path(event.path).expanduser().resolve()

        # Find which watched root this file belongs to
        matched: _WatchConfig | None = None
        for w in self._watches.values():
            # Check if file is under the watched root
            try:
                path.relative_to(w.root)
                matched = w
                break
            except ValueError:
                continue

        if matched is None:
            return {"action": "ignored", "reason": "no_matching_root"}

        # Debounce
        now = time.time()
        if now - matched.last_reindex < _DEBOUNCE_SECONDS:
            return {"action": "debounced", "root": str(matched.root)}

        matched.last_reindex = now

        # Ensure memory layer exists
        if self.memory is None:
            from animus.config import AnimusConfig
            from animus.memory import MemoryLayer

            cfg = AnimusConfig.load()
            self.memory = MemoryLayer(cfg.data_dir, backend=cfg.memory.backend)

        # Trigger incremental reindex
        try:
            from animus.workflows.code_ingest import ingest_codebase

            result = ingest_codebase(
                matched.root,
                memory=self.memory,
                tags=matched.tags,
                globs=matched.globs,
                exclude=matched.exclude,
                skip_existing=True,
                write_manifest=True,
            )
            logger.info(
                "Reindexed %s: stored=%d skipped=%d errors=%d",
                matched.root,
                result.stored_count,
                result.skipped_count,
                len(result.errors),
            )
            return {
                "action": "reindexed",
                "root": str(matched.root),
                "stored": str(result.stored_count),
                "skipped": str(result.skipped_count),
            }
        except Exception as exc:
            logger.error("Reindex failed for %s: %s", matched.root, exc)
            return {"action": "error", "root": str(matched.root), "error": str(exc)}
