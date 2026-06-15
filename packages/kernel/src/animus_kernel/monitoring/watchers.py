"""Proactive monitoring with file and system watchers.

Enables Clawdbot-style proactive assistance by monitoring:
- File system changes (new files, modifications, deletions)
- Directory changes
- Log files for patterns
- System resource usage

When events occur, triggers are sent to the notification system or
can invoke workflows automatically.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WatchEventType(str, Enum):
    """Types of watch events."""

    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"
    DIR_CREATED = "dir_created"
    DIR_MODIFIED = "dir_modified"
    DIR_DELETED = "dir_deleted"
    DIR_MOVED = "dir_moved"
    PATTERN_MATCH = "pattern_match"
    THRESHOLD_EXCEEDED = "threshold_exceeded"


@dataclass
class WatchEvent:
    """A watch event that occurred."""

    event_type: WatchEventType
    path: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    old_path: str | None = None  # For move events
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "path": self.path,
            "timestamp": self.timestamp.isoformat(),
            "old_path": self.old_path,
            "details": self.details,
        }


# Type alias for event handlers
EventHandler = Callable[[WatchEvent], None]


class BaseWatcher(ABC):
    """Abstract base class for watchers."""

    def __init__(self, name: str = "watcher"):
        """Initialize watcher.

        Args:
            name: Watcher name for identification.
        """
        self.name = name
        self._running = False
        self._handlers: list[EventHandler] = []
        self._thread: threading.Thread | None = None

    def add_handler(self, handler: EventHandler) -> None:
        """Add an event handler.

        Args:
            handler: Function to call when events occur.
        """
        self._handlers.append(handler)

    def remove_handler(self, handler: EventHandler) -> bool:
        """Remove an event handler.

        Args:
            handler: Handler to remove.

        Returns:
            True if removed, False if not found.
        """
        try:
            self._handlers.remove(handler)
            return True
        except ValueError:
            return False

    def _emit(self, event: WatchEvent) -> None:
        """Emit an event to all handlers.

        Args:
            event: Event to emit.
        """
        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.exception(f"Handler error: {e}")

    @abstractmethod
    def start(self) -> None:
        """Start the watcher."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the watcher."""
        pass

    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running


class FileWatcher(BaseWatcher):
    """Watch file system for changes.

    Uses polling to detect changes. For more efficient watching,
    consider using watchdog library.

    Usage:
        watcher = FileWatcher("/path/to/watch")
        watcher.add_handler(lambda e: print(f"Event: {e}"))
        watcher.start()
    """

    def __init__(
        self,
        path: str | Path,
        recursive: bool = True,
        patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        poll_interval: float = 1.0,
        name: str = "file_watcher",
    ):
        """Initialize file watcher.

        Args:
            path: Path to watch.
            recursive: Watch subdirectories.
            patterns: Glob patterns to include (e.g., ["*.py", "*.txt"]).
            ignore_patterns: Glob patterns to ignore.
            poll_interval: Seconds between checks.
            name: Watcher name.
        """
        super().__init__(name)
        self.path = Path(path)
        self.recursive = recursive
        self.patterns = patterns or ["*"]
        self.ignore_patterns = ignore_patterns or [
            "*.pyc",
            "__pycache__",
            ".git",
            ".DS_Store",
        ]
        self.poll_interval = poll_interval

        # Track file states
        self._file_states: dict[str, tuple[float, int]] = {}  # path -> (mtime, size)
        self._dir_states: set[str] = set()

    def _matches_patterns(self, path: Path) -> bool:
        """Check if path matches include patterns and not ignore patterns.

        Args:
            path: Path to check.

        Returns:
            True if should be watched.
        """
        str_path = str(path)

        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if path.match(pattern) or pattern in str_path:
                return False

        # Check include patterns
        for pattern in self.patterns:
            if path.match(pattern):
                return True

        return False

    def _scan_directory(self) -> tuple[dict[str, tuple[float, int]], set[str]]:
        """Scan directory for current state.

        Returns:
            Tuple of (file_states, dir_states).
        """
        files: dict[str, tuple[float, int]] = {}
        dirs: set[str] = set()

        if not self.path.exists():
            return files, dirs

        if self.recursive:
            items = self.path.rglob("*")
        else:
            items = self.path.iterdir()

        for item in items:
            if not self._matches_patterns(item):
                continue

            str_path = str(item)

            try:
                if item.is_file():
                    stat = item.stat()
                    files[str_path] = (stat.st_mtime, stat.st_size)
                elif item.is_dir():
                    dirs.add(str_path)
            except (OSError, PermissionError):
                continue

        return files, dirs

    def _check_changes(self) -> None:
        """Check for changes since last scan."""
        new_files, new_dirs = self._scan_directory()

        # Check for new and modified files
        for path, (mtime, size) in new_files.items():
            if path not in self._file_states:
                self._emit(
                    WatchEvent(
                        event_type=WatchEventType.FILE_CREATED,
                        path=path,
                        details={"size": size},
                    )
                )
            elif self._file_states[path] != (mtime, size):
                self._emit(
                    WatchEvent(
                        event_type=WatchEventType.FILE_MODIFIED,
                        path=path,
                        details={"size": size, "old_size": self._file_states[path][1]},
                    )
                )

        # Check for deleted files
        for path in self._file_states:
            if path not in new_files:
                self._emit(
                    WatchEvent(
                        event_type=WatchEventType.FILE_DELETED,
                        path=path,
                    )
                )

        # Check for new directories
        for path in new_dirs:
            if path not in self._dir_states:
                self._emit(
                    WatchEvent(
                        event_type=WatchEventType.DIR_CREATED,
                        path=path,
                    )
                )

        # Check for deleted directories
        for path in self._dir_states:
            if path not in new_dirs:
                self._emit(
                    WatchEvent(
                        event_type=WatchEventType.DIR_DELETED,
                        path=path,
                    )
                )

        # Update state
        self._file_states = new_files
        self._dir_states = new_dirs

    def _watch_loop(self) -> None:
        """Main watch loop."""
        # Initial scan
        self._file_states, self._dir_states = self._scan_directory()
        logger.info(
            f"FileWatcher started: {self.path} "
            f"({len(self._file_states)} files, {len(self._dir_states)} dirs)"
        )

        while self._running:
            try:
                self._check_changes()
            except Exception as e:
                logger.exception(f"Watch error: {e}")

            time.sleep(self.poll_interval)

    def start(self) -> None:
        """Start watching."""
        if self._running:
            return

        if not self.path.exists():
            logger.warning(f"Watch path does not exist: {self.path}")

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info(f"FileWatcher stopped: {self.path}")


class LogWatcher(BaseWatcher):
    r"""Watch log files for patterns.

    Monitors log files and triggers events when patterns are matched.
    Useful for alerting on errors, warnings, or specific events.

    Usage:
        watcher = LogWatcher("/var/log/app.log")
        watcher.add_pattern(r"ERROR", "error_detected")
        watcher.add_pattern(r"user (\w+) logged in", "user_login")
        watcher.add_handler(lambda e: print(f"Match: {e}"))
        watcher.start()
    """

    def __init__(
        self,
        path: str | Path,
        patterns: dict[str, str] | None = None,
        poll_interval: float = 1.0,
        name: str = "log_watcher",
    ):
        """Initialize log watcher.

        Args:
            path: Log file path to watch.
            patterns: Dict of regex pattern -> event name.
            poll_interval: Seconds between checks.
            name: Watcher name.
        """
        super().__init__(name)
        self.path = Path(path)
        self.poll_interval = poll_interval
        self._patterns: list[tuple[re.Pattern, str]] = []
        self._file_position = 0

        if patterns:
            for pattern, event_name in patterns.items():
                self.add_pattern(pattern, event_name)

    def add_pattern(self, pattern: str, event_name: str) -> None:
        """Add a pattern to watch for.

        Args:
            pattern: Regex pattern to match.
            event_name: Name for events when pattern matches.
        """
        compiled = re.compile(pattern)
        self._patterns.append((compiled, event_name))

    def _check_new_lines(self) -> None:
        """Check for new lines in log file."""
        if not self.path.exists():
            return

        try:
            with open(self.path) as f:
                # Check if file was truncated/rotated
                f.seek(0, 2)  # Go to end
                current_size = f.tell()

                if current_size < self._file_position:
                    # File was truncated, start from beginning
                    self._file_position = 0

                f.seek(self._file_position)
                new_lines = f.readlines()
                self._file_position = f.tell()

                for line in new_lines:
                    self._check_line(line)

        except (OSError, PermissionError) as e:
            logger.debug(f"Cannot read log file: {e}")

    def _check_line(self, line: str) -> None:
        """Check a line against all patterns.

        Args:
            line: Log line to check.
        """
        for pattern, event_name in self._patterns:
            match = pattern.search(line)
            if match:
                self._emit(
                    WatchEvent(
                        event_type=WatchEventType.PATTERN_MATCH,
                        path=str(self.path),
                        details={
                            "event_name": event_name,
                            "line": line.strip(),
                            "groups": match.groups(),
                            "match": match.group(0),
                        },
                    )
                )

    def _watch_loop(self) -> None:
        """Main watch loop."""
        # Start at end of file
        if self.path.exists():
            self._file_position = self.path.stat().st_size

        logger.info(f"LogWatcher started: {self.path}")

        while self._running:
            try:
                self._check_new_lines()
            except Exception as e:
                logger.exception(f"Log watch error: {e}")

            time.sleep(self.poll_interval)

    def start(self) -> None:
        """Start watching."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info(f"LogWatcher stopped: {self.path}")


class ResourceWatcher(BaseWatcher):
    """Watch system resources for threshold violations.

    Monitors CPU, memory, and disk usage and triggers events
    when thresholds are exceeded.

    Usage:
        watcher = ResourceWatcher()
        watcher.set_threshold("cpu", 80)  # 80% CPU
        watcher.set_threshold("memory", 90)  # 90% memory
        watcher.set_threshold("disk", 95)  # 95% disk
        watcher.add_handler(lambda e: print(f"Alert: {e}"))
        watcher.start()
    """

    def __init__(
        self,
        poll_interval: float = 30.0,
        name: str = "resource_watcher",
    ):
        """Initialize resource watcher.

        Args:
            poll_interval: Seconds between checks.
            name: Watcher name.
        """
        super().__init__(name)
        self.poll_interval = poll_interval
        self._thresholds: dict[str, float] = {}
        self._last_alert: dict[str, float] = {}
        self._alert_cooldown = 300  # 5 minutes between repeat alerts

    def set_threshold(self, resource: str, threshold: float) -> None:
        """Set a threshold for a resource.

        Args:
            resource: Resource name (cpu, memory, disk).
            threshold: Percentage threshold (0-100).
        """
        self._thresholds[resource] = threshold

    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            # Use /proc/stat on Linux
            if os.path.exists("/proc/stat"):
                with open("/proc/stat") as f:
                    line = f.readline()
                    parts = line.split()
                    idle = int(parts[4])
                    total = sum(int(p) for p in parts[1:])

                    # Need two readings to calculate usage
                    time.sleep(0.1)

                    with open("/proc/stat") as f2:
                        line2 = f2.readline()
                        parts2 = line2.split()
                        idle2 = int(parts2[4])
                        total2 = sum(int(p) for p in parts2[1:])

                    idle_delta = idle2 - idle
                    total_delta = total2 - total

                    if total_delta > 0:
                        return 100 * (1 - idle_delta / total_delta)

            # Fallback: use os.getloadavg
            load = os.getloadavg()[0]
            cpus = os.cpu_count() or 1
            return min(100, (load / cpus) * 100)

        except Exception:
            return 0.0

    def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo") as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            meminfo[parts[0].rstrip(":")] = int(parts[1])

                    total = meminfo.get("MemTotal", 0)
                    available = meminfo.get("MemAvailable", 0)

                    if total > 0:
                        return 100 * (1 - available / total)

            return 0.0

        except Exception:
            return 0.0

    def _get_disk_usage(self, path: str = "/") -> float:
        """Get disk usage percentage."""
        try:
            stat = os.statvfs(path)
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bfree * stat.f_frsize
            if total > 0:
                return 100 * (1 - free / total)
            return 0.0
        except Exception:
            return 0.0

    def _check_resources(self) -> None:
        """Check all resources against thresholds."""
        now = time.time()

        checks = [
            ("cpu", self._get_cpu_usage()),
            ("memory", self._get_memory_usage()),
            ("disk", self._get_disk_usage()),
        ]

        for resource, usage in checks:
            if resource not in self._thresholds:
                continue

            threshold = self._thresholds[resource]
            if usage > threshold:
                # Check cooldown
                last = self._last_alert.get(resource, 0)
                if now - last < self._alert_cooldown:
                    continue

                self._last_alert[resource] = now
                self._emit(
                    WatchEvent(
                        event_type=WatchEventType.THRESHOLD_EXCEEDED,
                        path=resource,
                        details={
                            "resource": resource,
                            "usage": round(usage, 1),
                            "threshold": threshold,
                        },
                    )
                )

    def _watch_loop(self) -> None:
        """Main watch loop."""
        logger.info(f"ResourceWatcher started with thresholds: {self._thresholds}")

        while self._running:
            try:
                self._check_resources()
            except Exception as e:
                logger.exception(f"Resource watch error: {e}")

            time.sleep(self.poll_interval)

    def start(self) -> None:
        """Start watching."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("ResourceWatcher stopped")


class WatchManager:
    """Manages multiple watchers and routes events.

    Usage:
        manager = WatchManager()

        # Add file watcher
        manager.add_file_watch("/path/to/watch", patterns=["*.log"])

        # Add log watcher
        manager.add_log_watch("/var/log/app.log", {
            r"ERROR": "error",
            r"WARN": "warning",
        })

        # Add resource watcher
        manager.add_resource_watch(cpu=80, memory=90)

        # Set handler for all events
        manager.set_handler(lambda e: print(e))

        # Start all watchers
        manager.start_all()
    """

    def __init__(self):
        """Initialize watch manager."""
        self._watchers: list[BaseWatcher] = []
        self._handlers: list[EventHandler] = []

    def add_watcher(self, watcher: BaseWatcher) -> None:
        """Add a watcher.

        Args:
            watcher: Watcher to add.
        """
        # Connect handlers
        for handler in self._handlers:
            watcher.add_handler(handler)

        self._watchers.append(watcher)

    def add_handler(self, handler: EventHandler) -> None:
        """Add an event handler to all watchers.

        Args:
            handler: Handler to add.
        """
        self._handlers.append(handler)
        for watcher in self._watchers:
            watcher.add_handler(handler)

    def set_handler(self, handler: EventHandler) -> None:
        """Set the event handler (replaces existing).

        Args:
            handler: Handler to set.
        """
        self._handlers = [handler]
        for watcher in self._watchers:
            watcher._handlers = [handler]

    def add_file_watch(
        self,
        path: str | Path,
        recursive: bool = True,
        patterns: list[str] | None = None,
        **kwargs,
    ) -> FileWatcher:
        """Add a file watcher.

        Args:
            path: Path to watch.
            recursive: Watch subdirectories.
            patterns: Glob patterns to include.
            **kwargs: Additional FileWatcher arguments.

        Returns:
            Created FileWatcher.
        """
        watcher = FileWatcher(path, recursive, patterns, **kwargs)
        self.add_watcher(watcher)
        return watcher

    def add_log_watch(
        self,
        path: str | Path,
        patterns: dict[str, str] | None = None,
        **kwargs,
    ) -> LogWatcher:
        """Add a log watcher.

        Args:
            path: Log file path.
            patterns: Dict of regex -> event name.
            **kwargs: Additional LogWatcher arguments.

        Returns:
            Created LogWatcher.
        """
        watcher = LogWatcher(path, patterns, **kwargs)
        self.add_watcher(watcher)
        return watcher

    def add_resource_watch(
        self,
        cpu: float | None = None,
        memory: float | None = None,
        disk: float | None = None,
        **kwargs,
    ) -> ResourceWatcher:
        """Add a resource watcher.

        Args:
            cpu: CPU threshold percentage.
            memory: Memory threshold percentage.
            disk: Disk threshold percentage.
            **kwargs: Additional ResourceWatcher arguments.

        Returns:
            Created ResourceWatcher.
        """
        watcher = ResourceWatcher(**kwargs)
        if cpu is not None:
            watcher.set_threshold("cpu", cpu)
        if memory is not None:
            watcher.set_threshold("memory", memory)
        if disk is not None:
            watcher.set_threshold("disk", disk)
        self.add_watcher(watcher)
        return watcher

    def start_all(self) -> None:
        """Start all watchers."""
        for watcher in self._watchers:
            watcher.start()

    def stop_all(self) -> None:
        """Stop all watchers."""
        for watcher in self._watchers:
            watcher.stop()

    def list_watchers(self) -> list[dict]:
        """List all watchers.

        Returns:
            List of watcher info dicts.
        """
        return [
            {
                "name": w.name,
                "type": type(w).__name__,
                "running": w.is_running,
            }
            for w in self._watchers
        ]
