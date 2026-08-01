"""Event sources and handlers for the daemon event loop.

Supports four event types:
- FileWatchEvent: React to filesystem changes
- TimerEvent: Periodic check triggers
- WebhookEvent: External HTTP callbacks
- MCPEvent: Messages from MCP servers/tools
"""

from __future__ import annotations

import fnmatch
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from animus.logging import get_logger

logger = get_logger("daemon.events")


class EventType(Enum):
    """Categories of daemon events."""

    FILE_WATCH = "file_watch"
    TIMER = "timer"
    WEBHOOK = "webhook"
    MCP = "mcp"
    SCHEDULED = "scheduled"
    SIGNAL = "signal"
    SHUTDOWN = "shutdown"


class EventPriority(Enum):
    """Event processing priority."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class DaemonEvent:
    """Base event for the daemon event loop."""

    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    source: str = "daemon"


@dataclass
class FileWatchEvent(DaemonEvent):
    """Filesystem change notification."""

    path: str = ""
    change_type: str = ""  # created, modified, deleted, moved
    file_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.event_type:
            self.event_type = EventType.FILE_WATCH


@dataclass
class TimerEvent(DaemonEvent):
    """Periodic timer tick."""

    tick_number: int = 0
    interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.event_type:
            self.event_type = EventType.TIMER


@dataclass
class WebhookEvent(DaemonEvent):
    """External HTTP callback."""

    endpoint: str = ""
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    body: str = ""

    def __post_init__(self) -> None:
        if not self.event_type:
            self.event_type = EventType.WEBHOOK


@dataclass
class MCPEvent(DaemonEvent):
    """Event from MCP server/tool."""

    tool_name: str = ""
    server_name: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.event_type:
            self.event_type = EventType.MCP


@dataclass
class ScheduledEvent(DaemonEvent):
    """Task scheduler trigger."""

    task_id: str = ""
    task_description: str = ""
    run_count: int = 0

    def __post_init__(self) -> None:
        if not self.event_type:
            self.event_type = EventType.SCHEDULED


@dataclass
class SignalEvent(DaemonEvent):
    """OS signal event."""

    signal_number: int = 0
    signal_name: str = ""

    def __post_init__(self) -> None:
        if not self.event_type:
            self.event_type = EventType.SIGNAL


class EventHandler(ABC):
    """Base class for event handlers."""

    @abstractmethod
    def can_handle(self, event: DaemonEvent) -> bool:
        """Check if this handler can process the event."""
        pass

    @abstractmethod
    async def handle(self, event: DaemonEvent) -> dict[str, Any]:
        """Process the event. Returns result dict."""
        pass


class FileWatchHandler(EventHandler):
    """Handler for file watch events."""

    def __init__(self, watch_path: str | Path, patterns: list[str] | None = None):
        self.watch_path = Path(watch_path)
        self.patterns = patterns or ["*"]
        self._last_seen: dict[str, float] = {}
        self._callbacks: list[Callable[[FileWatchEvent], None]] = []

    def add_callback(self, callback: Callable[[FileWatchEvent], None]) -> None:
        self._callbacks.append(callback)

    def can_handle(self, event: DaemonEvent) -> bool:
        return event.event_type == EventType.FILE_WATCH

    async def handle(self, event: DaemonEvent) -> dict[str, Any]:
        if not isinstance(event, FileWatchEvent):
            return {"handled": False, "reason": "wrong_type"}

        path = Path(event.path)
        if not any(fnmatch.fnmatch(path.name, p) for p in self.patterns):
            return {"handled": False, "reason": "pattern_mismatch"}

        result = {
            "handled": True,
            "path": event.path,
            "change_type": event.change_type,
            "callbacks_fired": len(self._callbacks),
        }

        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"File watch callback error: {e}")

        return result

    def scan(self) -> list[FileWatchEvent]:
        """Scan watch path and return changes since last scan."""
        if not self.watch_path.exists():
            return []

        events = []
        current_files = set()

        for path in self.watch_path.rglob("*"):
            if not path.is_file():
                continue
            if not any(fnmatch.fnmatch(path.name, p) for p in self.patterns):
                continue

            rel_path = str(path.relative_to(self.watch_path))
            current_files.add(rel_path)
            mtime = path.stat().st_mtime

            if rel_path in self._last_seen:
                if mtime > self._last_seen[rel_path]:
                    events.append(
                        FileWatchEvent(
                            event_type=EventType.FILE_WATCH,
                            path=str(path),
                            change_type="modified",
                        )
                    )
            else:
                events.append(
                    FileWatchEvent(
                        event_type=EventType.FILE_WATCH,
                        path=str(path),
                        change_type="created",
                    )
                )

            self._last_seen[rel_path] = mtime

        # Detect deletions
        for old_path in list(self._last_seen.keys()):
            if old_path not in current_files:
                events.append(
                    FileWatchEvent(
                        event_type=EventType.FILE_WATCH,
                        path=str(self.watch_path / old_path),
                        change_type="deleted",
                    )
                )
                del self._last_seen[old_path]

        return events


class TimerHandler(EventHandler):
    """Handler for periodic timer events."""

    def __init__(self, interval_seconds: float = 60.0):
        self.interval_seconds = interval_seconds
        self._callbacks: list[Callable[[TimerEvent], None]] = []
        self._tick_count = 0

    def add_callback(self, callback: Callable[[TimerEvent], None]) -> None:
        self._callbacks.append(callback)

    def can_handle(self, event: DaemonEvent) -> bool:
        return event.event_type == EventType.TIMER

    async def handle(self, event: DaemonEvent) -> dict[str, Any]:
        if not isinstance(event, TimerEvent):
            return {"handled": False, "reason": "wrong_type"}

        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Timer callback error: {e}")

        return {
            "handled": True,
            "tick": event.tick_number,
            "callbacks_fired": len(self._callbacks),
        }

    def create_tick(self) -> TimerEvent:
        self._tick_count += 1
        return TimerEvent(
            event_type=EventType.TIMER,
            tick_number=self._tick_count,
            interval_seconds=self.interval_seconds,
        )


class WebhookHandler(EventHandler):
    """Handler for webhook events."""

    def __init__(self, allowed_endpoints: list[str] | None = None):
        self.allowed_endpoints = set(allowed_endpoints or [])
        self._callbacks: dict[str, list[Callable[[WebhookEvent], None]]] = {}

    def add_callback(self, endpoint: str, callback: Callable[[WebhookEvent], None]) -> None:
        if endpoint not in self._callbacks:
            self._callbacks[endpoint] = []
        self._callbacks[endpoint].append(callback)

    def can_handle(self, event: DaemonEvent) -> bool:
        return event.event_type == EventType.WEBHOOK

    async def handle(self, event: DaemonEvent) -> dict[str, Any]:
        if not isinstance(event, WebhookEvent):
            return {"handled": False, "reason": "wrong_type"}

        if self.allowed_endpoints and event.endpoint not in self.allowed_endpoints:
            return {"handled": False, "reason": "endpoint_not_allowed"}

        callbacks = self._callbacks.get(event.endpoint, [])
        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Webhook callback error: {e}")

        return {
            "handled": True,
            "endpoint": event.endpoint,
            "callbacks_fired": len(callbacks),
        }


class MCPHandler(EventHandler):
    """Handler for MCP tool/server events."""

    def __init__(self):
        self._callbacks: dict[str, list[Callable[[MCPEvent], None]]] = {}

    def add_callback(self, tool_name: str, callback: Callable[[MCPEvent], None]) -> None:
        if tool_name not in self._callbacks:
            self._callbacks[tool_name] = []
        self._callbacks[tool_name].append(callback)

    def can_handle(self, event: DaemonEvent) -> bool:
        return event.event_type == EventType.MCP

    async def handle(self, event: DaemonEvent) -> dict[str, Any]:
        if not isinstance(event, MCPEvent):
            return {"handled": False, "reason": "wrong_type"}

        callbacks = self._callbacks.get(event.tool_name, [])
        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"MCP callback error: {e}")

        return {
            "handled": True,
            "tool": event.tool_name,
            "server": event.server_name,
            "callbacks_fired": len(callbacks),
        }
