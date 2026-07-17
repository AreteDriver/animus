"""Animus Daemon: persistent local agent operating environment.

Implements P-20260706-005: Citizen Zero as 24/7 background agent.
Inspired by Hermes framework (140K stars, persistent local operation)
and RTX Spark (70B local inference on edge).

Key design:
- Event-driven: file changes, webhooks, MCP server events, timers
- Session persistence: warm state, context replay across restarts
- Resource governance: token budgets, concurrency limits, cooldowns
- Signal-safe: SIGTERM/SIGINT graceful shutdown with state flush
- Singleton: PID file prevents multiple daemon instances
"""

from animus.daemon.code_watch import CodeIndexReindexer
from animus.daemon.core import (
    AnimusDaemon,
    DaemonConfig,
    DaemonState,
)
from animus.daemon.events import (
    DaemonEvent,
    EventType,
    EventPriority,
    FileWatchEvent,
    TimerEvent,
    WebhookEvent,
    MCPEvent,
    ScheduledEvent,
    SignalEvent,
)
from animus.daemon.scheduler import (
    TaskScheduler,
    ScheduledTask,
    ScheduleType,
)
from animus.daemon.session_manager import (
    SessionManager,
    WarmSession,
)
from animus.daemon.resource_guard import (
    ResourceGuard,
)

__all__ = [
    "AnimusDaemon",
    "CodeIndexReindexer",
    "DaemonConfig",
    "DaemonState",
    "DaemonEvent",
    "EventType",
    "EventPriority",
    "FileWatchEvent",
    "TimerEvent",
    "WebhookEvent",
    "MCPEvent",
    "ScheduledEvent",
    "SignalEvent",
    "TaskScheduler",
    "ScheduledTask",
    "ScheduleType",
    "SessionManager",
    "WarmSession",
    "ResourceGuard",
]