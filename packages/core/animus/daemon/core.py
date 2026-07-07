"""AnimusDaemon: Persistent local daemon with event loop and signal-safe shutdown.

Integrates ResourceGuard, SessionManager, TaskScheduler, and event handlers into
a single async event loop. Supports graceful shutdown, PID file locking, and
hooks into the Head and Meta-Thinker for intelligent background processing.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from animus.logging import get_logger

from .resource_guard import ResourceGuard, ResourceLimits
from .events import (
    DaemonEvent,
    EventPriority,
    EventType,
    FileWatchEvent,
    FileWatchHandler,
    MCPHandler,
    ScheduledEvent,
    SignalEvent,
    TimerEvent,
    TimerHandler,
    WebhookEvent,
    WebhookHandler,
)
from .resource_guard import ResourceGuard
from .scheduler import ScheduledTask, TaskScheduler
from .session_manager import SessionManager

logger = get_logger("daemon.core")


class DaemonState(Enum):
    """Lifecycle states of the daemon."""

    INIT = "init"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class DaemonConfig:
    """Configuration for daemon operation."""

    # Directories
    persistence_dir: str = "~/.animus/daemon"
    sessions_dir: str = "~/.animus/sessions"
    scheduler_dir: str = "~/.animus/scheduler"

    # Timing
    tick_interval: float = 1.0  # Main loop tick in seconds
    scheduler_check_interval: float = 15.0  # Check scheduled tasks every N seconds
    file_scan_interval: float = 5.0  # Scan files every N seconds
    session_save_interval: float = 300.0  # Save sessions every 5 minutes

    # Resource limits
    max_concurrent_tasks: int = 4
    max_tokens_per_minute: int = 100_000
    max_sessions: int = 10

    # Features
    enable_file_watch: bool = True
    enable_webhook: bool = False
    enable_mcp_events: bool = False
    enable_scheduler: bool = True

    # Meta-thinker integration
    meta_thinker_enabled: bool = True
    meta_thinker_check_interval: float = 60.0

    metadata: dict[str, Any] = field(default_factory=dict)


class AnimusDaemon:
    """Main daemon orchestrator for persistent background operation."""

    def __init__(self, config: DaemonConfig | None = None):
        self.config = config or DaemonConfig()
        self.state = DaemonState.INIT

        # Resolve paths
        self.persistence_dir = Path(self.config.persistence_dir).expanduser()
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file = self.persistence_dir / "daemon.pid"
        self.state_file = self.persistence_dir / "daemon.state"

        # Subsystems
        self.resource_guard = ResourceGuard(
            limits=ResourceLimits(
                max_concurrent_tasks=self.config.max_concurrent_tasks,
                max_tokens_per_minute=self.config.max_tokens_per_minute,
            )
        )
        self.session_manager = SessionManager(
            persistence_dir=self.config.sessions_dir,
            max_sessions=self.config.max_sessions,
        )
        self.scheduler = TaskScheduler(persistence_dir=self.config.scheduler_dir)

        # Event handlers
        self.timer_handler = TimerHandler(interval_seconds=self.config.tick_interval)
        self.file_handler = FileWatchHandler(
            watch_path=self.persistence_dir / "watch",
            patterns=["*.md", "*.json", "*.yaml"],
        )
        self.webhook_handler = WebhookHandler()
        self.mcp_handler = MCPHandler()

        # Event queue
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._tick_count = 0
        self._last_scheduler_check = 0.0
        self._last_file_scan = 0.0
        self._last_session_save = 0.0
        self._last_meta_check = 0.0

        # Task tracking
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()

        # Stats
        self.stats = {
            "events_processed": 0,
            "tasks_executed": 0,
            "sessions_warmed": 0,
            "start_time": None,
            "errors": 0,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────

    def _write_pid(self) -> None:
        self.pid_file.write_text(str(os.getpid()))

    def _read_pid(self) -> int | None:
        if self.pid_file.exists():
            try:
                return int(self.pid_file.read_text().strip())
            except ValueError:
                return None
        return None

    def _remove_pid(self) -> None:
        if self.pid_file.exists():
            self.pid_file.unlink()

    def is_running(self) -> bool:
        """Check if another daemon instance is running."""
        pid = self._read_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)  # Signal 0 is error check
            return True
        except ProcessLookupError:
            # Stale PID file
            self._remove_pid()
            return False
        except PermissionError:
            return True  # Can't check, assume running

    def _save_state(self) -> None:
        state = {
            "state": self.state.value,
            "tick_count": self._tick_count,
            "stats": self.stats,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            self.state_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.error(f"Failed to save daemon state: {e}")

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                pass
        return {}

    def _setup_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._on_signal(s)),
            )

        logger.info("Signal handlers registered")

    async def _on_signal(self, sig: int) -> None:
        """Handle OS signals."""
        sig_name = signal.Signals(sig).name
        logger.warning(f"Received signal {sig_name} ({sig})")
        await self._event_queue.put(
            SignalEvent(
                event_type=EventType.SIGNAL,
                signal_number=sig,
                signal_name=sig_name,
                priority=EventPriority.CRITICAL,
            )
        )
        self._shutdown_event.set()

    # ── Public API ──────────────────────────────────────────────────

    async def start(self) -> bool:
        """Start the daemon. Returns True if started, False if already running."""
        if self.is_running():
            logger.warning("Daemon is already running (PID file exists)")
            return False

        self.state = DaemonState.STARTING
        self._write_pid()
        self._running = True
        self.stats["start_time"] = time.time()

        # Ensure watch directory exists
        if self.config.enable_file_watch:
            self.file_handler.watch_path.mkdir(parents=True, exist_ok=True)

        self._setup_signal_handlers()
        atexit.register(self._cleanup_sync)

        logger.info(f"Daemon starting (PID: {os.getpid()})")
        self.state = DaemonState.RUNNING
        self._save_state()

        # Warm existing sessions
        self.stats["sessions_warmed"] = len(self.session_manager.list_sessions())

        # Prune old sessions
        self.session_manager.prune_old_sessions()

        return True

    async def run(self) -> None:
        """Main event loop. Blocks until shutdown."""
        if not self._running:
            await self.start()

        logger.info("Daemon event loop starting")
        try:
            while self._running and not self._shutdown_event.is_set():
                await self._tick()
                await asyncio.sleep(self.config.tick_interval)
        except asyncio.CancelledError:
            logger.info("Event loop cancelled")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Graceful shutdown."""
        if self.state in (DaemonState.STOPPED, DaemonState.SHUTTING_DOWN):
            return

        logger.info("Daemon shutting down...")
        self.state = DaemonState.SHUTTING_DOWN
        self._running = False
        self._shutdown_event.set()

        # Cancel active tasks
        for task_id, task in list(self._active_tasks.items()):
            if not task.done():
                logger.debug(f"Cancelling task: {task_id}")
                task.cancel()

        # Final state save
        self._save_state()
        self._remove_pid()

        self.state = DaemonState.STOPPED
        logger.info("Daemon stopped")

    def _cleanup_sync(self) -> None:
        """Synchronous cleanup for atexit handler."""
        if self.state != DaemonState.STOPPED:
            logger.info("Running synchronous cleanup")
            try:
                self._remove_pid()
            except Exception:
                pass

    # ── Event Loop ────────────────────────────────────────────────────

    async def _tick(self) -> None:
        """Process one tick of the event loop."""
        self._tick_count += 1
        now = time.time()

        # 1. Timer tick
        await self._event_queue.put(self.timer_handler.create_tick())

        # 2. Check scheduled tasks
        if self.config.enable_scheduler:
            if now - self._last_scheduler_check >= self.config.scheduler_check_interval:
                self._last_scheduler_check = now
                for task in self.scheduler.get_due_tasks():
                    await self._event_queue.put(
                        ScheduledEvent(
                            event_type=EventType.SCHEDULED,
                            task_id=task.task_id,
                            task_description=task.description,
                            priority=EventPriority.HIGH
                            if task.priority == "critical"
                            else EventPriority.NORMAL,
                        )
                    )

        # 3. File watch scan
        if self.config.enable_file_watch:
            if now - self._last_file_scan >= self.config.file_scan_interval:
                self._last_file_scan = now
                for event in self.file_handler.scan():
                    await self._event_queue.put(event)

        # 4. Periodic session prune
        if now - self._last_session_save >= self.config.session_save_interval:
            self._last_session_save = now
            self.session_manager.prune_old_sessions()
            logger.debug("Sessions pruned")

        # 5. Process events
        await self._process_events()

    async def _process_events(self) -> None:
        """Drain the event queue and dispatch to handlers."""
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if isinstance(event, SignalEvent):
                if event.signal_number in (signal.SIGINT, signal.SIGTERM):
                    await self.stop()
                    return

            # Dispatch to appropriate handler
            result = await self._dispatch_event(event)
            self.stats["events_processed"] += 1

            if event.event_type == EventType.SCHEDULED:
                await self._handle_scheduled_task(event)

    async def _dispatch_event(self, event: DaemonEvent) -> dict[str, Any]:
        """Route event to the correct handler."""
        handlers = [
            self.timer_handler,
            self.file_handler,
            self.webhook_handler,
            self.mcp_handler,
        ]

        for handler in handlers:
            if handler.can_handle(event):
                try:
                    return await handler.handle(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
                    self.stats["errors"] += 1
                    return {"handled": False, "error": str(e)}

        return {"handled": False, "reason": "no_handler"}

    async def _handle_scheduled_task(self, event: ScheduledEvent) -> None:
        """Execute a scheduled task."""
        task = self.scheduler.get_task(event.task_id)
        if not task:
            return

        # Check resources
        if not self.resource_guard.can_execute:
            logger.warning(f"Resources exhausted, skipping task: {event.task_id}")
            return

        granted, reason = self.resource_guard.acquire_task_slot(event.task_id)
        if not granted:
            logger.warning(f"No task slot for {event.task_id}: {reason}")
            return

        try:
            logger.info(f"Executing scheduled task: {event.task_id} — {event.task_description}")
            # Task execution would go here (hook into Head or direct processing)
            await self._execute_task_background(task)
            self.scheduler.mark_run(event.task_id)
            self.stats["tasks_executed"] += 1
            self.resource_guard.release_task_slot(event.task_id, tokens_used=0, success=True)
        except Exception as e:
            logger.error(f"Task execution error ({event.task_id}): {e}")
            self.stats["errors"] += 1
            self.resource_guard.release_task_slot(event.task_id, tokens_used=0, success=False)

    async def _execute_task_background(self, task: ScheduledTask) -> None:
        """Execute a task in the background.

        Placeholder for actual task execution. In production, this would
        invoke the Head or a specialized worker to process the task.
        """
        logger.debug(f"Background task: {task.description}")
        # NOTE: Head NL task execution and Meta-Thinker oversight are
        # planned integrations. Background tasks currently simulate work.
        await asyncio.sleep(0.1)  # Simulate work

    # ── External API ──────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Get current daemon status."""
        uptime = 0.0
        if self.stats["start_time"]:
            uptime = time.time() - self.stats["start_time"]

        return {
            "state": self.state.value,
            "running": self._running,
            "pid": self._read_pid(),
            "tick_count": self._tick_count,
            "uptime_seconds": round(uptime, 2),
            "events_processed": self.stats["events_processed"],
            "tasks_executed": self.stats["tasks_executed"],
            "sessions_warmed": self.stats["sessions_warmed"],
            "scheduler_tasks": self.scheduler.task_count,
            "errors": self.stats["errors"],
            "resource_usage": self.resource_guard.status,
        }

    def schedule_background_task(
        self,
        description: str,
        seconds: int = 3600,
        priority: str = "normal",
    ) -> ScheduledTask:
        """Schedule a background task via the daemon."""
        return self.scheduler.schedule_interval(
            description=description,
            seconds=seconds,
            priority=priority,
        )

    def add_file_watch_callback(
        self, callback: Any, patterns: list[str] | None = None
    ) -> None:
        """Add a callback for file watch events."""
        if patterns:
            self.file_handler.patterns = patterns
        self.file_handler.add_callback(callback)

    def add_webhook_endpoint(self, endpoint: str, callback: Any) -> None:
        """Register a webhook endpoint handler."""
        self.webhook_handler.add_callback(endpoint, callback)

    def add_mcp_callback(self, tool_name: str, callback: Any) -> None:
        """Register an MCP event handler."""
        self.mcp_handler.add_callback(tool_name, callback)