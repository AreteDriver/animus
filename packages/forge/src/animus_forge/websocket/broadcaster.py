"""Broadcaster for safely queuing updates from threads and broadcasting to WebSocket clients."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .messages import (
    ExecutionLogMessage,
    ExecutionMetricsMessage,
    ExecutionStatusMessage,
)

if TYPE_CHECKING:
    from .manager import ConnectionManager

logger = logging.getLogger(__name__)


class Broadcaster:
    """Queues execution updates from sync code and broadcasts to WebSocket clients.

    This class is thread-safe. The `on_*` callback methods can be called from
    any thread (e.g., ThreadPoolExecutor workers). Updates are queued and
    processed asynchronously in the event loop.
    """

    def __init__(self, manager: ConnectionManager):
        """Initialize the broadcaster.

        Args:
            manager: The ConnectionManager to broadcast messages through.
        """
        self._manager = manager
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the broadcast processing loop.

        Args:
            loop: The event loop to run in.
        """
        self._loop = loop
        self._running = True
        self._task = loop.create_task(self._process_queue())
        logger.info("Broadcaster started")

    async def stop(self) -> None:
        """Stop the broadcast processing loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # Graceful degradation: task cancellation is expected during stop()
        logger.info("Broadcaster stopped")

    async def _process_queue(self) -> None:
        """Process queued updates and broadcast to clients."""
        while self._running:
            try:
                update = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._handle_update(update)
            except Exception as e:
                logger.error(f"Error processing update: {e}")

    async def _handle_update(self, update: dict[str, Any]) -> None:
        """Handle a single update from the queue.

        Args:
            update: The update dictionary with type and data.
        """
        update_type = update.get("type")
        execution_id = update.get("execution_id")

        if not execution_id:
            logger.warning("Update missing execution_id")
            return

        message = None

        if update_type == "status":
            message = ExecutionStatusMessage(
                execution_id=execution_id,
                status=update.get("status", "unknown"),
                progress=update.get("progress", 0),
                current_step=update.get("current_step"),
                started_at=update.get("started_at"),
                completed_at=update.get("completed_at"),
                error=update.get("error"),
            )
        elif update_type == "log":
            message = ExecutionLogMessage(
                execution_id=execution_id,
                log=update.get("log", {}),
            )
        elif update_type == "metrics":
            message = ExecutionMetricsMessage(
                execution_id=execution_id,
                metrics=update.get("metrics", {}),
            )

        if message:
            sent = await self._manager.broadcast_to_execution(execution_id, message)
            if sent > 0:
                logger.debug(f"Broadcast {update_type} for {execution_id} to {sent} clients")

    def _enqueue(self, update: dict[str, Any]) -> None:
        """Thread-safe enqueue of an update.

        Args:
            update: The update dictionary.
        """
        if self._loop is None:
            logger.warning("Broadcaster not started, dropping update")
            return

        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, update)
        except RuntimeError:
            # Loop is closed
            logger.warning("Event loop closed, dropping update")

    # =========================================================================
    # Callback methods for ExecutionManager
    # =========================================================================

    def on_status_change(
        self,
        execution_id: str,
        status: str,
        progress: int = 0,
        current_step: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: str | None = None,
    ) -> None:
        """Callback for execution status changes.

        Thread-safe. Can be called from any thread.

        Args:
            execution_id: The execution ID.
            status: The new status.
            progress: Progress percentage (0-100).
            current_step: Current step ID.
            started_at: Started timestamp (ISO format).
            completed_at: Completed timestamp (ISO format).
            error: Error message if failed.
        """
        self._enqueue(
            {
                "type": "status",
                "execution_id": execution_id,
                "status": status,
                "progress": progress,
                "current_step": current_step,
                "started_at": started_at,
                "completed_at": completed_at,
                "error": error,
            }
        )

    def on_log(
        self,
        execution_id: str,
        level: str,
        message: str,
        step_id: str | None = None,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Callback for new log entries.

        Thread-safe. Can be called from any thread.

        Args:
            execution_id: The execution ID.
            level: Log level (debug, info, warning, error).
            message: Log message.
            step_id: Associated step ID.
            timestamp: Log timestamp (ISO format).
            metadata: Additional metadata.
        """
        self._enqueue(
            {
                "type": "log",
                "execution_id": execution_id,
                "log": {
                    "level": level,
                    "message": message,
                    "step_id": step_id,
                    "timestamp": timestamp,
                    "metadata": metadata,
                },
            }
        )

    def on_metrics(
        self,
        execution_id: str,
        total_tokens: int = 0,
        total_cost_cents: int = 0,
        duration_ms: int = 0,
        steps_completed: int = 0,
        steps_failed: int = 0,
    ) -> None:
        """Callback for metrics updates.

        Thread-safe. Can be called from any thread.

        Args:
            execution_id: The execution ID.
            total_tokens: Total tokens used.
            total_cost_cents: Total cost in cents.
            duration_ms: Total duration in milliseconds.
            steps_completed: Number of completed steps.
            steps_failed: Number of failed steps.
        """
        self._enqueue(
            {
                "type": "metrics",
                "execution_id": execution_id,
                "metrics": {
                    "total_tokens": total_tokens,
                    "total_cost_cents": total_cost_cents,
                    "duration_ms": duration_ms,
                    "steps_completed": steps_completed,
                    "steps_failed": steps_failed,
                },
            }
        )

    def create_execution_callback(self) -> Callable:
        """Create a callback function for ExecutionManager.

        Returns:
            A callback function that can be registered with ExecutionManager.
        """

        def callback(
            event_type: str,
            execution_id: str,
            **kwargs: Any,
        ) -> None:
            """Universal callback for execution events."""
            if event_type == "status":
                self.on_status_change(execution_id, **kwargs)
            elif event_type == "log":
                self.on_log(execution_id, **kwargs)
            elif event_type == "metrics":
                self.on_metrics(execution_id, **kwargs)

        return callback
