"""Bulkhead pattern for resource isolation.

Limits concurrent access to a resource to prevent cascading failures.
Uses semaphores to enforce concurrency limits with optional queueing.

Example:
    bulkhead = Bulkhead(name="openai", max_concurrent=10, max_waiting=20)

    async with bulkhead:
        result = await call_openai_api()

    # Or as decorator:
    @bulkhead
    async def call_api():
        return await external_service()
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BulkheadFull(Exception):
    """Raised when bulkhead is at capacity."""

    def __init__(
        self,
        message: str = "Bulkhead is full",
        name: str = "",
        active: int = 0,
        waiting: int = 0,
    ):
        super().__init__(message)
        self.name = name
        self.active = active
        self.waiting = waiting


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead behavior."""

    max_concurrent: int = 10
    max_waiting: int = 20
    timeout: float = 30.0  # Max time to wait for a slot
    name: str = ""


@dataclass
class BulkheadStats:
    """Statistics for bulkhead usage."""

    name: str
    max_concurrent: int
    max_waiting: int
    active_count: int
    waiting_count: int
    total_acquired: int
    total_rejected: int
    total_timeouts: int


class Bulkhead:
    """Bulkhead for limiting concurrent access to a resource.

    Uses semaphores to enforce concurrency limits. When the semaphore
    is full, requests queue up to max_waiting. Beyond that, requests
    are rejected immediately.
    """

    def __init__(
        self,
        name: str = "",
        max_concurrent: int = 10,
        max_waiting: int = 20,
        timeout: float = 30.0,
    ):
        """Initialize bulkhead.

        Args:
            name: Identifier for this bulkhead
            max_concurrent: Maximum concurrent executions
            max_waiting: Maximum requests waiting for a slot
            timeout: Maximum time to wait for a slot
        """
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_waiting = max_waiting
        self.timeout = timeout

        # Semaphores
        self._semaphore = threading.Semaphore(max_concurrent)
        self._async_semaphore = asyncio.Semaphore(max_concurrent)

        # Tracking
        self._active_count = 0
        self._waiting_count = 0
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

        # Stats
        self._total_acquired = 0
        self._total_rejected = 0
        self._total_timeouts = 0

    def _check_waiting_capacity(self) -> bool:
        """Check if there's room in the waiting queue."""
        return self._waiting_count < self.max_waiting

    def acquire(self, timeout: float | None = None) -> bool:
        """Acquire a slot in the bulkhead (sync).

        Args:
            timeout: Override default timeout

        Returns:
            True if acquired

        Raises:
            BulkheadFull: If queue is at capacity
        """
        timeout = timeout if timeout is not None else self.timeout

        # Try non-blocking acquire first
        if self._semaphore.acquire(blocking=False):
            with self._lock:
                self._active_count += 1
                self._total_acquired += 1
            return True

        # Need to wait - check if waiting queue has capacity
        with self._lock:
            if not self._check_waiting_capacity():
                self._total_rejected += 1
                raise BulkheadFull(
                    f"Bulkhead '{self.name}' is full",
                    name=self.name,
                    active=self._active_count,
                    waiting=self._waiting_count,
                )
            self._waiting_count += 1

        try:
            acquired = self._semaphore.acquire(timeout=timeout)

            with self._lock:
                self._waiting_count -= 1
                if acquired:
                    self._active_count += 1
                    self._total_acquired += 1
                else:
                    self._total_timeouts += 1

            return acquired

        except Exception:
            with self._lock:
                self._waiting_count -= 1
            raise

    def release(self) -> None:
        """Release a slot in the bulkhead (sync)."""
        with self._lock:
            if self._active_count > 0:
                self._active_count -= 1
        self._semaphore.release()

    async def acquire_async(self, timeout: float | None = None) -> bool:
        """Acquire a slot in the bulkhead (async).

        Args:
            timeout: Override default timeout

        Returns:
            True if acquired

        Raises:
            BulkheadFull: If queue is at capacity
        """
        timeout = timeout if timeout is not None else self.timeout

        # Try non-blocking acquire first
        if not self._async_semaphore.locked():
            try:
                # Use wait_for with 0 timeout for non-blocking
                await asyncio.wait_for(
                    self._async_semaphore.acquire(),
                    timeout=0.0,
                )
                async with self._async_lock:
                    self._active_count += 1
                    self._total_acquired += 1
                return True
            except TimeoutError:
                pass  # Graceful degradation: semaphore not immediately available, try waiting below

        # Need to wait - check if waiting queue has capacity
        async with self._async_lock:
            if not self._check_waiting_capacity():
                self._total_rejected += 1
                raise BulkheadFull(
                    f"Bulkhead '{self.name}' is full",
                    name=self.name,
                    active=self._active_count,
                    waiting=self._waiting_count,
                )
            self._waiting_count += 1

        try:
            acquired = await asyncio.wait_for(
                self._async_semaphore.acquire(),
                timeout=timeout,
            )

            async with self._async_lock:
                self._waiting_count -= 1
                if acquired:
                    self._active_count += 1
                    self._total_acquired += 1

            return acquired

        except TimeoutError:
            async with self._async_lock:
                self._waiting_count -= 1
                self._total_timeouts += 1
            return False

        except Exception:
            async with self._async_lock:
                self._waiting_count -= 1
            raise

    async def release_async(self) -> None:
        """Release a slot in the bulkhead (async)."""
        async with self._async_lock:
            if self._active_count > 0:
                self._active_count -= 1
        self._async_semaphore.release()

    def __enter__(self) -> Bulkhead:
        """Context manager entry (sync)."""
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit (sync)."""
        self.release()

    async def __aenter__(self) -> Bulkhead:
        """Async context manager entry."""
        await self.acquire_async()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.release_async()

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use bulkhead as decorator."""
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self:
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                with self:
                    return func(*args, **kwargs)

            return sync_wrapper

    def get_stats(self) -> dict[str, Any]:
        """Get bulkhead statistics."""
        with self._lock:
            return {
                "name": self.name,
                "max_concurrent": self.max_concurrent,
                "max_waiting": self.max_waiting,
                "active_count": self._active_count,
                "waiting_count": self._waiting_count,
                "total_acquired": self._total_acquired,
                "total_rejected": self._total_rejected,
                "total_timeouts": self._total_timeouts,
            }


# Global bulkheads
_bulkheads: dict[str, Bulkhead] = {}
_bulkheads_lock = threading.Lock()


def get_bulkhead(
    name: str,
    max_concurrent: int = 10,
    max_waiting: int = 20,
    timeout: float = 30.0,
) -> Bulkhead:
    """Get or create a named bulkhead.

    Args:
        name: Unique identifier
        max_concurrent: Max concurrent (only used on creation)
        max_waiting: Max waiting (only used on creation)
        timeout: Default timeout (only used on creation)

    Returns:
        Bulkhead instance
    """
    with _bulkheads_lock:
        if name not in _bulkheads:
            _bulkheads[name] = Bulkhead(
                name=name,
                max_concurrent=max_concurrent,
                max_waiting=max_waiting,
                timeout=timeout,
            )
            logger.info(
                f"Created bulkhead '{name}' "
                f"(max_concurrent={max_concurrent}, max_waiting={max_waiting})"
            )
        return _bulkheads[name]


def get_all_bulkhead_stats() -> dict[str, dict[str, Any]]:
    """Get statistics for all bulkheads."""
    with _bulkheads_lock:
        return {name: bh.get_stats() for name, bh in _bulkheads.items()}
