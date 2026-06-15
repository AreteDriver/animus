"""Semaphore-based concurrency limits.

Simple concurrency limiters for controlling parallel execution.
Designed for easy integration with API clients and external service calls.

Example:
    # Function decorator
    @limit_concurrency(max_concurrent=5)
    async def call_api():
        return await external_service()

    # Per-resource limits
    limiter = ConcurrencyLimiter("openai", max_concurrent=10)

    async with limiter:
        result = await call_openai()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ConcurrencyStats:
    """Statistics for concurrency limiter."""

    name: str
    max_concurrent: int
    current_active: int
    peak_active: int
    total_acquired: int
    total_rejected: int
    total_waited_ms: float


class ConcurrencyLimiter:
    """Simple semaphore-based concurrency limiter.

    Provides both sync and async interfaces for limiting concurrent
    operations. Simpler than Bulkhead when you don't need waiting queues.
    """

    def __init__(
        self,
        name: str = "",
        max_concurrent: int = 10,
        timeout: float | None = None,
    ):
        """Initialize concurrency limiter.

        Args:
            name: Identifier for this limiter
            max_concurrent: Maximum concurrent operations
            timeout: Optional timeout for acquiring (None = wait forever)
        """
        self.name = name
        self.max_concurrent = max_concurrent
        self.timeout = timeout

        # Semaphores
        self._semaphore = threading.Semaphore(max_concurrent)
        self._async_semaphore = asyncio.Semaphore(max_concurrent)

        # Tracking
        self._active = 0
        self._peak = 0
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

        # Stats
        self._total_acquired = 0
        self._total_rejected = 0
        self._total_waited_ms = 0.0

    def _update_active(self, delta: int) -> None:
        """Update active count and peak."""
        with self._lock:
            self._active += delta
            if self._active > self._peak:
                self._peak = self._active

    def acquire(self, timeout: float | None = None) -> bool:
        """Acquire a concurrency slot (sync).

        Args:
            timeout: Override default timeout

        Returns:
            True if acquired, False if timeout exceeded
        """
        timeout = timeout if timeout is not None else self.timeout
        start = time.monotonic()

        acquired = self._semaphore.acquire(
            blocking=timeout is None or timeout > 0,
            timeout=timeout,
        )

        if acquired:
            self._update_active(1)
            with self._lock:
                self._total_acquired += 1
                if timeout:
                    self._total_waited_ms += (time.monotonic() - start) * 1000
        else:
            with self._lock:
                self._total_rejected += 1

        return acquired

    def release(self) -> None:
        """Release a concurrency slot (sync)."""
        self._update_active(-1)
        self._semaphore.release()

    async def acquire_async(self, timeout: float | None = None) -> bool:
        """Acquire a concurrency slot (async).

        Args:
            timeout: Override default timeout

        Returns:
            True if acquired, False if timeout exceeded
        """
        timeout = timeout if timeout is not None else self.timeout
        start = time.monotonic()

        try:
            if timeout is not None:
                await asyncio.wait_for(
                    self._async_semaphore.acquire(),
                    timeout=timeout,
                )
            else:
                await self._async_semaphore.acquire()

            self._update_active(1)
            async with self._async_lock:
                self._total_acquired += 1
                if timeout:
                    self._total_waited_ms += (time.monotonic() - start) * 1000

            return True

        except TimeoutError:
            async with self._async_lock:
                self._total_rejected += 1
            return False

    async def release_async(self) -> None:
        """Release a concurrency slot (async)."""
        self._update_active(-1)
        self._async_semaphore.release()

    @contextmanager
    def __call__(self):
        """Use as sync context manager."""
        self.acquire()
        try:
            yield
        finally:
            self.release()

    def __enter__(self):
        """Sync context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        self.release()
        return False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.release_async()
        return False

    def get_stats(self) -> ConcurrencyStats:
        """Get limiter statistics."""
        with self._lock:
            return ConcurrencyStats(
                name=self.name,
                max_concurrent=self.max_concurrent,
                current_active=self._active,
                peak_active=self._peak,
                total_acquired=self._total_acquired,
                total_rejected=self._total_rejected,
                total_waited_ms=round(self._total_waited_ms, 2),
            )


# Global limiters registry
_limiters: dict[str, ConcurrencyLimiter] = {}
_limiters_lock = threading.Lock()


def get_limiter(
    name: str,
    max_concurrent: int = 10,
    timeout: float | None = None,
) -> ConcurrencyLimiter:
    """Get or create a named concurrency limiter.

    Args:
        name: Unique identifier
        max_concurrent: Max concurrent (only used on creation)
        timeout: Default timeout (only used on creation)

    Returns:
        ConcurrencyLimiter instance
    """
    with _limiters_lock:
        if name not in _limiters:
            _limiters[name] = ConcurrencyLimiter(
                name=name,
                max_concurrent=max_concurrent,
                timeout=timeout,
            )
            logger.info(f"Created concurrency limiter '{name}' (max={max_concurrent})")
        return _limiters[name]


def get_all_limiter_stats() -> dict[str, ConcurrencyStats]:
    """Get statistics for all limiters."""
    with _limiters_lock:
        return {name: lim.get_stats() for name, lim in _limiters.items()}


def limit_concurrency(
    max_concurrent: int = 10,
    name: str | None = None,
    timeout: float | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to limit concurrent executions of a function.

    Args:
        max_concurrent: Maximum concurrent calls
        name: Optional limiter name (uses function name if not provided)
        timeout: Optional timeout for acquiring slot

    Returns:
        Decorated function

    Example:
        @limit_concurrency(max_concurrent=5)
        async def call_api():
            return await external_service()

        @limit_concurrency(max_concurrent=3, name="database")
        def query_db():
            return db.execute(query)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        limiter_name = name or func.__name__
        limiter = get_limiter(limiter_name, max_concurrent, timeout)

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                async with limiter:
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore

        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                with limiter:
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


@asynccontextmanager
async def limit_async(
    name: str,
    max_concurrent: int = 10,
    timeout: float | None = None,
):
    """Async context manager for concurrency limiting.

    Args:
        name: Limiter name
        max_concurrent: Maximum concurrent operations
        timeout: Optional timeout

    Example:
        async with limit_async("external-api", max_concurrent=5):
            result = await call_api()
    """
    limiter = get_limiter(name, max_concurrent, timeout)
    await limiter.acquire_async(timeout)
    try:
        yield limiter
    finally:
        await limiter.release_async()


@contextmanager
def limit_sync(
    name: str,
    max_concurrent: int = 10,
    timeout: float | None = None,
):
    """Sync context manager for concurrency limiting.

    Args:
        name: Limiter name
        max_concurrent: Maximum concurrent operations
        timeout: Optional timeout

    Example:
        with limit_sync("database", max_concurrent=3):
            result = db.query(...)
    """
    limiter = get_limiter(name, max_concurrent, timeout)
    limiter.acquire(timeout)
    try:
        yield limiter
    finally:
        limiter.release()
