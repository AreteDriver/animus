"""Distributed rate limiting for cross-process coordination.

Provides rate limiting that works across multiple processes or instances.
Uses Redis when available, with SQLite fallback for simpler deployments.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    current_count: int
    limit: int
    reset_at: float  # Unix timestamp when window resets
    retry_after: float | None = None  # Seconds to wait if not allowed

    @property
    def remaining(self) -> int:
        """Remaining requests in current window."""
        return max(0, self.limit - self.current_count)


class DistributedRateLimiter(ABC):
    """Abstract base class for distributed rate limiters.

    Implements a sliding window rate limiting algorithm that works
    across multiple processes.
    """

    @abstractmethod
    async def acquire(self, key: str, limit: int, window_seconds: int = 60) -> RateLimitResult:
        """Try to acquire a rate limit slot.

        Args:
            key: Unique identifier (e.g., "anthropic:api")
            limit: Maximum requests per window
            window_seconds: Time window in seconds

        Returns:
            RateLimitResult indicating if request is allowed
        """
        pass

    @abstractmethod
    async def get_current(self, key: str, window_seconds: int = 60) -> int:
        """Get current request count for a key without incrementing."""
        pass

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        pass


class RedisRateLimiter(DistributedRateLimiter):
    """Redis-based distributed rate limiter.

    Uses Redis INCR with EXPIRE for atomic sliding window counting.
    Requires redis package: pip install redis

    Args:
        url: Redis URL (default: from REDIS_URL env or localhost)
        prefix: Key prefix for namespacing
    """

    def __init__(
        self,
        url: str | None = None,
        prefix: str = "gorgon:ratelimit:",
    ):
        if url is None:
            from animus_forge.config.settings import get_settings

            url = get_settings().redis_url
        self._url = url or "redis://localhost:6379/0"
        self._prefix = prefix
        self._client = None
        self._async_client = None

    def _get_client(self):
        """Lazy-load sync Redis client."""
        if self._client is None:
            try:
                import redis

                self._client = redis.from_url(self._url)
            except ImportError:
                raise ImportError("Redis package not installed. Install with: pip install redis")
        return self._client

    async def _get_async_client(self):
        """Lazy-load async Redis client."""
        if self._async_client is None:
            try:
                import redis.asyncio as aioredis

                self._async_client = await aioredis.from_url(self._url)
            except ImportError:
                raise ImportError("Redis package not installed. Install with: pip install redis")
        return self._async_client

    def _make_key(self, key: str, window_seconds: int) -> str:
        """Create Redis key with window timestamp."""
        window_start = int(time.time() // window_seconds) * window_seconds
        return f"{self._prefix}{key}:{window_start}"

    async def acquire(self, key: str, limit: int, window_seconds: int = 60) -> RateLimitResult:
        """Acquire rate limit slot using Redis INCR."""
        client = await self._get_async_client()
        redis_key = self._make_key(key, window_seconds)

        # Atomic increment
        current = await client.incr(redis_key)

        # Set expiry on first request in window
        if current == 1:
            await client.expire(redis_key, window_seconds)

        # Calculate window reset time
        window_start = int(time.time() // window_seconds) * window_seconds
        reset_at = window_start + window_seconds

        allowed = current <= limit
        retry_after = None if allowed else reset_at - time.time()

        return RateLimitResult(
            allowed=allowed,
            current_count=current,
            limit=limit,
            reset_at=reset_at,
            retry_after=retry_after,
        )

    async def get_current(self, key: str, window_seconds: int = 60) -> int:
        """Get current count without incrementing."""
        client = await self._get_async_client()
        redis_key = self._make_key(key, window_seconds)
        value = await client.get(redis_key)
        return int(value) if value else 0

    async def reset(self, key: str) -> None:
        """Reset by deleting all keys matching pattern."""
        client = await self._get_async_client()
        pattern = f"{self._prefix}{key}:*"
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break


class SQLiteRateLimiter(DistributedRateLimiter):
    """SQLite-based distributed rate limiter.

    Uses SQLite with file locking for cross-process coordination.
    Suitable for single-machine deployments without Redis.

    Args:
        db_path: Path to SQLite database (default: ~/.gorgon/rate_limits.db)
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_dir = Path.home() / ".gorgon"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "rate_limits.db")

        self._db_path = db_path
        self._local_lock = Lock()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Create table if not exists."""
        if self._initialized:
            return

        with self._local_lock:
            if self._initialized:
                return

            conn = sqlite3.connect(self._db_path, timeout=10.0)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS rate_limits (
                        key TEXT NOT NULL,
                        window_start INTEGER NOT NULL,
                        count INTEGER NOT NULL DEFAULT 1,
                        PRIMARY KEY (key, window_start)
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_rate_limits_key
                    ON rate_limits(key)
                """)
                conn.commit()
            finally:
                conn.close()

            self._initialized = True

    @contextmanager
    def _get_connection(self):
        """Get SQLite connection with proper locking."""
        self._ensure_initialized()
        conn = sqlite3.connect(self._db_path, timeout=10.0, isolation_level="EXCLUSIVE")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    async def acquire(self, key: str, limit: int, window_seconds: int = 60) -> RateLimitResult:
        """Acquire rate limit slot using SQLite."""
        # Run in executor to not block event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._acquire_sync, key, limit, window_seconds)

    def _acquire_sync(self, key: str, limit: int, window_seconds: int) -> RateLimitResult:
        """Synchronous acquire implementation."""
        window_start = int(time.time() // window_seconds) * window_seconds
        reset_at = window_start + window_seconds

        with self._get_connection() as conn:
            # Clean up old windows for this key
            conn.execute(
                "DELETE FROM rate_limits WHERE key = ? AND window_start < ?",
                (key, window_start),
            )

            # Try to increment existing record
            cursor = conn.execute(
                """
                UPDATE rate_limits
                SET count = count + 1
                WHERE key = ? AND window_start = ?
                RETURNING count
                """,
                (key, window_start),
            )
            row = cursor.fetchone()

            if row:
                current = row[0]
            else:
                # Insert new record
                conn.execute(
                    "INSERT INTO rate_limits (key, window_start, count) VALUES (?, ?, 1)",
                    (key, window_start),
                )
                current = 1

        allowed = current <= limit
        retry_after = None if allowed else reset_at - time.time()

        return RateLimitResult(
            allowed=allowed,
            current_count=current,
            limit=limit,
            reset_at=reset_at,
            retry_after=retry_after,
        )

    async def get_current(self, key: str, window_seconds: int = 60) -> int:
        """Get current count without incrementing."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_current_sync, key, window_seconds)

    def _get_current_sync(self, key: str, window_seconds: int) -> int:
        """Synchronous get_current implementation."""
        window_start = int(time.time() // window_seconds) * window_seconds

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT count FROM rate_limits WHERE key = ? AND window_start = ?",
                (key, window_start),
            )
            row = cursor.fetchone()
            return row[0] if row else 0

    async def reset(self, key: str) -> None:
        """Reset rate limit for key."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._reset_sync, key)

    def _reset_sync(self, key: str) -> None:
        """Synchronous reset implementation."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM rate_limits WHERE key = ?", (key,))

    async def cleanup_expired(self, older_than_seconds: int = 3600) -> int:
        """Remove expired rate limit records.

        Args:
            older_than_seconds: Remove windows older than this

        Returns:
            Number of records deleted
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._cleanup_expired_sync, older_than_seconds)

    def _cleanup_expired_sync(self, older_than_seconds: int) -> int:
        """Synchronous cleanup implementation."""
        cutoff = int(time.time()) - older_than_seconds

        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM rate_limits WHERE window_start < ?", (cutoff,))
            return cursor.rowcount


class MemoryRateLimiter(DistributedRateLimiter):
    """In-memory rate limiter for testing and single-process use.

    NOT suitable for cross-process coordination.
    """

    def __init__(self):
        self._counts: dict[str, tuple[int, int]] = {}  # key -> (count, window_start)
        self._lock = Lock()

    async def acquire(self, key: str, limit: int, window_seconds: int = 60) -> RateLimitResult:
        """Acquire rate limit slot."""
        window_start = int(time.time() // window_seconds) * window_seconds
        reset_at = window_start + window_seconds

        with self._lock:
            existing = self._counts.get(key)
            if existing and existing[1] == window_start:
                current = existing[0] + 1
            else:
                current = 1
            self._counts[key] = (current, window_start)

        allowed = current <= limit
        retry_after = None if allowed else reset_at - time.time()

        return RateLimitResult(
            allowed=allowed,
            current_count=current,
            limit=limit,
            reset_at=reset_at,
            retry_after=retry_after,
        )

    async def get_current(self, key: str, window_seconds: int = 60) -> int:
        """Get current count."""
        window_start = int(time.time() // window_seconds) * window_seconds

        with self._lock:
            existing = self._counts.get(key)
            if existing and existing[1] == window_start:
                return existing[0]
            return 0

    async def reset(self, key: str) -> None:
        """Reset rate limit."""
        with self._lock:
            self._counts.pop(key, None)


# Global instance management
_rate_limiter: DistributedRateLimiter | None = None


def get_rate_limiter() -> DistributedRateLimiter:
    """Get or create the global rate limiter.

    Auto-detects backend:
    - If REDIS_URL is set and redis installed: RedisRateLimiter
    - Otherwise: SQLiteRateLimiter
    """
    global _rate_limiter

    if _rate_limiter is None:
        _rate_limiter = _create_rate_limiter()

    return _rate_limiter


def _create_rate_limiter() -> DistributedRateLimiter:
    """Create rate limiter based on environment."""
    import importlib.util

    from animus_forge.config.settings import get_settings

    redis_url = get_settings().redis_url

    if redis_url:
        if importlib.util.find_spec("redis") is not None:
            logger.info(f"Using Redis rate limiter at {redis_url}")
            return RedisRateLimiter(url=redis_url)
        else:
            logger.warning(
                "REDIS_URL set but redis package not installed. "
                "Falling back to SQLite rate limiter."
            )

    logger.info("Using SQLite rate limiter")
    return SQLiteRateLimiter()


def reset_rate_limiter() -> None:
    """Reset the global rate limiter. Useful for testing."""
    global _rate_limiter
    _rate_limiter = None
