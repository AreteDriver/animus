"""Rate limiter implementations.

Provides token bucket and sliding window rate limiters.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
        limit_name: str = "",
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit_name = limit_name


@dataclass
class RateLimitConfig:
    """Configuration for rate limiters."""

    requests_per_second: float = 10.0
    burst_size: int = 20  # Max burst above steady rate
    max_wait_seconds: float = 30.0  # Max time to wait for token
    name: str = ""


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    def acquire(self, tokens: int = 1, wait: bool = True) -> bool:
        """Acquire tokens from the rate limiter.

        Args:
            tokens: Number of tokens to acquire
            wait: If True, wait for tokens; if False, return immediately

        Returns:
            True if tokens acquired, False if not (when wait=False)

        Raises:
            RateLimitExceeded: If wait=True and max_wait exceeded
        """
        pass

    @abstractmethod
    async def acquire_async(self, tokens: int = 1, wait: bool = True) -> bool:
        """Async version of acquire."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        pass


class TokenBucketLimiter(RateLimiter):
    """Token bucket rate limiter.

    Allows bursting up to bucket capacity, then limits to steady rate.
    Tokens are added at a constant rate up to the maximum capacity.
    """

    def __init__(self, config: RateLimitConfig):
        """Initialize token bucket limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self._capacity = config.burst_size
        self._tokens = float(config.burst_size)
        self._rate = config.requests_per_second
        self._last_update = time.monotonic()
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

        # Stats
        self._total_acquired = 0
        self._total_rejected = 0
        self._total_waited = 0.0

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(
            self._capacity,
            self._tokens + elapsed * self._rate,
        )
        self._last_update = now

    def _time_until_available(self, tokens: int) -> float:
        """Calculate time until tokens are available."""
        if self._tokens >= tokens:
            return 0.0
        needed = tokens - self._tokens
        return needed / self._rate

    def acquire(self, tokens: int = 1, wait: bool = True) -> bool:
        """Acquire tokens synchronously."""
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_acquired += tokens
                return True

            if not wait:
                self._total_rejected += tokens
                return False

            # Calculate wait time
            wait_time = self._time_until_available(tokens)

            if wait_time > self.config.max_wait_seconds:
                self._total_rejected += tokens
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {self.config.name or 'limiter'}",
                    retry_after=wait_time,
                    limit_name=self.config.name,
                )

        # Wait outside lock
        time.sleep(wait_time)
        self._total_waited += wait_time

        # Try again after waiting
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_acquired += tokens
                return True

            # Still not enough (shouldn't happen normally)
            self._total_rejected += tokens
            return False

    async def acquire_async(self, tokens: int = 1, wait: bool = True) -> bool:
        """Acquire tokens asynchronously."""
        async with self._async_lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_acquired += tokens
                return True

            if not wait:
                self._total_rejected += tokens
                return False

            wait_time = self._time_until_available(tokens)

            if wait_time > self.config.max_wait_seconds:
                self._total_rejected += tokens
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {self.config.name or 'limiter'}",
                    retry_after=wait_time,
                    limit_name=self.config.name,
                )

        # Wait outside lock
        await asyncio.sleep(wait_time)
        self._total_waited += wait_time

        async with self._async_lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_acquired += tokens
                return True

            self._total_rejected += tokens
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get limiter statistics."""
        with self._lock:
            self._refill()
            return {
                "name": self.config.name,
                "type": "token_bucket",
                "capacity": self._capacity,
                "available_tokens": round(self._tokens, 2),
                "rate_per_second": self._rate,
                "total_acquired": self._total_acquired,
                "total_rejected": self._total_rejected,
                "total_wait_seconds": round(self._total_waited, 2),
            }


@dataclass
class SlidingWindowEntry:
    """Entry in sliding window."""

    timestamp: float
    count: int = 1


class SlidingWindowLimiter(RateLimiter):
    """Sliding window rate limiter.

    Tracks requests in a time window and limits based on count.
    More accurate than fixed window but uses more memory.
    """

    def __init__(
        self,
        requests_per_window: int,
        window_seconds: float,
        name: str = "",
        max_wait_seconds: float = 30.0,
    ):
        """Initialize sliding window limiter.

        Args:
            requests_per_window: Max requests in window
            window_seconds: Window duration in seconds
            name: Limiter name for logging
            max_wait_seconds: Max time to wait for slot
        """
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.name = name
        self.max_wait_seconds = max_wait_seconds

        self._entries: list[SlidingWindowEntry] = []
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

        # Stats
        self._total_acquired = 0
        self._total_rejected = 0

    def _cleanup(self) -> None:
        """Remove expired entries."""
        cutoff = time.monotonic() - self.window_seconds
        self._entries = [e for e in self._entries if e.timestamp > cutoff]

    def _current_count(self) -> int:
        """Get current request count in window."""
        self._cleanup()
        return sum(e.count for e in self._entries)

    def _time_until_slot(self) -> float:
        """Calculate time until a slot opens."""
        if not self._entries:
            return 0.0

        # Find oldest entry that will expire
        oldest = min(e.timestamp for e in self._entries)
        return max(0.0, oldest + self.window_seconds - time.monotonic())

    def acquire(self, tokens: int = 1, wait: bool = True) -> bool:
        """Acquire tokens synchronously."""
        with self._lock:
            current = self._current_count()

            if current + tokens <= self.requests_per_window:
                self._entries.append(SlidingWindowEntry(timestamp=time.monotonic(), count=tokens))
                self._total_acquired += tokens
                return True

            if not wait:
                self._total_rejected += tokens
                return False

            wait_time = self._time_until_slot()

            if wait_time > self.max_wait_seconds:
                self._total_rejected += tokens
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {self.name or 'limiter'}",
                    retry_after=wait_time,
                    limit_name=self.name,
                )

        # Wait and retry
        time.sleep(wait_time + 0.01)  # Small buffer

        with self._lock:
            current = self._current_count()
            if current + tokens <= self.requests_per_window:
                self._entries.append(SlidingWindowEntry(timestamp=time.monotonic(), count=tokens))
                self._total_acquired += tokens
                return True

            self._total_rejected += tokens
            return False

    async def acquire_async(self, tokens: int = 1, wait: bool = True) -> bool:
        """Acquire tokens asynchronously."""
        async with self._async_lock:
            current = self._current_count()

            if current + tokens <= self.requests_per_window:
                self._entries.append(SlidingWindowEntry(timestamp=time.monotonic(), count=tokens))
                self._total_acquired += tokens
                return True

            if not wait:
                self._total_rejected += tokens
                return False

            wait_time = self._time_until_slot()

            if wait_time > self.max_wait_seconds:
                self._total_rejected += tokens
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {self.name or 'limiter'}",
                    retry_after=wait_time,
                    limit_name=self.name,
                )

        await asyncio.sleep(wait_time + 0.01)

        async with self._async_lock:
            current = self._current_count()
            if current + tokens <= self.requests_per_window:
                self._entries.append(SlidingWindowEntry(timestamp=time.monotonic(), count=tokens))
                self._total_acquired += tokens
                return True

            self._total_rejected += tokens
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get limiter statistics."""
        with self._lock:
            self._cleanup()
            return {
                "name": self.name,
                "type": "sliding_window",
                "window_seconds": self.window_seconds,
                "requests_per_window": self.requests_per_window,
                "current_count": self._current_count(),
                "total_acquired": self._total_acquired,
                "total_rejected": self._total_rejected,
            }
