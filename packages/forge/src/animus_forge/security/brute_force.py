"""Brute force protection.

Rate limits authentication attempts and sensitive endpoints
to prevent credential stuffing and brute force attacks.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class BruteForceBlocked(Exception):
    """Raised when request is blocked due to brute force protection."""

    def __init__(
        self,
        message: str = "Too many attempts",
        retry_after: float = 0,
        identifier: str = "",
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.identifier = identifier


@dataclass
class AttemptRecord:
    """Record of attempts from an identifier."""

    attempts: int = 0
    first_attempt: float = 0.0
    last_attempt: float = 0.0
    blocked_until: float = 0.0
    failed_attempts: int = 0


@dataclass
class BruteForceConfig:
    """Configuration for brute force protection."""

    # General rate limits
    max_attempts_per_minute: int = 60
    max_attempts_per_hour: int = 300

    # Auth-specific limits (stricter)
    max_auth_attempts_per_minute: int = 5
    max_auth_attempts_per_hour: int = 20

    # Block duration (exponential backoff)
    initial_block_seconds: float = 60.0
    max_block_seconds: float = 3600.0  # 1 hour max
    block_multiplier: float = 2.0

    # Failed attempt tracking
    max_failed_attempts: int = 10  # Before extended block
    failed_attempt_block_hours: float = 24.0

    # Paths considered authentication-related
    auth_paths: tuple[str, ...] = (
        "/auth/",
        "/login",
        "/token",
        "/api/token",
    )

    # Cleanup interval for expired records
    cleanup_interval_seconds: float = 300.0  # 5 minutes


class BruteForceProtection:
    """Tracks and blocks brute force attempts.

    Uses a combination of:
    - Per-IP rate limiting
    - Exponential backoff on repeated blocks
    - Extended blocks for repeated failed attempts
    """

    def __init__(self, config: BruteForceConfig | None = None):
        """Initialize protection.

        Args:
            config: Protection configuration
        """
        self.config = config or BruteForceConfig()
        self._attempts: dict[str, AttemptRecord] = defaultdict(AttemptRecord)
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._last_cleanup = time.monotonic()

        # Stats
        self._total_blocked = 0
        self._total_allowed = 0

    def _cleanup_expired(self) -> None:
        """Remove expired attempt records."""
        now = time.monotonic()
        if now - self._last_cleanup < self.config.cleanup_interval_seconds:
            return

        self._last_cleanup = now
        hour_ago = now - 3600

        # Remove records with no recent activity
        expired = [
            key
            for key, record in self._attempts.items()
            if record.last_attempt < hour_ago and record.blocked_until < now
        ]

        for key in expired:
            del self._attempts[key]

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired attempt records")

    def _get_identifier(self, request: Request) -> str:
        """Get identifier for rate limiting (usually IP)."""
        # Check for forwarded headers (behind proxy)
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Take first IP in chain
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        # Fall back to client host
        if request.client:
            return request.client.host

        return "unknown"

    def _is_auth_path(self, path: str) -> bool:
        """Check if path is authentication-related."""
        return any(path.startswith(auth_path) for auth_path in self.config.auth_paths)

    def _calculate_block_duration(self, record: AttemptRecord) -> float:
        """Calculate block duration with exponential backoff."""
        # Count how many times this identifier has been blocked
        block_count = record.failed_attempts // self.config.max_auth_attempts_per_minute

        duration = self.config.initial_block_seconds * (self.config.block_multiplier**block_count)

        return min(duration, self.config.max_block_seconds)

    def check_allowed(
        self,
        identifier: str,
        is_auth: bool = False,
    ) -> tuple[bool, float]:
        """Check if request is allowed.

        Args:
            identifier: Client identifier (usually IP)
            is_auth: Whether this is an auth-related request

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        with self._lock:
            self._cleanup_expired()

            now = time.monotonic()
            record = self._attempts[identifier]

            # Check if currently blocked
            if record.blocked_until > now:
                retry_after = record.blocked_until - now
                self._total_blocked += 1
                return False, retry_after

            # Reset window if first attempt or window expired
            minute_ago = now - 60
            hour_ago = now - 3600

            # Initialize first_attempt if not set (0.0), or reset if window expired
            if record.first_attempt == 0.0 or record.first_attempt < hour_ago:
                # Reset hourly counter
                record.attempts = 0
                record.first_attempt = now

            # Get appropriate limits
            if is_auth:
                max_per_minute = self.config.max_auth_attempts_per_minute
                max_per_hour = self.config.max_auth_attempts_per_hour
            else:
                max_per_minute = self.config.max_attempts_per_minute
                max_per_hour = self.config.max_attempts_per_hour

            # Simple rate check: if all attempts are within the last minute,
            # use attempt count directly. Otherwise, use hourly limit only.
            if record.first_attempt > minute_ago:
                # All attempts are within the last minute
                if record.attempts >= max_per_minute:
                    block_duration = self._calculate_block_duration(record)
                    record.blocked_until = now + block_duration
                    record.failed_attempts += 1
                    self._total_blocked += 1
                    return False, record.blocked_until - now

            # Check hourly limit
            if record.attempts >= max_per_hour:
                block_duration = self._calculate_block_duration(record)
                record.blocked_until = now + block_duration
                record.failed_attempts += 1

                # Extended block for repeated failures
                if record.failed_attempts >= self.config.max_failed_attempts:
                    record.blocked_until = now + (self.config.failed_attempt_block_hours * 3600)
                    logger.warning(
                        f"Extended block for {identifier}: {record.failed_attempts} failed attempts"
                    )

                self._total_blocked += 1
                return False, record.blocked_until - now

            # Allow request
            record.attempts += 1
            record.last_attempt = now

            self._total_allowed += 1
            return True, 0.0

    async def check_allowed_async(
        self,
        identifier: str,
        is_auth: bool = False,
    ) -> tuple[bool, float]:
        """Async version of check_allowed."""
        async with self._async_lock:
            return self.check_allowed(identifier, is_auth)

    def record_failed_attempt(self, identifier: str) -> None:
        """Record a failed attempt (e.g., wrong password).

        Call this after failed authentication to increase
        the likelihood of blocking.
        """
        with self._lock:
            record = self._attempts[identifier]
            record.failed_attempts += 1

            # Immediate short block after failed auth
            if record.failed_attempts >= 3:
                now = time.monotonic()
                block_duration = self._calculate_block_duration(record)
                record.blocked_until = now + block_duration
                logger.info(
                    f"Blocking {identifier} for {block_duration:.0f}s "
                    f"after {record.failed_attempts} failed attempts"
                )

    def record_success(self, identifier: str) -> None:
        """Record successful attempt (e.g., successful login).

        Reduces the failed attempt counter.
        """
        with self._lock:
            record = self._attempts[identifier]
            # Don't fully reset - keep some history
            record.failed_attempts = max(0, record.failed_attempts - 2)

    def get_stats(self) -> dict[str, Any]:
        """Get protection statistics."""
        with self._lock:
            blocked_count = sum(
                1 for r in self._attempts.values() if r.blocked_until > time.monotonic()
            )

            return {
                "total_blocked": self._total_blocked,
                "total_allowed": self._total_allowed,
                "currently_blocked": blocked_count,
                "tracked_identifiers": len(self._attempts),
            }


# Global instance
_protection: BruteForceProtection | None = None
_protection_lock = threading.Lock()


def get_brute_force_protection(
    config: BruteForceConfig | None = None,
) -> BruteForceProtection:
    """Get or create global brute force protection instance."""
    global _protection

    with _protection_lock:
        if _protection is None:
            _protection = BruteForceProtection(config=config)
        return _protection


class BruteForceMiddleware(BaseHTTPMiddleware):
    """Middleware for brute force protection.

    Checks each request against rate limits and blocks
    requests from identifiers that exceed limits.
    """

    def __init__(
        self,
        app: ASGIApp,
        protection: BruteForceProtection | None = None,
    ):
        """Initialize middleware.

        Args:
            app: ASGI application
            protection: Protection instance (uses global if not provided)
        """
        super().__init__(app)
        self.protection = protection or get_brute_force_protection()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Check request against brute force limits."""
        identifier = self.protection._get_identifier(request)
        is_auth = self.protection._is_auth_path(request.url.path)

        allowed, retry_after = await self.protection.check_allowed_async(identifier, is_auth)

        if not allowed:
            logger.warning(
                f"Brute force block: {identifier} blocked for "
                f"{retry_after:.0f}s on {request.url.path}"
            )
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(int(retry_after))},
                content={
                    "error": "Too Many Requests",
                    "message": "Rate limit exceeded. Please try again later.",
                    "retry_after": int(retry_after),
                },
            )

        response = await call_next(request)

        # Track failed auth attempts
        if is_auth and response.status_code in (401, 403):
            self.protection.record_failed_attempt(identifier)
        elif is_auth and response.status_code == 200:
            self.protection.record_success(identifier)

        return response
