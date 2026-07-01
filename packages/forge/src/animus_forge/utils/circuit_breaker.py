"""Circuit breaker pattern for resilient external service calls.

Prevents cascading failures by failing fast when a service is unhealthy.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service unhealthy, requests fail immediately
- HALF_OPEN: Testing if service recovered, one request allowed

Example:
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

    @breaker
    def call_external_api():
        return requests.get("https://api.example.com")

    # Or use as context manager:
    with breaker:
        response = requests.get("https://api.example.com")
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

from animus_forge.errors import GorgonError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerError(GorgonError):
    """Raised when circuit breaker is open."""

    pass


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before testing recovery
        success_threshold: Successes needed in half-open to close circuit
        excluded_exceptions: Exception types that don't count as failures
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 2
    excluded_exceptions: tuple[type[Exception], ...] = ()


@dataclass
class CircuitBreaker:
    """Circuit breaker for external service calls.

    Thread-safe implementation that tracks failures and prevents
    cascading failures by failing fast when a service is unhealthy.

    Args:
        name: Identifier for this circuit (used in logs/metrics)
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before testing recovery
        success_threshold: Successes needed to close from half-open
        excluded_exceptions: Exceptions that don't trigger the breaker
    """

    name: str = "default"
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 2
    excluded_exceptions: tuple[type[Exception], ...] = ()

    # Internal state (not part of init)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        with self._lock:
            self._check_recovery()
            return self._state

    @property
    def is_closed(self) -> bool:
        """True if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """True if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN

    def _check_recovery(self) -> None:
        """Check if enough time has passed to attempt recovery."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info(
                    f"Circuit '{self.name}' entering half-open state "
                    f"after {elapsed:.1f}s recovery timeout"
                )
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    logger.info(
                        f"Circuit '{self.name}' closed after {self._success_count} successful calls"
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def _record_failure(self, exc: Exception) -> None:
        """Record a failed call."""
        # Check if this exception type should be excluded
        if isinstance(exc, self.excluded_exceptions):
            logger.debug(f"Circuit '{self.name}' ignoring excluded exception: {type(exc).__name__}")
            return

        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens circuit
                logger.warning(f"Circuit '{self.name}' reopened after failure in half-open state")
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    logger.warning(
                        f"Circuit '{self.name}' opened after "
                        f"{self._failure_count} consecutive failures"
                    )
                    self._state = CircuitState.OPEN

    def _check_state(self) -> None:
        """Check if call should be allowed through."""
        with self._lock:
            self._check_recovery()

            if self._state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit '{self.name}' is open - service unavailable. "
                    f"Will retry after {self.recovery_timeout}s recovery timeout."
                )

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception from func (also recorded as failure)
        """
        self._check_state()

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    async def call_async(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute an async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception from func (also recorded as failure)
        """
        self._check_state()

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use circuit breaker as a decorator.

        Example:
            breaker = CircuitBreaker(name="api")

            @breaker
            def call_api():
                return requests.get("https://api.example.com")
        """
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await self.call_async(func, *args, **kwargs)

            return async_wrapper  # type: ignore
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                return self.call(func, *args, **kwargs)

            return sync_wrapper

    def __enter__(self) -> CircuitBreaker:
        """Context manager entry - check if call allowed."""
        self._check_state()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - record success or failure."""
        if exc_val is None:
            self._record_success()
        elif exc_type is not None:
            self._record_failure(exc_val)

    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        with self._lock:
            logger.info(f"Circuit '{self.name}' manually reset to closed")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get current circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
            }


# Global circuit breakers for common services
_circuit_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    success_threshold: int = 2,
) -> CircuitBreaker:
    """Get or create a named circuit breaker.

    Circuit breakers are cached by name, so multiple calls with the
    same name return the same instance.

    Args:
        name: Unique identifier for the circuit
        failure_threshold: Failures before opening (only used on creation)
        recovery_timeout: Recovery timeout in seconds (only used on creation)
        success_threshold: Successes to close (only used on creation)

    Returns:
        CircuitBreaker instance
    """
    with _breakers_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                success_threshold=success_threshold,
            )
        return _circuit_breakers[name]


def get_all_circuit_stats() -> dict[str, dict[str, Any]]:
    """Get statistics for all circuit breakers."""
    with _breakers_lock:
        return {name: breaker.get_stats() for name, breaker in _circuit_breakers.items()}


def reset_all_circuits() -> None:
    """Reset all circuit breakers to closed state."""
    with _breakers_lock:
        for breaker in _circuit_breakers.values():
            breaker.reset()
