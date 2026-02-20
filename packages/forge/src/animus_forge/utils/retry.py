"""Retry decorator with exponential backoff for transient failures.

Provides consistent retry behavior across all API clients.
Supports both synchronous and asynchronous functions.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import TypeVar

from animus_forge.errors import APIError, MaxRetriesError

logger = logging.getLogger(__name__)

# Type var for decorated function return type
T = TypeVar("T")

# HTTP status codes that should trigger retries
RETRYABLE_STATUS_CODES = frozenset(
    {
        408,  # Request Timeout
        429,  # Too Many Requests (rate limit)
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    }
)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retryable_exceptions: tuple[type[Exception], ...] = field(default_factory=tuple)

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed)."""
        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            # Add up to 25% jitter
            delay = delay * (0.75 + random.random() * 0.5)
        return delay


# Default retryable exceptions for common SDK errors
def _get_retryable_exceptions() -> tuple[type[Exception], ...]:
    """Build tuple of retryable exception types from available SDKs."""
    exceptions: list[type[Exception]] = [
        ConnectionError,
        TimeoutError,
        OSError,  # Covers network-level errors
    ]

    # OpenAI SDK exceptions
    try:
        import openai

        exceptions.extend(
            [
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ]
        )
    except (ImportError, AttributeError):
        pass  # Optional import: OpenAI SDK not installed

    # Anthropic SDK exceptions
    try:
        import anthropic

        exceptions.extend(
            [
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.InternalServerError,
            ]
        )
    except (ImportError, AttributeError):
        pass  # Optional import: Anthropic SDK not installed

    # GitHub SDK exceptions (PyGithub)
    try:
        from github import RateLimitExceededException

        exceptions.extend(
            [
                RateLimitExceededException,
            ]
        )
    except (ImportError, AttributeError):
        pass  # Optional import: PyGithub not installed

    # Requests library exceptions
    try:
        import requests

        exceptions.extend(
            [
                requests.ConnectionError,
                requests.Timeout,
            ]
        )
    except (ImportError, AttributeError):
        pass  # Optional import: requests library not installed

    return tuple(exceptions)


DEFAULT_RETRYABLE_EXCEPTIONS = _get_retryable_exceptions()


def is_retryable_error(exc: Exception) -> bool:
    """Check if an exception represents a retryable error."""
    # Check if it's a known retryable exception type
    if isinstance(exc, DEFAULT_RETRYABLE_EXCEPTIONS):
        return True

    # Check for Gorgon APIError with retryable status code
    if isinstance(exc, APIError) and exc.status_code in RETRYABLE_STATUS_CODES:
        return True

    # Check for HTTP status in exception attributes
    status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status_code and status_code in RETRYABLE_STATUS_CODES:
        return True

    return False


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (not including initial try)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        retryable_exceptions: Tuple of exception types to retry on.
            If None, uses default retryable exceptions.
        on_retry: Optional callback called before each retry with (exception, attempt)

    Returns:
        Decorated function with retry logic

    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        def call_external_api():
            return api.fetch_data()

        # With custom exceptions
        @with_retry(retryable_exceptions=(MyCustomError,))
        def call_service():
            return service.call()
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions or DEFAULT_RETRYABLE_EXCEPTIONS,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc

                    # Check if this error is retryable
                    should_retry = isinstance(
                        exc, config.retryable_exceptions
                    ) or is_retryable_error(exc)

                    if not should_retry:
                        # Non-retryable error, raise immediately
                        raise

                    if attempt >= config.max_retries:
                        # Exhausted all retries
                        logger.warning(
                            "Max retries (%d) exhausted for %s: %s",
                            config.max_retries,
                            func.__name__,
                            exc,
                        )
                        raise MaxRetriesError(
                            f"Max retries exceeded for {func.__name__}: {exc}",
                            stage=func.__name__,
                            attempts=attempt + 1,
                        ) from exc

                    # Calculate delay and wait
                    delay = config.calculate_delay(attempt)
                    logger.info(
                        "Retry %d/%d for %s after %.2fs: %s",
                        attempt + 1,
                        config.max_retries,
                        func.__name__,
                        delay,
                        exc,
                    )

                    # Call optional retry callback
                    if on_retry:
                        on_retry(exc, attempt)

                    time.sleep(delay)

            # Should never reach here, but satisfy type checker
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected state in retry loop for {func.__name__}")

        return wrapper

    return decorator


class RetryContext:
    """Context manager for retry logic when decorator isn't suitable.

    Example:
        with RetryContext(max_retries=3) as retry:
            for attempt in retry:
                try:
                    result = call_api()
                    break
                except Exception as e:
                    retry.handle_exception(e)
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        operation_name: str = "operation",
    ):
        self.config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
        self.operation_name = operation_name
        self._attempt = 0
        self._last_exception: Exception | None = None

    def __enter__(self) -> RetryContext:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False  # Don't suppress exceptions

    def __iter__(self):
        for attempt in range(self.config.max_retries + 1):
            self._attempt = attempt
            yield attempt

    def handle_exception(self, exc: Exception) -> None:
        """Handle an exception during retry loop."""
        self._last_exception = exc

        if not is_retryable_error(exc):
            raise exc

        if self._attempt >= self.config.max_retries:
            raise MaxRetriesError(
                f"Max retries exceeded for {self.operation_name}: {exc}",
                stage=self.operation_name,
                attempts=self._attempt + 1,
            ) from exc

        delay = self.config.calculate_delay(self._attempt)
        logger.info(
            "Retry %d/%d for %s after %.2fs: %s",
            self._attempt + 1,
            self.config.max_retries,
            self.operation_name,
            delay,
            exc,
        )
        time.sleep(delay)


def async_with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Async decorator to add retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (not including initial try)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        retryable_exceptions: Tuple of exception types to retry on.
            If None, uses default retryable exceptions.
        on_retry: Optional callback called before each retry with (exception, attempt)

    Returns:
        Decorated async function with retry logic

    Example:
        @async_with_retry(max_retries=3, base_delay=1.0)
        async def call_external_api():
            return await api.fetch_data()
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions or DEFAULT_RETRYABLE_EXCEPTIONS,
    )

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc

                    # Check if this error is retryable
                    should_retry = isinstance(
                        exc, config.retryable_exceptions
                    ) or is_retryable_error(exc)

                    if not should_retry:
                        # Non-retryable error, raise immediately
                        raise

                    if attempt >= config.max_retries:
                        # Exhausted all retries
                        logger.warning(
                            "Max retries (%d) exhausted for %s: %s",
                            config.max_retries,
                            func.__name__,
                            exc,
                        )
                        raise MaxRetriesError(
                            f"Max retries exceeded for {func.__name__}: {exc}",
                            stage=func.__name__,
                            attempts=attempt + 1,
                        ) from exc

                    # Calculate delay and wait asynchronously
                    delay = config.calculate_delay(attempt)
                    logger.info(
                        "Async retry %d/%d for %s after %.2fs: %s",
                        attempt + 1,
                        config.max_retries,
                        func.__name__,
                        delay,
                        exc,
                    )

                    # Call optional retry callback
                    if on_retry:
                        on_retry(exc, attempt)

                    await asyncio.sleep(delay)

            # Should never reach here, but satisfy type checker
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected state in retry loop for {func.__name__}")

        return wrapper

    return decorator


class AsyncRetryContext:
    """Async context manager for retry logic when decorator isn't suitable.

    Example:
        async with AsyncRetryContext(max_retries=3) as retry:
            for attempt in retry:
                try:
                    result = await call_api()
                    break
                except Exception as e:
                    await retry.handle_exception(e)
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        operation_name: str = "operation",
    ):
        self.config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
        self.operation_name = operation_name
        self._attempt = 0
        self._last_exception: Exception | None = None

    async def __aenter__(self) -> AsyncRetryContext:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False  # Don't suppress exceptions

    def __iter__(self):
        for attempt in range(self.config.max_retries + 1):
            self._attempt = attempt
            yield attempt

    async def handle_exception(self, exc: Exception) -> None:
        """Handle an exception during retry loop."""
        self._last_exception = exc

        if not is_retryable_error(exc):
            raise exc

        if self._attempt >= self.config.max_retries:
            raise MaxRetriesError(
                f"Max retries exceeded for {self.operation_name}: {exc}",
                stage=self.operation_name,
                attempts=self._attempt + 1,
            ) from exc

        delay = self.config.calculate_delay(self._attempt)
        logger.info(
            "Async retry %d/%d for %s after %.2fs: %s",
            self._attempt + 1,
            self.config.max_retries,
            self.operation_name,
            delay,
            exc,
        )
        await asyncio.sleep(delay)
