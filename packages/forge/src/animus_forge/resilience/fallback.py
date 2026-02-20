"""Fallback chains for graceful degradation.

Provides alternative execution paths when primary operations fail.
Supports both sync and async operations with configurable fallback order.

Example:
    # Simple fallback decorator
    @fallback(lambda: "default value")
    def get_data():
        return external_api_call()

    # Chain of alternatives
    chain = FallbackChain("completion")
    chain.add(call_claude)
    chain.add(call_openai)
    chain.add(return_cached_response)

    result = await chain.execute(prompt="Hello")
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""

    name: str = ""
    max_attempts: int = 3  # Max fallbacks to try
    fail_fast_exceptions: tuple[type[Exception], ...] = ()  # Don't fallback on these
    delay_between_fallbacks: float = 0.0  # Delay before trying next fallback


@dataclass
class FallbackResult(Generic[T]):
    """Result from fallback chain execution."""

    success: bool
    value: T | None
    source: str  # Name of handler that produced the result
    attempts: int
    errors: list[dict[str, Any]] = field(default_factory=list)
    total_time_ms: float = 0.0

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success


@dataclass
class FallbackHandler:
    """A handler in the fallback chain."""

    name: str
    handler: Callable[..., Any]
    is_async: bool
    priority: int = 0  # Lower = higher priority


class FallbackChain:
    """Chain of fallback handlers for graceful degradation.

    Tries handlers in order until one succeeds or all fail.
    Supports both sync and async handlers.
    """

    def __init__(
        self,
        name: str = "",
        config: FallbackConfig | None = None,
    ):
        """Initialize fallback chain.

        Args:
            name: Chain identifier for logging
            config: Fallback configuration
        """
        self.name = name
        self.config = config or FallbackConfig(name=name)
        self._handlers: list[FallbackHandler] = []

        # Stats
        self._total_executions = 0
        self._successful_executions = 0
        self._handler_success_counts: dict[str, int] = {}

    def add(
        self,
        handler: Callable[..., Any],
        name: str | None = None,
        priority: int = 0,
    ) -> FallbackChain:
        """Add a handler to the chain.

        Args:
            handler: Function to call (sync or async)
            name: Handler name for logging
            priority: Lower = tried first (default: order added)

        Returns:
            Self for chaining
        """
        handler_name = name or handler.__name__
        is_async = asyncio.iscoroutinefunction(handler)

        self._handlers.append(
            FallbackHandler(
                name=handler_name,
                handler=handler,
                is_async=is_async,
                priority=priority if priority else len(self._handlers),
            )
        )

        # Sort by priority
        self._handlers.sort(key=lambda h: h.priority)

        return self

    def execute(self, *args: Any, **kwargs: Any) -> FallbackResult:
        """Execute the fallback chain (sync).

        Tries handlers in priority order until one succeeds.

        Args:
            *args: Arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers

        Returns:
            FallbackResult with success status and value
        """
        start_time = time.monotonic()
        self._total_executions += 1
        errors: list[dict[str, Any]] = []
        attempts = 0

        for handler in self._handlers[: self.config.max_attempts]:
            attempts += 1

            try:
                logger.debug(f"Fallback chain '{self.name}' trying handler: {handler.name}")

                if handler.is_async:
                    # Run async handler in event loop
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    result = loop.run_until_complete(handler.handler(*args, **kwargs))
                else:
                    result = handler.handler(*args, **kwargs)

                self._successful_executions += 1
                self._handler_success_counts[handler.name] = (
                    self._handler_success_counts.get(handler.name, 0) + 1
                )

                logger.info(f"Fallback chain '{self.name}' succeeded with handler: {handler.name}")

                return FallbackResult(
                    success=True,
                    value=result,
                    source=handler.name,
                    attempts=attempts,
                    errors=errors,
                    total_time_ms=(time.monotonic() - start_time) * 1000,
                )

            except self.config.fail_fast_exceptions as e:
                # Don't fallback on certain exceptions
                logger.warning(f"Fallback chain '{self.name}' fast-failing on: {type(e).__name__}")
                errors.append(
                    {
                        "handler": handler.name,
                        "error": str(e),
                        "type": type(e).__name__,
                        "fail_fast": True,
                    }
                )
                break

            except Exception as e:
                logger.warning(f"Fallback chain '{self.name}' handler '{handler.name}' failed: {e}")
                errors.append(
                    {
                        "handler": handler.name,
                        "error": str(e),
                        "type": type(e).__name__,
                    }
                )

                if self.config.delay_between_fallbacks > 0:
                    time.sleep(self.config.delay_between_fallbacks)

        logger.error(f"Fallback chain '{self.name}' exhausted all {attempts} handlers")

        return FallbackResult(
            success=False,
            value=None,
            source="",
            attempts=attempts,
            errors=errors,
            total_time_ms=(time.monotonic() - start_time) * 1000,
        )

    async def execute_async(self, *args: Any, **kwargs: Any) -> FallbackResult:
        """Execute the fallback chain (async).

        Args:
            *args: Arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers

        Returns:
            FallbackResult with success status and value
        """
        start_time = time.monotonic()
        self._total_executions += 1
        errors: list[dict[str, Any]] = []
        attempts = 0

        for handler in self._handlers[: self.config.max_attempts]:
            attempts += 1

            try:
                logger.debug(f"Fallback chain '{self.name}' trying handler: {handler.name}")

                if handler.is_async:
                    result = await handler.handler(*args, **kwargs)
                else:
                    # Run sync handler in thread pool
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: handler.handler(*args, **kwargs)
                    )

                self._successful_executions += 1
                self._handler_success_counts[handler.name] = (
                    self._handler_success_counts.get(handler.name, 0) + 1
                )

                logger.info(f"Fallback chain '{self.name}' succeeded with handler: {handler.name}")

                return FallbackResult(
                    success=True,
                    value=result,
                    source=handler.name,
                    attempts=attempts,
                    errors=errors,
                    total_time_ms=(time.monotonic() - start_time) * 1000,
                )

            except self.config.fail_fast_exceptions as e:
                logger.warning(f"Fallback chain '{self.name}' fast-failing on: {type(e).__name__}")
                errors.append(
                    {
                        "handler": handler.name,
                        "error": str(e),
                        "type": type(e).__name__,
                        "fail_fast": True,
                    }
                )
                break

            except Exception as e:
                logger.warning(f"Fallback chain '{self.name}' handler '{handler.name}' failed: {e}")
                errors.append(
                    {
                        "handler": handler.name,
                        "error": str(e),
                        "type": type(e).__name__,
                    }
                )

                if self.config.delay_between_fallbacks > 0:
                    await asyncio.sleep(self.config.delay_between_fallbacks)

        logger.error(f"Fallback chain '{self.name}' exhausted all {attempts} handlers")

        return FallbackResult(
            success=False,
            value=None,
            source="",
            attempts=attempts,
            errors=errors,
            total_time_ms=(time.monotonic() - start_time) * 1000,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get chain statistics."""
        return {
            "name": self.name,
            "total_executions": self._total_executions,
            "successful_executions": self._successful_executions,
            "success_rate": (
                self._successful_executions / self._total_executions * 100
                if self._total_executions > 0
                else 0
            ),
            "handlers": [h.name for h in self._handlers],
            "handler_success_counts": self._handler_success_counts.copy(),
        }


def fallback(
    fallback_value: Callable[..., T] | T,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to provide a fallback value on exception.

    Args:
        fallback_value: Value or callable to return on failure
        exceptions: Exception types to catch (default: all)

    Returns:
        Decorated function

    Example:
        @fallback(lambda: "default")
        def get_data():
            return risky_call()

        @fallback("cached_value")
        async def fetch_data():
            return await api_call()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(
                        f"Function '{func.__name__}' failed with {type(e).__name__}, using fallback"
                    )
                    if callable(fallback_value):
                        fb = fallback_value()
                        if asyncio.iscoroutine(fb):
                            return await fb
                        return fb
                    return fallback_value  # type: ignore

            return async_wrapper  # type: ignore

        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(
                        f"Function '{func.__name__}' failed with {type(e).__name__}, using fallback"
                    )
                    if callable(fallback_value):
                        return fallback_value()
                    return fallback_value  # type: ignore

            return sync_wrapper

    return decorator
