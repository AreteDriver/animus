"""Resilience integration for API clients.

Provides unified rate limiting, bulkhead isolation, and circuit breaking
for all external API calls. Each provider has configured limits.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from animus_forge.ratelimit.provider import get_provider_limiter
from animus_forge.resilience.bulkhead import get_bulkhead

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _get_provider_configs() -> dict:
    """Get provider configurations from settings.

    Falls back to defaults if settings are not available.
    """
    try:
        from animus_forge.config import get_settings

        settings = get_settings()
        return {
            "openai": {
                "max_concurrent": settings.bulkhead_openai_concurrent,
                "max_waiting": settings.bulkhead_openai_concurrent * 2,
                "timeout": 60.0,
            },
            "anthropic": {
                "max_concurrent": settings.bulkhead_anthropic_concurrent,
                "max_waiting": settings.bulkhead_anthropic_concurrent * 2,
                "timeout": 60.0,
            },
            "github": {
                "max_concurrent": settings.bulkhead_github_concurrent,
                "max_waiting": settings.bulkhead_github_concurrent * 2,
                "timeout": settings.bulkhead_default_timeout,
            },
            "notion": {
                "max_concurrent": settings.bulkhead_notion_concurrent,
                "max_waiting": settings.bulkhead_notion_concurrent * 3,
                "timeout": settings.bulkhead_default_timeout,
            },
            "gmail": {
                "max_concurrent": settings.bulkhead_gmail_concurrent,
                "max_waiting": settings.bulkhead_gmail_concurrent * 2,
                "timeout": settings.bulkhead_default_timeout,
            },
        }
    except Exception:
        # Fall back to defaults if settings not available
        return {
            "openai": {"max_concurrent": 10, "max_waiting": 20, "timeout": 60.0},
            "anthropic": {"max_concurrent": 10, "max_waiting": 20, "timeout": 60.0},
            "github": {"max_concurrent": 5, "max_waiting": 10, "timeout": 30.0},
            "notion": {"max_concurrent": 3, "max_waiting": 10, "timeout": 30.0},
            "gmail": {"max_concurrent": 5, "max_waiting": 10, "timeout": 30.0},
        }


# Provider configurations loaded from settings
PROVIDER_CONFIGS = _get_provider_configs()


def get_provider_bulkhead(provider: str):
    """Get bulkhead for a provider with configured limits."""
    config = PROVIDER_CONFIGS.get(provider, {})
    return get_bulkhead(
        name=f"{provider}-bulkhead",
        max_concurrent=config.get("max_concurrent", 10),
        max_waiting=config.get("max_waiting", 20),
        timeout=config.get("timeout", 30.0),
    )


def resilient_call(
    provider: str,
    rate_limit: bool = True,
    bulkhead: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for resilient sync API calls.

    Applies rate limiting and bulkhead isolation to the decorated function.

    Args:
        provider: Provider name (openai, anthropic, github, notion, gmail)
        rate_limit: Whether to apply rate limiting
        bulkhead: Whether to apply bulkhead isolation

    Example:
        @resilient_call("openai")
        def call_openai_api():
            return client.chat.completions.create(...)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Apply rate limiting
            if rate_limit:
                limiter = get_provider_limiter(provider)
                if not limiter.acquire(wait=True):
                    raise RuntimeError(f"Rate limit exceeded for {provider}")

            # Apply bulkhead isolation
            if bulkhead:
                bh = get_provider_bulkhead(provider)
                if not bh.acquire():
                    raise RuntimeError(f"Bulkhead full for {provider}")
                try:
                    return func(*args, **kwargs)
                finally:
                    bh.release()
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def resilient_call_async(
    provider: str,
    rate_limit: bool = True,
    bulkhead: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for resilient async API calls.

    Applies rate limiting and bulkhead isolation to the decorated function.

    Args:
        provider: Provider name (openai, anthropic, github, notion, gmail)
        rate_limit: Whether to apply rate limiting
        bulkhead: Whether to apply bulkhead isolation

    Example:
        @resilient_call_async("openai")
        async def call_openai_api_async():
            return await client.chat.completions.create(...)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Apply rate limiting
            if rate_limit:
                limiter = get_provider_limiter(provider)
                if not await limiter.acquire_async(wait=True):
                    raise RuntimeError(f"Rate limit exceeded for {provider}")

            # Apply bulkhead isolation
            if bulkhead:
                bh = get_provider_bulkhead(provider)
                if not await bh.acquire_async():
                    raise RuntimeError(f"Bulkhead full for {provider}")
                try:
                    return await func(*args, **kwargs)
                finally:
                    await bh.release_async()
            else:
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


class ResilientClientMixin:
    """Mixin that provides resilience methods for API clients.

    Usage:
        class MyClient(ResilientClientMixin):
            PROVIDER = "openai"

            def call_api(self):
                with self.resilient_context():
                    return self._make_call()
    """

    PROVIDER: str = "unknown"

    def resilient_context(self, rate_limit: bool = True, bulkhead: bool = True):
        """Context manager for resilient sync calls."""
        return _ResilientContext(self.PROVIDER, rate_limit, bulkhead, is_async=False)

    def resilient_context_async(self, rate_limit: bool = True, bulkhead: bool = True):
        """Context manager for resilient async calls."""
        return _ResilientContextAsync(self.PROVIDER, rate_limit, bulkhead)


class _ResilientContext:
    """Sync context manager for resilient calls."""

    def __init__(self, provider: str, rate_limit: bool, bulkhead: bool, is_async: bool):
        self.provider = provider
        self.rate_limit = rate_limit
        self.bulkhead = bulkhead
        self._bh = None

    def __enter__(self):
        if self.rate_limit:
            limiter = get_provider_limiter(self.provider)
            limiter.acquire(wait=True)

        if self.bulkhead:
            self._bh = get_provider_bulkhead(self.provider)
            self._bh.acquire()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._bh:
            self._bh.release()
        return False


class _ResilientContextAsync:
    """Async context manager for resilient calls."""

    def __init__(self, provider: str, rate_limit: bool, bulkhead: bool):
        self.provider = provider
        self.rate_limit = rate_limit
        self.bulkhead = bulkhead
        self._bh = None

    async def __aenter__(self):
        if self.rate_limit:
            limiter = get_provider_limiter(self.provider)
            await limiter.acquire_async(wait=True)

        if self.bulkhead:
            self._bh = get_provider_bulkhead(self.provider)
            await self._bh.acquire_async()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._bh:
            await self._bh.release_async()
        return False


def get_all_provider_stats() -> dict:
    """Get stats for all providers."""
    from animus_forge.ratelimit.provider import get_provider_limiter
    from animus_forge.resilience.bulkhead import get_all_bulkhead_stats

    stats = {}
    for provider in PROVIDER_CONFIGS:
        try:
            limiter = get_provider_limiter(provider)
            stats[provider] = {
                "rate_limit": limiter.get_stats(),
                "bulkhead": get_all_bulkhead_stats().get(f"{provider}-bulkhead", {}),
            }
        except Exception as e:
            stats[provider] = {"error": str(e)}

    return stats
