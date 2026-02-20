"""Caching decorators for functions and methods.

Provides decorators for easy function-level caching:
- cached: For synchronous functions
- async_cached: For async functions

Usage:
    @cached(ttl=60, prefix="user")
    def get_user(user_id: str) -> dict:
        return fetch_user_from_db(user_id)

    @async_cached(ttl=300, prefix="api")
    async def fetch_data(endpoint: str) -> dict:
        return await client.get(endpoint)
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from animus_forge.cache.backends import get_cache, make_cache_key

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def cached(
    ttl: int | None = None,
    prefix: str = "",
    key_builder: Callable[..., str] | None = None,
    skip_cache_on: Callable[[Any], bool] | None = None,
) -> Callable[[F], F]:
    """Decorator to cache function results.

    Args:
        ttl: Time-to-live in seconds (None = use cache default)
        prefix: Key prefix for namespacing
        key_builder: Custom function to build cache key from args/kwargs
        skip_cache_on: Function that returns True if result should not be cached

    Example:
        @cached(ttl=60, prefix="user")
        def get_user(user_id: str) -> dict:
            return fetch_user_from_db(user_id)

        # With custom key builder
        @cached(ttl=300, key_builder=lambda x: f"item:{x.id}")
        def process_item(item: Item) -> dict:
            ...

        # Skip caching on None results
        @cached(ttl=60, skip_cache_on=lambda x: x is None)
        def find_user(email: str) -> Optional[dict]:
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            cache = get_cache()

            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = _build_key(func, args, kwargs)

            if prefix:
                cache_key = f"{prefix}:{cache_key}"

            # Try to get from cache
            cached_value = cache.get_sync(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_value

            # Call function
            result = func(*args, **kwargs)

            # Check if we should skip caching
            if skip_cache_on and skip_cache_on(result):
                logger.debug(f"Skipping cache for: {cache_key}")
                return result

            # Store in cache
            cache.set_sync(cache_key, result, ttl)
            logger.debug(f"Cache set: {cache_key}")

            return result

        # Add cache control methods
        wrapper.cache_clear = lambda: _clear_prefix(prefix or func.__name__)

        def _get_cache_key(*a, **kw):
            if key_builder:
                base_key = key_builder(*a, **kw)
            else:
                base_key = _build_key(func, a, kw)
            return f"{prefix}:{base_key}" if prefix else base_key

        wrapper.cache_key = _get_cache_key

        return wrapper  # type: ignore

    return decorator


def async_cached(
    ttl: int | None = None,
    prefix: str = "",
    key_builder: Callable[..., str] | None = None,
    skip_cache_on: Callable[[Any], bool] | None = None,
) -> Callable[[F], F]:
    """Decorator to cache async function results.

    Args:
        ttl: Time-to-live in seconds (None = use cache default)
        prefix: Key prefix for namespacing
        key_builder: Custom function to build cache key from args/kwargs
        skip_cache_on: Function that returns True if result should not be cached

    Example:
        @async_cached(ttl=300, prefix="api")
        async def fetch_data(endpoint: str) -> dict:
            return await client.get(endpoint)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            cache = get_cache()

            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = _build_key(func, args, kwargs)

            if prefix:
                cache_key = f"{prefix}:{cache_key}"

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_value

            # Call function
            result = await func(*args, **kwargs)

            # Check if we should skip caching
            if skip_cache_on and skip_cache_on(result):
                logger.debug(f"Skipping cache for: {cache_key}")
                return result

            # Store in cache
            await cache.set(cache_key, result, ttl)
            logger.debug(f"Cache set: {cache_key}")

            return result

        # Add cache control methods
        wrapper.cache_clear = lambda: asyncio.create_task(
            _async_clear_prefix(prefix or func.__name__)
        )

        def _get_cache_key(*a, **kw):
            if key_builder:
                base_key = key_builder(*a, **kw)
            else:
                base_key = _build_key(func, a, kw)
            return f"{prefix}:{base_key}" if prefix else base_key

        wrapper.cache_key = _get_cache_key

        return wrapper  # type: ignore

    return decorator


def _build_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Build cache key from function and arguments."""
    # Get function name (handle methods)
    func_name = func.__name__

    # Get bound arguments for methods (skip 'self')
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    # Filter out 'self' or 'cls' for methods
    cache_args = {}
    for k, v in bound.arguments.items():
        if k in ("self", "cls"):
            continue
        cache_args[k] = v

    # Build key
    key = f"{func_name}:{make_cache_key(**cache_args)}"
    return key


def _clear_prefix(prefix: str) -> None:
    """Clear all cache entries with given prefix (sync).

    Note: This is a simplified implementation that only logs the request.
    Full prefix-based clearing would require iteration for MemoryCache
    or SCAN for Redis.
    """
    # For memory cache, we could iterate; for Redis, would need scan
    logger.debug(f"Cache clear requested for prefix: {prefix}")


async def _async_clear_prefix(prefix: str) -> None:
    """Clear all cache entries with given prefix (async).

    Note: This is a simplified implementation that only logs the request.
    Full prefix-based clearing would require iteration for MemoryCache
    or SCAN for Redis.
    """
    logger.debug(f"Cache clear requested for prefix: {prefix}")


class CacheAside:
    """Cache-aside pattern helper for manual cache management.

    Usage:
        cache_aside = CacheAside(prefix="users", ttl=300)

        async def get_user(user_id: str) -> dict:
            # Try cache first
            cached = await cache_aside.get(user_id)
            if cached:
                return cached

            # Fetch from source
            user = await fetch_user_from_db(user_id)

            # Store in cache
            await cache_aside.set(user_id, user)
            return user
    """

    def __init__(
        self,
        prefix: str,
        ttl: int | None = None,
    ):
        self.prefix = prefix
        self.ttl = ttl
        self._cache = get_cache()

    def _make_key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        return await self._cache.get(self._make_key(key))

    def get_sync(self, key: str) -> Any | None:
        """Synchronous get."""
        return self._cache.get_sync(self._make_key(key))

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        await self._cache.set(self._make_key(key), value, ttl or self.ttl)

    def set_sync(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Synchronous set."""
        self._cache.set_sync(self._make_key(key), value, ttl or self.ttl)

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return await self._cache.delete(self._make_key(key))

    async def invalidate(self, key: str) -> None:
        """Alias for delete."""
        await self.delete(key)
