"""Cache backend implementations.

Provides:
- MemoryCache: In-memory cache with TTL (no external dependencies)
- RedisCache: Redis-backed cache for distributed deployments
- Cache: Abstract base class for custom backends
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry:
    """A single cache entry with expiration."""

    value: Any
    expires_at: float | None = None  # Unix timestamp, None = no expiration

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class Cache(ABC):
    """Abstract base class for cache implementations."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with optional TTL in seconds."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache. Returns True if key existed."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached values."""
        pass

    @abstractmethod
    def get_sync(self, key: str) -> Any | None:
        """Synchronous get for non-async contexts."""
        pass

    @abstractmethod
    def set_sync(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Synchronous set for non-async contexts."""
        pass


class MemoryCache(Cache):
    """In-memory cache with TTL support.

    Thread-safe implementation suitable for single-process deployments.
    Performs lazy cleanup of expired entries on access.

    Args:
        max_size: Maximum number of entries (default 1000)
        default_ttl: Default TTL in seconds (default None = no expiration)
        cleanup_interval: Cleanup expired entries every N operations (default 100)
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int | None = None,
        cleanup_interval: int = 100,
    ):
        self._cache: dict[str, CacheEntry] = {}
        self._lock = Lock()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        self._operation_count = 0
        self._stats = CacheStats()

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        return self.get_sync(key)

    def get_sync(self, key: str) -> Any | None:
        """Synchronous get."""
        self._maybe_cleanup()

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                return None

            self._stats.hits += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        self.set_sync(key, value, ttl)

    def set_sync(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Synchronous set."""
        self._maybe_cleanup()

        ttl = ttl if ttl is not None else self._default_ttl
        expires_at = time.time() + ttl if ttl else None

        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_oldest()

            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[key]
                return False
            return True

    async def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()

    def _maybe_cleanup(self) -> None:
        """Periodically clean up expired entries."""
        self._operation_count += 1
        if self._operation_count >= self._cleanup_interval:
            self._operation_count = 0
            self._cleanup_expired()

    def _cleanup_expired(self) -> None:
        """Remove all expired entries."""
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            for key in expired_keys:
                del self._cache[key]

    def _evict_oldest(self) -> None:
        """Evict oldest entry (FIFO). Must be called with lock held."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._cache)


class RedisCache(Cache):
    """Redis-backed cache for distributed deployments.

    Requires redis package: pip install redis

    Args:
        url: Redis URL (default: redis://localhost:6379/0)
        prefix: Key prefix for namespacing (default: "gorgon:")
        default_ttl: Default TTL in seconds (default: 3600)
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "gorgon:",
        default_ttl: int = 3600,
    ):
        self._url = url
        self._prefix = prefix
        self._default_ttl = default_ttl
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

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Any | None:
        """Get value from Redis."""
        client = await self._get_async_client()
        data = await client.get(self._make_key(key))
        if data is None:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data.decode() if isinstance(data, bytes) else data

    def get_sync(self, key: str) -> Any | None:
        """Synchronous get."""
        client = self._get_client()
        data = client.get(self._make_key(key))
        if data is None:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data.decode() if isinstance(data, bytes) else data

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in Redis."""
        client = await self._get_async_client()
        ttl = ttl if ttl is not None else self._default_ttl

        # Serialize value
        if isinstance(value, (str, int, float, bool)):
            data = json.dumps(value)
        else:
            data = json.dumps(value)

        if ttl:
            await client.setex(self._make_key(key), ttl, data)
        else:
            await client.set(self._make_key(key), data)

    def set_sync(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Synchronous set."""
        client = self._get_client()
        ttl = ttl if ttl is not None else self._default_ttl

        # Serialize value
        data = json.dumps(value)

        if ttl:
            client.setex(self._make_key(key), ttl, data)
        else:
            client.set(self._make_key(key), data)

    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        client = await self._get_async_client()
        return await client.delete(self._make_key(key)) > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        client = await self._get_async_client()
        return await client.exists(self._make_key(key)) > 0

    async def clear(self) -> None:
        """Clear all keys with our prefix."""
        client = await self._get_async_client()
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor=cursor, match=f"{self._prefix}*", count=100)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


# Global cache instance
_cache: Cache | None = None


def get_cache() -> Cache:
    """Get or create the global cache instance.

    Auto-detects backend based on configuration:
    - If REDIS_URL is set and redis is installed, uses RedisCache
    - Otherwise, uses MemoryCache
    """
    global _cache

    if _cache is None:
        _cache = _create_cache()

    return _cache


def _create_cache() -> Cache:
    """Create cache based on configuration."""
    import importlib.util

    from animus_forge.config.settings import get_settings

    redis_url = get_settings().redis_url

    if redis_url:
        if importlib.util.find_spec("redis") is not None:
            logger.info(f"Using Redis cache at {redis_url}")
            return RedisCache(url=redis_url)
        else:
            logger.warning(
                "REDIS_URL set but redis package not installed. Falling back to memory cache."
            )

    logger.debug("Using in-memory cache")
    return MemoryCache()


def reset_cache() -> None:
    """Reset the global cache instance. Useful for testing."""
    global _cache
    _cache = None


def make_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments.

    Creates a deterministic hash from the provided arguments.
    """
    key_parts = []

    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            # Hash complex objects
            key_parts.append(
                hashlib.md5(json.dumps(arg, sort_keys=True, default=str).encode()).hexdigest()[:8]
            )

    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}={v}")
        else:
            key_parts.append(
                f"{k}={hashlib.md5(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:8]}"
            )

    return ":".join(key_parts)
