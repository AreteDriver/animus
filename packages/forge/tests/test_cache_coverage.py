"""Additional coverage tests for cache backends."""

import asyncio
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from animus_forge.cache.backends import (
    CacheEntry,
    CacheStats,
    MemoryCache,
    RedisCache,
)


class TestCacheEntry:
    def test_not_expired(self):
        entry = CacheEntry(value="test", expires_at=time.time() + 3600)
        assert entry.is_expired() is False

    def test_expired(self):
        entry = CacheEntry(value="test", expires_at=time.time() - 1)
        assert entry.is_expired() is True

    def test_no_expiration(self):
        entry = CacheEntry(value="test", expires_at=None)
        assert entry.is_expired() is False


class TestMemoryCache:
    def test_get_set_sync(self):
        cache = MemoryCache()
        cache.set_sync("key", "value")
        assert cache.get_sync("key") == "value"

    def test_get_missing(self):
        cache = MemoryCache()
        assert cache.get_sync("missing") is None

    def test_get_expired(self):
        cache = MemoryCache()
        # Set with very short TTL
        cache._cache["key"] = CacheEntry(value="value", expires_at=time.time() - 1)
        assert cache.get_sync("key") is None

    def test_default_ttl(self):
        cache = MemoryCache(default_ttl=3600)
        cache.set_sync("key", "value")
        assert cache.get_sync("key") == "value"

    def test_delete(self):
        cache = MemoryCache()
        cache.set_sync("key", "value")

        async def _test():
            result1 = await cache.delete("key")
            assert result1 is True
            result2 = await cache.delete("key")
            assert result2 is False

        asyncio.run(_test())

    def test_exists(self):
        cache = MemoryCache()
        cache.set_sync("key", "value")

        async def _test():
            assert await cache.exists("key") is True
            assert await cache.exists("missing") is False

        asyncio.run(_test())

    def test_exists_expired(self):
        cache = MemoryCache()
        cache._cache["key"] = CacheEntry(value="value", expires_at=time.time() - 1)

        async def _test():
            assert await cache.exists("key") is False

        asyncio.run(_test())

    def test_clear(self):
        cache = MemoryCache()
        cache.set_sync("a", 1)
        cache.set_sync("b", 2)

        async def _test():
            await cache.clear()

        asyncio.run(_test())
        assert cache.size == 0

    def test_eviction(self):
        cache = MemoryCache(max_size=2)
        cache.set_sync("a", 1)
        cache.set_sync("b", 2)
        cache.set_sync("c", 3)
        assert cache.size == 2
        # Oldest (a) should be evicted
        assert cache.get_sync("a") is None

    def test_cleanup_interval(self):
        cache = MemoryCache(cleanup_interval=3)
        cache._cache["exp"] = CacheEntry(value="val", expires_at=time.time() - 1)
        # Trigger cleanup after N operations
        cache.get_sync("x")
        cache.get_sync("x")
        cache.get_sync("x")  # Should trigger cleanup
        # The expired entry should be cleaned up
        assert "exp" not in cache._cache

    def test_stats(self):
        cache = MemoryCache()
        cache.set_sync("key", "value")
        cache.get_sync("key")  # hit
        cache.get_sync("miss")  # miss
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1

    def test_async_get_set(self):
        cache = MemoryCache()

        async def _test():
            await cache.set("key", "async_val")
            result = await cache.get("key")
            assert result == "async_val"

        asyncio.run(_test())

    def test_size(self):
        cache = MemoryCache()
        assert cache.size == 0
        cache.set_sync("a", 1)
        assert cache.size == 1


class TestCacheStats:
    def test_defaults(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0

    def test_hit_rate_no_requests(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate(self):
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == 75.0


class TestRedisCache:
    def test_init(self):
        cache = RedisCache(url="redis://localhost:6379/0", prefix="test:")
        assert cache._prefix == "test:"
        assert cache._default_ttl == 3600

    def test_make_key(self):
        cache = RedisCache(prefix="gorgon:")
        assert cache._make_key("foo") == "gorgon:foo"

    def test_get_client_import_error(self):
        cache = RedisCache()
        with patch.dict("sys.modules", {"redis": None}):
            with pytest.raises(ImportError, match="Redis"):
                cache._get_client()

    def test_get_async_client_import_error(self):
        cache = RedisCache()

        async def _test():
            with patch.dict("sys.modules", {"redis.asyncio": None, "redis": None}):
                with pytest.raises(ImportError, match="Redis"):
                    await cache._get_async_client()

        asyncio.run(_test())

    def test_get_sync(self):
        cache = RedisCache()
        mock_client = MagicMock()
        mock_client.get.return_value = b'{"data": "test"}'
        cache._client = mock_client
        result = cache.get_sync("key")
        assert result == {"data": "test"}

    def test_get_sync_none(self):
        cache = RedisCache()
        mock_client = MagicMock()
        mock_client.get.return_value = None
        cache._client = mock_client
        assert cache.get_sync("key") is None

    def test_set_sync(self):
        cache = RedisCache()
        mock_client = MagicMock()
        cache._client = mock_client
        cache.set_sync("key", {"data": "test"}, ttl=60)
        mock_client.setex.assert_called_once()

    def test_set_sync_default_ttl(self):
        cache = RedisCache(default_ttl=300)
        mock_client = MagicMock()
        cache._client = mock_client
        cache.set_sync("key", "val")
        mock_client.setex.assert_called_once()
