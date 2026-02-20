"""Tests for caching module."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from animus_forge.cache.backends import (
    CacheEntry,
    CacheStats,
    MemoryCache,
    get_cache,
    make_cache_key,
    reset_cache,
)
from animus_forge.cache.decorators import CacheAside, async_cached, cached


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_not_expired_when_no_expiry(self):
        """Entry without expiry is never expired."""
        entry = CacheEntry(value="test", expires_at=None)
        assert not entry.is_expired()

    def test_not_expired_before_time(self):
        """Entry is not expired before expiry time."""
        entry = CacheEntry(value="test", expires_at=time.time() + 100)
        assert not entry.is_expired()

    def test_expired_after_time(self):
        """Entry is expired after expiry time."""
        entry = CacheEntry(value="test", expires_at=time.time() - 1)
        assert entry.is_expired()


class TestMemoryCache:
    """Tests for MemoryCache."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache instance."""
        return MemoryCache(max_size=10)

    def test_set_and_get(self, cache):
        """Basic set and get operations."""
        cache.set_sync("key", "value")
        assert cache.get_sync("key") == "value"

    def test_get_missing_key(self, cache):
        """Getting missing key returns None."""
        assert cache.get_sync("nonexistent") is None

    def test_set_with_ttl(self, cache):
        """Entry expires after TTL."""
        cache.set_sync("key", "value", ttl=1)
        assert cache.get_sync("key") == "value"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get_sync("key") is None

    def test_async_operations(self, cache):
        """Async get and set work correctly."""

        async def _test():
            await cache.set("async_key", {"data": 123})
            result = await cache.get("async_key")
            assert result == {"data": 123}

        asyncio.run(_test())

    def test_delete(self, cache):
        """Delete removes key from cache."""

        async def _test():
            await cache.set("key", "value")
            assert await cache.exists("key")

            result = await cache.delete("key")
            assert result is True
            assert not await cache.exists("key")

        asyncio.run(_test())

    def test_delete_nonexistent(self, cache):
        """Deleting nonexistent key returns False."""

        async def _test():
            result = await cache.delete("nonexistent")
            assert result is False

        asyncio.run(_test())

    def test_clear(self, cache):
        """Clear removes all entries."""

        async def _test():
            await cache.set("key1", "value1")
            await cache.set("key2", "value2")
            assert cache.size == 2

            await cache.clear()
            assert cache.size == 0

        asyncio.run(_test())

    def test_max_size_eviction(self, cache):
        """Cache evicts oldest when max size reached."""
        for i in range(15):  # Max size is 10
            cache.set_sync(f"key{i}", f"value{i}")

        assert cache.size <= 10
        # First entries should be evicted
        assert cache.get_sync("key0") is None
        # Recent entries should remain
        assert cache.get_sync("key14") == "value14"

    def test_stats_tracking(self, cache):
        """Cache tracks hit/miss statistics."""
        cache.set_sync("key", "value")

        # Hit
        cache.get_sync("key")
        assert cache.stats.hits == 1
        assert cache.stats.misses == 0

        # Miss
        cache.get_sync("missing")
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1

        assert cache.stats.hit_rate == 50.0


class TestCacheStats:
    """Tests for CacheStats."""

    def test_hit_rate_calculation(self):
        """Hit rate calculated correctly."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 80.0

    def test_hit_rate_no_accesses(self):
        """Hit rate is 0 with no accesses."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0


class TestGetCache:
    """Tests for get_cache factory."""

    def setup_method(self):
        """Reset cache before each test."""
        reset_cache()

    def teardown_method(self):
        """Reset cache after each test."""
        reset_cache()

    def test_returns_memory_cache_by_default(self):
        """Returns MemoryCache when no Redis URL configured."""
        with patch.dict("os.environ", {}, clear=True):
            cache = get_cache()
            assert isinstance(cache, MemoryCache)

    def test_same_instance_returned(self):
        """Same cache instance returned on subsequent calls."""
        cache1 = get_cache()
        cache2 = get_cache()
        assert cache1 is cache2


class TestMakeCacheKey:
    """Tests for make_cache_key utility."""

    def test_simple_args(self):
        """Creates key from simple arguments."""
        key = make_cache_key("user", 123)
        assert key == "user:123"

    def test_kwargs(self):
        """Creates key from kwargs."""
        key = make_cache_key(name="test", id=42)
        assert "id=42" in key
        assert "name=test" in key

    def test_complex_objects_hashed(self):
        """Complex objects are hashed."""
        key = make_cache_key({"nested": "data"})
        assert len(key) > 0
        # Hash should be deterministic
        key2 = make_cache_key({"nested": "data"})
        assert key == key2


class TestCachedDecorator:
    """Tests for @cached decorator."""

    def setup_method(self):
        """Reset cache before each test."""
        reset_cache()

    def test_caches_result(self):
        """Function result is cached."""
        call_count = 0

        @cached()
        def expensive_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call computes
        result1 = expensive_func(5)
        assert result1 == 10
        assert call_count == 1

        # Second call uses cache
        result2 = expensive_func(5)
        assert result2 == 10
        assert call_count == 1  # Not called again

    def test_different_args_different_cache(self):
        """Different arguments use different cache entries."""
        call_count = 0

        @cached()
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        func(2)
        func(1)  # Should hit cache

        assert call_count == 2  # Only 2 unique calls

    def test_prefix_namespacing(self):
        """Prefix creates namespace for cache keys."""

        @cached(prefix="ns1")
        def func1(x: int) -> str:
            return f"func1:{x}"

        @cached(prefix="ns2")
        def func2(x: int) -> str:
            return f"func2:{x}"

        result1 = func1(1)
        result2 = func2(1)

        assert result1 == "func1:1"
        assert result2 == "func2:1"

    def test_skip_cache_on_condition(self):
        """Result not cached when skip_cache_on returns True."""
        call_count = 0

        @cached(skip_cache_on=lambda x: x is None)
        def maybe_none(x: int):
            nonlocal call_count
            call_count += 1
            return None if x < 0 else x

        # None result should not be cached
        maybe_none(-1)
        maybe_none(-1)
        assert call_count == 2  # Called twice

        # Non-None result should be cached
        maybe_none(1)
        maybe_none(1)
        assert call_count == 3  # Only called once for x=1

    def test_custom_key_builder(self):
        """Custom key builder used for cache key."""

        @cached(key_builder=lambda x, y: f"custom:{x}:{y}")
        def func(x: int, y: int) -> int:
            return x + y

        result = func(1, 2)
        assert result == 3

        # Verify key was built correctly
        key = func.cache_key(1, 2)
        assert key == "custom:1:2"


class TestAsyncCachedDecorator:
    """Tests for @async_cached decorator."""

    def setup_method(self):
        """Reset cache before each test."""
        reset_cache()

    def test_caches_async_result(self):
        """Async function result is cached."""
        call_count = 0

        @async_cached()
        async def expensive_async(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 2

        async def _test():
            nonlocal call_count

            result1 = await expensive_async(5)
            assert result1 == 10
            assert call_count == 1

            result2 = await expensive_async(5)
            assert result2 == 10
            assert call_count == 1

        asyncio.run(_test())

    def test_ttl_expiration(self):
        """Cache entry expires after TTL."""
        call_count = 0

        @async_cached(ttl=1)
        async def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        async def _test():
            nonlocal call_count

            await func(1)
            assert call_count == 1

            await asyncio.sleep(1.1)

            await func(1)
            assert call_count == 2

        asyncio.run(_test())


class TestCacheAside:
    """Tests for CacheAside pattern helper."""

    def setup_method(self):
        """Reset cache before each test."""
        reset_cache()

    def test_basic_usage(self):
        """Basic get/set operations work."""

        async def _test():
            cache_aside = CacheAside(prefix="test", ttl=60)
            await cache_aside.set("key", {"value": 123})
            result = await cache_aside.get("key")
            assert result == {"value": 123}

        asyncio.run(_test())

    def test_invalidate(self):
        """Invalidate removes cache entry."""

        async def _test():
            cache_aside = CacheAside(prefix="test")
            await cache_aside.set("key", "value")
            await cache_aside.invalidate("key")
            result = await cache_aside.get("key")
            assert result is None

        asyncio.run(_test())

    def test_sync_operations(self):
        """Synchronous operations work."""
        cache_aside = CacheAside(prefix="sync")

        cache_aside.set_sync("key", "value")
        result = cache_aside.get_sync("key")
        assert result == "value"

    def test_delete(self):
        """Delete removes cache entry and returns True."""

        async def _test():
            cache_aside = CacheAside(prefix="del_test")
            await cache_aside.set("key", "value")
            result = await cache_aside.delete("key")
            assert result is True
            assert await cache_aside.get("key") is None

        asyncio.run(_test())

    def test_set_with_custom_ttl(self):
        """Set with explicit TTL overrides default."""

        async def _test():
            cache_aside = CacheAside(prefix="ttl_test", ttl=3600)
            await cache_aside.set("key", "value", ttl=1)
            assert await cache_aside.get("key") == "value"

        asyncio.run(_test())

    def test_make_key(self):
        """Internal key includes prefix."""
        cache_aside = CacheAside(prefix="ns")
        assert cache_aside._make_key("foo") == "ns:foo"

    def test_get_miss(self):
        """Getting nonexistent key returns None."""

        async def _test():
            cache_aside = CacheAside(prefix="miss_test")
            assert await cache_aside.get("nonexistent") is None

        asyncio.run(_test())


class TestMemoryCacheAdditional:
    """Additional edge case tests for MemoryCache."""

    def test_default_ttl_applied(self):
        """Default TTL is applied when no explicit TTL given."""
        cache = MemoryCache(default_ttl=1)
        cache.set_sync("key", "value")
        assert cache.get_sync("key") == "value"
        time.sleep(1.1)
        assert cache.get_sync("key") is None

    def test_explicit_ttl_overrides_default(self):
        """Explicit TTL overrides default_ttl."""
        cache = MemoryCache(default_ttl=1)
        cache.set_sync("key", "value", ttl=3600)
        time.sleep(1.1)
        # Should still be alive because explicit ttl is 3600
        assert cache.get_sync("key") == "value"

    def test_no_ttl_no_default_ttl(self):
        """Without TTL or default, entries never expire."""
        cache = MemoryCache()
        cache.set_sync("key", "value")
        entry = cache._cache["key"]
        assert entry.expires_at is None

    def test_update_existing_key(self):
        """Setting existing key updates value."""
        cache = MemoryCache(max_size=5)
        cache.set_sync("key", "old")
        cache.set_sync("key", "new")
        assert cache.get_sync("key") == "new"
        assert cache.size == 1

    def test_update_existing_key_does_not_evict(self):
        """Updating existing key at capacity does not evict."""
        cache = MemoryCache(max_size=2)
        cache.set_sync("a", 1)
        cache.set_sync("b", 2)
        cache.set_sync("a", 10)  # update, not new
        assert cache.size == 2
        assert cache.get_sync("a") == 10
        assert cache.get_sync("b") == 2

    def test_exists_for_valid_key(self):
        """exists returns True for valid, non-expired key."""

        async def _test():
            cache = MemoryCache()
            cache.set_sync("key", "value")
            assert await cache.exists("key") is True

        asyncio.run(_test())

    def test_stats_reset_on_clear(self):
        """Stats are reset when cache is cleared."""
        cache = MemoryCache()
        cache.set_sync("k", "v")
        cache.get_sync("k")  # hit
        assert cache.stats.hits == 1

        asyncio.run(cache.clear())
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0

    def test_cleanup_expired_removes_all(self):
        """_cleanup_expired removes all expired entries."""
        cache = MemoryCache()
        cache._cache["a"] = CacheEntry(value=1, expires_at=time.time() - 1)
        cache._cache["b"] = CacheEntry(value=2, expires_at=time.time() - 1)
        cache._cache["c"] = CacheEntry(value=3, expires_at=time.time() + 3600)
        cache._cleanup_expired()
        assert "a" not in cache._cache
        assert "b" not in cache._cache
        assert "c" in cache._cache

    def test_evict_oldest_empty_cache(self):
        """_evict_oldest on empty cache is a no-op."""
        cache = MemoryCache()
        cache._evict_oldest()  # Should not raise


class TestGetCacheAdditional:
    """Additional tests for get_cache and _create_cache."""

    def setup_method(self):
        reset_cache()

    def teardown_method(self):
        reset_cache()

    def test_redis_url_without_redis_installed(self):
        """Falls back to MemoryCache when redis_url set but redis not installed."""
        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"
        with patch("animus_forge.config.settings.get_settings", return_value=mock_settings):
            with patch("importlib.util.find_spec", return_value=None):
                reset_cache()
                cache = get_cache()
                assert isinstance(cache, MemoryCache)

    def test_reset_cache_clears_singleton(self):
        """reset_cache allows new instance creation."""
        c1 = get_cache()
        reset_cache()
        c2 = get_cache()
        assert c1 is not c2


class TestCachedDecoratorAdditional:
    """Additional tests for @cached decorator."""

    def setup_method(self):
        reset_cache()

    def test_cache_clear_method(self):
        """cached function has cache_clear method."""

        @cached(prefix="test_clear")
        def func(x: int) -> int:
            return x

        func(1)
        # cache_clear should be callable without error
        func.cache_clear()

    def test_cache_key_method(self):
        """cached function has cache_key method."""

        @cached(prefix="test_key")
        def func(x: int, y: int = 0) -> int:
            return x + y

        key = func.cache_key(1, y=2)
        assert "test_key:" in key

    def test_cache_key_without_prefix(self):
        """cache_key works without prefix."""

        @cached()
        def func(x: int) -> int:
            return x

        key = func.cache_key(42)
        assert "func:" in key


class TestAsyncCachedDecoratorAdditional:
    """Additional tests for @async_cached decorator."""

    def setup_method(self):
        reset_cache()

    def test_with_prefix(self):
        """async_cached with prefix namespaces correctly."""
        call_count = 0

        @async_cached(prefix="async_ns")
        async def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 3

        async def _test():
            nonlocal call_count

            result = await func(5)
            assert result == 15
            assert call_count == 1

            result = await func(5)
            assert result == 15
            assert call_count == 1

        asyncio.run(_test())

    def test_skip_cache_on(self):
        """async_cached skips caching when condition met."""
        call_count = 0

        @async_cached(skip_cache_on=lambda x: x is None)
        async def func(x: int):
            nonlocal call_count
            call_count += 1
            return None if x < 0 else x

        async def _test():
            nonlocal call_count
            await func(-1)
            await func(-1)
            assert call_count == 2

        asyncio.run(_test())

    def test_custom_key_builder(self):
        """async_cached with custom key builder."""

        @async_cached(key_builder=lambda x: f"async:{x}")
        async def func(x: int) -> int:
            return x

        async def _test():
            result = await func(1)
            assert result == 1
            key = func.cache_key(1)
            assert key == "async:1"

        asyncio.run(_test())

    def test_cache_clear_method(self):
        """async_cached function has cache_clear method."""

        @async_cached(prefix="async_clear")
        async def func(x: int) -> int:
            return x

        async def _test():
            await func(1)
            # cache_clear creates a task - should not raise
            func.cache_clear()

        asyncio.run(_test())


class TestMakeCacheKeyAdditional:
    """Additional tests for make_cache_key utility."""

    def test_mixed_args_and_kwargs(self):
        """Key with both args and kwargs."""
        key = make_cache_key("prefix", 42, name="test", flag=True)
        assert "prefix" in key
        assert "42" in key
        assert "name=test" in key
        assert "flag=True" in key

    def test_complex_kwarg_hashed(self):
        """Complex kwargs are hashed."""
        key = make_cache_key(data={"nested": [1, 2, 3]})
        assert "data=" in key
        # Should be deterministic
        key2 = make_cache_key(data={"nested": [1, 2, 3]})
        assert key == key2

    def test_bool_arg(self):
        """Boolean args work."""
        key = make_cache_key(True, False)
        assert "True" in key
        assert "False" in key

    def test_empty_args(self):
        """Empty args produce empty key."""
        key = make_cache_key()
        assert key == ""
