"""Additional coverage tests for rate limiters."""

import asyncio
import sys

import pytest

sys.path.insert(0, "src")

from animus_forge.ratelimit.limiter import (
    RateLimitConfig,
    RateLimitExceeded,
    SlidingWindowLimiter,
    TokenBucketLimiter,
)


class TestRateLimitExceeded:
    def test_attributes(self):
        exc = RateLimitExceeded("test", retry_after=5.0, limit_name="api")
        assert str(exc) == "test"
        assert exc.retry_after == 5.0
        assert exc.limit_name == "api"

    def test_defaults(self):
        exc = RateLimitExceeded()
        assert exc.retry_after is None
        assert exc.limit_name == ""


class TestTokenBucketLimiter:
    def test_acquire_no_wait(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=5)
        limiter = TokenBucketLimiter(config)
        # Drain all tokens
        for _ in range(5):
            assert limiter.acquire(wait=False) is True
        # Should fail without waiting
        assert limiter.acquire(wait=False) is False

    def test_acquire_exceeds_max_wait(self):
        config = RateLimitConfig(requests_per_second=0.1, burst_size=1, max_wait_seconds=0.01)
        limiter = TokenBucketLimiter(config)
        limiter.acquire()  # Use the one token
        with pytest.raises(RateLimitExceeded):
            limiter.acquire(tokens=5, wait=True)

    def test_get_stats(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=20, name="test-limiter")
        limiter = TokenBucketLimiter(config)
        limiter.acquire(tokens=3)
        stats = limiter.get_stats()
        assert stats["name"] == "test-limiter"
        assert stats["type"] == "token_bucket"
        assert stats["total_acquired"] == 3
        assert stats["capacity"] == 20

    def test_acquire_async(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=5)
        limiter = TokenBucketLimiter(config)

        async def _test():
            assert await limiter.acquire_async(wait=False) is True
            # Drain
            for _ in range(4):
                await limiter.acquire_async(wait=False)
            assert await limiter.acquire_async(wait=False) is False

        asyncio.run(_test())

    def test_acquire_async_exceeds_max_wait(self):
        config = RateLimitConfig(requests_per_second=0.1, burst_size=1, max_wait_seconds=0.01)
        limiter = TokenBucketLimiter(config)

        async def _test():
            await limiter.acquire_async()
            with pytest.raises(RateLimitExceeded):
                await limiter.acquire_async(tokens=5, wait=True)

        asyncio.run(_test())


class TestSlidingWindowLimiter:
    def test_acquire_within_limit(self):
        limiter = SlidingWindowLimiter(requests_per_window=5, window_seconds=1.0)
        for _ in range(5):
            assert limiter.acquire(wait=False) is True

    def test_acquire_exceeds_limit_no_wait(self):
        limiter = SlidingWindowLimiter(requests_per_window=2, window_seconds=1.0)
        limiter.acquire(wait=False)
        limiter.acquire(wait=False)
        assert limiter.acquire(wait=False) is False

    def test_acquire_exceeds_max_wait(self):
        limiter = SlidingWindowLimiter(
            requests_per_window=1, window_seconds=10.0, max_wait_seconds=0.01
        )
        limiter.acquire(wait=False)
        with pytest.raises(RateLimitExceeded):
            limiter.acquire(wait=True)

    def test_get_stats(self):
        limiter = SlidingWindowLimiter(requests_per_window=10, window_seconds=60.0, name="test-sw")
        limiter.acquire(tokens=3, wait=False)
        stats = limiter.get_stats()
        assert stats["name"] == "test-sw"
        assert stats["type"] == "sliding_window"
        assert stats["total_acquired"] == 3

    def test_acquire_async_within_limit(self):
        limiter = SlidingWindowLimiter(requests_per_window=3, window_seconds=1.0)

        async def _test():
            for _ in range(3):
                assert await limiter.acquire_async(wait=False) is True
            assert await limiter.acquire_async(wait=False) is False

        asyncio.run(_test())

    def test_acquire_async_exceeds_max_wait(self):
        limiter = SlidingWindowLimiter(
            requests_per_window=1, window_seconds=10.0, max_wait_seconds=0.01
        )

        async def _test():
            await limiter.acquire_async(wait=False)
            with pytest.raises(RateLimitExceeded):
                await limiter.acquire_async(wait=True)

        asyncio.run(_test())

    def test_time_until_slot_empty(self):
        limiter = SlidingWindowLimiter(requests_per_window=5, window_seconds=1.0)
        assert limiter._time_until_slot() == 0.0
