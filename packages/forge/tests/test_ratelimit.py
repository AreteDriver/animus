"""Tests for rate limiting module."""

import sys
import time

import pytest

sys.path.insert(0, "src")

from animus_forge.ratelimit.limiter import (
    RateLimitConfig,
    RateLimitExceeded,
    SlidingWindowLimiter,
    TokenBucketLimiter,
)
from animus_forge.ratelimit.provider import (
    ProviderLimitConfig,
    ProviderRateLimiter,
    configure_provider_limits,
    get_provider_limiter,
    get_quota_manager,
    reset_limiter,
)
from animus_forge.ratelimit.quota import (
    QuotaConfig,
    QuotaExceeded,
    QuotaManager,
    QuotaPeriod,
    QuotaUsage,
)


class TestTokenBucketLimiter:
    """Tests for TokenBucketLimiter."""

    def test_initial_burst(self):
        """Can burst up to capacity."""
        config = RateLimitConfig(
            requests_per_second=1,
            burst_size=10,
        )
        limiter = TokenBucketLimiter(config)

        # Should be able to acquire burst_size tokens immediately
        for i in range(10):
            assert limiter.acquire(wait=False) is True

        # Next one should fail without wait
        assert limiter.acquire(wait=False) is False

    def test_refill_rate(self):
        """Tokens refill at configured rate."""
        config = RateLimitConfig(
            requests_per_second=100,  # Fast for testing
            burst_size=5,
        )
        limiter = TokenBucketLimiter(config)

        # Exhaust tokens
        for _ in range(5):
            limiter.acquire(wait=False)

        # Wait a bit for refill
        time.sleep(0.05)  # 5 tokens at 100/s

        # Should have some tokens now
        assert limiter.acquire(wait=False) is True

    def test_wait_for_tokens(self):
        """Can wait for tokens to become available."""
        config = RateLimitConfig(
            requests_per_second=100,
            burst_size=1,
            max_wait_seconds=1.0,
        )
        limiter = TokenBucketLimiter(config)

        # Exhaust tokens
        limiter.acquire(wait=False)

        # Should wait and succeed
        start = time.monotonic()
        assert limiter.acquire(wait=True) is True
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # Should be fast at 100/s

    def test_max_wait_exceeded(self):
        """Raises exception when max wait exceeded."""
        config = RateLimitConfig(
            requests_per_second=0.1,  # Very slow
            burst_size=1,
            max_wait_seconds=0.01,  # Very short
        )
        limiter = TokenBucketLimiter(config)

        # Exhaust tokens
        limiter.acquire(wait=False)

        # Should raise
        with pytest.raises(RateLimitExceeded) as exc:
            limiter.acquire(wait=True)

        assert exc.value.retry_after is not None

    def test_stats(self):
        """Statistics are tracked correctly."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=5,
            name="test",
        )
        limiter = TokenBucketLimiter(config)

        limiter.acquire(wait=False)
        limiter.acquire(wait=False)
        limiter.acquire(wait=False)

        stats = limiter.get_stats()
        assert stats["name"] == "test"
        assert stats["type"] == "token_bucket"
        assert stats["total_acquired"] == 3

    @pytest.mark.asyncio
    async def test_async_acquire(self):
        """Async acquire works correctly."""
        config = RateLimitConfig(
            requests_per_second=100,
            burst_size=5,
        )
        limiter = TokenBucketLimiter(config)

        # Should succeed
        assert await limiter.acquire_async(wait=False) is True
        assert await limiter.acquire_async(wait=False) is True


class TestSlidingWindowLimiter:
    """Tests for SlidingWindowLimiter."""

    def test_basic_limit(self):
        """Respects basic request limit."""
        limiter = SlidingWindowLimiter(
            requests_per_window=3,
            window_seconds=60,
        )

        assert limiter.acquire(wait=False) is True
        assert limiter.acquire(wait=False) is True
        assert limiter.acquire(wait=False) is True
        assert limiter.acquire(wait=False) is False

    def test_window_expiry(self):
        """Old requests expire from window."""
        limiter = SlidingWindowLimiter(
            requests_per_window=2,
            window_seconds=0.05,  # 50ms window
        )

        limiter.acquire(wait=False)
        limiter.acquire(wait=False)
        assert limiter.acquire(wait=False) is False

        # Wait for window to expire
        time.sleep(0.06)

        # Should be able to acquire again
        assert limiter.acquire(wait=False) is True

    def test_stats(self):
        """Statistics are tracked."""
        limiter = SlidingWindowLimiter(
            requests_per_window=10,
            window_seconds=60,
            name="test-window",
        )

        limiter.acquire(wait=False)
        limiter.acquire(wait=False)

        stats = limiter.get_stats()
        assert stats["name"] == "test-window"
        assert stats["type"] == "sliding_window"
        assert stats["current_count"] == 2
        assert stats["total_acquired"] == 2


class TestQuotaUsage:
    """Tests for QuotaUsage."""

    def test_remaining_calculation(self):
        """Remaining is calculated correctly."""
        usage = QuotaUsage(
            period=QuotaPeriod.DAY,
            limit=100,
            used=30,
        )
        assert usage.remaining == 70

    def test_percent_used(self):
        """Percentage is calculated correctly."""
        usage = QuotaUsage(
            period=QuotaPeriod.DAY,
            limit=100,
            used=25,
        )
        assert usage.percent_used == 25.0

    def test_is_exceeded(self):
        """Exceeded check works."""
        usage = QuotaUsage(period=QuotaPeriod.DAY, limit=10, used=10)
        assert usage.is_exceeded() is True

        usage.used = 9
        assert usage.is_exceeded() is False


class TestQuotaManager:
    """Tests for QuotaManager."""

    @pytest.fixture
    def manager(self):
        """Create quota manager with test config."""
        manager = QuotaManager()
        manager.configure(
            QuotaConfig(
                provider="test",
                requests_per_minute=10,
                requests_per_day=100,
            )
        )
        return manager

    def test_check_within_limit(self, manager):
        """Check returns True within limit."""
        assert manager.check("test", 5) is True

    def test_check_exceeds_limit(self, manager):
        """Check returns False when limit exceeded."""
        # Use up quota
        for _ in range(10):
            manager.record_usage("test")

        assert manager.check("test") is False

    def test_acquire_success(self, manager):
        """Acquire succeeds within limit."""
        manager.acquire("test", 5)
        usage = manager.get_usage("test")
        assert usage["periods"]["minute"]["used"] == 5

    def test_acquire_raises_on_exceed(self, manager):
        """Acquire raises when quota exceeded."""
        # Fill quota
        for _ in range(10):
            manager.acquire("test")

        with pytest.raises(QuotaExceeded) as exc:
            manager.acquire("test")

        assert exc.value.provider == "test"
        assert exc.value.limit == 10

    def test_get_usage(self, manager):
        """Usage stats are returned correctly."""
        manager.record_usage("test", 3)
        usage = manager.get_usage("test")

        assert usage["provider"] == "test"
        assert usage["configured"] is True
        assert "minute" in usage["periods"]
        assert usage["periods"]["minute"]["used"] == 3

    def test_reset(self, manager):
        """Reset clears usage."""
        manager.record_usage("test", 5)
        manager.reset("test")

        usage = manager.get_usage("test")
        assert usage["periods"]["minute"]["used"] == 0

    def test_unconfigured_provider(self, manager):
        """Unconfigured providers pass through."""
        assert manager.check("unknown", 1000) is True


class TestProviderRateLimiter:
    """Tests for ProviderRateLimiter."""

    @pytest.fixture
    def limiter(self):
        """Create test provider limiter."""
        config = ProviderLimitConfig(
            provider="test-provider",
            requests_per_second=100,
            burst_size=10,
            requests_per_minute=100,
        )
        return ProviderRateLimiter(config)

    def test_acquire_success(self, limiter):
        """Acquire succeeds."""
        assert limiter.acquire(wait=False) is True

    def test_acquire_with_quota(self, limiter):
        """Acquire checks quota when manager provided."""
        quota_manager = QuotaManager()
        quota_manager.configure(QuotaConfig(provider="test-provider", requests_per_minute=5))

        # Should succeed until quota exhausted
        for _ in range(5):
            limiter.acquire(quota_manager=quota_manager)

        # Should fail on quota
        with pytest.raises(QuotaExceeded):
            limiter.acquire(quota_manager=quota_manager)

    def test_try_acquire(self, limiter):
        """try_acquire doesn't raise."""
        # Exhaust rate limit
        for _ in range(10):
            limiter.try_acquire()

        # Should return False, not raise
        assert limiter.try_acquire() is False

    def test_stats(self, limiter):
        """Stats are tracked."""
        limiter.acquire(wait=False)
        limiter.acquire(wait=False)

        stats = limiter.get_stats()
        assert stats["provider"] == "test-provider"
        assert stats["total_requests"] == 2


class TestProviderModule:
    """Tests for provider module functions."""

    def setup_method(self):
        """Reset limiters before each test."""
        reset_limiter("openai")
        reset_limiter("anthropic")
        reset_limiter("test")

    def test_get_provider_limiter_creates(self):
        """get_provider_limiter creates limiter with defaults."""
        limiter = get_provider_limiter("openai")
        assert limiter.provider == "openai"
        assert limiter.config.requests_per_second == 60

    def test_get_provider_limiter_caches(self):
        """Same limiter instance is returned."""
        limiter1 = get_provider_limiter("openai")
        limiter2 = get_provider_limiter("openai")
        assert limiter1 is limiter2

    def test_configure_provider_limits(self):
        """Custom limits can be configured."""
        limiter = configure_provider_limits(
            "test",
            requests_per_second=5,
            burst_size=3,
            requests_per_day=500,
        )
        assert limiter.config.requests_per_second == 5
        assert limiter.config.burst_size == 3
        assert limiter.config.requests_per_day == 500

    def test_get_quota_manager_singleton(self):
        """Quota manager is singleton."""
        manager1 = get_quota_manager()
        manager2 = get_quota_manager()
        assert manager1 is manager2


class TestAsyncRateLimiting:
    """Tests for async rate limiting."""

    @pytest.mark.asyncio
    async def test_provider_limiter_async(self):
        """Provider limiter async acquire works."""
        config = ProviderLimitConfig(
            provider="async-test",
            requests_per_second=100,
            burst_size=5,
        )
        limiter = ProviderRateLimiter(config)

        assert await limiter.acquire_async(wait=False) is True
        assert await limiter.acquire_async(wait=False) is True

    @pytest.mark.asyncio
    async def test_token_bucket_async_wait(self):
        """Token bucket async waits correctly."""
        config = RateLimitConfig(
            requests_per_second=100,
            burst_size=1,
            max_wait_seconds=1.0,
        )
        limiter = TokenBucketLimiter(config)

        # Exhaust
        await limiter.acquire_async(wait=False)

        # Should wait and succeed
        assert await limiter.acquire_async(wait=True) is True
