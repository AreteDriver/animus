"""Provider-specific rate limiters.

Combines rate limiting and quota management for each API provider.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from animus_forge.ratelimit.limiter import (
    RateLimitConfig,
    RateLimitExceeded,
    TokenBucketLimiter,
)
from animus_forge.ratelimit.quota import (
    QuotaConfig,
    QuotaExceeded,
    QuotaManager,
)

logger = logging.getLogger(__name__)

# Global instances
_provider_limiters: dict[str, ProviderRateLimiter] = {}
_quota_manager: QuotaManager | None = None


# Default rate limits for common providers
DEFAULT_PROVIDER_LIMITS = {
    "openai": {
        "requests_per_second": 60,  # Tier 1: 60 RPM = 1 RPS
        "burst_size": 20,
        "requests_per_minute": 60,
        "requests_per_day": 10000,
        "tokens_per_minute": 60000,
    },
    "anthropic": {
        "requests_per_second": 50,  # Varies by tier
        "burst_size": 15,
        "requests_per_minute": 50,
        "requests_per_day": 10000,
        "tokens_per_minute": 40000,
    },
    "github": {
        "requests_per_second": 1.67,  # 5000/hour = ~1.67/s
        "burst_size": 10,
        "requests_per_hour": 5000,
    },
    "notion": {
        "requests_per_second": 3,  # 3 requests per second
        "burst_size": 5,
        "requests_per_minute": 180,
    },
    "gmail": {
        "requests_per_second": 1,  # Conservative
        "burst_size": 5,
        "requests_per_day": 1000,
    },
}


@dataclass
class ProviderLimitConfig:
    """Configuration for provider rate limits."""

    provider: str
    requests_per_second: float = 10.0
    burst_size: int = 20
    max_wait_seconds: float = 30.0
    requests_per_minute: int | None = None
    requests_per_hour: int | None = None
    requests_per_day: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_day: int | None = None


class ProviderRateLimiter:
    """Rate limiter for a specific API provider.

    Combines token bucket rate limiting with quota management.
    """

    def __init__(self, config: ProviderLimitConfig):
        """Initialize provider rate limiter.

        Args:
            config: Provider limit configuration
        """
        self.config = config
        self.provider = config.provider

        # Create rate limiter
        self._limiter = TokenBucketLimiter(
            RateLimitConfig(
                requests_per_second=config.requests_per_second,
                burst_size=config.burst_size,
                max_wait_seconds=config.max_wait_seconds,
                name=config.provider,
            )
        )

        # Create quota config
        self._quota_config = QuotaConfig(
            provider=config.provider,
            requests_per_minute=config.requests_per_minute,
            requests_per_hour=config.requests_per_hour,
            requests_per_day=config.requests_per_day,
            tokens_per_minute=config.tokens_per_minute,
            tokens_per_day=config.tokens_per_day,
        )

        # Stats
        self._total_requests = 0
        self._total_tokens = 0

    def acquire(
        self,
        tokens: int = 1,
        wait: bool = True,
        quota_manager: QuotaManager | None = None,
    ) -> bool:
        """Acquire rate limit and quota.

        Args:
            tokens: Number of tokens/requests
            wait: Whether to wait for rate limit
            quota_manager: Optional quota manager for tracking

        Returns:
            True if acquired

        Raises:
            RateLimitExceeded: If rate limit exceeded and wait=True times out
            QuotaExceeded: If quota is exceeded
        """
        # Check quota first (no waiting)
        if quota_manager:
            quota_manager.acquire(self.provider, tokens)

        # Then check rate limit (may wait)
        result = self._limiter.acquire(tokens, wait)

        if result:
            self._total_requests += 1
            self._total_tokens += tokens

        return result

    async def acquire_async(
        self,
        tokens: int = 1,
        wait: bool = True,
        quota_manager: QuotaManager | None = None,
    ) -> bool:
        """Async version of acquire."""
        # Check quota first
        if quota_manager:
            quota_manager.acquire(self.provider, tokens)

        # Then check rate limit
        result = await self._limiter.acquire_async(tokens, wait)

        if result:
            self._total_requests += 1
            self._total_tokens += tokens

        return result

    def try_acquire(
        self,
        tokens: int = 1,
        quota_manager: QuotaManager | None = None,
    ) -> bool:
        """Try to acquire without waiting.

        Args:
            tokens: Number of tokens/requests
            quota_manager: Optional quota manager

        Returns:
            True if acquired, False otherwise (no exception)
        """
        try:
            # Check quota
            if quota_manager and not quota_manager.check(self.provider, tokens):
                return False

            # Check rate limit without waiting
            return self._limiter.acquire(tokens, wait=False)

        except (RateLimitExceeded, QuotaExceeded):
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get limiter statistics."""
        limiter_stats = self._limiter.get_stats()
        return {
            "provider": self.provider,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "rate_limiter": limiter_stats,
            "config": {
                "requests_per_second": self.config.requests_per_second,
                "burst_size": self.config.burst_size,
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "requests_per_day": self.config.requests_per_day,
            },
        }


def get_quota_manager() -> QuotaManager:
    """Get or create global quota manager."""
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = QuotaManager()
    return _quota_manager


def get_provider_limiter(provider: str) -> ProviderRateLimiter:
    """Get or create rate limiter for a provider.

    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')

    Returns:
        Provider rate limiter
    """
    global _provider_limiters

    provider_lower = provider.lower()

    if provider_lower not in _provider_limiters:
        # Get default config or create basic one
        defaults = DEFAULT_PROVIDER_LIMITS.get(provider_lower, {})

        config = ProviderLimitConfig(
            provider=provider_lower,
            requests_per_second=defaults.get("requests_per_second", 10.0),
            burst_size=defaults.get("burst_size", 20),
            requests_per_minute=defaults.get("requests_per_minute"),
            requests_per_hour=defaults.get("requests_per_hour"),
            requests_per_day=defaults.get("requests_per_day"),
            tokens_per_minute=defaults.get("tokens_per_minute"),
            tokens_per_day=defaults.get("tokens_per_day"),
        )

        _provider_limiters[provider_lower] = ProviderRateLimiter(config)

        # Configure quota manager
        quota_manager = get_quota_manager()
        quota_manager.configure(
            QuotaConfig(
                provider=provider_lower,
                requests_per_minute=config.requests_per_minute,
                requests_per_hour=config.requests_per_hour,
                requests_per_day=config.requests_per_day,
                tokens_per_minute=config.tokens_per_minute,
                tokens_per_day=config.tokens_per_day,
            )
        )

        logger.info(f"Created rate limiter for provider: {provider_lower}")

    return _provider_limiters[provider_lower]


def configure_provider_limits(
    provider: str,
    requests_per_second: float | None = None,
    burst_size: int | None = None,
    requests_per_minute: int | None = None,
    requests_per_hour: int | None = None,
    requests_per_day: int | None = None,
    tokens_per_minute: int | None = None,
    tokens_per_day: int | None = None,
) -> ProviderRateLimiter:
    """Configure custom limits for a provider.

    Args:
        provider: Provider name
        requests_per_second: Rate limit (RPS)
        burst_size: Maximum burst above steady rate
        requests_per_minute: Minute quota
        requests_per_hour: Hour quota
        requests_per_day: Day quota
        tokens_per_minute: Token limit per minute
        tokens_per_day: Token limit per day

    Returns:
        Configured provider limiter
    """
    global _provider_limiters

    provider_lower = provider.lower()

    # Get existing defaults or use basic defaults
    existing = DEFAULT_PROVIDER_LIMITS.get(provider_lower, {})

    config = ProviderLimitConfig(
        provider=provider_lower,
        requests_per_second=requests_per_second or existing.get("requests_per_second", 10.0),
        burst_size=burst_size or existing.get("burst_size", 20),
        requests_per_minute=requests_per_minute or existing.get("requests_per_minute"),
        requests_per_hour=requests_per_hour or existing.get("requests_per_hour"),
        requests_per_day=requests_per_day or existing.get("requests_per_day"),
        tokens_per_minute=tokens_per_minute or existing.get("tokens_per_minute"),
        tokens_per_day=tokens_per_day or existing.get("tokens_per_day"),
    )

    limiter = ProviderRateLimiter(config)
    _provider_limiters[provider_lower] = limiter

    # Update quota manager
    quota_manager = get_quota_manager()
    quota_manager.configure(
        QuotaConfig(
            provider=provider_lower,
            requests_per_minute=config.requests_per_minute,
            requests_per_hour=config.requests_per_hour,
            requests_per_day=config.requests_per_day,
            tokens_per_minute=config.tokens_per_minute,
            tokens_per_day=config.tokens_per_day,
        )
    )

    logger.info(f"Configured custom limits for provider: {provider_lower}")
    return limiter


def get_all_stats() -> dict[str, Any]:
    """Get statistics for all configured limiters."""
    quota_manager = get_quota_manager()

    return {
        "limiters": {
            provider: limiter.get_stats() for provider, limiter in _provider_limiters.items()
        },
        "quotas": quota_manager.get_all_usage(),
    }


def reset_limiter(provider: str) -> None:
    """Reset a provider's limiter (for testing)."""
    global _provider_limiters
    provider_lower = provider.lower()
    if provider_lower in _provider_limiters:
        del _provider_limiters[provider_lower]

    quota_manager = get_quota_manager()
    quota_manager.reset(provider_lower)
