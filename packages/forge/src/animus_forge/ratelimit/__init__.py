"""Rate limiting module for provider API calls.

Provides per-provider rate limiters with:
- Token bucket algorithm for smooth rate limiting
- Quota tracking (requests per minute/hour/day)
- Graceful backpressure and queue management
- Metrics and visibility
"""

from animus_forge.ratelimit.limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitExceeded,
    SlidingWindowLimiter,
    TokenBucketLimiter,
)
from animus_forge.ratelimit.provider import (
    ProviderRateLimiter,
    configure_provider_limits,
    get_provider_limiter,
)
from animus_forge.ratelimit.quota import (
    QuotaConfig,
    QuotaExceeded,
    QuotaManager,
    QuotaPeriod,
)

__all__ = [
    "RateLimiter",
    "TokenBucketLimiter",
    "SlidingWindowLimiter",
    "RateLimitConfig",
    "RateLimitExceeded",
    "QuotaManager",
    "QuotaConfig",
    "QuotaExceeded",
    "QuotaPeriod",
    "ProviderRateLimiter",
    "get_provider_limiter",
    "configure_provider_limits",
]
