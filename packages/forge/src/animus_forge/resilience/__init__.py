"""Resilience patterns for fault-tolerant systems.

Provides patterns for building resilient applications:
- Bulkhead: Isolates resources to prevent cascading failures
- Fallback: Provides alternative paths when primary fails
- Concurrency: Semaphore-based limits for parallel execution
- Integration with circuit breakers and rate limiters
"""

from animus_forge.resilience.bulkhead import (
    Bulkhead,
    BulkheadConfig,
    BulkheadFull,
    get_bulkhead,
)
from animus_forge.resilience.concurrency import (
    ConcurrencyLimiter,
    ConcurrencyStats,
    get_all_limiter_stats,
    get_limiter,
    limit_async,
    limit_concurrency,
    limit_sync,
)
from animus_forge.resilience.fallback import (
    FallbackChain,
    FallbackConfig,
    FallbackResult,
    fallback,
)

__all__ = [
    # Bulkhead
    "Bulkhead",
    "BulkheadConfig",
    "BulkheadFull",
    "get_bulkhead",
    # Fallback
    "FallbackChain",
    "FallbackConfig",
    "FallbackResult",
    "fallback",
    # Concurrency
    "ConcurrencyLimiter",
    "ConcurrencyStats",
    "get_limiter",
    "get_all_limiter_stats",
    "limit_concurrency",
    "limit_async",
    "limit_sync",
]
