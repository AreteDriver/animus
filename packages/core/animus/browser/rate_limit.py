"""Domain-aware rate limiting for browser fetches.

Integrates with PolicyDecisionPoint when available; falls back to a
simple in-memory token bucket.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from animus.logging import get_logger

logger = get_logger("browser.rate_limit")


class RateLimitedDomain:
    """Token-bucket rate limiter per domain.

    Default: 60 requests / minute per domain (matches turbowebfetch).
    """

    def __init__(self, requests_per_minute: int = 60, burst: int = 5) -> None:
        self.rate = requests_per_minute / 60.0
        self.burst = burst
        self._tokens: dict[str, float] = defaultdict(lambda: float(burst))
        self._last: dict[str, float] = defaultdict(float)
        self._lock = asyncio.Lock()

    async def acquire(self, url: str) -> None:
        """Block until a token is available for *url*'s domain."""
        from urllib.parse import urlparse

        domain = urlparse(url).netloc or "default"
        now = time.monotonic()

        async with self._lock:
            elapsed = now - self._last[domain]
            self._tokens[domain] = min(
                self.burst,
                self._tokens[domain] + elapsed * self.rate,
            )
            self._last[domain] = now

            if self._tokens[domain] < 1.0:
                wait = (1.0 - self._tokens[domain]) / self.rate
                logger.debug("Rate limit: waiting %.2fs for %s", wait, domain)
                await asyncio.sleep(wait)
                self._tokens[domain] = 1.0

            self._tokens[domain] -= 1.0
