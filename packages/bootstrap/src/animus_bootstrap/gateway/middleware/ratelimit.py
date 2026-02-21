"""Rate-limiting middleware — token-bucket per sender."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class _Bucket:
    """Internal token bucket state for a single sender."""

    tokens: float
    last_refill: float


class RateLimiter:
    """Token-bucket rate limiter keyed by sender ID.

    Each sender gets *max_tokens* tokens that refill at *refill_rate*
    tokens per second.  Calling :meth:`check` consumes one token and
    returns ``True`` when the sender is within their budget.

    All public methods are thread-safe.
    """

    def __init__(self, max_tokens: int = 10, refill_rate: float = 1.0) -> None:
        self._max_tokens = max_tokens
        self._refill_rate = refill_rate
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, sender_id: str) -> bool:
        """Consume one token for *sender_id*.

        Returns ``True`` if the sender has remaining budget, ``False``
        if they are rate-limited.
        """
        with self._lock:
            bucket = self._get_or_create(sender_id)
            self._refill(bucket)

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True

            logger.debug("ratelimit: sender %s exhausted tokens", sender_id)
            return False

    def reset(self, sender_id: str) -> None:
        """Reset the token bucket for *sender_id* to full capacity."""
        with self._lock:
            self._buckets.pop(sender_id, None)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_tokens(self) -> int:
        """Maximum tokens per sender."""
        return self._max_tokens

    @property
    def refill_rate(self) -> float:
        """Token refill rate (tokens per second)."""
        return self._refill_rate

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_or_create(self, sender_id: str) -> _Bucket:
        if sender_id not in self._buckets:
            self._buckets[sender_id] = _Bucket(
                tokens=float(self._max_tokens),
                last_refill=time.monotonic(),
            )
        return self._buckets[sender_id]

    def _refill(self, bucket: _Bucket) -> None:
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        if elapsed > 0:
            bucket.tokens = min(
                float(self._max_tokens),
                bucket.tokens + elapsed * self._refill_rate,
            )
            bucket.last_refill = now
