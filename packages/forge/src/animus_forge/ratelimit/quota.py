"""Quota management for API usage tracking.

Tracks usage against daily/hourly/monthly quotas with persistence.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class QuotaExceeded(Exception):
    """Raised when quota is exceeded."""

    def __init__(
        self,
        message: str = "Quota exceeded",
        provider: str = "",
        period: str = "",
        limit: int = 0,
        used: int = 0,
        resets_at: datetime | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.period = period
        self.limit = limit
        self.used = used
        self.resets_at = resets_at


class QuotaPeriod(str, Enum):
    """Quota period types."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"


@dataclass
class QuotaConfig:
    """Configuration for quota limits."""

    provider: str
    requests_per_minute: int | None = None
    requests_per_hour: int | None = None
    requests_per_day: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_day: int | None = None
    cost_per_day_usd: float | None = None  # Cost limit in USD


@dataclass
class QuotaUsage:
    """Track usage for a specific period."""

    period: QuotaPeriod
    limit: int
    used: int = 0
    period_start: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def remaining(self) -> int:
        """Remaining quota."""
        return max(0, self.limit - self.used)

    @property
    def percent_used(self) -> float:
        """Percentage of quota used."""
        if self.limit == 0:
            return 100.0
        return (self.used / self.limit) * 100

    def is_exceeded(self) -> bool:
        """Check if quota is exceeded."""
        return self.used >= self.limit

    def reset_if_expired(self) -> bool:
        """Reset usage if period has expired.

        Returns:
            True if reset occurred
        """
        now = datetime.now(UTC)

        if self.period == QuotaPeriod.MINUTE:
            if (now - self.period_start).total_seconds() >= 60:
                self.used = 0
                self.period_start = now
                return True
        elif self.period == QuotaPeriod.HOUR:
            if (now - self.period_start).total_seconds() >= 3600:
                self.used = 0
                self.period_start = now
                return True
        elif self.period == QuotaPeriod.DAY:
            if self.period_start.date() != now.date():
                self.used = 0
                self.period_start = now
                return True
        elif self.period == QuotaPeriod.MONTH:
            if self.period_start.year != now.year or self.period_start.month != now.month:
                self.used = 0
                self.period_start = now
                return True

        return False

    def time_until_reset(self) -> float:
        """Seconds until quota resets."""
        now = datetime.now(UTC)
        elapsed = (now - self.period_start).total_seconds()

        if self.period == QuotaPeriod.MINUTE:
            return max(0, 60 - elapsed)
        elif self.period == QuotaPeriod.HOUR:
            return max(0, 3600 - elapsed)
        elif self.period == QuotaPeriod.DAY:
            # Time until midnight UTC
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if now.hour > 0 or now.minute > 0:
                from datetime import timedelta

                tomorrow += timedelta(days=1)
            return (tomorrow - now).total_seconds()
        elif self.period == QuotaPeriod.MONTH:
            # Time until first of next month
            if now.month == 12:
                next_month = now.replace(
                    year=now.year + 1,
                    month=1,
                    day=1,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            else:
                next_month = now.replace(
                    month=now.month + 1,
                    day=1,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            return (next_month - now).total_seconds()

        return 0


class QuotaManager:
    """Manages quota limits across multiple providers."""

    def __init__(self):
        """Initialize quota manager."""
        self._quotas: dict[str, dict[QuotaPeriod, QuotaUsage]] = {}
        self._configs: dict[str, QuotaConfig] = {}
        self._lock = threading.Lock()

    def configure(self, config: QuotaConfig) -> None:
        """Configure quotas for a provider.

        Args:
            config: Quota configuration
        """
        with self._lock:
            self._configs[config.provider] = config
            self._quotas[config.provider] = {}

            # Create usage trackers for configured limits
            if config.requests_per_minute:
                self._quotas[config.provider][QuotaPeriod.MINUTE] = QuotaUsage(
                    period=QuotaPeriod.MINUTE,
                    limit=config.requests_per_minute,
                )
            if config.requests_per_hour:
                self._quotas[config.provider][QuotaPeriod.HOUR] = QuotaUsage(
                    period=QuotaPeriod.HOUR,
                    limit=config.requests_per_hour,
                )
            if config.requests_per_day:
                self._quotas[config.provider][QuotaPeriod.DAY] = QuotaUsage(
                    period=QuotaPeriod.DAY,
                    limit=config.requests_per_day,
                )

            logger.info(f"Configured quotas for provider: {config.provider}")

    def check(self, provider: str, count: int = 1) -> bool:
        """Check if quota allows the request.

        Args:
            provider: Provider name
            count: Number of requests/tokens to check

        Returns:
            True if quota allows request
        """
        with self._lock:
            if provider not in self._quotas:
                return True  # No quota configured

            for usage in self._quotas[provider].values():
                usage.reset_if_expired()
                if usage.used + count > usage.limit:
                    return False

            return True

    def acquire(self, provider: str, count: int = 1) -> None:
        """Acquire quota (raises if exceeded).

        Args:
            provider: Provider name
            count: Number of requests/tokens

        Raises:
            QuotaExceeded: If quota is exceeded
        """
        with self._lock:
            if provider not in self._quotas:
                return  # No quota configured

            # Check all periods first
            for usage in self._quotas[provider].values():
                usage.reset_if_expired()
                if usage.used + count > usage.limit:
                    raise QuotaExceeded(
                        f"Quota exceeded for {provider} ({usage.period.value})",
                        provider=provider,
                        period=usage.period.value,
                        limit=usage.limit,
                        used=usage.used,
                        resets_at=datetime.now(UTC),
                    )

            # Record usage
            for usage in self._quotas[provider].values():
                usage.used += count

    def record_usage(self, provider: str, count: int = 1) -> None:
        """Record usage without checking limits.

        Use this to track usage for monitoring when limits are soft.

        Args:
            provider: Provider name
            count: Number of requests/tokens
        """
        with self._lock:
            if provider not in self._quotas:
                return

            for usage in self._quotas[provider].values():
                usage.reset_if_expired()
                usage.used += count

    def get_usage(self, provider: str) -> dict[str, Any]:
        """Get usage statistics for a provider.

        Args:
            provider: Provider name

        Returns:
            Usage statistics
        """
        with self._lock:
            if provider not in self._quotas:
                return {"provider": provider, "configured": False}

            periods = {}
            for period, usage in self._quotas[provider].items():
                usage.reset_if_expired()
                periods[period.value] = {
                    "limit": usage.limit,
                    "used": usage.used,
                    "remaining": usage.remaining,
                    "percent_used": round(usage.percent_used, 1),
                    "time_until_reset": round(usage.time_until_reset(), 0),
                }

            return {
                "provider": provider,
                "configured": True,
                "periods": periods,
            }

    def get_all_usage(self) -> dict[str, Any]:
        """Get usage for all providers.

        Returns:
            Dict of provider usage
        """
        return {provider: self.get_usage(provider) for provider in self._quotas.keys()}

    def reset(self, provider: str, period: QuotaPeriod | None = None) -> None:
        """Reset quota usage for a provider.

        Args:
            provider: Provider name
            period: Specific period to reset, or None for all
        """
        with self._lock:
            if provider not in self._quotas:
                return

            if period:
                if period in self._quotas[provider]:
                    self._quotas[provider][period].used = 0
                    self._quotas[provider][period].period_start = datetime.now(UTC)
            else:
                for usage in self._quotas[provider].values():
                    usage.used = 0
                    usage.period_start = datetime.now(UTC)

            logger.info(f"Reset quota for {provider} (period={period})")
