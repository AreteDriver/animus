"""Additional coverage tests for quota management."""

import sys
from datetime import UTC, datetime, timedelta

import pytest

sys.path.insert(0, "src")

from animus_forge.ratelimit.quota import (
    QuotaConfig,
    QuotaExceeded,
    QuotaManager,
    QuotaPeriod,
    QuotaUsage,
)


class TestQuotaExceeded:
    def test_attributes(self):
        exc = QuotaExceeded(
            "over limit",
            provider="anthropic",
            period="hour",
            limit=100,
            used=101,
        )
        assert str(exc) == "over limit"
        assert exc.provider == "anthropic"
        assert exc.period == "hour"
        assert exc.limit == 100
        assert exc.used == 101

    def test_defaults(self):
        exc = QuotaExceeded()
        assert exc.provider == ""
        assert exc.resets_at is None


class TestQuotaUsage:
    def test_remaining(self):
        usage = QuotaUsage(period=QuotaPeriod.HOUR, limit=100, used=40)
        assert usage.remaining == 60

    def test_remaining_over_limit(self):
        usage = QuotaUsage(period=QuotaPeriod.HOUR, limit=100, used=150)
        assert usage.remaining == 0

    def test_percent_used(self):
        usage = QuotaUsage(period=QuotaPeriod.HOUR, limit=100, used=25)
        assert usage.percent_used == 25.0

    def test_percent_used_zero_limit(self):
        usage = QuotaUsage(period=QuotaPeriod.HOUR, limit=0, used=0)
        assert usage.percent_used == 100.0

    def test_is_exceeded(self):
        usage = QuotaUsage(period=QuotaPeriod.HOUR, limit=10, used=10)
        assert usage.is_exceeded() is True

    def test_is_not_exceeded(self):
        usage = QuotaUsage(period=QuotaPeriod.HOUR, limit=10, used=5)
        assert usage.is_exceeded() is False

    def test_reset_if_expired_minute(self):
        usage = QuotaUsage(
            period=QuotaPeriod.MINUTE,
            limit=10,
            used=5,
            period_start=datetime.now(UTC) - timedelta(seconds=120),
        )
        assert usage.reset_if_expired() is True
        assert usage.used == 0

    def test_reset_if_not_expired_minute(self):
        usage = QuotaUsage(
            period=QuotaPeriod.MINUTE,
            limit=10,
            used=5,
            period_start=datetime.now(UTC),
        )
        assert usage.reset_if_expired() is False
        assert usage.used == 5

    def test_reset_if_expired_hour(self):
        usage = QuotaUsage(
            period=QuotaPeriod.HOUR,
            limit=10,
            used=5,
            period_start=datetime.now(UTC) - timedelta(hours=2),
        )
        assert usage.reset_if_expired() is True
        assert usage.used == 0

    def test_reset_if_expired_day(self):
        usage = QuotaUsage(
            period=QuotaPeriod.DAY,
            limit=10,
            used=5,
            period_start=datetime.now(UTC) - timedelta(days=2),
        )
        assert usage.reset_if_expired() is True

    def test_reset_if_not_expired_day(self):
        usage = QuotaUsage(
            period=QuotaPeriod.DAY,
            limit=10,
            used=5,
            period_start=datetime.now(UTC),
        )
        assert usage.reset_if_expired() is False

    def test_reset_if_expired_month(self):
        # Set period_start to 2 months ago
        old_start = datetime.now(UTC) - timedelta(days=60)
        usage = QuotaUsage(
            period=QuotaPeriod.MONTH,
            limit=10,
            used=5,
            period_start=old_start,
        )
        assert usage.reset_if_expired() is True

    def test_reset_if_not_expired_month(self):
        usage = QuotaUsage(
            period=QuotaPeriod.MONTH,
            limit=10,
            used=5,
            period_start=datetime.now(UTC),
        )
        assert usage.reset_if_expired() is False

    def test_time_until_reset_minute(self):
        usage = QuotaUsage(
            period=QuotaPeriod.MINUTE,
            limit=10,
            period_start=datetime.now(UTC),
        )
        t = usage.time_until_reset()
        assert 0 <= t <= 60

    def test_time_until_reset_hour(self):
        usage = QuotaUsage(
            period=QuotaPeriod.HOUR,
            limit=10,
            period_start=datetime.now(UTC),
        )
        t = usage.time_until_reset()
        assert 0 <= t <= 3600

    def test_time_until_reset_day(self):
        usage = QuotaUsage(
            period=QuotaPeriod.DAY,
            limit=10,
            period_start=datetime.now(UTC),
        )
        t = usage.time_until_reset()
        assert t >= 0

    def test_time_until_reset_month(self):
        usage = QuotaUsage(
            period=QuotaPeriod.MONTH,
            limit=10,
            period_start=datetime.now(UTC),
        )
        t = usage.time_until_reset()
        assert t >= 0


class TestQuotaManager:
    def test_configure(self):
        mgr = QuotaManager()
        config = QuotaConfig(
            provider="anthropic",
            requests_per_minute=60,
            requests_per_hour=1000,
        )
        mgr.configure(config)
        assert "anthropic" in mgr._quotas
        assert QuotaPeriod.MINUTE in mgr._quotas["anthropic"]
        assert QuotaPeriod.HOUR in mgr._quotas["anthropic"]

    def test_check_unconfigured(self):
        mgr = QuotaManager()
        assert mgr.check("unknown") is True

    def test_check_within_limit(self):
        mgr = QuotaManager()
        mgr.configure(QuotaConfig(provider="test", requests_per_minute=10))
        assert mgr.check("test") is True

    def test_check_exceeds_limit(self):
        mgr = QuotaManager()
        mgr.configure(QuotaConfig(provider="test", requests_per_minute=2))
        mgr.acquire("test")
        mgr.acquire("test")
        assert mgr.check("test") is False

    def test_acquire_unconfigured(self):
        mgr = QuotaManager()
        mgr.acquire("unknown")  # Should not raise

    def test_acquire_success(self):
        mgr = QuotaManager()
        mgr.configure(QuotaConfig(provider="test", requests_per_minute=10))
        mgr.acquire("test")
        usage = mgr.get_usage("test")
        assert usage["periods"]["minute"]["used"] == 1

    def test_acquire_exceeded(self):
        mgr = QuotaManager()
        mgr.configure(QuotaConfig(provider="test", requests_per_minute=1))
        mgr.acquire("test")
        with pytest.raises(QuotaExceeded):
            mgr.acquire("test")

    def test_record_usage(self):
        mgr = QuotaManager()
        mgr.configure(QuotaConfig(provider="test", requests_per_minute=100))
        mgr.record_usage("test", 5)
        usage = mgr.get_usage("test")
        assert usage["periods"]["minute"]["used"] == 5

    def test_record_usage_unconfigured(self):
        mgr = QuotaManager()
        mgr.record_usage("unknown", 5)  # Should not raise

    def test_get_usage_unconfigured(self):
        mgr = QuotaManager()
        usage = mgr.get_usage("unknown")
        assert usage["configured"] is False

    def test_get_all_usage(self):
        mgr = QuotaManager()
        mgr.configure(QuotaConfig(provider="a", requests_per_minute=10))
        mgr.configure(QuotaConfig(provider="b", requests_per_hour=100))
        all_usage = mgr.get_all_usage()
        assert "a" in all_usage
        assert "b" in all_usage

    def test_reset_all_periods(self):
        mgr = QuotaManager()
        mgr.configure(QuotaConfig(provider="test", requests_per_minute=10, requests_per_hour=100))
        mgr.acquire("test", 5)
        mgr.reset("test")
        usage = mgr.get_usage("test")
        assert usage["periods"]["minute"]["used"] == 0
        assert usage["periods"]["hour"]["used"] == 0

    def test_reset_specific_period(self):
        mgr = QuotaManager()
        mgr.configure(QuotaConfig(provider="test", requests_per_minute=10, requests_per_hour=100))
        mgr.acquire("test", 5)
        mgr.reset("test", QuotaPeriod.MINUTE)
        usage = mgr.get_usage("test")
        assert usage["periods"]["minute"]["used"] == 0
        assert usage["periods"]["hour"]["used"] == 5

    def test_reset_unconfigured(self):
        mgr = QuotaManager()
        mgr.reset("unknown")  # Should not raise

    def test_configure_with_daily_quota(self):
        mgr = QuotaManager()
        mgr.configure(QuotaConfig(provider="test", requests_per_day=1000))
        assert QuotaPeriod.DAY in mgr._quotas["test"]
