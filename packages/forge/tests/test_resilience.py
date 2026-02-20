"""Tests for resilience patterns (bulkhead and fallback)."""

import sys
import threading
import time

import pytest

sys.path.insert(0, "src")

from animus_forge.resilience.bulkhead import (
    Bulkhead,
    BulkheadFull,
    get_bulkhead,
)
from animus_forge.resilience.concurrency import (
    ConcurrencyLimiter,
    get_limiter,
    limit_concurrency,
)
from animus_forge.resilience.fallback import (
    FallbackChain,
    FallbackConfig,
    FallbackResult,
    fallback,
)


class TestBulkhead:
    """Tests for Bulkhead pattern."""

    def test_basic_acquire_release(self):
        """Can acquire and release slots."""
        bulkhead = Bulkhead(max_concurrent=2)

        assert bulkhead.acquire() is True
        assert bulkhead.acquire() is True

        stats = bulkhead.get_stats()
        assert stats["active_count"] == 2

        bulkhead.release()
        stats = bulkhead.get_stats()
        assert stats["active_count"] == 1

    def test_blocks_when_full(self):
        """Rejects when max concurrent reached and no waiting allowed."""
        bulkhead = Bulkhead(max_concurrent=1, max_waiting=0, timeout=0.01)

        # First acquire succeeds
        assert bulkhead.acquire() is True

        # Second should raise immediately (no waiting allowed)
        with pytest.raises(BulkheadFull) as exc:
            bulkhead.acquire()

        assert exc.value.active == 1

        # Clean up
        bulkhead.release()

    def test_waiting_queue(self):
        """Requests queue when semaphore full."""
        bulkhead = Bulkhead(max_concurrent=1, max_waiting=5, timeout=1.0)

        bulkhead.acquire()
        stats = bulkhead.get_stats()
        assert stats["active_count"] == 1

        # Start thread that will wait
        result = [None]

        def waiting_acquirer():
            result[0] = bulkhead.acquire(timeout=2.0)

        thread = threading.Thread(target=waiting_acquirer)
        thread.start()

        time.sleep(0.05)  # Let thread start waiting
        stats = bulkhead.get_stats()
        assert stats["waiting_count"] == 1

        # Release to allow waiting thread
        bulkhead.release()
        thread.join(timeout=1.0)

        assert result[0] is True

    def test_context_manager(self):
        """Context manager acquires and releases."""
        bulkhead = Bulkhead(max_concurrent=2)

        with bulkhead:
            stats = bulkhead.get_stats()
            assert stats["active_count"] == 1

        stats = bulkhead.get_stats()
        assert stats["active_count"] == 0

    def test_decorator(self):
        """Can use bulkhead as decorator."""
        bulkhead = Bulkhead(max_concurrent=5)

        @bulkhead
        def decorated_func():
            return "result"

        result = decorated_func()
        assert result == "result"

        stats = bulkhead.get_stats()
        assert stats["total_acquired"] == 1

    def test_max_waiting_enforced(self):
        """Rejects when waiting queue is full."""
        bulkhead = Bulkhead(max_concurrent=1, max_waiting=1, timeout=5.0)

        # Fill semaphore
        bulkhead.acquire()

        # Fill waiting queue
        def fill_waiting():
            bulkhead.acquire(timeout=5.0)

        thread = threading.Thread(target=fill_waiting)
        thread.start()
        time.sleep(0.05)

        # Should reject immediately
        with pytest.raises(BulkheadFull):
            bulkhead.acquire()

        # Cleanup
        bulkhead.release()
        thread.join()
        bulkhead.release()

    def test_stats_tracking(self):
        """Statistics are tracked correctly."""
        bulkhead = Bulkhead(name="test", max_concurrent=2, max_waiting=5)

        bulkhead.acquire()
        bulkhead.release()
        bulkhead.acquire()
        bulkhead.release()

        stats = bulkhead.get_stats()
        assert stats["name"] == "test"
        assert stats["total_acquired"] == 2

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Async context manager works correctly."""
        bulkhead = Bulkhead(max_concurrent=2)

        async with bulkhead:
            stats = bulkhead.get_stats()
            assert stats["active_count"] == 1

        stats = bulkhead.get_stats()
        assert stats["active_count"] == 0

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Async decorator works correctly."""
        bulkhead = Bulkhead(max_concurrent=5)

        @bulkhead
        async def async_func():
            return "async result"

        result = await async_func()
        assert result == "async result"


class TestGetBulkhead:
    """Tests for global bulkhead management."""

    def test_creates_bulkhead(self):
        """Creates bulkhead with specified config."""
        bulkhead = get_bulkhead("test-create", max_concurrent=5)
        assert bulkhead.name == "test-create"
        assert bulkhead.max_concurrent == 5

    def test_returns_same_instance(self):
        """Returns cached instance."""
        bh1 = get_bulkhead("test-cache")
        bh2 = get_bulkhead("test-cache")
        assert bh1 is bh2


class TestFallbackChain:
    """Tests for FallbackChain."""

    def test_first_handler_succeeds(self):
        """Uses first handler when it succeeds."""
        chain = FallbackChain("test")
        chain.add(lambda: "first")
        chain.add(lambda: "second")

        result = chain.execute()

        assert result.success is True
        assert result.value == "first"
        assert result.source == "<lambda>"
        assert result.attempts == 1

    def test_fallback_on_failure(self):
        """Falls back to second handler when first fails."""

        def failing_handler():
            raise ValueError("primary failed")

        chain = FallbackChain("test")
        chain.add(failing_handler, name="primary")
        chain.add(lambda: "fallback", name="fallback")

        result = chain.execute()

        assert result.success is True
        assert result.value == "fallback"
        assert result.source == "fallback"
        assert result.attempts == 2
        assert len(result.errors) == 1

    def test_all_handlers_fail(self):
        """Returns failure when all handlers fail."""

        def failing1():
            raise ValueError("fail 1")

        def failing2():
            raise ValueError("fail 2")

        chain = FallbackChain("test")
        chain.add(failing1, name="handler1")
        chain.add(failing2, name="handler2")

        result = chain.execute()

        assert result.success is False
        assert result.value is None
        assert result.attempts == 2
        assert len(result.errors) == 2

    def test_fail_fast_exceptions(self):
        """Stops on fail-fast exceptions."""

        def auth_error():
            raise PermissionError("unauthorized")

        config = FallbackConfig(fail_fast_exceptions=(PermissionError,))
        chain = FallbackChain("test", config=config)
        chain.add(auth_error, name="auth")
        chain.add(lambda: "backup", name="backup")

        result = chain.execute()

        assert result.success is False
        assert result.attempts == 1  # Didn't try backup

    def test_passes_arguments(self):
        """Passes arguments to handlers."""

        def handler(x, y=0):
            return x + y

        chain = FallbackChain("test")
        chain.add(handler)

        result = chain.execute(5, y=3)

        assert result.success is True
        assert result.value == 8

    def test_priority_ordering(self):
        """Handlers execute in priority order."""
        calls = []

        def handler_a():
            calls.append("a")
            raise ValueError("a failed")

        def handler_b():
            calls.append("b")
            return "b result"

        chain = FallbackChain("test")
        chain.add(handler_a, name="a", priority=2)
        chain.add(handler_b, name="b", priority=1)

        result = chain.execute()

        assert calls == ["b"]  # b has higher priority (lower number)
        assert result.value == "b result"

    def test_stats_tracking(self):
        """Statistics are tracked correctly."""
        chain = FallbackChain("stats-test")
        chain.add(lambda: "result", name="handler")

        chain.execute()
        chain.execute()

        stats = chain.get_stats()
        assert stats["total_executions"] == 2
        assert stats["successful_executions"] == 2
        assert stats["success_rate"] == 100.0

    @pytest.mark.asyncio
    async def test_async_execution(self):
        """Async execution works correctly."""

        async def async_handler():
            return "async result"

        chain = FallbackChain("async-test")
        chain.add(async_handler)

        result = await chain.execute_async()

        assert result.success is True
        assert result.value == "async result"

    @pytest.mark.asyncio
    async def test_async_fallback(self):
        """Async fallback works correctly."""

        async def failing_async():
            raise ValueError("async fail")

        async def backup_async():
            return "backup"

        chain = FallbackChain("async-fallback")
        chain.add(failing_async, name="primary")
        chain.add(backup_async, name="backup")

        result = await chain.execute_async()

        assert result.success is True
        assert result.value == "backup"


class TestFallbackDecorator:
    """Tests for @fallback decorator."""

    def test_returns_result_on_success(self):
        """Returns function result when successful."""

        @fallback("default")
        def get_value():
            return "real value"

        assert get_value() == "real value"

    def test_returns_fallback_on_exception(self):
        """Returns fallback value on exception."""

        @fallback("default")
        def failing_func():
            raise ValueError("oops")

        assert failing_func() == "default"

    def test_callable_fallback(self):
        """Fallback can be callable."""

        @fallback(lambda: "computed default")
        def failing_func():
            raise ValueError("oops")

        assert failing_func() == "computed default"

    def test_specific_exceptions(self):
        """Only catches specified exceptions."""

        @fallback("default", exceptions=(ValueError,))
        def func(raise_type):
            if raise_type:
                raise ValueError("expected")
            raise RuntimeError("unexpected")

        # Should use fallback for ValueError
        assert func(True) == "default"

        # Should raise RuntimeError (not caught)
        with pytest.raises(RuntimeError):
            func(False)

    @pytest.mark.asyncio
    async def test_async_fallback_decorator(self):
        """Async fallback decorator works."""

        @fallback("async default")
        async def async_failing():
            raise ValueError("async error")

        result = await async_failing()
        assert result == "async default"


class TestFallbackResult:
    """Tests for FallbackResult."""

    def test_bool_conversion(self):
        """Result is truthy when successful."""
        success = FallbackResult(success=True, value="x", source="h", attempts=1)
        failure = FallbackResult(success=False, value=None, source="", attempts=1)

        assert bool(success) is True
        assert bool(failure) is False

    def test_result_fields(self):
        """All fields are accessible."""
        result = FallbackResult(
            success=True,
            value="test",
            source="handler",
            attempts=2,
            errors=[{"error": "test"}],
            total_time_ms=150.5,
        )

        assert result.value == "test"
        assert result.source == "handler"
        assert result.attempts == 2
        assert len(result.errors) == 1
        assert result.total_time_ms == 150.5


class TestConcurrencyLimiter:
    """Tests for ConcurrencyLimiter."""

    def test_basic_acquire_release(self):
        """Can acquire and release slots."""
        limiter = ConcurrencyLimiter(max_concurrent=3)

        assert limiter.acquire() is True
        stats = limiter.get_stats()
        assert stats.current_active == 1

        limiter.release()
        stats = limiter.get_stats()
        assert stats.current_active == 0

    def test_respects_limit(self):
        """Respects max concurrent limit."""
        limiter = ConcurrencyLimiter(max_concurrent=2, timeout=0.01)

        assert limiter.acquire() is True
        assert limiter.acquire() is True
        # Third should timeout
        assert limiter.acquire(timeout=0.01) is False

        limiter.release()
        limiter.release()

    def test_context_manager(self):
        """Works as context manager."""
        limiter = ConcurrencyLimiter(max_concurrent=3)

        with limiter:
            stats = limiter.get_stats()
            assert stats.current_active == 1

        stats = limiter.get_stats()
        assert stats.current_active == 0

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Works as async context manager."""
        limiter = ConcurrencyLimiter(max_concurrent=3)

        async with limiter:
            stats = limiter.get_stats()
            assert stats.current_active == 1

        stats = limiter.get_stats()
        assert stats.current_active == 0

    def test_stats_tracking(self):
        """Statistics are tracked correctly."""
        limiter = ConcurrencyLimiter(name="test", max_concurrent=5)

        limiter.acquire()
        limiter.release()
        limiter.acquire()
        limiter.release()

        stats = limiter.get_stats()
        assert stats.name == "test"
        assert stats.total_acquired == 2
        assert stats.peak_active == 1


class TestLimitConcurrencyDecorator:
    """Tests for @limit_concurrency decorator."""

    def test_sync_function(self):
        """Decorates sync functions."""

        @limit_concurrency(max_concurrent=3, name="test-sync")
        def my_func():
            return "result"

        result = my_func()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Decorates async functions."""

        @limit_concurrency(max_concurrent=3, name="test-async")
        async def my_async_func():
            return "async result"

        result = await my_async_func()
        assert result == "async result"


class TestGetLimiter:
    """Tests for get_limiter function."""

    def test_creates_limiter(self):
        """Creates limiter with specified config."""
        limiter = get_limiter("test-get", max_concurrent=5)
        assert limiter.name == "test-get"
        assert limiter.max_concurrent == 5

    def test_returns_same_instance(self):
        """Returns cached instance."""
        lim1 = get_limiter("test-cache-lim")
        lim2 = get_limiter("test-cache-lim")
        assert lim1 is lim2
