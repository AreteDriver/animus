"""Tests for circuit breaker functionality."""

import asyncio
import time

import pytest

from animus_forge.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    get_all_circuit_stats,
    get_circuit_breaker,
    reset_all_circuits,
)


class TestCircuitBreakerBasics:
    """Test basic circuit breaker functionality."""

    def test_starts_closed(self):
        """Circuit starts in closed state."""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    def test_successful_calls_stay_closed(self):
        """Successful calls keep circuit closed."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        def success():
            return "ok"

        for _ in range(10):
            result = breaker.call(success)
            assert result == "ok"

        assert breaker.is_closed

    def test_failures_open_circuit(self):
        """Consecutive failures open the circuit."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        def fail():
            raise ValueError("error")

        # First 2 failures - still closed
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail)
        assert breaker.is_closed

        # Third failure - opens circuit
        with pytest.raises(ValueError):
            breaker.call(fail)
        assert breaker.is_open

    def test_open_circuit_fails_fast(self):
        """Open circuit raises CircuitBreakerError immediately."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=60)

        def fail():
            raise ValueError("error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail)

        # Should fail fast without calling function
        call_count = 0

        def tracked_call():
            nonlocal call_count
            call_count += 1
            return "ok"

        with pytest.raises(CircuitBreakerError):
            breaker.call(tracked_call)

        assert call_count == 0  # Function was never called

    def test_success_resets_failure_count(self):
        """A success resets the failure counter."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        def fail():
            raise ValueError("error")

        def success():
            return "ok"

        # Two failures
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail)

        # One success resets count
        breaker.call(success)

        # Two more failures shouldn't open circuit (count reset)
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail)

        assert breaker.is_closed


class TestCircuitBreakerRecovery:
    """Test circuit recovery behavior."""

    def test_recovery_timeout_enters_half_open(self):
        """After recovery timeout, circuit enters half-open state."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.1)

        def fail():
            raise ValueError("error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail)

        assert breaker.is_open

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should be half-open now
        assert breaker.state == CircuitState.HALF_OPEN

    def test_success_in_half_open_closes_circuit(self):
        """Success in half-open state closes the circuit."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=1,
        )

        def fail():
            raise ValueError("error")

        def success():
            return "ok"

        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail)

        time.sleep(0.15)  # Wait for half-open

        # Success closes circuit
        breaker.call(success)
        assert breaker.is_closed

    def test_failure_in_half_open_reopens_circuit(self):
        """Failure in half-open state reopens the circuit."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.1)

        def fail():
            raise ValueError("error")

        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail)

        time.sleep(0.15)  # Wait for half-open

        # Failure reopens circuit
        with pytest.raises(ValueError):
            breaker.call(fail)

        assert breaker.is_open

    def test_success_threshold(self):
        """Multiple successes needed to close from half-open."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=3,
        )

        def fail():
            raise ValueError("error")

        def success():
            return "ok"

        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail)

        time.sleep(0.15)  # Wait for half-open

        # First two successes - still half-open
        breaker.call(success)
        assert breaker.state == CircuitState.HALF_OPEN
        breaker.call(success)
        assert breaker.state == CircuitState.HALF_OPEN

        # Third success - closes circuit
        breaker.call(success)
        assert breaker.is_closed


class TestCircuitBreakerDecorator:
    """Test circuit breaker as decorator."""

    def test_decorator_sync(self):
        """Circuit breaker works as sync decorator."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)

        call_count = 0

        @breaker
        def tracked_function():
            nonlocal call_count
            call_count += 1
            return "result"

        result = tracked_function()
        assert result == "result"
        assert call_count == 1

    def test_decorator_async(self):
        """Circuit breaker works as async decorator."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)

        call_count = 0

        @breaker
        async def tracked_async():
            nonlocal call_count
            call_count += 1
            return "async_result"

        result = asyncio.run(tracked_async())
        assert result == "async_result"
        assert call_count == 1

    def test_decorator_opens_on_failure(self):
        """Decorator opens circuit on failures."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)

        @breaker
        def failing_function():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            failing_function()
        with pytest.raises(RuntimeError):
            failing_function()

        # Circuit should be open
        with pytest.raises(CircuitBreakerError):
            failing_function()


class TestCircuitBreakerContextManager:
    """Test circuit breaker as context manager."""

    def test_context_manager_success(self):
        """Context manager records success on clean exit."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)

        with breaker:
            pass  # Success

        assert breaker.is_closed
        stats = breaker.get_stats()
        assert stats["failure_count"] == 0

    def test_context_manager_failure(self):
        """Context manager records failure on exception."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)

        with pytest.raises(ValueError):
            with breaker:
                raise ValueError("error")

        stats = breaker.get_stats()
        assert stats["failure_count"] == 1

    def test_context_manager_opens_circuit(self):
        """Context manager opens circuit after threshold failures."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)

        for _ in range(2):
            with pytest.raises(ValueError):
                with breaker:
                    raise ValueError("error")

        # Circuit is open
        with pytest.raises(CircuitBreakerError):
            with breaker:
                pass


class TestExcludedExceptions:
    """Test excluded exception handling."""

    def test_excluded_exceptions_dont_count(self):
        """Excluded exceptions don't increment failure count."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            excluded_exceptions=(KeyError,),
        )

        def raise_keyerror():
            raise KeyError("not found")

        # KeyError is excluded - shouldn't count as failure
        for _ in range(5):
            with pytest.raises(KeyError):
                breaker.call(raise_keyerror)

        assert breaker.is_closed  # Should still be closed

    def test_non_excluded_exceptions_count(self):
        """Non-excluded exceptions still count as failures."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            excluded_exceptions=(KeyError,),
        )

        def raise_valueerror():
            raise ValueError("bad value")

        # ValueError is not excluded
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(raise_valueerror)

        assert breaker.is_open


class TestCircuitBreakerManagement:
    """Test circuit breaker management functions."""

    def test_get_circuit_breaker_creates(self):
        """get_circuit_breaker creates new breaker."""
        reset_all_circuits()
        breaker = get_circuit_breaker("new_service")
        assert breaker.name == "new_service"
        assert breaker.is_closed

    def test_get_circuit_breaker_reuses(self):
        """get_circuit_breaker returns same instance."""
        reset_all_circuits()
        breaker1 = get_circuit_breaker("shared_service")
        breaker2 = get_circuit_breaker("shared_service")
        assert breaker1 is breaker2

    def test_get_all_circuit_stats(self):
        """get_all_circuit_stats returns stats for all circuits."""
        reset_all_circuits()
        get_circuit_breaker("service_a")
        get_circuit_breaker("service_b")

        stats = get_all_circuit_stats()
        assert "service_a" in stats
        assert "service_b" in stats
        assert stats["service_a"]["state"] == "closed"

    def test_reset_all_circuits(self):
        """reset_all_circuits resets all circuits to closed."""
        reset_all_circuits()
        breaker = get_circuit_breaker("test_reset", failure_threshold=2)

        # Open the circuit
        def fail():
            raise ValueError("error")

        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail)

        assert breaker.is_open

        # Reset all
        reset_all_circuits()

        # Should be closed now
        assert breaker.is_closed


class TestCircuitBreakerStats:
    """Test circuit breaker statistics."""

    def test_get_stats(self):
        """get_stats returns current state."""
        breaker = CircuitBreaker(
            name="stats_test",
            failure_threshold=5,
            recovery_timeout=30.0,
        )

        stats = breaker.get_stats()
        assert stats["name"] == "stats_test"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 0
        assert stats["failure_threshold"] == 5
        assert stats["recovery_timeout"] == 30.0

    def test_stats_update_on_failure(self):
        """Stats update after failures."""
        breaker = CircuitBreaker(name="test", failure_threshold=5)

        def fail():
            raise ValueError("error")

        with pytest.raises(ValueError):
            breaker.call(fail)

        stats = breaker.get_stats()
        assert stats["failure_count"] == 1
        assert stats["last_failure_time"] > 0


class TestManualReset:
    """Test manual circuit reset."""

    def test_manual_reset(self):
        """reset() closes circuit and clears counters."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)

        def fail():
            raise ValueError("error")

        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail)

        assert breaker.is_open

        # Manual reset
        breaker.reset()

        assert breaker.is_closed
        stats = breaker.get_stats()
        assert stats["failure_count"] == 0
        assert stats["success_count"] == 0


class TestAsyncCircuitBreaker:
    """Test async functionality."""

    def test_call_async(self):
        """call_async works with coroutines."""
        breaker = CircuitBreaker(name="async_test", failure_threshold=2)

        async def async_operation():
            return "async_result"

        result = asyncio.run(breaker.call_async(async_operation))
        assert result == "async_result"

    def test_call_async_failure(self):
        """call_async records failures."""
        breaker = CircuitBreaker(name="async_test", failure_threshold=2)

        async def async_fail():
            raise RuntimeError("async error")

        async def run_test():
            for _ in range(2):
                with pytest.raises(RuntimeError):
                    await breaker.call_async(async_fail)

            # Circuit should be open
            with pytest.raises(CircuitBreakerError):
                await breaker.call_async(async_fail)

        asyncio.run(run_test())
        assert breaker.is_open


class TestThreadSafety:
    """Test thread safety of circuit breaker."""

    def test_concurrent_calls(self):
        """Circuit breaker handles concurrent calls safely."""
        import threading

        breaker = CircuitBreaker(name="concurrent", failure_threshold=100)
        success_count = 0
        lock = threading.Lock()

        def worker():
            nonlocal success_count
            for _ in range(50):
                try:
                    breaker.call(lambda: "ok")
                    with lock:
                        success_count += 1
                except CircuitBreakerError:
                    pass

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All calls should have succeeded
        assert success_count == 500
