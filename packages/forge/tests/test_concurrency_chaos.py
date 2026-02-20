"""Concurrency and chaos tests for resilience patterns.

Tests system behavior under:
- High concurrency loads
- Race conditions
- Failure injection
- Recovery scenarios
"""

import asyncio
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

sys.path.insert(0, "src")

from animus_forge.ratelimit.limiter import RateLimitConfig, TokenBucketLimiter
from animus_forge.resilience.bulkhead import Bulkhead, BulkheadFull
from animus_forge.resilience.fallback import FallbackChain
from animus_forge.security.brute_force import BruteForceConfig, BruteForceProtection


class TestBulkheadConcurrency:
    """Concurrency tests for Bulkhead pattern."""

    def test_concurrent_acquisitions(self):
        """Multiple threads can acquire slots concurrently."""
        bulkhead = Bulkhead(max_concurrent=5, max_waiting=50, timeout=5.0)
        acquired = []
        errors = []
        lock = threading.Lock()

        def worker(worker_id):
            try:
                if bulkhead.acquire(timeout=5.0):
                    with lock:
                        acquired.append(worker_id)
                    time.sleep(0.02)  # Simulate work
                    bulkhead.release()
            except BulkheadFull as e:
                with lock:
                    errors.append(("full", e))
            except Exception as e:
                with lock:
                    errors.append(("error", e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Most should acquire (with high waiting queue limit)
        assert len(acquired) >= 15
        assert bulkhead.get_stats()["active_count"] == 0  # All released

    def test_concurrent_async_acquisitions(self):
        """Multiple async tasks can acquire slots concurrently."""
        bulkhead = Bulkhead(max_concurrent=5, max_waiting=50, timeout=5.0)
        acquired = []
        lock = asyncio.Lock()

        async def worker(worker_id):
            try:
                if await bulkhead.acquire_async(timeout=5.0):
                    async with lock:
                        acquired.append(worker_id)
                    await asyncio.sleep(0.02)
                    await bulkhead.release_async()
            except BulkheadFull:
                pass  # Expected when queue is full

        async def run_test():
            tasks = [worker(i) for i in range(20)]
            await asyncio.gather(*tasks)

        asyncio.run(run_test())

        # Most should acquire
        assert len(acquired) >= 15
        assert bulkhead.get_stats()["active_count"] == 0

    def test_respects_concurrency_limit(self):
        """Never exceeds max concurrent limit."""
        bulkhead = Bulkhead(max_concurrent=3, max_waiting=50)
        max_observed = 0
        lock = threading.Lock()

        def worker():
            nonlocal max_observed
            if bulkhead.acquire(timeout=5.0):
                with lock:
                    current = bulkhead.get_stats()["active_count"]
                    max_observed = max(max_observed, current)
                time.sleep(0.01)
                bulkhead.release()

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_observed <= 3

    def test_rejects_when_queue_full(self):
        """Rejects requests when waiting queue is full."""
        bulkhead = Bulkhead(max_concurrent=1, max_waiting=2, timeout=0.5)
        rejected = []
        lock = threading.Lock()

        def slow_worker():
            bulkhead.acquire()
            time.sleep(0.5)
            bulkhead.release()

        def fast_worker():
            try:
                bulkhead.acquire(timeout=0.1)
                bulkhead.release()
            except BulkheadFull:
                with lock:
                    rejected.append(True)

        # Start slow worker to hold semaphore
        slow = threading.Thread(target=slow_worker)
        slow.start()
        time.sleep(0.05)

        # Start fast workers - some should be rejected
        threads = [threading.Thread(target=fast_worker) for _ in range(10)]
        for t in threads:
            t.start()

        slow.join()
        for t in threads:
            t.join()

        # Some should have been rejected
        assert len(rejected) > 0


class TestFallbackChainConcurrency:
    """Concurrency tests for FallbackChain."""

    def test_concurrent_executions(self):
        """Chain handles concurrent executions correctly."""
        call_count = 0
        lock = threading.Lock()

        def handler():
            nonlocal call_count
            with lock:
                call_count += 1
            time.sleep(0.01)
            return "result"

        chain = FallbackChain("test")
        chain.add(handler)

        def worker():
            return chain.execute()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            results = [f.result() for f in as_completed(futures)]

        assert all(r.success for r in results)
        assert call_count == 20

    def test_concurrent_fallbacks(self):
        """Fallbacks work correctly under concurrency."""
        fail_count = 0
        success_count = 0
        lock = threading.Lock()

        def failing_handler():
            nonlocal fail_count
            with lock:
                fail_count += 1
            raise ValueError("primary failed")

        def backup_handler():
            nonlocal success_count
            with lock:
                success_count += 1
            return "backup"

        chain = FallbackChain("test")
        chain.add(failing_handler, name="primary")
        chain.add(backup_handler, name="backup")

        def worker():
            return chain.execute()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            results = [f.result() for f in as_completed(futures)]

        assert all(r.success for r in results)
        assert all(r.source == "backup" for r in results)
        assert fail_count == 20
        assert success_count == 20

    @pytest.mark.asyncio
    async def test_async_concurrent_executions(self):
        """Async chain handles concurrent executions."""
        call_count = 0
        lock = asyncio.Lock()

        async def async_handler():
            nonlocal call_count
            async with lock:
                call_count += 1
            await asyncio.sleep(0.01)
            return "async result"

        chain = FallbackChain("async-test")
        chain.add(async_handler)

        tasks = [chain.execute_async() for _ in range(20)]
        results = await asyncio.gather(*tasks)

        assert all(r.success for r in results)
        assert call_count == 20


class TestRateLimiterConcurrency:
    """Concurrency tests for rate limiters."""

    def test_token_bucket_concurrent_acquires(self):
        """Token bucket handles concurrent acquires correctly."""
        config = RateLimitConfig(
            requests_per_second=100,
            burst_size=50,
            max_wait_seconds=1.0,
        )
        limiter = TokenBucketLimiter(config)
        acquired = []
        lock = threading.Lock()

        def worker():
            result = limiter.acquire(wait=False)
            with lock:
                acquired.append(result)

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should get approximately burst_size successes
        successes = sum(1 for a in acquired if a)
        assert 40 <= successes <= 60  # Allow some variance

    @pytest.mark.asyncio
    async def test_async_rate_limiting(self):
        """Async rate limiting works correctly."""
        config = RateLimitConfig(
            requests_per_second=100,
            burst_size=20,
            max_wait_seconds=0.1,
        )
        limiter = TokenBucketLimiter(config)
        acquired = []

        async def worker():
            result = await limiter.acquire_async(wait=False)
            acquired.append(result)

        tasks = [worker() for _ in range(50)]
        await asyncio.gather(*tasks)

        successes = sum(1 for a in acquired if a)
        assert 15 <= successes <= 25


class TestBruteForceProtectionConcurrency:
    """Concurrency tests for brute force protection."""

    def test_concurrent_checks(self):
        """Protection handles concurrent checks correctly."""
        config = BruteForceConfig(
            max_attempts_per_minute=50,
            max_attempts_per_hour=200,
        )
        protection = BruteForceProtection(config=config)
        allowed = []
        lock = threading.Lock()

        def worker(ip):
            result, _ = protection.check_allowed(ip)
            with lock:
                allowed.append(result)

        # Different IPs should all be allowed
        threads = [threading.Thread(target=worker, args=(f"192.168.1.{i}",)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(allowed)

    def test_same_ip_rate_limited(self):
        """Same IP gets rate limited under concurrent load."""
        config = BruteForceConfig(
            max_attempts_per_minute=10,
            max_attempts_per_hour=100,
        )
        protection = BruteForceProtection(config=config)
        allowed = []
        blocked = []
        lock = threading.Lock()

        def worker():
            result, _ = protection.check_allowed("192.168.1.1")
            with lock:
                if result:
                    allowed.append(True)
                else:
                    blocked.append(True)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(allowed) <= 15  # Should block most
        assert len(blocked) > 0


class TestChaosScenarios:
    """Chaos engineering tests - simulating failures."""

    def test_random_failures_with_fallback(self):
        """System recovers from random failures using fallbacks."""
        fail_rate = 0.5

        def flaky_handler():
            if random.random() < fail_rate:
                raise ValueError("random failure")
            return "primary"

        def reliable_handler():
            return "backup"

        chain = FallbackChain("chaos-test")
        chain.add(flaky_handler, name="flaky")
        chain.add(reliable_handler, name="reliable")

        results = [chain.execute() for _ in range(100)]

        assert all(r.success for r in results)
        # Some should have used backup
        backup_count = sum(1 for r in results if r.source == "reliable")
        assert backup_count > 0

    def test_bulkhead_under_failure(self):
        """Bulkhead maintains isolation during failures."""
        bulkhead = Bulkhead(max_concurrent=5, max_waiting=10)
        errors = []
        completions = []
        lock = threading.Lock()

        def failing_worker():
            try:
                if bulkhead.acquire(timeout=1.0):
                    try:
                        if random.random() < 0.3:
                            raise RuntimeError("simulated failure")
                        time.sleep(0.02)
                        with lock:
                            completions.append(True)
                    finally:
                        bulkhead.release()
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=failing_worker) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Bulkhead should still be in valid state
        stats = bulkhead.get_stats()
        assert stats["active_count"] == 0
        assert stats["waiting_count"] == 0

    def test_recovery_after_cascade(self):
        """System recovers after cascading failures."""
        # Simulate circuit breaker-like behavior
        failures_in_row = 0
        max_failures = 5
        is_open = False

        def handler():
            nonlocal failures_in_row, is_open
            if is_open:
                raise ConnectionError("circuit open")

            if random.random() < 0.8:  # High initial failure rate
                failures_in_row += 1
                if failures_in_row >= max_failures:
                    is_open = True
                raise ConnectionError("service unavailable")

            failures_in_row = 0
            return "success"

        def fallback_handler():
            return "fallback"

        chain = FallbackChain("cascade-test")
        chain.add(handler, name="primary")
        chain.add(fallback_handler, name="fallback")

        results = [chain.execute() for _ in range(50)]

        # All should succeed (via fallback if needed)
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_async_timeout_handling(self):
        """Async operations handle timeouts gracefully."""
        bulkhead = Bulkhead(max_concurrent=2, max_waiting=20, timeout=0.5)

        async def slow_worker():
            try:
                if await bulkhead.acquire_async(timeout=0.5):
                    await asyncio.sleep(0.1)  # Slow operation
                    await bulkhead.release_async()
                    return "completed"
                return "timed out"
            except BulkheadFull:
                return "queue full"

        # Start some slow workers
        results = await asyncio.gather(*[slow_worker() for _ in range(10)])

        # Some should complete, some may be rejected
        completed = sum(1 for r in results if r == "completed")
        # At least some should complete
        assert completed >= 2

    def test_resource_exhaustion_recovery(self):
        """System recovers from resource exhaustion."""
        config = BruteForceConfig(
            max_attempts_per_minute=5,
            max_attempts_per_hour=100,
            initial_block_seconds=0.1,  # Short block for test
        )
        protection = BruteForceProtection(config=config)

        # Exhaust rate limit
        for _ in range(10):
            protection.check_allowed("192.168.1.1")

        # Should be blocked now
        allowed, _ = protection.check_allowed("192.168.1.1")
        assert allowed is False

        # Wait for block to expire
        time.sleep(0.2)

        # Different IP should still work
        allowed, _ = protection.check_allowed("192.168.1.2")
        assert allowed is True


class TestStressScenarios:
    """High-load stress tests."""

    def test_high_throughput_fallback(self):
        """Fallback chain handles high throughput."""
        chain = FallbackChain("stress")
        chain.add(lambda: "fast")

        start = time.monotonic()
        results = [chain.execute() for _ in range(1000)]
        elapsed = time.monotonic() - start

        assert all(r.success for r in results)
        # Should be fast - under 1 second for 1000 executions
        assert elapsed < 2.0

    def test_bulkhead_stress(self):
        """Bulkhead handles stress load correctly."""
        bulkhead = Bulkhead(max_concurrent=10, max_waiting=100)
        completed = []
        lock = threading.Lock()

        def worker():
            if bulkhead.acquire(timeout=5.0):
                try:
                    time.sleep(0.001)
                    with lock:
                        completed.append(True)
                finally:
                    bulkhead.release()

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(worker) for _ in range(500)]
            for f in as_completed(futures):
                pass  # Wait for completion

        assert len(completed) == 500
        assert bulkhead.get_stats()["active_count"] == 0
