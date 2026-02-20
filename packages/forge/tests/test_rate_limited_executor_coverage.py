"""Additional coverage tests for rate_limited_executor module."""

import sys
import time

sys.path.insert(0, "src")

from animus_forge.workflow.parallel import ParallelStrategy, ParallelTask
from animus_forge.workflow.rate_limited_executor import (
    AdaptiveRateLimitConfig,
    AdaptiveRateLimitState,
    ProviderRateLimits,
    RateLimitedParallelExecutor,
    create_rate_limited_executor,
)

# =============================================================================
# Dataclasses
# =============================================================================


class TestProviderRateLimits:
    def test_defaults(self):
        limits = ProviderRateLimits()
        assert limits.anthropic == 5
        assert limits.openai == 8
        assert limits.default == 10


class TestAdaptiveRateLimitConfig:
    def test_defaults(self):
        config = AdaptiveRateLimitConfig()
        assert config.min_concurrent == 1
        assert config.backoff_factor == 0.5
        assert config.recovery_factor == 1.2
        assert config.recovery_threshold == 10
        assert config.cooldown_seconds == 30.0


class TestAdaptiveRateLimitState:
    def test_record_success_no_recovery(self):
        config = AdaptiveRateLimitConfig(recovery_threshold=10)
        state = AdaptiveRateLimitState(base_limit=5, current_limit=5)
        # Not enough successes and not below base
        assert state.record_success(config) is False
        assert state.consecutive_successes == 1
        assert state.consecutive_failures == 0

    def test_record_success_triggers_recovery(self):
        config = AdaptiveRateLimitConfig(recovery_threshold=2, cooldown_seconds=0)
        state = AdaptiveRateLimitState(
            base_limit=10,
            current_limit=3,
            last_adjustment_time=time.time() - 100,
        )
        state.record_success(config)
        adjusted = state.record_success(config)
        assert adjusted is True
        assert state.current_limit > 3

    def test_record_rate_limit_error_backoff(self):
        config = AdaptiveRateLimitConfig(backoff_factor=0.5, cooldown_seconds=0)
        state = AdaptiveRateLimitState(
            base_limit=10,
            current_limit=10,
            last_adjustment_time=time.time() - 100,
        )
        adjusted = state.record_rate_limit_error(config)
        assert adjusted is True
        assert state.current_limit == 5
        assert state.total_429s == 1

    def test_record_rate_limit_error_cooldown_blocks(self):
        config = AdaptiveRateLimitConfig(backoff_factor=0.5, cooldown_seconds=9999)
        state = AdaptiveRateLimitState(
            base_limit=10, current_limit=10, last_adjustment_time=time.time()
        )
        adjusted = state.record_rate_limit_error(config)
        assert adjusted is False  # Cooldown hasn't passed

    def test_record_rate_limit_error_at_min(self):
        config = AdaptiveRateLimitConfig(backoff_factor=0.5, min_concurrent=5, cooldown_seconds=0)
        state = AdaptiveRateLimitState(
            base_limit=10,
            current_limit=5,
            last_adjustment_time=time.time() - 100,
        )
        adjusted = state.record_rate_limit_error(config)
        assert adjusted is False  # new_limit == current_limit (both 5)


# =============================================================================
# RateLimitedParallelExecutor
# =============================================================================


class TestRateLimitedParallelExecutorInit:
    def test_default_init(self):
        executor = RateLimitedParallelExecutor()
        assert executor.strategy == ParallelStrategy.ASYNCIO
        assert executor._adaptive is True
        assert "anthropic" in executor._provider_limits
        assert "openai" in executor._provider_limits

    def test_custom_limits(self):
        executor = RateLimitedParallelExecutor(provider_limits={"anthropic": 3, "openai": 5})
        assert executor._provider_limits["anthropic"] == 3
        assert executor._provider_limits["openai"] == 5

    def test_non_adaptive(self):
        executor = RateLimitedParallelExecutor(adaptive=False)
        assert executor._adaptive is False
        assert len(executor._adaptive_state) == 0


class TestRateLimitedExecutorIsRateLimitError:
    def test_429_in_message(self):
        executor = RateLimitedParallelExecutor()
        assert executor._is_rate_limit_error(Exception("Error 429")) is True

    def test_rate_limit_in_message(self):
        executor = RateLimitedParallelExecutor()
        assert executor._is_rate_limit_error(Exception("rate limit exceeded")) is True

    def test_too_many_requests(self):
        executor = RateLimitedParallelExecutor()
        assert executor._is_rate_limit_error(Exception("too many requests")) is True

    def test_ratelimit_in_type(self):
        RateLimitError = type("RateLimitError", (Exception,), {})
        executor = RateLimitedParallelExecutor()
        assert executor._is_rate_limit_error(RateLimitError("x")) is True

    def test_status_code_429(self):
        exc = Exception("error")
        exc.status_code = 429
        executor = RateLimitedParallelExecutor()
        assert executor._is_rate_limit_error(exc) is True

    def test_normal_error(self):
        executor = RateLimitedParallelExecutor()
        assert executor._is_rate_limit_error(ValueError("bad input")) is False


class TestRateLimitedExecutorGetProvider:
    def test_explicit_provider_kwarg(self):
        executor = RateLimitedParallelExecutor()
        task = ParallelTask(
            id="t1",
            step_id="s1",
            handler=lambda: None,
            kwargs={"provider": "Anthropic"},
        )
        assert executor._get_provider_for_task(task) == "anthropic"

    def test_step_type_claude(self):
        executor = RateLimitedParallelExecutor()
        task = ParallelTask(
            id="t1",
            step_id="s1",
            handler=lambda: None,
            kwargs={"step_type": "claude_code"},
        )
        assert executor._get_provider_for_task(task) == "anthropic"

    def test_step_type_openai(self):
        executor = RateLimitedParallelExecutor()
        task = ParallelTask(
            id="t1",
            step_id="s1",
            handler=lambda: None,
            kwargs={"step_type": "gpt_task"},
        )
        assert executor._get_provider_for_task(task) == "openai"

    def test_handler_name_anthropic(self):
        def anthropic_handler():
            pass

        executor = RateLimitedParallelExecutor()
        task = ParallelTask(id="t1", step_id="s1", handler=anthropic_handler)
        assert executor._get_provider_for_task(task) == "anthropic"

    def test_handler_name_openai(self):
        def openai_handler():
            pass

        executor = RateLimitedParallelExecutor()
        task = ParallelTask(id="t1", step_id="s1", handler=openai_handler)
        assert executor._get_provider_for_task(task) == "openai"

    def test_default_provider(self):
        executor = RateLimitedParallelExecutor()
        task = ParallelTask(id="t1", step_id="s1", handler=lambda: None)
        assert executor._get_provider_for_task(task) == "default"


class TestRateLimitedExecutorProviderStats:
    def test_get_provider_stats_adaptive(self):
        executor = RateLimitedParallelExecutor(adaptive=True)
        stats = executor.get_provider_stats()
        assert "anthropic" in stats
        assert "current_limit" in stats["anthropic"]
        assert stats["anthropic"]["is_throttled"] is False
        assert stats["anthropic"]["distributed_enabled"] is False

    def test_get_provider_stats_non_adaptive(self):
        executor = RateLimitedParallelExecutor(adaptive=False)
        stats = executor.get_provider_stats()
        assert stats["anthropic"]["current_limit"] == stats["anthropic"]["base_limit"]

    def test_get_provider_stats_distributed(self):
        executor = RateLimitedParallelExecutor(distributed=True)
        stats = executor.get_provider_stats()
        assert stats["anthropic"]["distributed_enabled"] is True
        assert "distributed_rpm" in stats["anthropic"]


class TestRateLimitedExecutorResetState:
    def test_reset_all(self):
        executor = RateLimitedParallelExecutor(adaptive=True)
        # Simulate throttling
        state = executor._adaptive_state["anthropic"]
        state.current_limit = 1
        state.consecutive_successes = 5

        executor.reset_adaptive_state()
        assert state.current_limit == state.base_limit
        assert state.consecutive_successes == 0

    def test_reset_single_provider(self):
        executor = RateLimitedParallelExecutor(adaptive=True)
        state = executor._adaptive_state["openai"]
        state.current_limit = 1

        executor.reset_adaptive_state("openai")
        assert state.current_limit == state.base_limit

    def test_reset_non_adaptive_noop(self):
        executor = RateLimitedParallelExecutor(adaptive=False)
        executor.reset_adaptive_state()  # Should not raise


class TestRateLimitedExecutorExecution:
    def test_execute_parallel_asyncio(self):
        executor = RateLimitedParallelExecutor(strategy=ParallelStrategy.ASYNCIO, max_workers=2)
        tasks = [
            ParallelTask(id="t1", step_id="s1", handler=lambda: "r1"),
            ParallelTask(id="t2", step_id="s2", handler=lambda: "r2"),
        ]
        result = executor.execute_parallel(tasks)
        assert len(result.successful) == 2

    def test_execute_parallel_threading_fallback(self):
        executor = RateLimitedParallelExecutor(strategy=ParallelStrategy.THREADING, max_workers=2)
        tasks = [
            ParallelTask(id="t1", step_id="s1", handler=lambda: "r1"),
        ]
        result = executor.execute_parallel(tasks)
        assert len(result.successful) == 1

    def test_execute_with_failure(self):
        def fail():
            raise ValueError("rate limit test")

        executor = RateLimitedParallelExecutor(strategy=ParallelStrategy.ASYNCIO, max_workers=2)
        tasks = [
            ParallelTask(id="t1", step_id="s1", handler=lambda: "ok"),
            ParallelTask(id="t2", step_id="s2", handler=fail),
        ]
        result = executor.execute_parallel(tasks)
        assert "t2" in result.failed
        assert "t1" in result.successful

    def test_execute_fail_fast(self):
        def fail():
            raise ValueError("fail fast")

        executor = RateLimitedParallelExecutor(strategy=ParallelStrategy.ASYNCIO, max_workers=2)
        tasks = [
            ParallelTask(id="t1", step_id="s1", handler=fail),
            ParallelTask(id="t2", step_id="s2", handler=lambda: "ok", dependencies=["t1"]),
        ]
        result = executor.execute_parallel(tasks, fail_fast=True)
        assert "t1" in result.failed


# =============================================================================
# create_rate_limited_executor factory
# =============================================================================


class TestCreateRateLimitedExecutor:
    def test_default_factory(self):
        executor = create_rate_limited_executor()
        assert executor.strategy == ParallelStrategy.ASYNCIO
        assert executor._adaptive is True
        assert executor.max_workers == 4

    def test_custom_factory(self):
        executor = create_rate_limited_executor(
            max_workers=8,
            anthropic_concurrent=3,
            openai_concurrent=5,
            timeout=600.0,
            adaptive=False,
        )
        assert executor.max_workers == 8
        assert executor._adaptive is False
        assert executor._provider_limits["anthropic"] == 3

    def test_factory_with_distributed(self):
        executor = create_rate_limited_executor(
            distributed=True,
            anthropic_rpm=30,
            openai_rpm=45,
        )
        assert executor._distributed is True
