"""Rate-limited parallel execution for AI workflows.

Provides per-provider rate limiting to prevent 429 errors during
parallel agent execution. Includes adaptive rate limiting that
dynamically adjusts limits based on 429 responses.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .parallel import (
    ParallelExecutor,
    ParallelResult,
    ParallelStrategy,
    ParallelTask,
)

logger = logging.getLogger(__name__)


@dataclass
class ProviderRateLimits:
    """Rate limit configuration for providers.

    Default limits are conservative to prevent 429 errors:
    - Anthropic: ~60 RPM = 5 concurrent safe with headroom
    - OpenAI: ~90 RPM = 8 concurrent safe with headroom
    """

    anthropic: int = 5
    openai: int = 8
    default: int = 10


@dataclass
class AdaptiveRateLimitConfig:
    """Configuration for adaptive rate limiting.

    Attributes:
        min_concurrent: Minimum concurrent requests (floor)
        backoff_factor: Factor to reduce limit by on 429 (e.g., 0.5 = halve)
        recovery_factor: Factor to increase limit after successes (e.g., 1.1)
        recovery_threshold: Consecutive successes before recovery
        cooldown_seconds: Minimum time between limit adjustments
    """

    min_concurrent: int = 1
    backoff_factor: float = 0.5
    recovery_factor: float = 1.2
    recovery_threshold: int = 10
    cooldown_seconds: float = 30.0


@dataclass
class AdaptiveRateLimitState:
    """Tracks state for adaptive rate limiting per provider.

    Attributes:
        base_limit: Original configured limit
        current_limit: Current effective limit
        consecutive_successes: Count of consecutive successful calls
        consecutive_failures: Count of consecutive 429 errors
        last_adjustment_time: Timestamp of last limit adjustment
        total_429s: Total 429 errors since start
    """

    base_limit: int
    current_limit: int
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_adjustment_time: float = field(default_factory=time.time)
    total_429s: int = 0

    def record_success(self, config: AdaptiveRateLimitConfig) -> bool:
        """Record a successful call. Returns True if limit was adjusted."""
        self.consecutive_successes += 1
        self.consecutive_failures = 0

        # Check if we should recover
        if (
            self.consecutive_successes >= config.recovery_threshold
            and self.current_limit < self.base_limit
            and time.time() - self.last_adjustment_time >= config.cooldown_seconds
        ):
            # Ensure at least +1 increase on recovery (handles low current_limit values)
            new_limit = min(
                self.base_limit,
                max(
                    self.current_limit + 1,
                    int(self.current_limit * config.recovery_factor),
                ),
            )
            logger.info(f"Rate limit recovery: {self.current_limit} -> {new_limit}")
            self.current_limit = new_limit
            self.consecutive_successes = 0
            self.last_adjustment_time = time.time()
            return True
        return False

    def record_rate_limit_error(self, config: AdaptiveRateLimitConfig) -> bool:
        """Record a 429 error. Returns True if limit was adjusted."""
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.total_429s += 1

        # Apply backoff if cooldown has passed
        if time.time() - self.last_adjustment_time >= config.cooldown_seconds:
            new_limit = max(
                config.min_concurrent,
                int(self.current_limit * config.backoff_factor),
            )
            if new_limit < self.current_limit:
                logger.warning(
                    f"Rate limit backoff (429 received): {self.current_limit} -> {new_limit}"
                )
                self.current_limit = new_limit
                self.consecutive_failures = 0
                self.last_adjustment_time = time.time()
                return True
        return False


class RateLimitedParallelExecutor(ParallelExecutor):
    """Parallel executor with per-provider rate limiting.

    Extends ParallelExecutor to add semaphore-based rate limiting
    for AI provider calls, preventing 429 rate limit errors during
    parallel execution.

    Usage:
        executor = RateLimitedParallelExecutor(
            strategy=ParallelStrategy.ASYNCIO,
            max_workers=4,
            provider_limits={"anthropic": 3, "openai": 5},
        )

        # Tasks with provider metadata
        tasks = [
            ParallelTask(id="t1", handler=agent_fn, kwargs={"provider": "anthropic"}),
            ParallelTask(id="t2", handler=agent_fn, kwargs={"provider": "openai"}),
        ]

        result = await executor.execute_parallel_rate_limited(tasks)
    """

    def __init__(
        self,
        strategy: ParallelStrategy = ParallelStrategy.ASYNCIO,
        max_workers: int = 4,
        timeout: float = 300.0,
        provider_limits: dict[str, int] | None = None,
        adaptive: bool = True,
        adaptive_config: AdaptiveRateLimitConfig | None = None,
        distributed: bool = False,
        distributed_window: int = 60,
        distributed_rpm: dict[str, int] | None = None,
    ):
        """Initialize rate-limited parallel executor.

        Args:
            strategy: Execution strategy (asyncio recommended for rate limiting)
            max_workers: Maximum concurrent workers overall
            timeout: Default timeout in seconds
            provider_limits: Dict of provider name -> max concurrent calls
            adaptive: Enable adaptive rate limiting (adjust on 429s)
            adaptive_config: Configuration for adaptive rate limiting
            distributed: Enable cross-process rate limiting
            distributed_window: Time window for distributed limits (seconds)
            distributed_rpm: Dict of provider -> requests per minute for distributed
        """
        super().__init__(strategy, max_workers, timeout)

        # Distributed rate limiting (cross-process)
        self._distributed = distributed
        self._distributed_window = distributed_window
        self._distributed_limiter = None

        # Default RPM limits (conservative for API rate limits)
        self._distributed_rpm = distributed_rpm or {
            "anthropic": 60,  # 60 RPM typical
            "openai": 90,  # Higher for GPT
            "default": 120,
        }

        # Set up per-provider limits
        defaults = ProviderRateLimits()
        limits = provider_limits or {}

        self._provider_limits = {
            "anthropic": limits.get("anthropic", defaults.anthropic),
            "openai": limits.get("openai", defaults.openai),
            "default": limits.get("default", defaults.default),
        }

        # Adaptive rate limiting
        self._adaptive = adaptive
        self._adaptive_config = adaptive_config or AdaptiveRateLimitConfig()
        self._adaptive_state: dict[str, AdaptiveRateLimitState] = {}
        self._state_lock = asyncio.Lock() if adaptive else None

        # Initialize adaptive state for each provider
        if adaptive:
            for provider, limit in self._provider_limits.items():
                self._adaptive_state[provider] = AdaptiveRateLimitState(
                    base_limit=limit,
                    current_limit=limit,
                )

        # Semaphores are created lazily in async context
        self._semaphores: dict[str, asyncio.Semaphore] | None = None

    def _get_distributed_limiter(self):
        """Get or create distributed rate limiter."""
        if not self._distributed:
            return None

        if self._distributed_limiter is None:
            from .distributed_rate_limiter import get_rate_limiter

            self._distributed_limiter = get_rate_limiter()
            logger.info("Distributed rate limiting enabled")

        return self._distributed_limiter

    async def _check_distributed_limit(self, provider: str) -> bool:
        """Check if request is allowed under distributed rate limit.

        Args:
            provider: Provider to check

        Returns:
            True if allowed, False if rate limited
        """
        limiter = self._get_distributed_limiter()
        if limiter is None:
            return True

        rpm = self._distributed_rpm.get(provider, self._distributed_rpm["default"])
        key = f"provider:{provider}"

        result = await limiter.acquire(key, rpm, self._distributed_window)

        if not result.allowed:
            logger.warning(
                f"Distributed rate limit hit for {provider}: "
                f"{result.current_count}/{rpm} in {self._distributed_window}s window. "
                f"Retry after {result.retry_after:.1f}s"
            )

        return result.allowed

    def _get_semaphores(self) -> dict[str, asyncio.Semaphore]:
        """Get or create semaphores for rate limiting.

        Must be called from async context.
        Uses adaptive limits if enabled.
        """
        if self._semaphores is None:
            if self._adaptive:
                # Use current adaptive limits
                self._semaphores = {
                    name: asyncio.Semaphore(state.current_limit)
                    for name, state in self._adaptive_state.items()
                }
            else:
                self._semaphores = {
                    name: asyncio.Semaphore(limit) for name, limit in self._provider_limits.items()
                }
        return self._semaphores

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an exception indicates a rate limit (429) error."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Check for common rate limit patterns
        if "429" in error_str or "rate limit" in error_str:
            return True
        if "too many requests" in error_str:
            return True
        if "ratelimit" in error_type:
            return True

        # Check for Anthropic/OpenAI specific errors
        if hasattr(error, "status_code") and getattr(error, "status_code") == 429:
            return True

        return False

    async def _adjust_rate_limit(
        self, provider: str, is_success: bool, error: Exception | None = None
    ) -> None:
        """Adjust rate limit based on call outcome.

        Args:
            provider: Provider name
            is_success: Whether the call succeeded
            error: The exception if call failed
        """
        if not self._adaptive or self._state_lock is None:
            return

        async with self._state_lock:
            state = self._adaptive_state.get(provider)
            if not state:
                return

            if is_success:
                adjusted = state.record_success(self._adaptive_config)
            elif error and self._is_rate_limit_error(error):
                adjusted = state.record_rate_limit_error(self._adaptive_config)
            else:
                # Non-rate-limit error, don't adjust
                return

            # If limit changed, recreate semaphore
            if adjusted and self._semaphores:
                self._semaphores[provider] = asyncio.Semaphore(state.current_limit)

    def _get_provider_for_task(self, task: ParallelTask) -> str:
        """Determine which provider a task uses.

        Checks task kwargs and handler for provider hints.
        """
        # Check explicit provider in kwargs
        provider = task.kwargs.get("provider")
        if provider:
            return provider.lower()

        # Check step_type hint
        step_type = task.kwargs.get("step_type", "").lower()
        if "claude" in step_type or "anthropic" in step_type:
            return "anthropic"
        if "openai" in step_type or "gpt" in step_type:
            return "openai"

        # Check handler name or docstring for hints
        handler_name = getattr(task.handler, "__name__", "").lower()
        if "claude" in handler_name or "anthropic" in handler_name:
            return "anthropic"
        if "openai" in handler_name or "gpt" in handler_name:
            return "openai"

        return "default"

    async def _run_task_with_rate_limit(
        self,
        task: ParallelTask,
        semaphores: dict[str, asyncio.Semaphore],
    ) -> tuple[str, Any, Exception | None]:
        """Execute a task with rate limiting based on its provider.

        Args:
            task: Task to execute
            semaphores: Provider semaphores for rate limiting

        Returns:
            Tuple of (task_id, result, error)
        """
        provider = self._get_provider_for_task(task)
        semaphore = semaphores.get(provider, semaphores.get("default"))
        if semaphore is None:
            semaphore = asyncio.Semaphore(10)  # Fallback

        task.started_at = datetime.now(UTC)

        # Check distributed rate limit (cross-process)
        if self._distributed:
            max_retries = 3
            for attempt in range(max_retries):
                allowed = await self._check_distributed_limit(provider)
                if allowed:
                    break

                # Wait and retry
                wait_time = min(5.0 * (attempt + 1), 30.0)
                logger.info(f"Task {task.id} waiting {wait_time}s for distributed rate limit")
                await asyncio.sleep(wait_time)
            else:
                # All retries exhausted
                error = RuntimeError(
                    f"Distributed rate limit exceeded for {provider} after {max_retries} retries"
                )
                task.completed_at = datetime.now(UTC)
                return task.id, None, error

        logger.debug(
            f"Task {task.id} waiting for {provider} semaphore (available: {semaphore._value})"  # noqa: SLF001
        )

        async with semaphore:
            logger.debug(f"Task {task.id} acquired {provider} semaphore")
            try:
                if asyncio.iscoroutinefunction(task.handler):
                    result = await task.handler(*task.args, **task.kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, lambda: task.handler(*task.args, **task.kwargs)
                    )
                task.completed_at = datetime.now(UTC)

                # Record success for adaptive rate limiting
                await self._adjust_rate_limit(provider, is_success=True)

                return task.id, result, None
            except asyncio.CancelledError:
                task.completed_at = datetime.now(UTC)
                raise
            except Exception as e:
                task.completed_at = datetime.now(UTC)
                logger.warning(f"Task {task.id} failed: {e}")

                # Record failure for adaptive rate limiting
                await self._adjust_rate_limit(provider, is_success=False, error=e)

                return task.id, None, e

    async def execute_parallel_rate_limited(
        self,
        tasks: list[ParallelTask],
        on_complete: Callable[[str, Any], None] | None = None,
        on_error: Callable[[str, Exception], None] | None = None,
        fail_fast: bool = False,
    ) -> ParallelResult:
        """Execute tasks in parallel with per-provider rate limiting.

        This is the primary method for rate-limited parallel execution.
        Uses asyncio for best rate limiting control.

        Args:
            tasks: List of tasks to execute
            on_complete: Callback when task completes successfully
            on_error: Callback when task fails
            fail_fast: If True, cancel remaining tasks on first failure

        Returns:
            ParallelResult with all task outcomes
        """
        result = ParallelResult()
        start_time = datetime.now(UTC)

        pending = {t.id: t for t in tasks}
        completed_ids: set[str] = set()
        should_cancel = False
        semaphores = self._get_semaphores()

        # Overall concurrency limit
        overall_semaphore = asyncio.Semaphore(self.max_workers)

        async def bounded_run(task: ParallelTask):
            async with overall_semaphore:
                return await asyncio.wait_for(
                    self._run_task_with_rate_limit(task, semaphores),
                    timeout=self.timeout,
                )

        while pending and not should_cancel:
            # Get tasks whose dependencies are satisfied
            ready = self._get_ready_tasks(pending, completed_ids)
            if not ready:
                if pending:
                    raise ValueError("Deadlock: no tasks ready but some pending")
                break

            # Create async tasks for ready parallel tasks
            async_tasks = {asyncio.create_task(bounded_run(task)): task for task in ready}

            for coro in asyncio.as_completed(async_tasks.keys()):
                try:
                    item = await coro
                except (TimeoutError, asyncio.CancelledError, Exception) as e:
                    # Find the task that failed
                    for async_task, ptask in async_tasks.items():
                        if async_task.done() and ptask.id not in completed_ids:
                            ptask.error = e if isinstance(e, Exception) else None
                            result.failed.append(ptask.id)
                            result.tasks[ptask.id] = ptask
                            completed_ids.add(ptask.id)
                            if ptask.id in pending:
                                del pending[ptask.id]
                            if on_error and isinstance(e, Exception):
                                on_error(ptask.id, e)
                    continue

                if item is None:
                    continue

                task_id, res, err = item
                task = pending.get(task_id)
                if not task:
                    continue

                task.result = res
                task.error = err
                result.tasks[task_id] = task

                if err:
                    result.failed.append(task_id)
                    if on_error:
                        on_error(task_id, err)
                    if fail_fast:
                        should_cancel = True
                        # Cancel remaining async tasks
                        for async_task, ptask in async_tasks.items():
                            if ptask.id != task_id and not async_task.done():
                                async_task.cancel()
                                result.cancelled.append(ptask.id)
                                result.tasks[ptask.id] = ptask
                                if ptask.id in pending:
                                    del pending[ptask.id]
                else:
                    result.successful.append(task_id)
                    if on_complete:
                        on_complete(task_id, res)

                completed_ids.add(task_id)
                if task_id in pending:
                    del pending[task_id]

                if should_cancel:
                    break

        # Mark remaining pending tasks as cancelled
        if should_cancel and pending:
            self._cancel_pending_tasks(pending, result)

        result.total_duration_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)
        return result

    def execute_parallel(
        self,
        tasks: list[ParallelTask],
        on_complete: Callable[[str, Any], None] | None = None,
        on_error: Callable[[str, Exception], None] | None = None,
        fail_fast: bool = False,
    ) -> ParallelResult:
        """Execute tasks in parallel with rate limiting.

        Overrides parent to use rate-limited execution for asyncio strategy.
        Falls back to parent implementation for other strategies.

        Args:
            tasks: List of tasks to execute
            on_complete: Callback when task completes
            on_error: Callback when task fails
            fail_fast: If True, cancel remaining tasks on first failure

        Returns:
            ParallelResult with all task outcomes
        """
        if self.strategy == ParallelStrategy.ASYNCIO:
            return asyncio.run(
                self.execute_parallel_rate_limited(tasks, on_complete, on_error, fail_fast)
            )
        # For threading/process, fall back to parent (no async rate limiting)
        logger.warning(
            "Rate limiting is only effective with ASYNCIO strategy. "
            f"Current strategy: {self.strategy}"
        )
        return super().execute_parallel(tasks, on_complete, on_error, fail_fast)

    def get_provider_stats(self) -> dict[str, dict]:
        """Get current stats for provider rate limits.

        Returns:
            Dict with provider name -> stats including adaptive state
        """
        stats = {}
        for provider, limit in self._provider_limits.items():
            provider_stats = {"base_limit": limit}

            if self._adaptive and provider in self._adaptive_state:
                state = self._adaptive_state[provider]
                provider_stats.update(
                    {
                        "current_limit": state.current_limit,
                        "consecutive_successes": state.consecutive_successes,
                        "total_429s": state.total_429s,
                        "is_throttled": state.current_limit < state.base_limit,
                    }
                )
            else:
                provider_stats["current_limit"] = limit

            if self._semaphores and provider in self._semaphores:
                provider_stats["available"] = self._semaphores[provider]._value  # noqa: SLF001

            # Add distributed rate limit info
            if self._distributed:
                provider_stats["distributed_enabled"] = True
                provider_stats["distributed_rpm"] = self._distributed_rpm.get(
                    provider, self._distributed_rpm["default"]
                )
                provider_stats["distributed_window"] = self._distributed_window
            else:
                provider_stats["distributed_enabled"] = False

            stats[provider] = provider_stats
        return stats

    def reset_adaptive_state(self, provider: str | None = None) -> None:
        """Reset adaptive rate limiting state to base limits.

        Args:
            provider: Specific provider to reset, or None for all
        """
        if not self._adaptive:
            return

        providers = [provider] if provider else list(self._adaptive_state.keys())
        for p in providers:
            if p in self._adaptive_state:
                state = self._adaptive_state[p]
                state.current_limit = state.base_limit
                state.consecutive_successes = 0
                state.consecutive_failures = 0
                state.last_adjustment_time = time.time()
                logger.info(f"Reset adaptive state for {p} to base limit {state.base_limit}")

                # Update semaphore if exists
                if self._semaphores and p in self._semaphores:
                    self._semaphores[p] = asyncio.Semaphore(state.base_limit)


def create_rate_limited_executor(
    max_workers: int = 4,
    anthropic_concurrent: int = 5,
    openai_concurrent: int = 8,
    timeout: float = 300.0,
    adaptive: bool = True,
    backoff_factor: float = 0.5,
    recovery_threshold: int = 10,
    distributed: bool = False,
    distributed_window: int = 60,
    anthropic_rpm: int = 60,
    openai_rpm: int = 90,
) -> RateLimitedParallelExecutor:
    """Create a rate-limited executor with common defaults.

    Args:
        max_workers: Maximum overall concurrent tasks
        anthropic_concurrent: Max concurrent Anthropic API calls (per process)
        openai_concurrent: Max concurrent OpenAI API calls (per process)
        timeout: Default timeout in seconds
        adaptive: Enable adaptive rate limiting (adjusts on 429s)
        backoff_factor: How much to reduce limit on 429 (0.5 = halve)
        recovery_threshold: Consecutive successes before recovery
        distributed: Enable cross-process rate limiting
        distributed_window: Time window for distributed limits (seconds)
        anthropic_rpm: Anthropic requests per minute (distributed)
        openai_rpm: OpenAI requests per minute (distributed)

    Returns:
        Configured RateLimitedParallelExecutor
    """
    adaptive_config = (
        AdaptiveRateLimitConfig(
            backoff_factor=backoff_factor,
            recovery_threshold=recovery_threshold,
        )
        if adaptive
        else None
    )

    distributed_rpm = (
        {
            "anthropic": anthropic_rpm,
            "openai": openai_rpm,
            "default": max(anthropic_rpm, openai_rpm),
        }
        if distributed
        else None
    )

    return RateLimitedParallelExecutor(
        strategy=ParallelStrategy.ASYNCIO,
        max_workers=max_workers,
        timeout=timeout,
        provider_limits={
            "anthropic": anthropic_concurrent,
            "openai": openai_concurrent,
        },
        adaptive=adaptive,
        adaptive_config=adaptive_config,
        distributed=distributed,
        distributed_window=distributed_window,
        distributed_rpm=distributed_rpm,
    )
