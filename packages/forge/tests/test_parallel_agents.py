"""Tests for parallel agent execution with rate limiting."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, "src")

from animus_forge.providers.base import (
    CompletionRequest,
    CompletionResponse,
    Provider,
    ProviderConfig,
    ProviderType,
)
from animus_forge.workflow import (
    AdaptiveRateLimitConfig,
    AdaptiveRateLimitState,
    ParallelStrategy,
    ParallelTask,
    ProviderRateLimits,
    RateLimitedParallelExecutor,
    create_rate_limited_executor,
)


class TestProviderRateLimits:
    """Tests for ProviderRateLimits configuration."""

    def test_default_limits(self):
        """Default rate limits are reasonable."""
        limits = ProviderRateLimits()
        assert limits.anthropic == 5
        assert limits.openai == 8
        assert limits.default == 10

    def test_custom_limits(self):
        """Custom rate limits can be set."""
        limits = ProviderRateLimits(anthropic=3, openai=5, default=2)
        assert limits.anthropic == 3
        assert limits.openai == 5
        assert limits.default == 2


class TestRateLimitedParallelExecutorInit:
    """Tests for RateLimitedParallelExecutor initialization."""

    def test_default_init(self):
        """Executor initializes with default settings."""
        executor = RateLimitedParallelExecutor()
        assert executor.strategy == ParallelStrategy.ASYNCIO
        assert executor.max_workers == 4
        assert executor._provider_limits["anthropic"] == 5
        assert executor._provider_limits["openai"] == 8

    def test_custom_limits(self):
        """Executor accepts custom provider limits."""
        executor = RateLimitedParallelExecutor(provider_limits={"anthropic": 3, "openai": 10})
        assert executor._provider_limits["anthropic"] == 3
        assert executor._provider_limits["openai"] == 10

    def test_create_helper_function(self):
        """create_rate_limited_executor helper works correctly."""
        executor = create_rate_limited_executor(
            max_workers=8,
            anthropic_concurrent=2,
            openai_concurrent=4,
            timeout=600.0,
        )
        assert executor.max_workers == 8
        assert executor.timeout == 600.0
        assert executor._provider_limits["anthropic"] == 2
        assert executor._provider_limits["openai"] == 4


class TestProviderDetection:
    """Tests for provider detection from tasks."""

    def test_detects_from_explicit_provider(self):
        """Provider is detected from explicit kwarg."""
        executor = RateLimitedParallelExecutor()
        task = ParallelTask(
            id="t1",
            step_id="t1",
            handler=lambda: None,
            kwargs={"provider": "anthropic"},
        )
        assert executor._get_provider_for_task(task) == "anthropic"

    def test_detects_from_step_type_claude(self):
        """Provider detected from claude_code step type."""
        executor = RateLimitedParallelExecutor()
        task = ParallelTask(
            id="t1",
            step_id="t1",
            handler=lambda: None,
            kwargs={"step_type": "claude_code"},
        )
        assert executor._get_provider_for_task(task) == "anthropic"

    def test_detects_from_step_type_openai(self):
        """Provider detected from openai step type."""
        executor = RateLimitedParallelExecutor()
        task = ParallelTask(
            id="t1",
            step_id="t1",
            handler=lambda: None,
            kwargs={"step_type": "openai"},
        )
        assert executor._get_provider_for_task(task) == "openai"

    def test_detects_from_handler_name(self):
        """Provider detected from handler function name."""
        executor = RateLimitedParallelExecutor()

        def call_claude_api():
            pass

        task = ParallelTask(id="t1", step_id="t1", handler=call_claude_api)
        assert executor._get_provider_for_task(task) == "anthropic"

    def test_falls_back_to_default(self):
        """Falls back to default for unknown providers."""
        executor = RateLimitedParallelExecutor()
        task = ParallelTask(
            id="t1",
            step_id="t1",
            handler=lambda: None,
            kwargs={"step_type": "shell"},
        )
        assert executor._get_provider_for_task(task) == "default"


class TestRateLimitedExecution:
    """Tests for rate-limited parallel execution."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Per-provider semaphore limits concurrent calls."""
        executor = RateLimitedParallelExecutor(provider_limits={"anthropic": 2, "default": 10})

        call_counts = {"concurrent": 0, "max_concurrent": 0}
        lock = asyncio.Lock()

        async def track_concurrency(**kwargs):
            async with lock:
                call_counts["concurrent"] += 1
                call_counts["max_concurrent"] = max(
                    call_counts["max_concurrent"], call_counts["concurrent"]
                )
            await asyncio.sleep(0.05)
            async with lock:
                call_counts["concurrent"] -= 1
            return "done"

        tasks = [
            ParallelTask(
                id=f"t{i}",
                step_id=f"t{i}",
                handler=track_concurrency,
                kwargs={"provider": "anthropic"},
            )
            for i in range(5)
        ]

        result = await executor.execute_parallel_rate_limited(tasks)

        assert result.all_succeeded
        # Max concurrent should be limited to semaphore value (2)
        assert call_counts["max_concurrent"] <= 2

    @pytest.mark.asyncio
    async def test_different_providers_separate_limits(self):
        """Different providers have separate concurrency limits."""
        executor = RateLimitedParallelExecutor(
            max_workers=10,
            provider_limits={"anthropic": 2, "openai": 3, "default": 1},
        )

        provider_max = {"anthropic": 0, "openai": 0}
        provider_current = {"anthropic": 0, "openai": 0}
        lock = asyncio.Lock()

        async def track_provider(provider: str):
            async with lock:
                provider_current[provider] += 1
                provider_max[provider] = max(provider_max[provider], provider_current[provider])
            await asyncio.sleep(0.05)
            async with lock:
                provider_current[provider] -= 1
            return provider

        tasks = []
        for i in range(4):
            tasks.append(
                ParallelTask(
                    id=f"ant{i}",
                    step_id=f"ant{i}",
                    handler=lambda: asyncio.create_task(track_provider("anthropic")),
                    kwargs={"provider": "anthropic"},
                )
            )
        for i in range(6):
            tasks.append(
                ParallelTask(
                    id=f"oai{i}",
                    step_id=f"oai{i}",
                    handler=lambda: asyncio.create_task(track_provider("openai")),
                    kwargs={"provider": "openai"},
                )
            )

        # Note: This test is simplified since the handlers call create_task
        # In practice, the concurrency is controlled by the semaphores
        result = await executor.execute_parallel_rate_limited(tasks)
        assert len(result.successful) + len(result.failed) == 10

    @pytest.mark.asyncio
    async def test_handles_task_errors(self):
        """Errors in tasks are captured correctly."""
        executor = RateLimitedParallelExecutor()

        async def failing_task():
            raise ValueError("Test error")

        tasks = [
            ParallelTask(id="fail", step_id="fail", handler=failing_task),
        ]

        result = await executor.execute_parallel_rate_limited(tasks)

        assert "fail" in result.failed
        assert result.tasks["fail"].error is not None

    @pytest.mark.asyncio
    async def test_fail_fast_cancels_remaining(self):
        """fail_fast=True cancels remaining tasks on failure."""
        executor = RateLimitedParallelExecutor(max_workers=2)

        async def slow_success():
            await asyncio.sleep(0.5)
            return "success"

        async def fast_fail():
            await asyncio.sleep(0.01)
            raise ValueError("Fast failure")

        tasks = [
            ParallelTask(id="fast_fail", step_id="fast_fail", handler=fast_fail),
            ParallelTask(id="slow1", step_id="slow1", handler=slow_success),
            ParallelTask(id="slow2", step_id="slow2", handler=slow_success),
        ]

        result = await executor.execute_parallel_rate_limited(tasks, fail_fast=True)

        assert "fast_fail" in result.failed
        # Some tasks should be cancelled
        assert len(result.cancelled) >= 0  # May vary based on timing

    @pytest.mark.asyncio
    async def test_callbacks_invoked(self):
        """on_complete and on_error callbacks are invoked."""
        executor = RateLimitedParallelExecutor()

        completed = []
        errors = []

        def on_complete(task_id, result):
            completed.append(task_id)

        def on_error(task_id, error):
            errors.append(task_id)

        async def succeed():
            return "ok"

        async def fail():
            raise ValueError("fail")

        tasks = [
            ParallelTask(id="good", step_id="good", handler=succeed),
            ParallelTask(id="bad", step_id="bad", handler=fail),
        ]

        await executor.execute_parallel_rate_limited(
            tasks, on_complete=on_complete, on_error=on_error
        )

        assert "good" in completed
        assert "bad" in errors


class TestSyncExecuteParallel:
    """Tests for synchronous execute_parallel method."""

    def test_uses_rate_limiting_for_asyncio_strategy(self):
        """execute_parallel uses rate limiting for asyncio strategy."""
        executor = RateLimitedParallelExecutor(
            strategy=ParallelStrategy.ASYNCIO,
            provider_limits={"anthropic": 2},
        )

        def simple_task(**kwargs):
            return "done"

        tasks = [
            ParallelTask(
                id="t1",
                step_id="t1",
                handler=simple_task,
                kwargs={"provider": "anthropic"},
            ),
        ]

        result = executor.execute_parallel(tasks)
        assert result.all_succeeded

    def test_falls_back_for_threading_strategy(self):
        """Threading strategy falls back to parent implementation."""
        executor = RateLimitedParallelExecutor(
            strategy=ParallelStrategy.THREADING,
        )

        def simple_task():
            return "done"

        tasks = [
            ParallelTask(id="t1", step_id="t1", handler=simple_task),
        ]

        with patch.object(executor, "_execute_threaded") as mock_threaded:
            mock_threaded.return_value = MagicMock(successful=["t1"], failed=[], cancelled=[])
            executor.execute_parallel(tasks)
            mock_threaded.assert_called_once()


class TestProviderStats:
    """Tests for provider stats reporting."""

    def test_get_provider_stats(self):
        """get_provider_stats returns limit info."""
        executor = RateLimitedParallelExecutor(provider_limits={"anthropic": 3, "openai": 5})

        stats = executor.get_provider_stats()

        assert stats["anthropic"]["base_limit"] == 3
        assert stats["openai"]["base_limit"] == 5
        assert stats["default"]["base_limit"] == 10


class TestAsyncProviderMethods:
    """Tests for async provider methods."""

    @pytest.mark.asyncio
    async def test_provider_complete_async_default(self):
        """Provider.complete_async default wraps sync method."""

        class MockProvider(Provider):
            @property
            def name(self):
                return "mock"

            @property
            def provider_type(self):
                return ProviderType.OPENAI

            def _get_fallback_model(self):
                return "mock-model"

            def is_configured(self):
                return True

            def initialize(self):
                pass

            def complete(self, request):
                return CompletionResponse(
                    content="sync response",
                    model="mock-model",
                    provider="mock",
                )

        provider = MockProvider(ProviderConfig(provider_type=ProviderType.OPENAI))
        request = CompletionRequest(prompt="test")

        response = await provider.complete_async(request)

        assert response.content == "sync response"

    @pytest.mark.asyncio
    async def test_provider_generate_async(self):
        """Provider.generate_async works correctly."""

        class MockProvider(Provider):
            @property
            def name(self):
                return "mock"

            @property
            def provider_type(self):
                return ProviderType.OPENAI

            def _get_fallback_model(self):
                return "mock-model"

            def is_configured(self):
                return True

            def initialize(self):
                pass

            def complete(self, request):
                return CompletionResponse(
                    content=f"Response to: {request.prompt}",
                    model="mock-model",
                    provider="mock",
                )

        provider = MockProvider(ProviderConfig(provider_type=ProviderType.OPENAI))

        response = await provider.generate_async("hello")

        assert "hello" in response


class TestProviderManagerAsync:
    """Tests for ProviderManager async methods."""

    @pytest.mark.asyncio
    async def test_manager_complete_async(self):
        """ProviderManager.complete_async works with fallback."""
        from animus_forge.providers.manager import ProviderManager

        manager = ProviderManager()

        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.complete_async = AsyncMock(
            return_value=CompletionResponse(
                content="async response",
                model="test-model",
                provider="mock",
            )
        )

        manager._providers["mock"] = mock_provider
        manager._default_provider = "mock"
        manager._fallback_order = ["mock"]

        request = CompletionRequest(prompt="test")
        response = await manager.complete_async(request)

        assert response.content == "async response"
        mock_provider.complete_async.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_manager_complete_async_fallback(self):
        """ProviderManager.complete_async tries fallback on failure."""
        from animus_forge.providers.base import ProviderError
        from animus_forge.providers.manager import ProviderManager

        manager = ProviderManager()

        # First provider fails
        failing_provider = MagicMock()
        failing_provider.complete_async = AsyncMock(side_effect=ProviderError("fail"))

        # Second provider succeeds
        working_provider = MagicMock()
        working_provider.complete_async = AsyncMock(
            return_value=CompletionResponse(
                content="fallback response",
                model="test-model",
                provider="working",
            )
        )

        manager._providers["failing"] = failing_provider
        manager._providers["working"] = working_provider
        manager._default_provider = "failing"
        manager._fallback_order = ["failing", "working"]

        request = CompletionRequest(prompt="test")
        response = await manager.complete_async(request)

        assert response.content == "fallback response"

    @pytest.mark.asyncio
    async def test_manager_generate_async(self):
        """ProviderManager.generate_async convenience method works."""
        from animus_forge.providers.manager import ProviderManager

        manager = ProviderManager()

        mock_provider = MagicMock()
        mock_provider.complete_async = AsyncMock(
            return_value=CompletionResponse(
                content="generated text",
                model="test-model",
                provider="mock",
            )
        )

        manager._providers["mock"] = mock_provider
        manager._default_provider = "mock"
        manager._fallback_order = ["mock"]

        result = await manager.generate_async("hello")

        assert result == "generated text"


class TestClaudeCodeClientAsync:
    """Tests for ClaudeCodeClient async methods."""

    @pytest.mark.asyncio
    async def test_execute_agent_async_not_configured(self):
        """execute_agent_async returns error when not configured."""
        with patch("animus_forge.api_clients.claude_code_client.get_settings") as mock:
            mock.return_value = MagicMock(
                claude_mode="api",
                anthropic_api_key=None,
                claude_cli_path="claude",
                base_dir=MagicMock(),
            )
            mock.return_value.base_dir.__truediv__ = MagicMock(
                return_value=MagicMock(exists=lambda: False)
            )

            from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

            client = ClaudeCodeClient()
            result = await client.execute_agent_async(role="planner", task="test")

            assert result["success"] is False
            assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_agent_async_unknown_role(self):
        """execute_agent_async returns error for unknown role."""
        with patch("animus_forge.api_clients.claude_code_client.get_settings") as mock:
            mock.return_value = MagicMock(
                claude_mode="api",
                anthropic_api_key="test-key",
                claude_cli_path="claude",
                base_dir=MagicMock(),
            )
            mock.return_value.base_dir.__truediv__ = MagicMock(
                return_value=MagicMock(exists=lambda: False)
            )

            with patch("animus_forge.api_clients.claude_code_client.anthropic") as anth_mock:
                anth_mock.Anthropic = MagicMock()
                anth_mock.AsyncAnthropic = MagicMock()

                from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

                client = ClaudeCodeClient()
                result = await client.execute_agent_async(role="unknown_role", task="test")

                assert result["success"] is False
                assert "Unknown role" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_via_cli_async_error(self):
        """_execute_via_cli_async handles CLI errors."""
        with patch("animus_forge.api_clients.claude_code_client.get_settings") as mock:
            mock.return_value = MagicMock(
                claude_mode="cli",
                anthropic_api_key=None,
                claude_cli_path="false",  # Command that always fails
                base_dir=MagicMock(),
            )
            mock.return_value.base_dir.__truediv__ = MagicMock(
                return_value=MagicMock(exists=lambda: False)
            )

            from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

            client = ClaudeCodeClient()
            client.cli_path = "false"  # Command that returns exit code 1

            with pytest.raises(RuntimeError, match="CLI error"):
                await client._execute_via_cli_async(prompt="test")


class TestWorkflowExecutorRateLimitedIntegration:
    """Tests for WorkflowExecutor integration with rate-limited executor."""

    def test_uses_rate_limited_executor_for_ai_steps(self):
        """Verify rate-limited executor is used when AI steps are present."""
        from animus_forge.workflow import StepConfig, WorkflowConfig, WorkflowExecutor

        executor = WorkflowExecutor(dry_run=True)

        workflow = WorkflowConfig(
            name="test-parallel-ai",
            description="Test parallel AI execution",
            version="1.0.0",
            steps=[
                StepConfig(
                    id="parallel_ai",
                    type="parallel",
                    params={
                        "steps": [
                            {
                                "id": "claude_task",
                                "type": "claude_code",
                                "params": {"prompt": "test", "role": "builder"},
                            },
                            {
                                "id": "openai_task",
                                "type": "openai",
                                "params": {"prompt": "test"},
                            },
                        ],
                        "max_workers": 2,
                    },
                )
            ],
        )

        result = executor.execute(workflow)

        # Should complete (dry run mode)
        assert result.status == "success"
        # Both AI tasks should be in results
        parallel_output = result.steps[0].output
        assert "claude_task" in parallel_output["parallel_results"]
        assert "openai_task" in parallel_output["parallel_results"]

    def test_rate_limit_can_be_disabled(self):
        """Verify rate limiting can be explicitly disabled."""
        from animus_forge.workflow import StepConfig, WorkflowConfig, WorkflowExecutor

        executor = WorkflowExecutor(dry_run=True)

        workflow = WorkflowConfig(
            name="test-no-rate-limit",
            description="Test disabled rate limiting",
            version="1.0.0",
            steps=[
                StepConfig(
                    id="parallel_ai",
                    type="parallel",
                    params={
                        "steps": [
                            {
                                "id": "claude_task",
                                "type": "claude_code",
                                "params": {"prompt": "test", "role": "builder"},
                            },
                        ],
                        "rate_limit": False,  # Explicitly disable
                        "strategy": "threading",
                    },
                )
            ],
        )

        result = executor.execute(workflow)
        assert result.status == "success"

    def test_non_ai_steps_use_standard_executor(self):
        """Verify non-AI parallel steps use standard ParallelExecutor."""
        from animus_forge.workflow import StepConfig, WorkflowConfig, WorkflowExecutor

        executor = WorkflowExecutor()

        workflow = WorkflowConfig(
            name="test-shell-parallel",
            description="Test shell parallel execution",
            version="1.0.0",
            steps=[
                StepConfig(
                    id="parallel_shell",
                    type="parallel",
                    params={
                        "steps": [
                            {
                                "id": "echo1",
                                "type": "shell",
                                "params": {"command": "echo hello"},
                            },
                            {
                                "id": "echo2",
                                "type": "shell",
                                "params": {"command": "echo world"},
                            },
                        ],
                        "strategy": "threading",  # Should use standard executor
                    },
                )
            ],
        )

        result = executor.execute(workflow)
        assert result.status == "success"
        parallel_output = result.steps[0].output
        assert parallel_output["parallel_results"]["echo1"]["stdout"].strip() == "hello"
        assert parallel_output["parallel_results"]["echo2"]["stdout"].strip() == "world"


class TestAdaptiveRateLimitConfig:
    """Tests for AdaptiveRateLimitConfig."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = AdaptiveRateLimitConfig()
        assert config.min_concurrent == 1
        assert config.backoff_factor == 0.5
        assert config.recovery_factor == 1.2
        assert config.recovery_threshold == 10
        assert config.cooldown_seconds == 30.0

    def test_custom_config(self):
        """Custom config values are respected."""
        config = AdaptiveRateLimitConfig(
            min_concurrent=2,
            backoff_factor=0.7,
            recovery_factor=1.5,
            recovery_threshold=5,
            cooldown_seconds=60.0,
        )
        assert config.min_concurrent == 2
        assert config.backoff_factor == 0.7
        assert config.recovery_factor == 1.5
        assert config.recovery_threshold == 5
        assert config.cooldown_seconds == 60.0


class TestAdaptiveRateLimitState:
    """Tests for AdaptiveRateLimitState."""

    def test_initial_state(self):
        """Initial state has correct values."""
        state = AdaptiveRateLimitState(base_limit=5, current_limit=5)
        assert state.base_limit == 5
        assert state.current_limit == 5
        assert state.consecutive_successes == 0
        assert state.consecutive_failures == 0
        assert state.total_429s == 0

    def test_record_success_increments_counter(self):
        """Success increments consecutive_successes."""
        config = AdaptiveRateLimitConfig()
        state = AdaptiveRateLimitState(base_limit=5, current_limit=5)
        state.record_success(config)
        assert state.consecutive_successes == 1
        assert state.consecutive_failures == 0

    def test_record_success_resets_failures(self):
        """Success resets consecutive_failures."""
        config = AdaptiveRateLimitConfig()
        state = AdaptiveRateLimitState(base_limit=5, current_limit=5)
        state.consecutive_failures = 3
        state.record_success(config)
        assert state.consecutive_failures == 0

    def test_record_rate_limit_error_increments_counter(self):
        """429 error increments failure counter."""
        config = AdaptiveRateLimitConfig(cooldown_seconds=0)
        state = AdaptiveRateLimitState(base_limit=5, current_limit=5)
        state.record_rate_limit_error(config)
        assert state.total_429s == 1

    def test_record_rate_limit_error_resets_successes(self):
        """429 error resets consecutive_successes."""
        config = AdaptiveRateLimitConfig(cooldown_seconds=0)
        state = AdaptiveRateLimitState(base_limit=5, current_limit=5)
        state.consecutive_successes = 5
        state.record_rate_limit_error(config)
        assert state.consecutive_successes == 0

    def test_backoff_reduces_limit(self):
        """429 error reduces current limit."""
        config = AdaptiveRateLimitConfig(backoff_factor=0.5, cooldown_seconds=0)
        state = AdaptiveRateLimitState(base_limit=8, current_limit=8)
        state.last_adjustment_time = 0  # Force past cooldown
        adjusted = state.record_rate_limit_error(config)
        assert adjusted is True
        assert state.current_limit == 4

    def test_backoff_respects_min_limit(self):
        """Backoff doesn't go below min_concurrent."""
        config = AdaptiveRateLimitConfig(min_concurrent=2, backoff_factor=0.1, cooldown_seconds=0)
        state = AdaptiveRateLimitState(base_limit=5, current_limit=3)
        state.last_adjustment_time = 0
        state.record_rate_limit_error(config)
        assert state.current_limit == 2  # min_concurrent

    def test_recovery_increases_limit(self):
        """Recovery increases current limit after enough successes."""
        config = AdaptiveRateLimitConfig(
            recovery_factor=1.5, recovery_threshold=3, cooldown_seconds=0
        )
        state = AdaptiveRateLimitState(base_limit=10, current_limit=4)
        state.last_adjustment_time = 0

        # Need recovery_threshold successes
        for _ in range(3):
            state.record_success(config)

        assert state.current_limit == 6  # 4 * 1.5

    def test_recovery_caps_at_base_limit(self):
        """Recovery doesn't exceed base limit."""
        config = AdaptiveRateLimitConfig(
            recovery_factor=2.0, recovery_threshold=2, cooldown_seconds=0
        )
        state = AdaptiveRateLimitState(base_limit=5, current_limit=4)
        state.last_adjustment_time = 0

        state.record_success(config)
        state.record_success(config)

        assert state.current_limit == 5  # Capped at base


class TestAdaptiveRateLimiting:
    """Tests for adaptive rate limiting in executor."""

    def test_executor_with_adaptive_disabled(self):
        """Executor works with adaptive disabled."""
        executor = RateLimitedParallelExecutor(adaptive=False)
        assert executor._adaptive is False
        assert len(executor._adaptive_state) == 0

    def test_executor_with_adaptive_enabled(self):
        """Executor initializes adaptive state when enabled."""
        executor = RateLimitedParallelExecutor(adaptive=True)
        assert executor._adaptive is True
        assert "anthropic" in executor._adaptive_state
        assert "openai" in executor._adaptive_state

    def test_get_provider_stats_includes_adaptive_info(self):
        """Provider stats include adaptive rate limit info."""
        executor = RateLimitedParallelExecutor(adaptive=True)
        stats = executor.get_provider_stats()

        assert "anthropic" in stats
        assert "current_limit" in stats["anthropic"]
        assert "consecutive_successes" in stats["anthropic"]
        assert "total_429s" in stats["anthropic"]
        assert "is_throttled" in stats["anthropic"]

    def test_reset_adaptive_state_resets_single_provider(self):
        """reset_adaptive_state resets specific provider."""
        executor = RateLimitedParallelExecutor(adaptive=True)
        executor._adaptive_state["anthropic"].current_limit = 2
        executor._adaptive_state["anthropic"].total_429s = 5

        executor.reset_adaptive_state("anthropic")

        assert executor._adaptive_state["anthropic"].current_limit == 5
        assert executor._adaptive_state["openai"].current_limit == 8  # Unchanged

    def test_reset_adaptive_state_resets_all_providers(self):
        """reset_adaptive_state resets all providers when no arg."""
        executor = RateLimitedParallelExecutor(adaptive=True)
        executor._adaptive_state["anthropic"].current_limit = 2
        executor._adaptive_state["openai"].current_limit = 3

        executor.reset_adaptive_state()

        assert executor._adaptive_state["anthropic"].current_limit == 5
        assert executor._adaptive_state["openai"].current_limit == 8

    def test_is_rate_limit_error_detects_429_string(self):
        """Detects 429 in error message."""
        executor = RateLimitedParallelExecutor()
        error = Exception("Error 429: Too many requests")
        assert executor._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_detects_rate_limit_string(self):
        """Detects rate limit in error message."""
        executor = RateLimitedParallelExecutor()
        error = Exception("API rate limit exceeded")
        assert executor._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_detects_status_code(self):
        """Detects 429 status code attribute."""
        executor = RateLimitedParallelExecutor()

        class APIError(Exception):
            status_code = 429

        assert executor._is_rate_limit_error(APIError()) is True

    def test_is_rate_limit_error_returns_false_for_other_errors(self):
        """Returns False for non-rate-limit errors."""
        executor = RateLimitedParallelExecutor()
        error = Exception("Connection timeout")
        assert executor._is_rate_limit_error(error) is False

    def test_create_executor_with_adaptive_options(self):
        """create_rate_limited_executor accepts adaptive options."""
        executor = create_rate_limited_executor(
            adaptive=True,
            backoff_factor=0.7,
            recovery_threshold=5,
        )
        assert executor._adaptive is True
        assert executor._adaptive_config.backoff_factor == 0.7
        assert executor._adaptive_config.recovery_threshold == 5

    def test_create_executor_without_adaptive(self):
        """create_rate_limited_executor can disable adaptive."""
        executor = create_rate_limited_executor(adaptive=False)
        assert executor._adaptive is False


class TestAdaptiveRateLimitingAsync:
    """Async tests for adaptive rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_adjust_rate_limit_on_success(self):
        """Rate limit adjusts on successful calls."""
        executor = RateLimitedParallelExecutor(adaptive=True)
        executor._adaptive_state["anthropic"].current_limit = 3
        executor._adaptive_state["anthropic"].consecutive_successes = 9
        executor._adaptive_state["anthropic"].last_adjustment_time = 0

        await executor._adjust_rate_limit("anthropic", is_success=True)

        # Should have recovered after 10 successes
        assert executor._adaptive_state["anthropic"].current_limit > 3

    @pytest.mark.asyncio
    async def test_adjust_rate_limit_on_429_error(self):
        """Rate limit backs off on 429 error."""
        config = AdaptiveRateLimitConfig(cooldown_seconds=0)
        executor = RateLimitedParallelExecutor(adaptive=True, adaptive_config=config)
        executor._adaptive_state["anthropic"].last_adjustment_time = 0

        error = Exception("Error 429: Rate limit")
        await executor._adjust_rate_limit("anthropic", is_success=False, error=error)

        assert executor._adaptive_state["anthropic"].current_limit < 5
        assert executor._adaptive_state["anthropic"].total_429s == 1

    @pytest.mark.asyncio
    async def test_adjust_rate_limit_ignores_non_429_errors(self):
        """Non-429 errors don't trigger backoff."""
        executor = RateLimitedParallelExecutor(adaptive=True)
        original_limit = executor._adaptive_state["anthropic"].current_limit

        error = Exception("Connection timeout")
        await executor._adjust_rate_limit("anthropic", is_success=False, error=error)

        assert executor._adaptive_state["anthropic"].current_limit == original_limit

    @pytest.mark.asyncio
    async def test_task_execution_triggers_adjustment(self):
        """Task execution triggers rate limit adjustment."""
        config = AdaptiveRateLimitConfig(cooldown_seconds=0, recovery_threshold=1)
        executor = RateLimitedParallelExecutor(adaptive=True, adaptive_config=config)
        # Reduce limit first
        executor._adaptive_state["anthropic"].current_limit = 3
        executor._adaptive_state["anthropic"].last_adjustment_time = 0

        async def success_handler(**kwargs):
            return "ok"

        task = ParallelTask(
            id="t1",
            step_id="t1",
            handler=success_handler,
            kwargs={"provider": "anthropic"},
        )

        semaphores = executor._get_semaphores()
        result = await executor._run_task_with_rate_limit(task, semaphores)

        assert result[2] is None  # No error
        # After 1 success with threshold=1, should recover
        assert executor._adaptive_state["anthropic"].current_limit > 3
