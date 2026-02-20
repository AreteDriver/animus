"""Tests for async retry functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from animus_forge.errors import MaxRetriesError
from animus_forge.utils.retry import (
    AsyncRetryContext,
    RetryConfig,
    async_with_retry,
)


class TestAsyncWithRetry:
    """Tests for async_with_retry decorator."""

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self):
        """Successful call should not retry."""
        call_count = 0

        @async_with_retry(max_retries=3)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Should retry on ConnectionError."""
        call_count = 0

        @async_with_retry(max_retries=3, base_delay=0.01)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        result = await flaky_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Should raise MaxRetriesError after max retries."""
        call_count = 0

        @async_with_retry(max_retries=2, base_delay=0.01)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(MaxRetriesError) as exc_info:
            await always_fails()

        assert call_count == 3  # Initial + 2 retries
        assert "Max retries exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self):
        """Non-retryable errors should raise immediately."""
        call_count = 0

        @async_with_retry(max_retries=3, base_delay=0.01)
        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid value")

        with pytest.raises(ValueError):
            await raises_value_error()

        assert call_count == 1  # No retries for non-retryable error

    @pytest.mark.asyncio
    async def test_custom_retryable_exceptions(self):
        """Should retry on custom exception types."""
        call_count = 0

        class CustomError(Exception):
            pass

        @async_with_retry(max_retries=2, base_delay=0.01, retryable_exceptions=(CustomError,))
        async def raises_custom_error():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise CustomError("Custom error")
            return "success"

        result = await raises_custom_error()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Should call on_retry callback before each retry."""
        call_count = 0
        retry_calls = []

        def on_retry(exc, attempt):
            retry_calls.append((str(exc), attempt))

        @async_with_retry(max_retries=2, base_delay=0.01, on_retry=on_retry)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError(f"Fail {call_count}")
            return "success"

        await flaky_func()
        assert len(retry_calls) == 2
        assert retry_calls[0][1] == 0  # First retry (attempt 0)
        assert retry_calls[1][1] == 1  # Second retry (attempt 1)

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self):
        """Decorator should preserve function metadata."""

        @async_with_retry()
        async def documented_func():
            """This is a documented function."""
            return True

        assert documented_func.__name__ == "documented_func"
        assert "documented function" in documented_func.__doc__


class TestAsyncRetryContext:
    """Tests for AsyncRetryContext context manager."""

    @pytest.mark.asyncio
    async def test_successful_operation(self):
        """Successful operation in first attempt."""
        async with AsyncRetryContext(max_retries=3) as retry:
            for attempt in retry:
                result = "success"
                break

        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Should retry on failure and succeed."""
        attempts = []

        async with AsyncRetryContext(max_retries=3, base_delay=0.01) as retry:
            for attempt in retry:
                attempts.append(attempt)
                try:
                    if attempt < 2:
                        raise ConnectionError("Fail")
                    result = "success"
                    break
                except Exception as e:
                    await retry.handle_exception(e)

        assert attempts == [0, 1, 2]
        assert result == "success"

    @pytest.mark.asyncio
    async def test_max_retries_raises(self):
        """Should raise MaxRetriesError when exhausted."""
        async with AsyncRetryContext(
            max_retries=2, base_delay=0.01, operation_name="test_op"
        ) as retry:
            with pytest.raises(MaxRetriesError) as exc_info:
                for attempt in retry:
                    try:
                        raise ConnectionError("Always fails")
                    except Exception as e:
                        await retry.handle_exception(e)

        assert "test_op" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self):
        """Non-retryable error raises immediately."""
        async with AsyncRetryContext(max_retries=3) as retry:
            with pytest.raises(ValueError):
                for attempt in retry:
                    try:
                        raise ValueError("Not retryable")
                    except Exception as e:
                        await retry.handle_exception(e)


class TestRetryConfig:
    """Tests for RetryConfig delay calculation."""

    def test_exponential_backoff(self):
        """Delay should increase exponentially."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=100.0, jitter=False)

        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0

    def test_max_delay_cap(self):
        """Delay should not exceed max_delay."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=5.0, jitter=False)

        assert config.calculate_delay(10) == 5.0  # Capped at max

    def test_jitter_adds_randomness(self):
        """Jitter should add randomness to delay."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=100.0, jitter=True)

        # With jitter, delays should vary
        delays = [config.calculate_delay(1) for _ in range(10)]
        unique_delays = set(delays)

        # With randomness, we should have some variation
        assert len(unique_delays) > 1


class TestAsyncRetryWithMockedSleep:
    """Tests that verify asyncio.sleep is used correctly."""

    @pytest.mark.asyncio
    async def test_uses_asyncio_sleep(self):
        """Should use asyncio.sleep for delays."""
        with patch("animus_forge.utils.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            call_count = 0

            @async_with_retry(max_retries=2, base_delay=1.0, jitter=False)
            async def flaky_func():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("Fail")
                return "success"

            await flaky_func()

            # Should have slept twice (before retry 1 and retry 2)
            assert mock_sleep.call_count == 2
            # First sleep should be base_delay (1.0)
            assert mock_sleep.call_args_list[0][0][0] == 1.0
            # Second sleep should be 2.0 (exponential backoff)
            assert mock_sleep.call_args_list[1][0][0] == 2.0
