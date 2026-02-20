"""Tests for retry decorator and utilities."""

from unittest.mock import Mock, patch

import pytest

from animus_forge.errors import APIError, MaxRetriesError
from animus_forge.utils.retry import (
    RetryConfig,
    RetryContext,
    is_retryable_error,
    with_retry,
)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_calculate_delay_exponential(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

        assert config.calculate_delay(0) == 1.0  # 1 * 2^0 = 1
        assert config.calculate_delay(1) == 2.0  # 1 * 2^1 = 2
        assert config.calculate_delay(2) == 4.0  # 1 * 2^2 = 4
        assert config.calculate_delay(3) == 8.0  # 1 * 2^3 = 8

    def test_calculate_delay_max_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(base_delay=1.0, max_delay=10.0, jitter=False)

        # 2^5 = 32, but should be capped at 10
        assert config.calculate_delay(5) == 10.0
        assert config.calculate_delay(10) == 10.0

    def test_calculate_delay_with_jitter(self):
        """Test that jitter adds randomness to delay."""
        config = RetryConfig(base_delay=10.0, jitter=True)

        # With jitter, delay should be in range [7.5, 12.5] (±25%)
        delays = [config.calculate_delay(0) for _ in range(100)]

        assert all(7.5 <= d <= 12.5 for d in delays)
        # Should have some variation
        assert len(set(delays)) > 1


class TestIsRetryableError:
    """Tests for is_retryable_error function."""

    def test_api_error_with_retryable_status(self):
        """Test APIError with retryable status codes."""
        assert is_retryable_error(APIError("Rate limited", status_code=429))
        assert is_retryable_error(APIError("Server error", status_code=500))
        assert is_retryable_error(APIError("Service unavailable", status_code=503))

    def test_api_error_with_non_retryable_status(self):
        """Test APIError with non-retryable status codes."""
        assert not is_retryable_error(APIError("Not found", status_code=404))
        assert not is_retryable_error(APIError("Bad request", status_code=400))
        assert not is_retryable_error(APIError("Unauthorized", status_code=401))

    def test_connection_errors(self):
        """Test connection-related errors are retryable."""
        assert is_retryable_error(ConnectionError("Connection refused"))
        assert is_retryable_error(TimeoutError("Timed out"))
        assert is_retryable_error(OSError("Network unreachable"))

    def test_generic_exception_not_retryable(self):
        """Test generic exceptions are not retryable."""
        assert not is_retryable_error(ValueError("Invalid input"))
        assert not is_retryable_error(TypeError("Wrong type"))
        assert not is_retryable_error(RuntimeError("Runtime error"))


class TestWithRetryDecorator:
    """Tests for with_retry decorator."""

    def test_success_no_retry(self):
        """Test successful call doesn't retry."""
        call_count = 0

        @with_retry(max_retries=3)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_retryable_error(self):
        """Test retry on retryable error."""
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 3

    def test_max_retries_exceeded(self):
        """Test MaxRetriesError raised when retries exhausted."""
        call_count = 0

        @with_retry(max_retries=2, base_delay=0.01)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Connection failed")

        with pytest.raises(MaxRetriesError) as exc_info:
            always_fail()

        assert "Max retries exceeded" in str(exc_info.value)
        assert exc_info.value.attempts == 3  # Initial + 2 retries
        assert call_count == 3

    def test_non_retryable_error_raises_immediately(self):
        """Test non-retryable errors are raised immediately."""
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input")

        with pytest.raises(ValueError):
            raise_value_error()

        # Should not retry on ValueError
        assert call_count == 1

    def test_custom_retryable_exceptions(self):
        """Test custom retryable exceptions."""

        class CustomError(Exception):
            pass

        call_count = 0

        @with_retry(
            max_retries=2,
            base_delay=0.01,
            retryable_exceptions=(CustomError,),
        )
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise CustomError("Custom failure")
            return "success"

        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 2

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        callbacks = []

        def record_retry(exc, attempt):
            callbacks.append((str(exc), attempt))

        @with_retry(max_retries=2, base_delay=0.01, on_retry=record_retry)
        def fail_twice_then_succeed():
            if len(callbacks) < 2:
                raise ConnectionError(f"Failure {len(callbacks) + 1}")
            return "success"

        result = fail_twice_then_succeed()
        assert result == "success"
        assert len(callbacks) == 2
        assert callbacks[0] == ("Failure 1", 0)
        assert callbacks[1] == ("Failure 2", 1)

    @patch("animus_forge.utils.retry.time.sleep")
    def test_exponential_backoff_timing(self, mock_sleep):
        """Test that exponential backoff delays are applied."""
        call_count = 0

        @with_retry(max_retries=3, base_delay=1.0, jitter=False)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Failed")

        with pytest.raises(MaxRetriesError):
            always_fail()

        # Check sleep was called with correct delays
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert sleep_calls == [1.0, 2.0, 4.0]

    def test_preserves_function_metadata(self):
        """Test decorator preserves function name and docstring."""

        @with_retry()
        def documented_function():
            """This is the docstring."""
            return "result"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is the docstring."


class TestRetryContext:
    """Tests for RetryContext context manager."""

    def test_success_no_retry(self):
        """Test successful operation in context."""
        with RetryContext(max_retries=3) as retry:
            for attempt in retry:
                result = "success"
                break

        assert result == "success"

    def test_retry_on_error(self):
        """Test retry on transient error."""
        attempts = []

        with RetryContext(max_retries=3, base_delay=0.01) as retry:
            for attempt in retry:
                attempts.append(attempt)
                try:
                    if attempt < 2:
                        raise ConnectionError("Failed")
                    result = "success"
                    break
                except Exception as e:
                    retry.handle_exception(e)

        assert len(attempts) == 3
        assert result == "success"

    def test_max_retries_raises(self):
        """Test MaxRetriesError when retries exhausted."""
        with pytest.raises(MaxRetriesError):
            with RetryContext(max_retries=2, base_delay=0.01) as retry:
                for attempt in retry:
                    try:
                        raise ConnectionError("Always fails")
                    except Exception as e:
                        retry.handle_exception(e)

    def test_non_retryable_raises_immediately(self):
        """Test non-retryable error raises immediately."""
        attempts = []

        with pytest.raises(ValueError):
            with RetryContext(max_retries=3) as retry:
                for attempt in retry:
                    attempts.append(attempt)
                    try:
                        raise ValueError("Invalid")
                    except Exception as e:
                        retry.handle_exception(e)

        assert len(attempts) == 1


class TestRetryIntegration:
    """Integration tests for retry with mocked SDK errors."""

    def test_openai_rate_limit_retry(self):
        """Test retry on OpenAI rate limit error."""
        try:
            import openai

            call_count = 0

            @with_retry(max_retries=2, base_delay=0.01)
            def mock_openai_call():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise openai.RateLimitError(
                        "Rate limit exceeded",
                        response=Mock(status_code=429),
                        body=None,
                    )
                return "success"

            result = mock_openai_call()
            assert result == "success"
            assert call_count == 2

        except ImportError:
            pytest.skip("OpenAI not installed")

    def test_anthropic_rate_limit_retry(self):
        """Test retry on Anthropic rate limit error."""
        try:
            import anthropic

            call_count = 0

            @with_retry(max_retries=2, base_delay=0.01)
            def mock_anthropic_call():
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise anthropic.RateLimitError(
                        "Rate limit exceeded",
                        response=Mock(status_code=429),
                        body=None,
                    )
                return "success"

            result = mock_anthropic_call()
            assert result == "success"
            assert call_count == 2

        except ImportError:
            pytest.skip("Anthropic not installed")
