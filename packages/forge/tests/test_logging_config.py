"""Tests for logging configuration module."""

import json
import logging
import sys
from io import StringIO

import pytest

sys.path.insert(0, "src")

from animus_forge.config.logging import (
    JSONFormatter,
    SanitizingFilter,
    TextFormatter,
    configure_logging,
)


class TestSanitizingFilter:
    """Tests for SanitizingFilter class."""

    @pytest.fixture
    def filter(self):
        """Create SanitizingFilter instance."""
        return SanitizingFilter()

    def test_filter_always_returns_true(self, filter):
        """Filter always returns True (passes all records)."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = filter.filter(record)
        assert result is True

    def test_sanitizes_api_key_in_message(self, filter):
        """Sanitizes API keys in log messages."""
        # OpenAI key pattern: sk- followed by 20+ alphanumeric chars
        # Using obviously fake test value (FAKE repeated)
        api_key = "sk-FAKEFAKEFAKEFAKEFAKE"
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=f"Request with key {api_key} failed",
            args=(),
            exc_info=None,
        )
        filter.filter(record)
        assert api_key not in record.msg
        assert "[REDACTED_API_KEY]" in record.msg

    def test_sanitizes_github_token_in_message(self, filter):
        """Sanitizes GitHub tokens in log messages."""
        # GitHub PAT pattern: ghp_ followed by 36 alphanumeric chars
        # Using obviously fake test value (FAKE repeated)
        github_token = "ghp_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE"
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=f"Authorization: token {github_token}",
            args=(),
            exc_info=None,
        )
        filter.filter(record)
        assert github_token not in record.msg
        assert "[REDACTED_GITHUB_TOKEN]" in record.msg

    def test_sanitizes_string_args(self, filter):
        """Sanitizes string arguments in log record."""
        # OpenAI key pattern in args
        # Using obviously fake test value (FAKE repeated)
        api_key = "sk-FAKEFAKEFAKEFAKEFAKE"
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="User %s made request with key %s",
            args=("alice", api_key),
            exc_info=None,
        )
        filter.filter(record)
        assert record.args[0] == "alice"  # Non-sensitive unchanged
        assert api_key not in record.args[1]

    def test_preserves_non_string_args(self, filter):
        """Preserves non-string arguments unchanged."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Count: %d, Rate: %f",
            args=(42, 3.14),
            exc_info=None,
        )
        filter.filter(record)
        assert record.args == (42, 3.14)

    def test_handles_empty_message(self, filter):
        """Handles empty message gracefully."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="",
            args=(),
            exc_info=None,
        )
        result = filter.filter(record)
        assert result is True
        assert record.msg == ""

    def test_handles_none_args(self, filter):
        """Handles None args gracefully."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="No args",
            args=None,
            exc_info=None,
        )
        result = filter.filter(record)
        assert result is True


class TestJSONFormatter:
    """Tests for JSONFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create JSONFormatter instance."""
        return JSONFormatter()

    def test_format_returns_valid_json(self, formatter):
        """Format returns valid JSON string."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_format_includes_required_fields(self, formatter):
        """Format includes timestamp, level, logger, message."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        parsed = json.loads(result)

        assert "timestamp" in parsed
        assert parsed["level"] == "WARNING"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Warning message"

    def test_timestamp_format(self, formatter):
        """Timestamp is in ISO format with Z suffix."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["timestamp"].endswith("+00:00")
        assert "T" in parsed["timestamp"]

    def test_includes_optional_fields_when_present(self, formatter):
        """Includes optional fields when set on record."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Request",
            args=(),
            exc_info=None,
        )
        record.trace_id = "trace-123"
        record.request_id = "req-456"
        record.duration_ms = 150

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["trace_id"] == "trace-123"
        assert parsed["request_id"] == "req-456"
        assert parsed["duration_ms"] == 150

    def test_includes_http_fields(self, formatter):
        """Includes HTTP-related fields when present."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="HTTP request",
            args=(),
            exc_info=None,
        )
        setattr(record, "http.method", "POST")
        setattr(record, "http.path", "/api/v1/users")
        setattr(record, "http.status_code", 200)

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["http.method"] == "POST"
        assert parsed["http.path"] == "/api/v1/users"
        assert parsed["http.status_code"] == 200

    def test_includes_exception_info(self, formatter):
        """Includes exception info when present."""
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert "Test error" in parsed["exception"]

    def test_formats_message_with_args(self, formatter):
        """Formats message with arguments."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="User %s logged in from %s",
            args=("alice", "192.168.1.1"),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["message"] == "User alice logged in from 192.168.1.1"


class TestTextFormatter:
    """Tests for TextFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create TextFormatter instance."""
        return TextFormatter()

    def test_format_includes_all_parts(self, formatter):
        """Format includes time, name, level, message."""
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "test.module" in result
        assert "INFO" in result
        assert "Test message" in result

    def test_format_date_format(self, formatter):
        """Uses correct date format."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        # Should have YYYY-MM-DD HH:MM:SS format
        import re

        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", result)

    def test_format_separator(self, formatter):
        """Uses dash separators between fields."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert " - " in result


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def setup_method(self):
        """Reset root logger before each test."""
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    def test_sets_log_level(self):
        """Sets correct log level on root logger."""
        configure_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_sets_warning_level(self):
        """Can set WARNING level."""
        configure_logging(level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_case_insensitive_level(self):
        """Level parameter is case insensitive."""
        configure_logging(level="debug")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_uses_text_formatter_by_default(self):
        """Uses TextFormatter when format='text'."""
        configure_logging(format="text")
        root = logging.getLogger()

        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, TextFormatter)

    def test_uses_json_formatter(self):
        """Uses JSONFormatter when format='json'."""
        configure_logging(format="json")
        root = logging.getLogger()

        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_adds_sanitizing_filter_by_default(self):
        """Adds SanitizingFilter by default."""
        configure_logging(sanitize_logs=True)
        root = logging.getLogger()

        handler = root.handlers[0]
        filter_types = [type(f) for f in handler.filters]
        assert SanitizingFilter in filter_types

    def test_no_sanitizing_filter_when_disabled(self):
        """No SanitizingFilter when sanitize_logs=False."""
        configure_logging(sanitize_logs=False)
        root = logging.getLogger()

        handler = root.handlers[0]
        filter_types = [type(f) for f in handler.filters]
        assert SanitizingFilter not in filter_types

    def test_removes_existing_handlers(self):
        """Removes existing handlers before adding new one."""
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())

        configure_logging()

        assert len(root.handlers) == 1

    def test_silences_noisy_libraries(self):
        """Sets WARNING level on noisy third-party loggers."""
        configure_logging()

        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert logging.getLogger("uvicorn.access").level == logging.WARNING
        assert logging.getLogger("apscheduler").level == logging.WARNING

    def test_handler_uses_stdout(self):
        """Console handler writes to stdout."""
        configure_logging()
        root = logging.getLogger()

        handler = root.handlers[0]
        assert handler.stream == sys.stdout

    def test_invalid_level_defaults_to_info(self):
        """Invalid log level defaults to INFO."""
        configure_logging(level="INVALID")
        root = logging.getLogger()
        assert root.level == logging.INFO


class TestLoggingIntegration:
    """Integration tests for logging configuration."""

    def setup_method(self):
        """Reset root logger before each test."""
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    def test_json_output_captures_to_stream(self):
        """JSON logging produces valid JSON output."""
        stream = StringIO()

        configure_logging(level="INFO", format="json")
        root = logging.getLogger()

        # Replace handler stream for testing
        root.handlers[0].stream = stream

        logger = logging.getLogger("test.integration")
        logger.info("Test log message")

        output = stream.getvalue()
        parsed = json.loads(output.strip())

        assert parsed["message"] == "Test log message"
        assert parsed["level"] == "INFO"

    def test_text_output_format(self):
        """Text logging produces expected format."""
        stream = StringIO()

        configure_logging(level="INFO", format="text")
        root = logging.getLogger()

        root.handlers[0].stream = stream

        logger = logging.getLogger("test.integration")
        logger.warning("Warning message")

        output = stream.getvalue()

        assert "WARNING" in output
        assert "test.integration" in output
        assert "Warning message" in output

    def test_sanitization_in_json_output(self):
        """Sanitization works with JSON output."""
        stream = StringIO()

        configure_logging(level="INFO", format="json", sanitize_logs=True)
        root = logging.getLogger()
        root.handlers[0].stream = stream

        # Use pattern that matches sanitization regex
        # Using obviously fake test value (FAKE repeated)
        api_key = "sk-FAKEFAKEFAKEFAKEFAKE"
        logger = logging.getLogger("test.sanitize")
        logger.info(f"API key: {api_key}")

        output = stream.getvalue()
        parsed = json.loads(output.strip())

        assert api_key not in parsed["message"]
        assert "[REDACTED_API_KEY]" in parsed["message"]

    def test_debug_messages_filtered_at_info_level(self):
        """Debug messages filtered when level is INFO."""
        stream = StringIO()

        configure_logging(level="INFO", format="text")
        root = logging.getLogger()
        root.handlers[0].stream = stream

        logger = logging.getLogger("test.level")
        logger.debug("Debug message")
        logger.info("Info message")

        output = stream.getvalue()

        assert "Debug message" not in output
        assert "Info message" in output
