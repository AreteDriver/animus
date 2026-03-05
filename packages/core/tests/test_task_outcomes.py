"""
Tests for Phase 5: Task outcome tracking and learning.

Covers TaskOutcome serialization, TaskPattern formatting,
TaskOutcomeTracker operations, and _suggest_fix heuristics.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock

from animus.memory import Memory, MemoryType
from animus.task_outcomes import (
    OUTCOME_TAG,
    TaskOutcome,
    TaskOutcomeTracker,
    TaskPattern,
    _suggest_fix,
)

# ---------------------------------------------------------------------------
# TaskOutcome
# ---------------------------------------------------------------------------


class TestTaskOutcome:
    """Tests for TaskOutcome dataclass."""

    def test_to_memory_content_success(self):
        outcome = TaskOutcome(request="Read file.py", success=True)
        content = outcome.to_memory_content()
        assert "[SUCCESS]" in content
        assert "Read file.py" in content

    def test_to_memory_content_failure(self):
        outcome = TaskOutcome(
            request="Delete everything",
            success=False,
            error="Permission denied",
        )
        content = outcome.to_memory_content()
        assert "[FAILURE]" in content
        assert "Permission denied" in content

    def test_to_memory_content_with_tools(self):
        outcome = TaskOutcome(
            request="Edit code",
            success=True,
            tools_used=["read_file", "edit_file"],
        )
        content = outcome.to_memory_content()
        assert "read_file" in content
        assert "edit_file" in content

    def test_to_memory_content_with_summary(self):
        outcome = TaskOutcome(
            request="Summarize",
            success=True,
            response_summary="A brief summary of the document.",
        )
        content = outcome.to_memory_content()
        assert "Result:" in content
        assert "brief summary" in content

    def test_to_memory_content_truncates_summary(self):
        outcome = TaskOutcome(
            request="Long task",
            success=True,
            response_summary="x" * 300,
        )
        content = outcome.to_memory_content()
        # Summary should be truncated to 200 chars
        assert len(content.split("Result: ")[1]) == 200

    def test_to_json_roundtrip(self):
        outcome = TaskOutcome(
            request="Test task",
            success=True,
            tools_used=["run_command"],
            error=None,
            response_summary="All good",
            timestamp="2026-03-05T12:00:00",
        )
        json_str = outcome.to_json()
        restored = TaskOutcome.from_json(json_str)
        assert restored.request == outcome.request
        assert restored.success == outcome.success
        assert restored.tools_used == outcome.tools_used
        assert restored.error is None
        assert restored.response_summary == outcome.response_summary
        assert restored.timestamp == outcome.timestamp

    def test_to_json_is_valid_json(self):
        outcome = TaskOutcome(request="test", success=False, error="boom")
        parsed = json.loads(outcome.to_json())
        assert parsed["request"] == "test"
        assert parsed["success"] is False
        assert parsed["error"] == "boom"

    def test_from_json_failure(self):
        outcome = TaskOutcome(
            request="Fix bug",
            success=False,
            error="ImportError: no module named foo",
            tools_used=["edit_file"],
        )
        json_str = outcome.to_json()
        restored = TaskOutcome.from_json(json_str)
        assert restored.success is False
        assert "ImportError" in restored.error

    def test_timestamp_auto_generated(self):
        outcome = TaskOutcome(request="test", success=True)
        # Should be a valid ISO timestamp
        datetime.fromisoformat(outcome.timestamp)


# ---------------------------------------------------------------------------
# TaskPattern
# ---------------------------------------------------------------------------


class TestTaskPattern:
    """Tests for TaskPattern dataclass."""

    def test_to_memory_content_basic(self):
        pattern = TaskPattern(
            description="Import errors after refactor",
            occurrences=5,
        )
        content = pattern.to_memory_content()
        assert "[PATTERN]" in content
        assert "Import errors" in content
        assert "5x" in content

    def test_to_memory_content_with_suggestion(self):
        pattern = TaskPattern(
            description="Lint failures",
            occurrences=3,
            suggestion="Run ruff before committing",
        )
        content = pattern.to_memory_content()
        assert "Suggestion:" in content
        assert "ruff" in content

    def test_to_memory_content_no_suggestion(self):
        pattern = TaskPattern(description="Timeouts", occurrences=2)
        content = pattern.to_memory_content()
        assert "Suggestion:" not in content


# ---------------------------------------------------------------------------
# TaskOutcomeTracker
# ---------------------------------------------------------------------------


def _make_memory(content: str, tags: list[str], metadata: dict | None = None) -> Memory:
    """Helper to create a Memory object for mocking."""
    return Memory(
        id="mem-001",
        content=content,
        memory_type=MemoryType.PROCEDURAL,
        created_at=datetime(2026, 3, 5),
        updated_at=datetime(2026, 3, 5),
        metadata=metadata or {},
        tags=tags,
    )


class TestTaskOutcomeTrackerRecord:
    """Tests for TaskOutcomeTracker.record()."""

    def test_record_success(self):
        mock_memory = MagicMock()
        mock_memory.remember.return_value = _make_memory("test", [OUTCOME_TAG])
        tracker = TaskOutcomeTracker(mock_memory)

        outcome = TaskOutcome(request="Read a file", success=True, tools_used=["read_file"])
        mem_id = tracker.record(outcome)

        assert mem_id == "mem-001"
        call_args = mock_memory.remember.call_args
        assert "success" in call_args.kwargs["tags"]
        assert "tool:read_file" in call_args.kwargs["tags"]

    def test_record_failure(self):
        mock_memory = MagicMock()
        mock_memory.remember.return_value = _make_memory("test", [OUTCOME_TAG])
        tracker = TaskOutcomeTracker(mock_memory)

        outcome = TaskOutcome(request="Bad task", success=False, error="Boom")
        tracker.record(outcome)

        call_args = mock_memory.remember.call_args
        assert "failure" in call_args.kwargs["tags"]

    def test_record_stores_json_metadata(self):
        mock_memory = MagicMock()
        mock_memory.remember.return_value = _make_memory("test", [OUTCOME_TAG])
        tracker = TaskOutcomeTracker(mock_memory)

        outcome = TaskOutcome(request="Test", success=True)
        tracker.record(outcome)

        call_args = mock_memory.remember.call_args
        metadata = call_args.kwargs["metadata"]
        assert "outcome_json" in metadata
        parsed = json.loads(metadata["outcome_json"])
        assert parsed["request"] == "Test"


class TestTaskOutcomeTrackerRecall:
    """Tests for TaskOutcomeTracker.recall_similar()."""

    def test_recall_from_json_metadata(self):
        outcome = TaskOutcome(request="Deploy app", success=True, tools_used=["run_command"])
        mem = _make_memory(
            content="[SUCCESS] Deploy app",
            tags=[OUTCOME_TAG, "success"],
            metadata={"outcome_json": outcome.to_json()},
        )
        mock_memory = MagicMock()
        mock_memory.recall.return_value = [mem]

        tracker = TaskOutcomeTracker(mock_memory)
        results = tracker.recall_similar("Deploy the application")

        assert len(results) == 1
        assert results[0].request == "Deploy app"
        assert results[0].tools_used == ["run_command"]

    def test_recall_fallback_from_content(self):
        mem = _make_memory(
            content="[SUCCESS] Some old task",
            tags=[OUTCOME_TAG, "success"],
            metadata={},  # No JSON metadata
        )
        mock_memory = MagicMock()
        mock_memory.recall.return_value = [mem]

        tracker = TaskOutcomeTracker(mock_memory)
        results = tracker.recall_similar("old task")

        assert len(results) == 1
        assert results[0].success is True
        assert "SUCCESS" in results[0].request

    def test_recall_fallback_failure_content(self):
        mem = _make_memory(
            content="[FAILURE] Broken build",
            tags=[OUTCOME_TAG, "failure"],
            metadata={},
        )
        mock_memory = MagicMock()
        mock_memory.recall.return_value = [mem]

        tracker = TaskOutcomeTracker(mock_memory)
        results = tracker.recall_similar("build")

        assert len(results) == 1
        assert results[0].success is False

    def test_recall_handles_bad_json(self):
        mem = _make_memory(
            content="[SUCCESS] Task with bad json",
            tags=[OUTCOME_TAG],
            metadata={"outcome_json": "not valid json{{{"},
        )
        mock_memory = MagicMock()
        mock_memory.recall.return_value = [mem]

        tracker = TaskOutcomeTracker(mock_memory)
        results = tracker.recall_similar("task")

        assert len(results) == 1
        # Falls back to content parsing
        assert results[0].success is True

    def test_recall_empty(self):
        mock_memory = MagicMock()
        mock_memory.recall.return_value = []

        tracker = TaskOutcomeTracker(mock_memory)
        results = tracker.recall_similar("anything")

        assert results == []

    def test_recall_respects_limit(self):
        mock_memory = MagicMock()
        mock_memory.recall.return_value = []

        tracker = TaskOutcomeTracker(mock_memory)
        tracker.recall_similar("test", limit=5)

        mock_memory.recall.assert_called_once_with(
            query="test",
            tags=[OUTCOME_TAG],
            limit=5,
        )


class TestTaskOutcomeTrackerContext:
    """Tests for TaskOutcomeTracker.get_context_for_task()."""

    def test_context_with_results(self):
        outcome = TaskOutcome(
            request="Run tests",
            success=True,
            tools_used=["run_command"],
        )
        mem = _make_memory(
            content="[SUCCESS] Run tests",
            tags=[OUTCOME_TAG],
            metadata={"outcome_json": outcome.to_json()},
        )
        mock_memory = MagicMock()
        mock_memory.recall.return_value = [mem]

        tracker = TaskOutcomeTracker(mock_memory)
        ctx = tracker.get_context_for_task("Run pytest")

        assert ctx is not None
        assert "Past similar tasks:" in ctx
        assert "[OK]" in ctx
        assert "run_command" in ctx

    def test_context_with_failure(self):
        outcome = TaskOutcome(
            request="Deploy broken",
            success=False,
            error="Connection refused",
        )
        mem = _make_memory(
            content="[FAILURE] Deploy broken",
            tags=[OUTCOME_TAG],
            metadata={"outcome_json": outcome.to_json()},
        )
        mock_memory = MagicMock()
        mock_memory.recall.return_value = [mem]

        tracker = TaskOutcomeTracker(mock_memory)
        ctx = tracker.get_context_for_task("Deploy app")

        assert "[FAILED]" in ctx
        assert "Connection refused" in ctx

    def test_context_none_when_no_results(self):
        mock_memory = MagicMock()
        mock_memory.recall.return_value = []

        tracker = TaskOutcomeTracker(mock_memory)
        ctx = tracker.get_context_for_task("Something new")

        assert ctx is None


class TestTaskOutcomeTrackerPatterns:
    """Tests for TaskOutcomeTracker.get_failure_patterns()."""

    def test_detects_repeated_errors(self):
        mems = [
            _make_memory("[FAILURE] Task A | Error: ImportError no module named foo", ["task_outcome", "failure"]),
            _make_memory("[FAILURE] Task B | Error: ImportError no module named foo", ["task_outcome", "failure"]),
            _make_memory("[FAILURE] Task C | Error: ImportError no module named foo", ["task_outcome", "failure"]),
        ]
        mock_memory = MagicMock()
        mock_memory.recall_by_tags.return_value = mems

        tracker = TaskOutcomeTracker(mock_memory)
        patterns = tracker.get_failure_patterns()

        assert len(patterns) == 1
        assert patterns[0].occurrences == 3
        assert "ImportError" in patterns[0].description

    def test_ignores_single_occurrences(self):
        mems = [
            _make_memory("[FAILURE] Task A | Error: ImportError foo", ["task_outcome", "failure"]),
            _make_memory("[FAILURE] Task B | Error: TimeoutError bar", ["task_outcome", "failure"]),
        ]
        mock_memory = MagicMock()
        mock_memory.recall_by_tags.return_value = mems

        tracker = TaskOutcomeTracker(mock_memory)
        patterns = tracker.get_failure_patterns()

        assert len(patterns) == 0

    def test_no_failures(self):
        mock_memory = MagicMock()
        mock_memory.recall_by_tags.return_value = []

        tracker = TaskOutcomeTracker(mock_memory)
        patterns = tracker.get_failure_patterns()

        assert patterns == []

    def test_pattern_has_suggestion(self):
        mems = [
            _make_memory("[FAILURE] Lint A | Error: ruff check failed on line 42", ["task_outcome", "failure"]),
            _make_memory("[FAILURE] Lint B | Error: ruff check failed on line 42", ["task_outcome", "failure"]),
        ]
        mock_memory = MagicMock()
        mock_memory.recall_by_tags.return_value = mems

        tracker = TaskOutcomeTracker(mock_memory)
        patterns = tracker.get_failure_patterns()

        assert len(patterns) == 1
        assert "ruff" in patterns[0].suggestion.lower()


class TestTaskOutcomeTrackerStats:
    """Tests for TaskOutcomeTracker.get_success_rate()."""

    def test_success_rate_calculation(self):
        mems = [
            _make_memory("[SUCCESS] A", [OUTCOME_TAG]),
            _make_memory("[SUCCESS] B", [OUTCOME_TAG]),
            _make_memory("[FAILURE] C", [OUTCOME_TAG]),
        ]
        mock_memory = MagicMock()
        mock_memory.recall_by_tags.return_value = mems

        tracker = TaskOutcomeTracker(mock_memory)
        stats = tracker.get_success_rate()

        assert stats["total"] == 3
        assert stats["successes"] == 2
        assert stats["failures"] == 1
        assert abs(stats["rate"] - 2 / 3) < 0.01

    def test_success_rate_empty(self):
        mock_memory = MagicMock()
        mock_memory.recall_by_tags.return_value = []

        tracker = TaskOutcomeTracker(mock_memory)
        stats = tracker.get_success_rate()

        assert stats["total"] == 0
        assert stats["rate"] == 0.0

    def test_success_rate_all_success(self):
        mems = [_make_memory("[SUCCESS] X", [OUTCOME_TAG]) for _ in range(5)]
        mock_memory = MagicMock()
        mock_memory.recall_by_tags.return_value = mems

        tracker = TaskOutcomeTracker(mock_memory)
        stats = tracker.get_success_rate()

        assert stats["rate"] == 1.0
        assert stats["failures"] == 0


# ---------------------------------------------------------------------------
# _suggest_fix
# ---------------------------------------------------------------------------


class TestSuggestFix:
    """Tests for the _suggest_fix helper."""

    def test_ruff_pattern(self):
        assert "ruff" in _suggest_fix("ruff check failed").lower()

    def test_lint_pattern(self):
        assert "ruff" in _suggest_fix("lint error on line 42").lower()

    def test_import_pattern(self):
        assert "import" in _suggest_fix("ImportError: no module").lower()

    def test_test_pattern(self):
        assert "test" in _suggest_fix("AssertionError in test_foo").lower()

    def test_permission_pattern(self):
        assert "permission" in _suggest_fix("PermissionError: access denied").lower()

    def test_timeout_pattern(self):
        assert "timeout" in _suggest_fix("TimeoutError after 30s").lower()

    def test_connection_pattern(self):
        assert "service" in _suggest_fix("ConnectionRefusedError").lower()

    def test_unknown_pattern(self):
        result = _suggest_fix("Some unknown error xyz")
        assert "review" in result.lower()
