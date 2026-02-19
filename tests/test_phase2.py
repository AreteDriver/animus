"""
Tests for Phase 2: Cognitive Capabilities

Tools, Tasks, Decisions, Mode Detection
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from animus.cognitive import ReasoningMode, detect_mode
from animus.decision import Decision
from animus.tasks import Task, TaskStatus, TaskTracker
from animus.tools import (
    Tool,
    ToolRegistry,
    ToolResult,
    create_default_registry,
)

# =============================================================================
# Tool Tests
# =============================================================================


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_tool_result_success(self):
        result = ToolResult(
            tool_name="test_tool",
            success=True,
            output="Test output",
        )
        assert result.success is True
        assert result.output == "Test output"
        assert result.error is None

    def test_tool_result_failure(self):
        result = ToolResult(
            tool_name="test_tool",
            success=False,
            output=None,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_tool_result_to_context(self):
        result = ToolResult(
            tool_name="my_tool",
            success=True,
            output="The answer is 42",
        )
        context = result.to_context()
        assert "[Tool: my_tool]" in context
        assert "The answer is 42" in context

    def test_tool_result_error_to_context(self):
        result = ToolResult(
            tool_name="my_tool",
            success=False,
            output=None,
            error="Failed",
        )
        context = result.to_context()
        assert "ERROR" in context
        assert "Failed" in context


class TestTool:
    """Tests for Tool dataclass."""

    def test_tool_creation(self):
        def handler(params):
            return ToolResult("test", True, "ok")

        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            handler=handler,
        )
        assert tool.name == "test_tool"
        assert tool.requires_approval is False

    def test_tool_with_approval(self):
        def handler(params):
            return ToolResult("test", True, "ok")

        tool = Tool(
            name="dangerous_tool",
            description="Requires approval",
            parameters={},
            handler=handler,
            requires_approval=True,
        )
        assert tool.requires_approval is True

    def test_tool_get_schema(self):
        def handler(params):
            return ToolResult("test", True, "ok")

        tool = Tool(
            name="my_tool",
            description="Does things",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
            handler=handler,
        )
        schema = tool.get_schema()
        assert schema["name"] == "my_tool"
        assert schema["description"] == "Does things"
        assert "parameters" in schema


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_get(self):
        registry = ToolRegistry()

        def handler(params):
            return ToolResult("echo", True, params.get("text", ""))

        tool = Tool(
            name="echo",
            description="Echo text",
            parameters={},
            handler=handler,
        )
        registry.register(tool)

        retrieved = registry.get("echo")
        assert retrieved is not None
        assert retrieved.name == "echo"

    def test_list_tools(self):
        registry = ToolRegistry()

        def handler(params):
            return ToolResult("test", True, "ok")

        registry.register(Tool(name="a", description="A", parameters={}, handler=handler))
        registry.register(Tool(name="b", description="B", parameters={}, handler=handler))

        tools = registry.list_tools()
        assert len(tools) == 2

    def test_execute_tool(self):
        registry = ToolRegistry()

        def handler(params):
            return ToolResult("adder", True, params.get("a", 0) + params.get("b", 0))

        registry.register(Tool(name="adder", description="Add", parameters={}, handler=handler))

        result = registry.execute("adder", {"a": 2, "b": 3})
        assert result.success is True
        assert result.output == 5

    def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute("nonexistent", {})
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_unregister_tool(self):
        registry = ToolRegistry()

        def handler(params):
            return ToolResult("test", True, "ok")

        registry.register(Tool(name="temp", description="T", parameters={}, handler=handler))
        assert registry.get("temp") is not None

        registry.unregister("temp")
        assert registry.get("temp") is None


class TestBuiltinTools:
    """Tests for built-in tools."""

    def test_default_registry_has_tools(self):
        registry = create_default_registry()
        tools = registry.list_tools()
        assert len(tools) >= 5  # At least 5 built-in tools

    def test_get_datetime_tool(self):
        registry = create_default_registry()
        result = registry.execute("get_datetime", {})
        assert result.success is True
        assert len(result.output) > 0
        # Should be a date string
        assert "-" in result.output  # Contains date separator

    def test_list_files_tool(self):
        registry = create_default_registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            Path(tmpdir, "test.txt").touch()
            Path(tmpdir, "other.py").touch()

            result = registry.execute("list_files", {"directory": tmpdir, "pattern": "*.txt"})
            assert result.success is True
            assert "test.txt" in result.output

    def test_read_file_tool(self):
        registry = create_default_registry()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, World!")
            f.flush()

            result = registry.execute("read_file", {"path": f.name})
            assert result.success is True
            assert "Hello, World!" in result.output

    def test_read_file_not_found(self):
        registry = create_default_registry()
        result = registry.execute("read_file", {"path": "/nonexistent/file.txt"})
        assert result.success is False
        assert "not found" in result.error.lower()


# =============================================================================
# Task Tests
# =============================================================================


class TestTask:
    """Tests for Task dataclass."""

    def test_task_creation(self):
        now = datetime.now()
        task = Task(
            id="test-id",
            description="Test task",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING

    def test_task_to_dict(self):
        now = datetime.now()
        task = Task(
            id="test-id",
            description="Test",
            status=TaskStatus.IN_PROGRESS,
            created_at=now,
            updated_at=now,
            tags=["work"],
        )
        data = task.to_dict()
        assert data["status"] == "in_progress"
        assert data["tags"] == ["work"]

    def test_task_from_dict(self):
        data = {
            "id": "test-id",
            "description": "Test",
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "tags": ["done"],
        }
        task = Task.from_dict(data)
        assert task.status == TaskStatus.COMPLETED
        assert task.tags == ["done"]

    def test_task_is_overdue(self):
        from datetime import timedelta

        now = datetime.now()
        past = now - timedelta(days=1)

        task = Task(
            id="test",
            description="Overdue",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            due_at=past,
        )
        assert task.is_overdue() is True

    def test_completed_task_not_overdue(self):
        from datetime import timedelta

        now = datetime.now()
        past = now - timedelta(days=1)

        task = Task(
            id="test",
            description="Done",
            status=TaskStatus.COMPLETED,
            created_at=now,
            updated_at=now,
            due_at=past,
        )
        assert task.is_overdue() is False


class TestTaskTracker:
    """Tests for TaskTracker."""

    def test_add_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TaskTracker(Path(tmpdir))
            task = tracker.add("New task")

            assert task.id is not None
            assert task.description == "New task"
            assert task.status == TaskStatus.PENDING

    def test_get_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TaskTracker(Path(tmpdir))
            task = tracker.add("Test task")

            retrieved = tracker.get(task.id)
            assert retrieved is not None
            assert retrieved.description == "Test task"

    def test_get_task_partial_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TaskTracker(Path(tmpdir))
            task = tracker.add("Test task")
            partial_id = task.id[:8]

            retrieved = tracker.get(partial_id)
            assert retrieved is not None
            assert retrieved.id == task.id

    def test_update_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TaskTracker(Path(tmpdir))
            task = tracker.add("Test task")

            tracker.start(task.id)
            task = tracker.get(task.id)
            assert task.status == TaskStatus.IN_PROGRESS

            tracker.complete(task.id)
            task = tracker.get(task.id)
            assert task.status == TaskStatus.COMPLETED
            assert task.completed_at is not None

    def test_delete_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TaskTracker(Path(tmpdir))
            task = tracker.add("To delete")

            deleted = tracker.delete(task.id)
            assert deleted is True
            assert tracker.get(task.id) is None

    def test_list_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TaskTracker(Path(tmpdir))
            tracker.add("Task 1")
            tracker.add("Task 2")
            task3 = tracker.add("Task 3")
            tracker.complete(task3.id)

            # By default, excludes completed
            tasks = tracker.list()
            assert len(tasks) == 2

            # Include completed
            all_tasks = tracker.list(include_completed=True)
            assert len(all_tasks) == 3

    def test_task_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            tracker1 = TaskTracker(path)
            task = tracker1.add("Persistent task")

            tracker2 = TaskTracker(path)
            retrieved = tracker2.get(task.id)
            assert retrieved is not None
            assert retrieved.description == "Persistent task"


# =============================================================================
# Mode Detection Tests
# =============================================================================


class TestModeDetection:
    """Tests for reasoning mode detection."""

    def test_detect_quick_mode_default(self):
        mode = detect_mode("What is the weather?")
        assert mode == ReasoningMode.QUICK

    def test_detect_deep_mode_think(self):
        mode = detect_mode("Think about the implications of AI")
        assert mode == ReasoningMode.DEEP

    def test_detect_deep_mode_analyze(self):
        mode = detect_mode("Analyze this code for bugs")
        assert mode == ReasoningMode.DEEP

    def test_detect_research_mode_research(self):
        mode = detect_mode("Research the latest Python features")
        assert mode == ReasoningMode.RESEARCH

    def test_detect_research_mode_find_out(self):
        mode = detect_mode("Find out what happened in the news")
        assert mode == ReasoningMode.RESEARCH

    def test_detect_case_insensitive(self):
        mode = detect_mode("THINK ABOUT this problem")
        assert mode == ReasoningMode.DEEP


# =============================================================================
# Decision Tests
# =============================================================================


class TestDecision:
    """Tests for Decision dataclass."""

    def test_decision_creation(self):
        decision = Decision(
            question="Which database?",
            options=["PostgreSQL", "SQLite"],
            criteria=["Performance", "Simplicity"],
            analysis={
                "PostgreSQL": {"Performance": "Good", "Simplicity": "Medium"},
                "SQLite": {"Performance": "Medium", "Simplicity": "Good"},
            },
            recommendation="SQLite",
            reasoning="For simple use cases",
        )
        assert decision.question == "Which database?"
        assert len(decision.options) == 2

    def test_decision_format_analysis(self):
        decision = Decision(
            question="Test?",
            options=["A", "B"],
            criteria=["Speed"],
            analysis={"A": {"Speed": "Fast"}, "B": {"Speed": "Slow"}},
            recommendation="A",
        )
        formatted = decision.format_analysis()
        assert "Test?" in formatted
        assert "Options" in formatted
        assert "Recommendation" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
