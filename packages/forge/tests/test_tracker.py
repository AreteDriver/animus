"""Tests for execution tracker module."""

from unittest.mock import MagicMock, patch

import pytest

from animus_forge.monitoring.tracker import (
    AgentTracker,
    ExecutionTracker,
    get_tracker,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_metrics_store():
    """Create mock metrics store."""
    with patch("animus_forge.monitoring.tracker.MetricsStore") as mock_class:
        mock_store = MagicMock()
        mock_store.get_summary.return_value = {
            "total": 10,
            "completed": 8,
            "failed": 2,
        }
        mock_store.get_active_workflows.return_value = []
        mock_store.get_recent_executions.return_value = []
        mock_store.get_step_performance.return_value = {}
        mock_class.return_value = mock_store
        yield mock_store


@pytest.fixture
def tracker(mock_metrics_store):
    """Create execution tracker with mocked store."""
    return ExecutionTracker()


@pytest.fixture
def agent_tracker():
    """Create agent tracker."""
    return AgentTracker()


@pytest.fixture(autouse=True)
def reset_global_tracker():
    """Reset global tracker before each test."""
    import animus_forge.monitoring.tracker as tracker_module

    tracker_module._tracker = None
    yield
    tracker_module._tracker = None


# =============================================================================
# Test get_tracker Function
# =============================================================================


class TestGetTracker:
    """Tests for get_tracker function."""

    def test_creates_tracker(self, mock_metrics_store):
        """Test get_tracker creates new tracker."""
        tracker = get_tracker()
        assert tracker is not None
        assert isinstance(tracker, ExecutionTracker)

    def test_returns_same_instance(self, mock_metrics_store):
        """Test get_tracker returns same instance."""
        tracker1 = get_tracker()
        tracker2 = get_tracker()
        assert tracker1 is tracker2

    def test_passes_db_path(self):
        """Test get_tracker passes db_path to MetricsStore."""
        with patch("animus_forge.monitoring.tracker.MetricsStore") as mock_class:
            get_tracker("/path/to/db.sqlite")
            mock_class.assert_called_with("/path/to/db.sqlite")


# =============================================================================
# Test ExecutionTracker Initialization
# =============================================================================


class TestExecutionTrackerInit:
    """Tests for ExecutionTracker initialization."""

    def test_init_creates_store(self, mock_metrics_store):
        """Test tracker creates metrics store."""
        tracker = ExecutionTracker()
        assert tracker.store is not None

    def test_init_no_current_execution(self, tracker):
        """Test tracker starts with no current execution."""
        assert tracker._current_execution is None
        assert tracker._current_workflow is None

    def test_init_with_db_path(self):
        """Test tracker initializes with db_path."""
        with patch("animus_forge.monitoring.tracker.MetricsStore") as mock_class:
            ExecutionTracker("/custom/path.db")
            mock_class.assert_called_with("/custom/path.db")


# =============================================================================
# Test track_workflow Context Manager
# =============================================================================


class TestTrackWorkflow:
    """Tests for track_workflow context manager."""

    def test_track_workflow_yields_execution_id(self, tracker):
        """Test track_workflow yields execution ID."""
        with tracker.track_workflow("wf1", "Workflow 1") as exec_id:
            assert exec_id is not None
            assert exec_id.startswith("exec_")

    def test_track_workflow_starts_workflow(self, tracker, mock_metrics_store):
        """Test track_workflow calls start_workflow."""
        with tracker.track_workflow("wf1", "Workflow 1"):
            pass
        mock_metrics_store.start_workflow.assert_called_once()

    def test_track_workflow_completes_on_success(self, tracker, mock_metrics_store):
        """Test track_workflow marks completed on success."""
        with tracker.track_workflow("wf1", "Workflow 1") as exec_id:
            pass
        mock_metrics_store.complete_workflow.assert_called_with(exec_id, "completed")

    def test_track_workflow_marks_failed_on_exception(self, tracker, mock_metrics_store):
        """Test track_workflow marks failed on exception."""
        with pytest.raises(ValueError):
            with tracker.track_workflow("wf1", "Workflow 1") as exec_id:
                raise ValueError("Test error")

        mock_metrics_store.complete_workflow.assert_called_with(exec_id, "failed", "Test error")

    def test_track_workflow_sets_current_execution(self, tracker):
        """Test track_workflow sets current execution."""
        with tracker.track_workflow("wf1", "Workflow 1") as exec_id:
            assert tracker._current_execution == exec_id
            assert tracker._current_workflow is not None

    def test_track_workflow_clears_current_on_exit(self, tracker):
        """Test track_workflow clears current execution on exit."""
        with tracker.track_workflow("wf1", "Workflow 1"):
            pass
        assert tracker._current_execution is None
        assert tracker._current_workflow is None

    def test_track_workflow_clears_current_on_exception(self, tracker):
        """Test track_workflow clears current execution on exception."""
        with pytest.raises(ValueError):
            with tracker.track_workflow("wf1", "Workflow 1"):
                raise ValueError("Test error")

        assert tracker._current_execution is None
        assert tracker._current_workflow is None

    def test_track_workflow_uses_id_as_name_if_not_provided(self, tracker, mock_metrics_store):
        """Test track_workflow uses workflow_id as name if not provided."""
        with tracker.track_workflow("my_workflow"):
            pass

        call_args = mock_metrics_store.start_workflow.call_args
        workflow = call_args[0][0]
        assert workflow.workflow_name == "my_workflow"


# =============================================================================
# Test track_step Context Manager
# =============================================================================


class TestTrackStep:
    """Tests for track_step context manager."""

    def test_track_step_within_workflow(self, tracker, mock_metrics_store):
        """Test track_step within workflow context."""
        with tracker.track_workflow("wf1", "Workflow 1"):
            with tracker.track_step("step1", "transform", "format") as step:
                assert step is not None
                assert step.step_id == "step1"

    def test_track_step_starts_step(self, tracker, mock_metrics_store):
        """Test track_step calls start_step."""
        with tracker.track_workflow("wf1", "Workflow 1"):
            with tracker.track_step("step1", "transform", "format"):
                pass
        mock_metrics_store.start_step.assert_called()

    def test_track_step_completes_on_success(self, tracker, mock_metrics_store):
        """Test track_step marks success on completion."""
        with tracker.track_workflow("wf1", "Workflow 1") as exec_id:
            with tracker.track_step("step1", "transform", "format"):
                pass

        mock_metrics_store.complete_step.assert_called_with(exec_id, "step1", "success", tokens=0)

    def test_track_step_marks_failed_on_exception(self, tracker, mock_metrics_store):
        """Test track_step marks failed on exception."""
        with pytest.raises(RuntimeError):
            with tracker.track_workflow("wf1", "Workflow 1") as exec_id:
                with tracker.track_step("step1", "transform", "format"):
                    raise RuntimeError("Step failed")

        mock_metrics_store.complete_step.assert_called_with(
            exec_id, "step1", "failed", "Step failed"
        )

    def test_track_step_records_tokens(self, tracker, mock_metrics_store):
        """Test track_step records token usage."""
        with tracker.track_workflow("wf1", "Workflow 1") as exec_id:
            with tracker.track_step("step1", "openai", "generate") as step:
                step.tokens_used = 500

        mock_metrics_store.complete_step.assert_called_with(exec_id, "step1", "success", tokens=500)

    def test_track_step_without_workflow_raises(self, tracker):
        """Test track_step without active workflow raises error."""
        with pytest.raises(ValueError, match="No active workflow"):
            with tracker.track_step("step1", "transform", "format"):
                pass

    def test_track_step_with_explicit_execution_id(self, tracker, mock_metrics_store):
        """Test track_step with explicit execution_id."""
        mock_metrics_store.start_step.return_value = None
        mock_metrics_store.complete_step.return_value = None

        with tracker.track_step("step1", "transform", "format", execution_id="exec_123"):
            pass

        mock_metrics_store.start_step.assert_called()
        call_args = mock_metrics_store.start_step.call_args
        assert call_args[0][0] == "exec_123"


# =============================================================================
# Test track_step_decorator
# =============================================================================


class TestTrackStepDecorator:
    """Tests for track_step_decorator."""

    def test_decorator_tracks_function(self, tracker, mock_metrics_store):
        """Test decorator tracks function execution."""

        @tracker.track_step_decorator("transform", "format")
        def my_function(x):
            return x * 2

        with tracker.track_workflow("wf1", "Workflow 1"):
            result = my_function(5)

        assert result == 10
        mock_metrics_store.start_step.assert_called()

    def test_decorator_preserves_function_name(self, tracker):
        """Test decorator preserves function name."""

        @tracker.track_step_decorator("transform", "format")
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_decorator_generates_step_id(self, tracker, mock_metrics_store):
        """Test decorator generates unique step ID."""

        @tracker.track_step_decorator("transform", "format")
        def my_func():
            pass

        with tracker.track_workflow("wf1", "Workflow 1"):
            my_func()

        call_args = mock_metrics_store.start_step.call_args
        step = call_args[0][1]
        assert step.step_id.startswith("my_func_")

    def test_decorator_passes_arguments(self, tracker, mock_metrics_store):
        """Test decorator passes arguments to function."""

        @tracker.track_step_decorator("transform", "format")
        def add(a, b, c=0):
            return a + b + c

        with tracker.track_workflow("wf1", "Workflow 1"):
            result = add(1, 2, c=3)

        assert result == 6


# =============================================================================
# Test record_tokens
# =============================================================================


class TestRecordTokens:
    """Tests for record_tokens method."""

    def test_record_tokens(self, tracker, mock_metrics_store):
        """Test recording token usage."""
        tracker.record_tokens("exec_123", "step1", 1000)
        mock_metrics_store.complete_step.assert_called_with(
            "exec_123", "step1", "success", tokens=1000
        )


# =============================================================================
# Test get_status
# =============================================================================


class TestGetStatus:
    """Tests for get_status method."""

    def test_get_status_returns_dict(self, tracker):
        """Test get_status returns status dictionary."""
        status = tracker.get_status()
        assert isinstance(status, dict)
        assert "summary" in status
        assert "active_workflows" in status
        assert "current_execution" in status

    def test_get_status_includes_current_execution(self, tracker):
        """Test get_status includes current execution."""
        with tracker.track_workflow("wf1", "Workflow 1") as exec_id:
            status = tracker.get_status()
            assert status["current_execution"] == exec_id


# =============================================================================
# Test get_dashboard_data
# =============================================================================


class TestGetDashboardData:
    """Tests for get_dashboard_data method."""

    def test_get_dashboard_data_returns_dict(self, tracker):
        """Test get_dashboard_data returns data dictionary."""
        data = tracker.get_dashboard_data()
        assert isinstance(data, dict)
        assert "summary" in data
        assert "active_workflows" in data
        assert "recent_executions" in data
        assert "step_performance" in data

    def test_get_dashboard_data_calls_store_methods(self, tracker, mock_metrics_store):
        """Test get_dashboard_data calls all store methods."""
        tracker.get_dashboard_data()

        mock_metrics_store.get_summary.assert_called()
        mock_metrics_store.get_active_workflows.assert_called()
        mock_metrics_store.get_recent_executions.assert_called_with(20)
        mock_metrics_store.get_step_performance.assert_called()


# =============================================================================
# Test AgentTracker Initialization
# =============================================================================


class TestAgentTrackerInit:
    """Tests for AgentTracker initialization."""

    def test_init_empty_agents(self, agent_tracker):
        """Test tracker starts with no agents."""
        assert len(agent_tracker._active_agents) == 0
        assert len(agent_tracker._agent_history) == 0

    def test_init_max_history(self, agent_tracker):
        """Test tracker has max history limit."""
        assert agent_tracker._max_history == 50


# =============================================================================
# Test AgentTracker register_agent
# =============================================================================


class TestAgentTrackerRegister:
    """Tests for register_agent method."""

    def test_register_agent(self, agent_tracker):
        """Test registering an agent."""
        agent_tracker.register_agent("agent1", "builder", "wf1")

        assert "agent1" in agent_tracker._active_agents
        agent = agent_tracker._active_agents["agent1"]
        assert agent["role"] == "builder"
        assert agent["workflow_id"] == "wf1"
        assert agent["status"] == "active"

    def test_register_agent_without_workflow(self, agent_tracker):
        """Test registering agent without workflow."""
        agent_tracker.register_agent("agent1", "planner")

        agent = agent_tracker._active_agents["agent1"]
        assert agent["workflow_id"] is None

    def test_register_agent_sets_started_at(self, agent_tracker):
        """Test register sets started_at timestamp."""
        agent_tracker.register_agent("agent1", "builder")

        agent = agent_tracker._active_agents["agent1"]
        assert "started_at" in agent

    def test_register_agent_initializes_tasks(self, agent_tracker):
        """Test register initializes tasks_completed."""
        agent_tracker.register_agent("agent1", "builder")

        agent = agent_tracker._active_agents["agent1"]
        assert agent["tasks_completed"] == 0


# =============================================================================
# Test AgentTracker update_agent
# =============================================================================


class TestAgentTrackerUpdate:
    """Tests for update_agent method."""

    def test_update_agent(self, agent_tracker):
        """Test updating an agent."""
        agent_tracker.register_agent("agent1", "builder")
        agent_tracker.update_agent("agent1", tasks_completed=5, status="busy")

        agent = agent_tracker._active_agents["agent1"]
        assert agent["tasks_completed"] == 5
        assert agent["status"] == "busy"

    def test_update_nonexistent_agent(self, agent_tracker):
        """Test updating nonexistent agent is safe."""
        agent_tracker.update_agent("nonexistent", status="error")
        # Should not raise


# =============================================================================
# Test AgentTracker complete_agent
# =============================================================================


class TestAgentTrackerComplete:
    """Tests for complete_agent method."""

    def test_complete_agent(self, agent_tracker):
        """Test completing an agent."""
        agent_tracker.register_agent("agent1", "builder")
        agent_tracker.complete_agent("agent1")

        assert "agent1" not in agent_tracker._active_agents
        assert len(agent_tracker._agent_history) == 1

    def test_complete_agent_sets_status(self, agent_tracker):
        """Test complete sets status."""
        agent_tracker.register_agent("agent1", "builder")
        agent_tracker.complete_agent("agent1", status="failed")

        agent = agent_tracker._agent_history[0]
        assert agent["status"] == "failed"

    def test_complete_agent_sets_completed_at(self, agent_tracker):
        """Test complete sets completed_at."""
        agent_tracker.register_agent("agent1", "builder")
        agent_tracker.complete_agent("agent1")

        agent = agent_tracker._agent_history[0]
        assert "completed_at" in agent

    def test_complete_agent_adds_to_history(self, agent_tracker):
        """Test complete adds to history."""
        for i in range(3):
            agent_tracker.register_agent(f"agent{i}", "builder")
            agent_tracker.complete_agent(f"agent{i}")

        assert len(agent_tracker._agent_history) == 3

    def test_complete_agent_respects_max_history(self, agent_tracker):
        """Test complete respects max history limit."""
        agent_tracker._max_history = 3

        for i in range(5):
            agent_tracker.register_agent(f"agent{i}", "builder")
            agent_tracker.complete_agent(f"agent{i}")

        assert len(agent_tracker._agent_history) == 3

    def test_complete_nonexistent_agent(self, agent_tracker):
        """Test completing nonexistent agent is safe."""
        agent_tracker.complete_agent("nonexistent")
        # Should not raise


# =============================================================================
# Test AgentTracker get_active_agents
# =============================================================================


class TestAgentTrackerGetActive:
    """Tests for get_active_agents method."""

    def test_get_active_agents_empty(self, agent_tracker):
        """Test get_active_agents with no agents."""
        result = agent_tracker.get_active_agents()
        assert result == []

    def test_get_active_agents(self, agent_tracker):
        """Test get_active_agents returns active agents."""
        agent_tracker.register_agent("agent1", "builder")
        agent_tracker.register_agent("agent2", "tester")

        result = agent_tracker.get_active_agents()
        assert len(result) == 2

    def test_get_active_agents_excludes_completed(self, agent_tracker):
        """Test get_active_agents excludes completed agents."""
        agent_tracker.register_agent("agent1", "builder")
        agent_tracker.register_agent("agent2", "tester")
        agent_tracker.complete_agent("agent1")

        result = agent_tracker.get_active_agents()
        assert len(result) == 1
        assert result[0]["agent_id"] == "agent2"


# =============================================================================
# Test AgentTracker get_agent_history
# =============================================================================


class TestAgentTrackerGetHistory:
    """Tests for get_agent_history method."""

    def test_get_agent_history_empty(self, agent_tracker):
        """Test get_agent_history with no history."""
        result = agent_tracker.get_agent_history()
        assert result == []

    def test_get_agent_history(self, agent_tracker):
        """Test get_agent_history returns history."""
        for i in range(5):
            agent_tracker.register_agent(f"agent{i}", "builder")
            agent_tracker.complete_agent(f"agent{i}")

        result = agent_tracker.get_agent_history()
        assert len(result) == 5

    def test_get_agent_history_limit(self, agent_tracker):
        """Test get_agent_history respects limit."""
        for i in range(10):
            agent_tracker.register_agent(f"agent{i}", "builder")
            agent_tracker.complete_agent(f"agent{i}")

        result = agent_tracker.get_agent_history(limit=5)
        assert len(result) == 5

    def test_get_agent_history_most_recent_first(self, agent_tracker):
        """Test get_agent_history returns most recent first."""
        for i in range(3):
            agent_tracker.register_agent(f"agent{i}", "builder")
            agent_tracker.complete_agent(f"agent{i}")

        result = agent_tracker.get_agent_history()
        # Most recently completed should be first
        assert result[0]["agent_id"] == "agent2"


# =============================================================================
# Test AgentTracker get_agent_summary
# =============================================================================


class TestAgentTrackerGetSummary:
    """Tests for get_agent_summary method."""

    def test_get_agent_summary_empty(self, agent_tracker):
        """Test get_agent_summary with no agents."""
        result = agent_tracker.get_agent_summary()
        assert result["active_count"] == 0
        assert result["by_role"] == {}
        assert result["recent_count"] == 0

    def test_get_agent_summary_counts(self, agent_tracker):
        """Test get_agent_summary counts correctly."""
        agent_tracker.register_agent("a1", "builder")
        agent_tracker.register_agent("a2", "builder")
        agent_tracker.register_agent("a3", "tester")

        result = agent_tracker.get_agent_summary()
        assert result["active_count"] == 3
        assert result["by_role"]["builder"] == 2
        assert result["by_role"]["tester"] == 1

    def test_get_agent_summary_includes_history_count(self, agent_tracker):
        """Test get_agent_summary includes history count."""
        for i in range(5):
            agent_tracker.register_agent(f"agent{i}", "builder")
            agent_tracker.complete_agent(f"agent{i}")

        result = agent_tracker.get_agent_summary()
        assert result["recent_count"] == 5
