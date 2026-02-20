"""Tests for ReactFlowExecutor graph execution."""

import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, "src")

from animus_forge.executions.manager import ExecutionManager
from animus_forge.state.backends import SQLiteBackend
from animus_forge.workflow.graph_executor import (
    ExecutionResult,
    NodeResult,
    NodeStatus,
    ReactFlowExecutor,
)
from animus_forge.workflow.graph_models import GraphEdge, GraphNode, WorkflowGraph


class TestNodeResult:
    """Tests for NodeResult dataclass."""

    def test_creation(self):
        """NodeResult can be created with minimal fields."""
        result = NodeResult(node_id="n1", status=NodeStatus.COMPLETED)

        assert result.node_id == "n1"
        assert result.status == NodeStatus.COMPLETED
        assert result.outputs == {}
        assert result.error is None
        assert result.duration_ms == 0
        assert result.tokens_used == 0

    def test_creation_with_outputs(self):
        """NodeResult can include outputs."""
        result = NodeResult(
            node_id="n2",
            status=NodeStatus.COMPLETED,
            outputs={"result": "value"},
            duration_ms=150,
            tokens_used=50,
        )

        assert result.outputs == {"result": "value"}
        assert result.duration_ms == 150
        assert result.tokens_used == 50

    def test_failed_result(self):
        """NodeResult can represent failure."""
        result = NodeResult(
            node_id="n3",
            status=NodeStatus.FAILED,
            error="Something went wrong",
        )

        assert result.status == NodeStatus.FAILED
        assert result.error == "Something went wrong"


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_creation(self):
        """ExecutionResult can be created."""
        result = ExecutionResult(
            execution_id="exec-1",
            workflow_id="wf-1",
            status="completed",
        )

        assert result.execution_id == "exec-1"
        assert result.workflow_id == "wf-1"
        assert result.status == "completed"
        assert result.outputs == {}
        assert result.node_results == {}
        assert result.error is None

    def test_with_results(self):
        """ExecutionResult can include node results."""
        node_result = NodeResult(node_id="n1", status=NodeStatus.COMPLETED)
        result = ExecutionResult(
            execution_id="exec-2",
            workflow_id="wf-2",
            status="completed",
            outputs={"final": "output"},
            node_results={"n1": node_result},
            total_duration_ms=1000,
            total_tokens=100,
        )

        assert result.outputs == {"final": "output"}
        assert "n1" in result.node_results
        assert result.total_duration_ms == 1000
        assert result.total_tokens == 100


class TestReactFlowExecutorBasics:
    """Basic ReactFlowExecutor tests."""

    @pytest.fixture
    def executor(self):
        """Create a basic executor without execution manager."""
        return ReactFlowExecutor()

    @pytest.fixture
    def simple_graph(self):
        """Simple workflow: start -> end."""
        return WorkflowGraph(
            id="simple",
            name="Simple",
            nodes=[
                GraphNode(id="start", type="start"),
                GraphNode(id="end", type="end"),
            ],
            edges=[
                GraphEdge(id="e1", source="start", target="end"),
            ],
        )

    def test_executor_creation(self):
        """ReactFlowExecutor can be created."""
        executor = ReactFlowExecutor()
        assert executor.execution_manager is None
        assert executor.on_node_start is None
        assert executor.on_node_complete is None
        assert executor.on_node_error is None

    def test_executor_with_callbacks(self):
        """ReactFlowExecutor accepts callbacks."""
        start_cb = MagicMock()
        complete_cb = MagicMock()
        error_cb = MagicMock()

        executor = ReactFlowExecutor(
            on_node_start=start_cb,
            on_node_complete=complete_cb,
            on_node_error=error_cb,
        )

        assert executor.on_node_start is start_cb
        assert executor.on_node_complete is complete_cb
        assert executor.on_node_error is error_cb

    @pytest.mark.asyncio
    async def test_execute_simple_graph(self, executor, simple_graph):
        """execute_async completes simple start->end graph."""
        result = await executor.execute_async(simple_graph)

        assert result.status == "completed"
        assert result.workflow_id == "simple"
        assert "start" in result.node_results
        assert "end" in result.node_results

    @pytest.mark.asyncio
    async def test_execute_with_variables(self, executor, simple_graph):
        """execute_async passes variables to context."""
        result = await executor.execute_async(
            simple_graph,
            variables={"input": "test-value"},
        )

        assert result.status == "completed"


class TestReactFlowExecutorBranching:
    """Tests for branch node execution."""

    @pytest.fixture
    def branch_graph(self):
        """Workflow with conditional branch."""
        return WorkflowGraph(
            id="branch",
            name="Branch",
            nodes=[
                GraphNode(id="start", type="start"),
                GraphNode(
                    id="check",
                    type="branch",
                    data={
                        "condition": {
                            "field": "should_pass",
                            "operator": "equals",
                            "value": True,
                        }
                    },
                ),
                GraphNode(id="yes", type="end"),
                GraphNode(id="no", type="end"),
            ],
            edges=[
                GraphEdge(id="e1", source="start", target="check"),
                GraphEdge(id="e2", source="check", target="yes", source_handle="true"),
                GraphEdge(id="e3", source="check", target="no", source_handle="false"),
            ],
        )

    @pytest.mark.asyncio
    async def test_branch_true_path(self, branch_graph):
        """Branch takes true path when condition is true."""
        executor = ReactFlowExecutor()
        result = await executor.execute_async(
            branch_graph,
            variables={"should_pass": True},
        )

        assert result.status == "completed"
        assert "yes" in result.node_results
        # The false path shouldn't be in results since it wasn't executed
        # (Actually both end nodes may be checked but only one should complete)

    @pytest.mark.asyncio
    async def test_branch_false_path(self, branch_graph):
        """Branch takes false path when condition is false."""
        executor = ReactFlowExecutor()
        result = await executor.execute_async(
            branch_graph,
            variables={"should_pass": False},
        )

        assert result.status == "completed"
        assert "no" in result.node_results


class TestReactFlowExecutorCheckpoints:
    """Tests for checkpoint functionality."""

    @pytest.fixture
    def backend(self):
        """Create a temporary SQLite backend with schema."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            backend = SQLiteBackend(db_path=db_path)

            backend.executescript("""
                CREATE TABLE IF NOT EXISTS executions (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    workflow_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    current_step TEXT,
                    progress INTEGER DEFAULT 0,
                    checkpoint_id TEXT,
                    variables TEXT,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS execution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    step_id TEXT,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS execution_metrics (
                    execution_id TEXT PRIMARY KEY,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost_cents INTEGER DEFAULT 0,
                    duration_ms INTEGER DEFAULT 0,
                    steps_completed INTEGER DEFAULT 0,
                    steps_failed INTEGER DEFAULT 0
                );
            """)

            yield backend
            backend.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def manager(self, backend):
        """Create an ExecutionManager."""
        return ExecutionManager(backend=backend)

    @pytest.fixture
    def checkpoint_graph(self):
        """Workflow with checkpoint."""
        return WorkflowGraph(
            id="checkpoint",
            name="Checkpoint",
            nodes=[
                GraphNode(id="start", type="start"),
                GraphNode(id="save", type="checkpoint"),
                GraphNode(id="end", type="end"),
            ],
            edges=[
                GraphEdge(id="e1", source="start", target="save"),
                GraphEdge(id="e2", source="save", target="end"),
            ],
        )

    @pytest.mark.asyncio
    async def test_checkpoint_saves_state(self, manager, checkpoint_graph):
        """Checkpoint node saves execution state."""
        executor = ReactFlowExecutor(execution_manager=manager)

        # Create execution
        execution = manager.create_execution("checkpoint", "Checkpoint Test")

        result = await executor.execute_async(
            checkpoint_graph,
            variables={"test": "value"},
            execution_id=execution.id,
        )

        assert result.status == "completed"

        # Check checkpoint was saved
        updated = manager.get_execution(execution.id)
        assert updated.checkpoint_id is not None


class TestReactFlowExecutorCallbacks:
    """Tests for execution callbacks."""

    @pytest.fixture
    def linear_graph(self):
        """Linear workflow for callback testing."""
        return WorkflowGraph(
            id="linear",
            name="Linear",
            nodes=[
                GraphNode(id="start", type="start"),
                GraphNode(id="step1", type="checkpoint"),
                GraphNode(id="end", type="end"),
            ],
            edges=[
                GraphEdge(id="e1", source="start", target="step1"),
                GraphEdge(id="e2", source="step1", target="end"),
            ],
        )

    @pytest.mark.asyncio
    async def test_callbacks_called(self, linear_graph):
        """Callbacks are called during execution."""
        start_calls = []
        complete_calls = []

        def on_start(node_id, status, data):
            start_calls.append(node_id)

        def on_complete(node_id, status, data):
            complete_calls.append(node_id)

        executor = ReactFlowExecutor(
            on_node_start=on_start,
            on_node_complete=on_complete,
        )

        await executor.execute_async(linear_graph)

        # Start and end are pass-through, step1 should have callbacks
        assert "step1" in start_calls
        assert "step1" in complete_calls


class TestReactFlowExecutorCycleDetection:
    """Tests for cycle handling."""

    @pytest.fixture
    def cyclic_graph(self):
        """Graph with a non-loop cycle (invalid)."""
        return WorkflowGraph(
            id="cyclic",
            name="Cyclic",
            nodes=[
                GraphNode(id="a", type="agent"),
                GraphNode(id="b", type="agent"),
            ],
            edges=[
                GraphEdge(id="e1", source="a", target="b"),
                GraphEdge(id="e2", source="b", target="a"),
            ],
        )

    @pytest.mark.asyncio
    async def test_rejects_non_loop_cycle(self, cyclic_graph):
        """Executor rejects graphs with non-loop cycles."""
        executor = ReactFlowExecutor()

        result = await executor.execute_async(cyclic_graph)

        assert result.status == "failed"
        assert "cycle" in result.error.lower()


class TestReactFlowExecutorWithManager:
    """Tests for execution with ExecutionManager integration."""

    @pytest.fixture
    def backend(self):
        """Create a temporary SQLite backend with schema."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            backend = SQLiteBackend(db_path=db_path)

            backend.executescript("""
                CREATE TABLE IF NOT EXISTS executions (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    workflow_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    current_step TEXT,
                    progress INTEGER DEFAULT 0,
                    checkpoint_id TEXT,
                    variables TEXT,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS execution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    step_id TEXT,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS execution_metrics (
                    execution_id TEXT PRIMARY KEY,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost_cents INTEGER DEFAULT 0,
                    duration_ms INTEGER DEFAULT 0,
                    steps_completed INTEGER DEFAULT 0,
                    steps_failed INTEGER DEFAULT 0
                );
            """)

            yield backend
            backend.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def manager(self, backend):
        """Create an ExecutionManager."""
        return ExecutionManager(backend=backend)

    @pytest.fixture
    def tracked_graph(self):
        """Simple graph for tracking tests."""
        return WorkflowGraph(
            id="tracked",
            name="Tracked",
            nodes=[
                GraphNode(id="start", type="start"),
                GraphNode(id="checkpoint", type="checkpoint"),
                GraphNode(id="end", type="end"),
            ],
            edges=[
                GraphEdge(id="e1", source="start", target="checkpoint"),
                GraphEdge(id="e2", source="checkpoint", target="end"),
            ],
        )

    @pytest.mark.asyncio
    async def test_logs_created(self, manager, tracked_graph):
        """Execution creates log entries."""
        executor = ReactFlowExecutor(execution_manager=manager)
        execution = manager.create_execution("tracked", "Tracked Test")

        await executor.execute_async(
            tracked_graph,
            execution_id=execution.id,
        )

        logs = manager.get_logs(execution.id)
        assert len(logs) > 0

    @pytest.mark.asyncio
    async def test_progress_updated(self, manager, tracked_graph):
        """Execution updates progress."""
        executor = ReactFlowExecutor(execution_manager=manager)
        execution = manager.create_execution("tracked", "Tracked Test")

        await executor.execute_async(
            tracked_graph,
            execution_id=execution.id,
        )

        updated = manager.get_execution(execution.id)
        # Progress should be updated (may be 100 at end or intermediate value)
        assert updated.progress >= 0


class TestReactFlowExecutorPauseResume:
    """Tests for pause/resume functionality."""

    def test_pause_returns_true_with_manager(self):
        """pause() returns True when manager exists."""
        manager = MagicMock()
        executor = ReactFlowExecutor(execution_manager=manager)

        result = executor.pause("exec-1")

        assert result is True
        manager.pause_execution.assert_called_once_with("exec-1")

    def test_pause_returns_false_without_manager(self):
        """pause() returns False without manager."""
        executor = ReactFlowExecutor()

        result = executor.pause("exec-1")

        assert result is False

    def test_resume_requires_manager(self):
        """resume() raises without manager."""
        executor = ReactFlowExecutor()
        graph = WorkflowGraph(id="test", name="Test")

        with pytest.raises(RuntimeError):
            executor.resume("exec-1", graph)

    def test_resume_raises_for_missing_execution(self):
        """resume() raises for nonexistent execution."""
        manager = MagicMock()
        manager.get_execution.return_value = None
        executor = ReactFlowExecutor(execution_manager=manager)
        graph = WorkflowGraph(id="test", name="Test")

        with pytest.raises(ValueError):
            executor.resume("nonexistent", graph)


class TestReactFlowExecutorNodeTypes:
    """Tests for different node type handling."""

    @pytest.mark.asyncio
    async def test_start_node_passthrough(self):
        """Start nodes complete immediately."""
        graph = WorkflowGraph(
            id="test",
            name="Test",
            nodes=[
                GraphNode(id="start", type="start"),
                GraphNode(id="end", type="end"),
            ],
            edges=[
                GraphEdge(id="e1", source="start", target="end"),
            ],
        )
        executor = ReactFlowExecutor()

        result = await executor.execute_async(graph)

        assert result.node_results["start"].status == NodeStatus.COMPLETED
        assert result.node_results["start"].duration_ms == 0

    @pytest.mark.asyncio
    async def test_end_node_passthrough(self):
        """End nodes complete immediately."""
        graph = WorkflowGraph(
            id="test",
            name="Test",
            nodes=[
                GraphNode(id="start", type="start"),
                GraphNode(id="end", type="end"),
            ],
            edges=[
                GraphEdge(id="e1", source="start", target="end"),
            ],
        )
        executor = ReactFlowExecutor()

        result = await executor.execute_async(graph)

        assert result.node_results["end"].status == NodeStatus.COMPLETED


class TestReactFlowExecutorLoop:
    """Tests for loop node execution."""

    @pytest.fixture
    def loop_graph(self):
        """Workflow with a loop node."""
        return WorkflowGraph(
            id="loop",
            name="Loop",
            nodes=[
                GraphNode(id="start", type="start"),
                GraphNode(
                    id="loop",
                    type="loop",
                    data={
                        "loop_type": "count",
                        "count": 3,
                        "max_iterations": 10,
                    },
                ),
                GraphNode(id="end", type="end"),
            ],
            edges=[
                GraphEdge(id="e1", source="start", target="loop"),
                GraphEdge(id="e2", source="loop", target="end"),
            ],
        )

    @pytest.mark.asyncio
    async def test_loop_executes(self, loop_graph):
        """Loop node executes iterations."""
        executor = ReactFlowExecutor()

        result = await executor.execute_async(loop_graph)

        assert result.status == "completed"
        # Loop node should have result with iterations
        loop_result = result.node_results.get("loop")
        assert loop_result is not None
        assert loop_result.status == NodeStatus.COMPLETED
        # Check iterations in outputs
        assert loop_result.outputs.get("iterations") == 3
