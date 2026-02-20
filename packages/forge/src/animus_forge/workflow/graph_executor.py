"""ReactFlow workflow executor - connects visual graphs to execution engine."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from .graph_models import GraphNode, WorkflowGraph
from .graph_walker import GraphWalker
from .loader import ConditionConfig, StepConfig, WorkflowConfig

if TYPE_CHECKING:
    from animus_forge.executions import ExecutionManager

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    """Status of a node during execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    """Result of executing a single node."""

    node_id: str
    status: NodeStatus
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: int = 0
    tokens_used: int = 0


@dataclass
class ExecutionResult:
    """Result of executing a complete workflow graph."""

    execution_id: str
    workflow_id: str
    status: str  # completed, failed, cancelled
    outputs: dict[str, Any] = field(default_factory=dict)
    node_results: dict[str, NodeResult] = field(default_factory=dict)
    total_duration_ms: int = 0
    total_tokens: int = 0
    error: str | None = None


# Type for execution callbacks
NodeCallback = Callable[[str, NodeStatus, dict[str, Any] | None], None]


class ReactFlowExecutor:
    """Executes workflow graphs from the visual builder.

    Bridges the ReactFlow-style graph representation with the
    existing workflow executor infrastructure.
    """

    def __init__(
        self,
        execution_manager: ExecutionManager | None = None,
        on_node_start: NodeCallback | None = None,
        on_node_complete: NodeCallback | None = None,
        on_node_error: NodeCallback | None = None,
    ):
        """Initialize the executor.

        Args:
            execution_manager: Optional manager for tracking execution state
            on_node_start: Callback when a node starts executing
            on_node_complete: Callback when a node completes
            on_node_error: Callback when a node fails
        """
        self.execution_manager = execution_manager
        self.on_node_start = on_node_start
        self.on_node_complete = on_node_complete
        self.on_node_error = on_node_error

        # Lazy import to avoid circular dependencies
        self._workflow_executor = None

    def _get_workflow_executor(self):
        """Get or create the workflow executor."""
        if self._workflow_executor is None:
            from .executor import WorkflowExecutor

            self._workflow_executor = WorkflowExecutor()
        return self._workflow_executor

    def execute(
        self,
        graph: WorkflowGraph,
        variables: dict[str, Any] | None = None,
        execution_id: str | None = None,
    ) -> ExecutionResult:
        """Execute a workflow graph synchronously.

        Args:
            graph: The workflow graph to execute
            variables: Initial variables/inputs
            execution_id: Optional execution ID for tracking

        Returns:
            ExecutionResult with outputs and status
        """
        return asyncio.get_event_loop().run_until_complete(
            self.execute_async(graph, variables, execution_id)
        )

    async def execute_async(
        self,
        graph: WorkflowGraph,
        variables: dict[str, Any] | None = None,
        execution_id: str | None = None,
    ) -> ExecutionResult:
        """Execute a workflow graph asynchronously.

        Args:
            graph: The workflow graph to execute
            variables: Initial variables/inputs
            execution_id: Optional execution ID for tracking

        Returns:
            ExecutionResult with outputs and status
        """
        start_time = time.monotonic()
        execution_id = execution_id or f"exec-{int(time.time())}"

        # Initialize execution context
        context = dict(graph.variables)
        if variables:
            context.update(variables)

        # Initialize walker and tracking
        walker = GraphWalker(graph)
        completed: set[str] = set()
        branch_decisions: dict[str, str] = {}
        node_results: dict[str, NodeResult] = {}
        total_tokens = 0

        # Check for cycles
        cycles = walker.detect_cycles()
        if cycles:
            # Cycles are OK if they're loops, otherwise error
            for cycle in cycles:
                cycle_nodes = [graph.get_node(nid) for nid in cycle]
                if not any(n and n.type == "loop" for n in cycle_nodes):
                    error = f"Graph contains non-loop cycle: {cycle}"
                    return ExecutionResult(
                        execution_id=execution_id,
                        workflow_id=graph.id,
                        status="failed",
                        error=error,
                        total_duration_ms=int((time.monotonic() - start_time) * 1000),
                    )

        # Execute nodes in topological order
        try:
            while True:
                ready_nodes = walker.get_ready_nodes(completed, branch_decisions)

                if not ready_nodes:
                    # No more nodes to execute
                    break

                # Execute all ready nodes (could be parallelized)
                for node_id in ready_nodes:
                    node = graph.get_node(node_id)
                    if not node:
                        continue

                    # Handle special node types
                    if node.type == "start":
                        # Start nodes just pass through
                        completed.add(node_id)
                        node_results[node_id] = NodeResult(
                            node_id=node_id,
                            status=NodeStatus.COMPLETED,
                        )
                        continue

                    if node.type == "end":
                        # End nodes just pass through
                        completed.add(node_id)
                        node_results[node_id] = NodeResult(
                            node_id=node_id,
                            status=NodeStatus.COMPLETED,
                        )
                        continue

                    # Execute the node
                    result = await self._execute_node(node, context, walker, execution_id)
                    node_results[node_id] = result
                    total_tokens += result.tokens_used

                    if result.status == NodeStatus.COMPLETED:
                        completed.add(node_id)
                        # Update context with outputs
                        context.update(result.outputs)

                        # For branch nodes, record the decision
                        if node.type == "branch":
                            branch_taken = result.outputs.get("branch_taken", "true")
                            branch_decisions[node_id] = branch_taken

                    elif result.status == NodeStatus.FAILED:
                        # Abort on failure (unless configured otherwise)
                        error = result.error or f"Node {node_id} failed"
                        return ExecutionResult(
                            execution_id=execution_id,
                            workflow_id=graph.id,
                            status="failed",
                            node_results=node_results,
                            error=error,
                            total_duration_ms=int((time.monotonic() - start_time) * 1000),
                            total_tokens=total_tokens,
                        )

                    # Update progress if execution manager is available
                    if self.execution_manager:
                        progress = int(len(completed) / len(graph.nodes) * 100)
                        self.execution_manager.update_progress(execution_id, progress, node_id)

        except Exception as e:
            logger.exception(f"Execution failed: {e}")
            return ExecutionResult(
                execution_id=execution_id,
                workflow_id=graph.id,
                status="failed",
                node_results=node_results,
                error=str(e),
                total_duration_ms=int((time.monotonic() - start_time) * 1000),
                total_tokens=total_tokens,
            )

        # Collect outputs from end nodes
        outputs = {}
        for node in graph.get_end_nodes():
            if node.id in node_results:
                outputs.update(node_results[node.id].outputs)

        # Also include any context variables marked as outputs
        for output_name in graph.variables.get("_outputs", []):
            if output_name in context:
                outputs[output_name] = context[output_name]

        return ExecutionResult(
            execution_id=execution_id,
            workflow_id=graph.id,
            status="completed",
            outputs=outputs,
            node_results=node_results,
            total_duration_ms=int((time.monotonic() - start_time) * 1000),
            total_tokens=total_tokens,
        )

    async def _execute_node(
        self,
        node: GraphNode,
        context: dict[str, Any],
        walker: GraphWalker,
        execution_id: str,
    ) -> NodeResult:
        """Execute a single node.

        Args:
            node: The node to execute
            context: Current execution context
            walker: Graph walker for branch/loop handling
            execution_id: Execution ID for logging

        Returns:
            NodeResult with outputs and status
        """
        start_time = time.monotonic()

        # Notify start callback
        if self.on_node_start:
            self.on_node_start(node.id, NodeStatus.RUNNING, None)

        # Log to execution manager
        if self.execution_manager:
            from animus_forge.executions import LogLevel

            self.execution_manager.add_log(
                execution_id,
                LogLevel.INFO,
                f"Starting node: {node.id} ({node.type})",
                step_id=node.id,
            )

        try:
            # Execute based on node type
            if node.type == "branch":
                result = self._execute_branch(node, context, walker)
            elif node.type == "loop":
                result = await self._execute_loop(node, context, walker, execution_id)
            elif node.type == "checkpoint":
                result = self._execute_checkpoint(node, context, execution_id)
            elif node.type == "parallel":
                result = await self._execute_parallel(node, context, execution_id)
            else:
                # Standard step execution (agent, shell, etc.)
                result = await self._execute_step(node, context)

            duration_ms = int((time.monotonic() - start_time) * 1000)

            # Notify complete callback
            if self.on_node_complete:
                self.on_node_complete(node.id, NodeStatus.COMPLETED, result)

            # Log success
            if self.execution_manager:
                from animus_forge.executions import LogLevel

                self.execution_manager.add_log(
                    execution_id,
                    LogLevel.INFO,
                    f"Completed node: {node.id} in {duration_ms}ms",
                    step_id=node.id,
                )

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                outputs=result,
                duration_ms=duration_ms,
                tokens_used=result.get("_tokens", 0),
            )

        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            error = str(e)

            # Notify error callback
            if self.on_node_error:
                self.on_node_error(node.id, NodeStatus.FAILED, {"error": error})

            # Log error
            if self.execution_manager:
                from animus_forge.executions import LogLevel

                self.execution_manager.add_log(
                    execution_id,
                    LogLevel.ERROR,
                    f"Node {node.id} failed: {error}",
                    step_id=node.id,
                )

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=error,
                duration_ms=duration_ms,
            )

    def _execute_branch(
        self,
        node: GraphNode,
        context: dict[str, Any],
        walker: GraphWalker,
    ) -> dict[str, Any]:
        """Execute a branch node.

        Args:
            node: Branch node
            context: Execution context
            walker: Graph walker

        Returns:
            Dict with branch_taken key
        """
        branch_taken = walker.evaluate_branch(node.id, context)
        return {"branch_taken": branch_taken}

    async def _execute_loop(
        self,
        node: GraphNode,
        context: dict[str, Any],
        walker: GraphWalker,
        execution_id: str,
    ) -> dict[str, Any]:
        """Execute a loop node.

        Args:
            node: Loop node
            context: Execution context
            walker: Graph walker
            execution_id: Execution ID

        Returns:
            Dict with loop results
        """
        max_iterations = node.data.get("max_iterations", 10)
        results = []
        iteration = 0

        while walker.should_continue_loop(node.id, context, iteration):
            if iteration >= max_iterations:
                logger.warning(f"Loop {node.id} hit max iterations ({max_iterations})")
                break

            # For for-each loops, set the current item
            loop_item = walker.get_loop_item(node.id, context, iteration)
            if loop_item is not None:
                item_var = node.data.get("item_variable", "item")
                context[item_var] = loop_item

            # Set iteration counter
            context["_loop_iteration"] = iteration
            context["_loop_index"] = iteration

            # Execute loop body would go here
            # For now, just collect iteration info
            results.append(
                {
                    "iteration": iteration,
                    "item": loop_item,
                }
            )

            iteration += 1

        return {
            "iterations": iteration,
            "results": results,
        }

    def _execute_checkpoint(
        self,
        node: GraphNode,
        context: dict[str, Any],
        execution_id: str,
    ) -> dict[str, Any]:
        """Execute a checkpoint node (save state for resume).

        Args:
            node: Checkpoint node
            context: Execution context
            execution_id: Execution ID

        Returns:
            Empty dict (checkpoint is a no-op for outputs)
        """
        if self.execution_manager:
            checkpoint_id = f"{execution_id}-{node.id}"
            self.execution_manager.save_checkpoint(execution_id, checkpoint_id)
            self.execution_manager.update_variables(execution_id, context)

        return {}

    async def _execute_parallel(
        self,
        node: GraphNode,
        context: dict[str, Any],
        execution_id: str,
    ) -> dict[str, Any]:
        """Execute a parallel node (multiple concurrent steps).

        Args:
            node: Parallel node
            context: Execution context
            execution_id: Execution ID

        Returns:
            Combined outputs from parallel branches
        """
        # Get steps to run in parallel from node data
        steps = node.data.get("steps", [])
        if not steps:
            return {}

        # Execute steps concurrently
        async def run_step(step_data: dict) -> dict:
            step_node = GraphNode.from_dict(
                {
                    "id": f"{node.id}-{step_data.get('id', 'step')}",
                    "type": step_data.get("type", "agent"),
                    "data": step_data,
                    "position": {"x": 0, "y": 0},
                }
            )
            result = await self._execute_step(step_node, context)
            return result

        results = await asyncio.gather(
            *[run_step(s) for s in steps],
            return_exceptions=True,
        )

        # Combine outputs
        combined = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Parallel step {i} failed: {result}")
                continue
            if isinstance(result, dict):
                combined.update(result)

        return combined

    async def _execute_step(
        self,
        node: GraphNode,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a standard step (agent, shell, etc.).

        Args:
            node: Step node
            context: Execution context

        Returns:
            Step outputs
        """
        # Map node type to step type
        step_type = node.type
        if step_type == "agent":
            # Determine provider from node data
            provider = node.data.get("provider", "openai")
            step_type = "claude_code" if provider == "anthropic" else "openai"

        # Build step config
        step = StepConfig(
            id=node.id,
            type=step_type,
            params=node.data.get("params", {}),
            timeout_seconds=node.data.get("timeout", 300),
        )

        # Handle condition if present
        if "condition" in node.data:
            cond = node.data["condition"]
            step.condition = ConditionConfig(
                field=cond.get("field", ""),
                operator=cond.get("operator", "equals"),
                value=cond.get("value"),
            )

        # Build minimal workflow config for executor
        workflow = WorkflowConfig(
            name=f"step-{node.id}",
            version="1.0",
            description="",
            steps=[step],
        )

        # Execute using the existing workflow executor
        executor = self._get_workflow_executor()
        result = executor.execute(workflow, inputs=context)

        # Extract outputs
        outputs = result.outputs or {}

        # Add token count if available
        if result.total_tokens:
            outputs["_tokens"] = result.total_tokens

        return outputs

    def pause(self, execution_id: str) -> bool:
        """Pause an execution.

        Args:
            execution_id: Execution ID to pause

        Returns:
            True if paused successfully
        """
        if self.execution_manager:
            self.execution_manager.pause_execution(execution_id)
            return True
        return False

    def resume(
        self,
        execution_id: str,
        graph: WorkflowGraph,
    ) -> ExecutionResult:
        """Resume a paused execution.

        Args:
            execution_id: Execution ID to resume
            graph: Original workflow graph

        Returns:
            ExecutionResult from resumed execution
        """
        if not self.execution_manager:
            raise RuntimeError("Cannot resume without execution manager")

        # Get execution state
        execution = self.execution_manager.get_execution(execution_id)
        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")

        # Resume
        self.execution_manager.resume_execution(execution_id)

        # Continue from checkpoint
        variables = execution.variables

        return self.execute(graph, variables, execution_id)
