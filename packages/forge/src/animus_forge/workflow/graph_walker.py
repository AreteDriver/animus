"""Graph walker for topological traversal of workflow graphs."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from .graph_models import GraphEdge, WorkflowGraph

logger = logging.getLogger(__name__)


class GraphWalker:
    """Traverses a workflow graph in topological order.

    Handles branching, loops, and parallel execution paths.
    Determines which nodes are ready to execute based on completed dependencies.
    """

    def __init__(self, graph: WorkflowGraph):
        """Initialize the graph walker.

        Args:
            graph: The workflow graph to traverse
        """
        self.graph = graph
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        """Build adjacency lists for efficient traversal."""
        # Outgoing edges from each node
        self._outgoing: dict[str, list[GraphEdge]] = defaultdict(list)
        # Incoming edges to each node
        self._incoming: dict[str, list[GraphEdge]] = defaultdict(list)

        for edge in self.graph.edges:
            self._outgoing[edge.source].append(edge)
            self._incoming[edge.target].append(edge)

    def get_start_nodes(self) -> list[str]:
        """Get IDs of all start nodes (no incoming edges).

        Returns:
            List of node IDs that have no dependencies
        """
        start_nodes = []
        for node in self.graph.nodes:
            # Nodes with type "start" are always start nodes
            if node.type == "start":
                start_nodes.append(node.id)
            # Otherwise, nodes with no incoming edges are start nodes
            elif node.id not in self._incoming or not self._incoming[node.id]:
                start_nodes.append(node.id)
        return start_nodes

    def get_ready_nodes(
        self,
        completed: set[str],
        branch_decisions: dict[str, str] | None = None,
    ) -> list[str]:
        """Get IDs of nodes that are ready to execute.

        A node is ready if all its dependencies are completed.
        For branch nodes, only the taken path's successors are considered.

        Args:
            completed: Set of completed node IDs
            branch_decisions: Map of branch node ID -> taken handle ("true"/"false")

        Returns:
            List of node IDs ready for execution
        """
        branch_decisions = branch_decisions or {}
        ready = []

        for node in self.graph.nodes:
            # Skip already completed nodes
            if node.id in completed:
                continue

            # Check if all dependencies are satisfied
            incoming = self._incoming.get(node.id, [])

            if not incoming:
                # Start nodes with no incoming edges are ready
                if node.id not in completed:
                    ready.append(node.id)
                continue

            # For each incoming edge, check if the source is completed
            # and if it's a branch, check if this edge was taken
            all_deps_satisfied = True

            for edge in incoming:
                source_node = self.graph.get_node(edge.source)

                if edge.source not in completed:
                    # Source not completed yet
                    all_deps_satisfied = False
                    break

                # For branch nodes, only the taken path should be followed
                if source_node and source_node.type == "branch":
                    taken_handle = branch_decisions.get(edge.source)
                    if taken_handle and edge.source_handle != taken_handle:
                        # This edge wasn't taken, so this node shouldn't be ready
                        # unless there's another path to it
                        all_deps_satisfied = False
                        break

            if all_deps_satisfied:
                ready.append(node.id)

        return ready

    def get_downstream_nodes(
        self,
        node_id: str,
        handle: str | None = None,
    ) -> list[str]:
        """Get IDs of immediate downstream nodes.

        For branch nodes, optionally filter by the taken handle.

        Args:
            node_id: Source node ID
            handle: Optional handle to filter by (for branch nodes)

        Returns:
            List of downstream node IDs
        """
        downstream = []
        for edge in self._outgoing.get(node_id, []):
            if handle is None or edge.source_handle == handle:
                downstream.append(edge.target)
        return downstream

    def get_all_downstream(
        self,
        node_id: str,
        handle: str | None = None,
    ) -> set[str]:
        """Get all downstream node IDs (transitive closure).

        Args:
            node_id: Source node ID
            handle: Optional handle for initial branch

        Returns:
            Set of all reachable downstream node IDs
        """
        visited = set()
        to_visit = self.get_downstream_nodes(node_id, handle)

        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
            visited.add(current)
            to_visit.extend(self.get_downstream_nodes(current))

        return visited

    def evaluate_branch(
        self,
        node_id: str,
        context: dict[str, Any],
    ) -> str:
        """Evaluate a branch condition and return the handle to follow.

        Args:
            node_id: Branch node ID
            context: Execution context with variables

        Returns:
            Handle to follow: "true" or "false"
        """
        node = self.graph.get_node(node_id)
        if not node or node.type != "branch":
            raise ValueError(f"Node {node_id} is not a branch node")

        # Get condition from node data
        condition = node.data.get("condition", {})
        field = condition.get("field", "")
        operator = condition.get("operator", "equals")
        value = condition.get("value")

        # Get actual value from context
        actual = context.get(field)

        # Evaluate condition
        result = self._evaluate_condition(actual, operator, value)

        logger.debug(f"Branch {node_id}: {field} {operator} {value} = {result} (actual: {actual})")

        return "true" if result else "false"

    def _evaluate_condition(
        self,
        actual: Any,
        operator: str,
        value: Any,
    ) -> bool:
        """Evaluate a single condition.

        Args:
            actual: Actual value from context
            operator: Comparison operator
            value: Expected value

        Returns:
            Boolean result of the condition
        """
        if actual is None:
            return False

        if operator == "equals":
            return actual == value
        elif operator == "not_equals":
            return actual != value
        elif operator == "contains":
            return value in actual if isinstance(actual, (str, list)) else False
        elif operator == "greater_than":
            return actual > value if isinstance(actual, (int, float)) else False
        elif operator == "less_than":
            return actual < value if isinstance(actual, (int, float)) else False
        elif operator == "in":
            return actual in value if isinstance(value, (list, str)) else False
        elif operator == "not_empty":
            return bool(actual)
        elif operator == "is_true":
            return bool(actual)
        elif operator == "is_false":
            return not bool(actual)
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False

    def should_continue_loop(
        self,
        node_id: str,
        context: dict[str, Any],
        iteration: int,
    ) -> bool:
        """Determine if a loop should continue.

        Args:
            node_id: Loop node ID
            context: Execution context with variables
            iteration: Current iteration number (0-indexed)

        Returns:
            True if the loop should continue, False otherwise
        """
        node = self.graph.get_node(node_id)
        if not node or node.type != "loop":
            raise ValueError(f"Node {node_id} is not a loop node")

        loop_type = node.data.get("loop_type", "while")
        max_iterations = node.data.get("max_iterations", 10)

        # Check max iterations
        if iteration >= max_iterations:
            logger.debug(f"Loop {node_id}: stopping at max iterations ({max_iterations})")
            return False

        if loop_type == "count":
            # Count loop: iterate a fixed number of times
            count = node.data.get("count", max_iterations)
            return iteration < count

        elif loop_type == "while":
            # While loop: continue while condition is true
            condition = node.data.get("condition", {})
            field = condition.get("field", "")
            operator = condition.get("operator", "equals")
            value = condition.get("value")

            actual = context.get(field)
            result = self._evaluate_condition(actual, operator, value)

            logger.debug(
                f"Loop {node_id} iteration {iteration}: {field} {operator} {value} = {result}"
            )
            return result

        elif loop_type == "for_each":
            # For-each loop: iterate over a collection
            collection_field = node.data.get("collection", "")
            collection = context.get(collection_field, [])

            if not isinstance(collection, (list, tuple)):
                return False

            return iteration < len(collection)

        else:
            logger.warning(f"Unknown loop type: {loop_type}")
            return False

    def get_loop_item(
        self,
        node_id: str,
        context: dict[str, Any],
        iteration: int,
    ) -> Any:
        """Get the current item for a for-each loop.

        Args:
            node_id: Loop node ID
            context: Execution context
            iteration: Current iteration

        Returns:
            The current item from the collection, or None
        """
        node = self.graph.get_node(node_id)
        if not node or node.type != "loop":
            return None

        if node.data.get("loop_type") != "for_each":
            return None

        collection_field = node.data.get("collection", "")
        collection = context.get(collection_field, [])

        if isinstance(collection, (list, tuple)) and iteration < len(collection):
            return collection[iteration]

        return None

    def detect_cycles(self) -> list[list[str]]:
        """Detect cycles in the graph.

        Returns:
            List of cycles, where each cycle is a list of node IDs
        """
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node_id: str, path: list[str]) -> None:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for edge in self._outgoing.get(node_id, []):
                next_id = edge.target
                if next_id not in visited:
                    dfs(next_id, path)
                elif next_id in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(next_id)
                    cycles.append(path[cycle_start:] + [next_id])

            path.pop()
            rec_stack.remove(node_id)

        for node in self.graph.nodes:
            if node.id not in visited:
                dfs(node.id, [])

        return cycles

    def topological_sort(self) -> list[str]:
        """Return nodes in topological order.

        Returns:
            List of node IDs in execution order

        Raises:
            ValueError: If the graph contains cycles
        """
        in_degree = {n.id: 0 for n in self.graph.nodes}

        for edge in self.graph.edges:
            in_degree[edge.target] += 1

        queue = [nid for nid, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for edge in self._outgoing.get(node_id, []):
                in_degree[edge.target] -= 1
                if in_degree[edge.target] == 0:
                    queue.append(edge.target)

        if len(result) != len(self.graph.nodes):
            raise ValueError("Graph contains cycles")

        return result
