"""Auto-parallel execution analysis for workflows.

Analyzes step dependencies to automatically identify and group
independent steps for concurrent execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loader import StepConfig, WorkflowConfig

logger = logging.getLogger(__name__)


@dataclass
class DependencyGraph:
    """Graph representation of step dependencies.

    Attributes:
        nodes: Set of all step IDs
        edges: Dict mapping step_id -> set of step_ids it depends on
        reverse_edges: Dict mapping step_id -> set of step_ids that depend on it
    """

    nodes: set[str] = field(default_factory=set)
    edges: dict[str, set[str]] = field(default_factory=dict)
    reverse_edges: dict[str, set[str]] = field(default_factory=dict)

    def add_node(self, step_id: str) -> None:
        """Add a node to the graph."""
        self.nodes.add(step_id)
        if step_id not in self.edges:
            self.edges[step_id] = set()
        if step_id not in self.reverse_edges:
            self.reverse_edges[step_id] = set()

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add a dependency edge (from_id depends on to_id)."""
        self.add_node(from_id)
        self.add_node(to_id)
        self.edges[from_id].add(to_id)
        self.reverse_edges[to_id].add(from_id)

    def get_dependencies(self, step_id: str) -> set[str]:
        """Get all direct dependencies of a step."""
        return self.edges.get(step_id, set())

    def get_dependents(self, step_id: str) -> set[str]:
        """Get all steps that directly depend on this step."""
        return self.reverse_edges.get(step_id, set())

    def get_roots(self) -> set[str]:
        """Get all nodes with no dependencies."""
        return {node for node in self.nodes if not self.edges.get(node)}

    def get_leaves(self) -> set[str]:
        """Get all nodes with no dependents."""
        return {node for node in self.nodes if not self.reverse_edges.get(node)}


@dataclass
class ParallelGroup:
    """A group of steps that can execute concurrently.

    Attributes:
        step_ids: Set of step IDs in this group
        level: Execution level (groups at same level can't run until
               all groups at lower levels complete)
    """

    step_ids: set[str]
    level: int


def build_dependency_graph(steps: list[StepConfig]) -> DependencyGraph:
    """Build a dependency graph from workflow steps.

    Args:
        steps: List of StepConfig objects

    Returns:
        DependencyGraph representing step dependencies
    """
    graph = DependencyGraph()

    for step in steps:
        graph.add_node(step.id)
        for dep_id in step.depends_on:
            graph.add_edge(step.id, dep_id)

    return graph


def find_parallel_groups(graph: DependencyGraph) -> list[ParallelGroup]:
    """Find groups of steps that can execute in parallel.

    Uses topological sorting to identify execution levels.
    Steps at the same level have no dependencies on each other
    and can run concurrently.

    Args:
        graph: DependencyGraph of step dependencies

    Returns:
        List of ParallelGroups ordered by execution level
    """
    groups: list[ParallelGroup] = []
    completed: set[str] = set()
    remaining = graph.nodes.copy()

    level = 0
    while remaining:
        # Find all steps whose dependencies are satisfied
        ready = {
            step_id for step_id in remaining if graph.get_dependencies(step_id).issubset(completed)
        }

        if not ready:
            # Circular dependency detected
            raise ValueError(f"Circular dependency detected. Remaining steps: {remaining}")

        groups.append(ParallelGroup(step_ids=ready, level=level))
        completed.update(ready)
        remaining -= ready
        level += 1

    return groups


def analyze_parallelism(workflow: WorkflowConfig) -> dict:
    """Analyze a workflow's parallelism potential.

    Args:
        workflow: WorkflowConfig to analyze

    Returns:
        Dict containing:
        - total_steps: Total number of steps
        - parallel_groups: List of ParallelGroups
        - max_parallelism: Maximum concurrent steps possible
        - sequential_depth: Number of sequential levels
        - speedup_potential: Theoretical speedup ratio
    """
    if not workflow.steps:
        return {
            "total_steps": 0,
            "parallel_groups": [],
            "max_parallelism": 0,
            "sequential_depth": 0,
            "speedup_potential": 1.0,
        }

    graph = build_dependency_graph(workflow.steps)
    groups = find_parallel_groups(graph)

    max_parallelism = max(len(g.step_ids) for g in groups) if groups else 0
    sequential_depth = len(groups)
    total_steps = len(workflow.steps)

    # Speedup potential: ratio of total steps to sequential depth
    # A fully parallel workflow would have speedup_potential = total_steps
    # A fully sequential workflow would have speedup_potential = 1.0
    speedup_potential = total_steps / sequential_depth if sequential_depth > 0 else 1.0

    return {
        "total_steps": total_steps,
        "parallel_groups": groups,
        "max_parallelism": max_parallelism,
        "sequential_depth": sequential_depth,
        "speedup_potential": round(speedup_potential, 2),
    }


def get_step_execution_order(
    workflow: WorkflowConfig,
    max_concurrent: int = 4,
) -> list[list[str]]:
    """Get optimal step execution order respecting dependencies.

    Returns steps grouped into batches that can run concurrently,
    limited by max_concurrent.

    Args:
        workflow: WorkflowConfig to analyze
        max_concurrent: Maximum steps to run concurrently

    Returns:
        List of batches, each containing step IDs to execute concurrently
    """
    if not workflow.steps:
        return []

    graph = build_dependency_graph(workflow.steps)
    groups = find_parallel_groups(graph)

    batches: list[list[str]] = []
    for group in groups:
        step_list = list(group.step_ids)
        # Split into sub-batches if group exceeds max_concurrent
        for i in range(0, len(step_list), max_concurrent):
            batches.append(step_list[i : i + max_concurrent])

    return batches


def can_run_parallel(
    step_id: str,
    completed: set[str],
    graph: DependencyGraph,
) -> bool:
    """Check if a step can run given completed steps.

    Args:
        step_id: Step to check
        completed: Set of completed step IDs
        graph: DependencyGraph

    Returns:
        True if all dependencies are satisfied
    """
    return graph.get_dependencies(step_id).issubset(completed)


def get_ready_steps(
    remaining: set[str],
    completed: set[str],
    graph: DependencyGraph,
) -> set[str]:
    """Get all steps that are ready to execute.

    Args:
        remaining: Set of remaining step IDs
        completed: Set of completed step IDs
        graph: DependencyGraph

    Returns:
        Set of step IDs ready to execute
    """
    return {step_id for step_id in remaining if can_run_parallel(step_id, completed, graph)}


def validate_no_cycles(graph: DependencyGraph) -> bool:
    """Validate that the dependency graph has no cycles.

    Args:
        graph: DependencyGraph to validate

    Returns:
        True if no cycles, raises ValueError if cycles detected
    """
    visited: set[str] = set()
    rec_stack: set[str] = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph.get_dependencies(node):
            if neighbor not in visited:
                if not dfs(neighbor):
                    return False
            elif neighbor in rec_stack:
                return False

        rec_stack.remove(node)
        return True

    for node in graph.nodes:
        if node not in visited:
            if not dfs(node):
                raise ValueError(f"Cycle detected involving step: {node}")

    return True
