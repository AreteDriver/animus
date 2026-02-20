"""Dependency cycle detection for the intent graph.

Detects circular dependencies in the provides/requires edges of the intent
graph. When Agent A requires something that Agent B provides, and B requires
something that A provides (directly or transitively), agents can deadlock
waiting on each other.

This module builds a directed dependency graph from provides/requires
relationships and uses DFS-based cycle detection (Tarjan-style) to find
all cycles. It also provides topological ordering for valid execution
sequencing.

Usage::

    from convergent.cycles import find_cycles, topological_order

    resolver = IntentResolver(backend=my_backend)
    cycles = find_cycles(resolver)
    if cycles:
        print(f"Found {len(cycles)} dependency cycles!")
    else:
        order = topological_order(resolver)
        print(f"Safe execution order: {order}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from convergent.intent import Intent
from convergent.matching import names_overlap
from convergent.resolver import IntentResolver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DependencyCycle:
    """A circular dependency in the intent graph.

    Attributes:
        intent_ids: Ordered list of intent IDs forming the cycle.
            The last element depends on the first, closing the loop.
        agent_ids: Corresponding agent IDs for each intent.
    """

    intent_ids: tuple[str, ...]
    agent_ids: tuple[str, ...]

    def __str__(self) -> str:
        parts = [f"{iid}({aid})" for iid, aid in zip(self.intent_ids, self.agent_ids, strict=True)]
        return " -> ".join(parts) + f" -> {parts[0]}"


class DependencyGraph:
    """Directed graph of intent dependencies based on provides/requires edges.

    An edge from intent A to intent B means A *requires* something that B
    *provides*. If these edges form a cycle, agents cannot be sequenced.

    Args:
        intents: List of intents to analyze.
    """

    def __init__(self, intents: list[Intent]) -> None:
        self._intents = {i.intent: i for i in intents}
        self._adjacency: dict[str, set[str]] = {i.intent: set() for i in intents}
        self._build_edges(intents)

    def _build_edges(self, intents: list[Intent]) -> None:
        """Build directed edges from requires -> provides relationships.

        For each intent A's requires specs, find intents B whose provides
        specs overlap with A's requirement. Edge: A -> B (A depends on B).
        """
        for a in intents:
            for req_spec in a.requires:
                for b in intents:
                    if a.intent == b.intent:
                        continue
                    for prov_spec in b.provides:
                        if names_overlap(req_spec.name, prov_spec.name):
                            self._adjacency[a.intent].add(b.intent)
                            break

    @property
    def nodes(self) -> list[str]:
        """All intent IDs in the graph."""
        return list(self._adjacency.keys())

    @property
    def edges(self) -> list[tuple[str, str]]:
        """All directed edges (from, to) in the graph."""
        result = []
        for src, dsts in self._adjacency.items():
            for dst in sorted(dsts):
                result.append((src, dst))
        return result

    def neighbors(self, intent_id: str) -> set[str]:
        """Get the dependencies (outgoing edges) for an intent.

        Args:
            intent_id: The intent to look up.

        Returns:
            Set of intent IDs that this intent depends on.
        """
        return self._adjacency.get(intent_id, set())


def find_cycles(resolver: IntentResolver) -> list[DependencyCycle]:
    """Find all dependency cycles in the intent graph.

    Uses DFS-based cycle detection. Each cycle is reported as a
    ``DependencyCycle`` containing the intent and agent IDs.

    Args:
        resolver: IntentResolver with published intents.

    Returns:
        List of detected dependency cycles. Empty if no cycles.
    """
    intents = resolver.backend.query_all()
    if not intents:
        return []

    graph = DependencyGraph(intents)
    intent_map = {i.intent: i for i in intents}
    cycles: list[DependencyCycle] = []
    visited: set[str] = set()
    path: list[str] = []
    on_path: set[str] = set()

    def _dfs(node: str) -> None:
        if node in on_path:
            # Found a cycle — extract it from the path
            cycle_start = path.index(node)
            cycle_nodes = path[cycle_start:]
            intent_ids = tuple(cycle_nodes)
            agent_ids = tuple(intent_map[n].agent_id for n in cycle_nodes)
            cycle = DependencyCycle(intent_ids=intent_ids, agent_ids=agent_ids)
            if cycle not in cycles:
                cycles.append(cycle)
            return

        if node in visited:
            return

        visited.add(node)
        on_path.add(node)
        path.append(node)

        for neighbor in sorted(graph.neighbors(node)):
            _dfs(neighbor)

        path.pop()
        on_path.discard(node)

    for node in sorted(graph.nodes):
        if node not in visited:
            _dfs(node)

    return cycles


def topological_order(resolver: IntentResolver) -> list[str]:
    """Compute a valid execution order for intents (topological sort).

    If there are cycles, raises ``ValueError`` with cycle details.

    Args:
        resolver: IntentResolver with published intents.

    Returns:
        List of intent IDs in a valid execution order (dependencies first).

    Raises:
        ValueError: If the dependency graph contains cycles.
    """
    cycles = find_cycles(resolver)
    if cycles:
        cycle_strs = [str(c) for c in cycles]
        msg = f"Cannot compute execution order: {len(cycles)} cycle(s) found: "
        msg += "; ".join(cycle_strs)
        raise ValueError(msg)

    intents = resolver.backend.query_all()
    if not intents:
        return []

    graph = DependencyGraph(intents)

    # Kahn's algorithm on reversed edges.
    # Edge A->B means "A depends on B", so B must execute first.
    # Build reverse adjacency: B->A, and use in-degree on the reversed graph.
    reverse_adj: dict[str, set[str]] = {n: set() for n in graph.nodes}
    in_degree: dict[str, int] = {n: 0 for n in graph.nodes}
    for src, dsts in graph._adjacency.items():
        for dst in dsts:
            reverse_adj[dst].add(src)
            in_degree[src] = in_degree.get(src, 0) + 1

    queue = sorted([n for n, d in in_degree.items() if d == 0])
    order: list[str] = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for dependent in sorted(reverse_adj[node]):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    return order
