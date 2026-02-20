"""Tests for dependency cycle detection."""

from __future__ import annotations

import pytest
from convergent.cycles import DependencyCycle, DependencyGraph, find_cycles, topological_order
from convergent.intent import Evidence, EvidenceKind, Intent, InterfaceKind, InterfaceSpec
from convergent.resolver import IntentResolver, PythonGraphBackend


def _make_intent(
    agent_id: str,
    name: str,
    provides: list[str] | None = None,
    requires: list[str] | None = None,
) -> Intent:
    """Helper to create test intents with named provides/requires."""
    prov = [
        InterfaceSpec(name=n, kind=InterfaceKind.FUNCTION, signature="") for n in (provides or [])
    ]
    req = [
        InterfaceSpec(name=n, kind=InterfaceKind.FUNCTION, signature="") for n in (requires or [])
    ]
    return Intent(
        agent_id=agent_id,
        intent=name,
        provides=prov,
        requires=req,
        evidence=[Evidence(kind=EvidenceKind.TEST_PASS, description="test")],
    )


def _make_resolver(*intents: Intent) -> IntentResolver:
    backend = PythonGraphBackend()
    for i in intents:
        backend.publish(i)
    return IntentResolver(backend=backend)


class TestDependencyGraph:
    """Tests for DependencyGraph construction."""

    def test_basic_edges(self) -> None:
        i1 = _make_intent("a1", "auth", provides=["AuthService"])
        i2 = _make_intent("a2", "ui", requires=["AuthService"])
        graph = DependencyGraph([i1, i2])

        assert ("ui", "auth") in graph.edges  # ui depends on auth
        assert ("auth", "ui") not in graph.edges  # not the reverse

    def test_no_self_edges(self) -> None:
        i = _make_intent("a1", "self_ref", provides=["Foo"], requires=["Foo"])
        graph = DependencyGraph([i])
        assert graph.edges == []

    def test_no_edges_when_no_overlap(self) -> None:
        i1 = _make_intent("a1", "auth", provides=["AuthService"])
        i2 = _make_intent("a2", "db", provides=["DBService"])
        graph = DependencyGraph([i1, i2])
        assert graph.edges == []

    def test_multiple_dependencies(self) -> None:
        i1 = _make_intent("a1", "auth", provides=["AuthService"])
        i2 = _make_intent("a2", "db", provides=["DBService"])
        i3 = _make_intent("a3", "api", requires=["AuthService", "DBService"])
        graph = DependencyGraph([i1, i2, i3])

        assert ("api", "auth") in graph.edges
        assert ("api", "db") in graph.edges

    def test_neighbors(self) -> None:
        i1 = _make_intent("a1", "auth", provides=["Auth"])
        i2 = _make_intent("a2", "ui", requires=["Auth"])
        graph = DependencyGraph([i1, i2])
        assert graph.neighbors("ui") == {"auth"}
        assert graph.neighbors("auth") == set()

    def test_neighbors_unknown_node(self) -> None:
        graph = DependencyGraph([])
        assert graph.neighbors("nonexistent") == set()


class TestFindCycles:
    """Tests for cycle detection."""

    def test_no_cycles_linear(self) -> None:
        """A -> B -> C has no cycles."""
        i1 = _make_intent("a1", "C", provides=["CService"])
        i2 = _make_intent("a2", "B", provides=["BService"], requires=["CService"])
        i3 = _make_intent("a3", "A", requires=["BService"])
        resolver = _make_resolver(i3, i2, i1)
        assert find_cycles(resolver) == []

    def test_simple_cycle(self) -> None:
        """A requires B, B requires A = cycle."""
        i1 = _make_intent("a1", "A", provides=["AService"], requires=["BService"])
        i2 = _make_intent("a2", "B", provides=["BService"], requires=["AService"])
        resolver = _make_resolver(i1, i2)
        cycles = find_cycles(resolver)
        assert len(cycles) == 1
        assert len(cycles[0].intent_ids) == 2

    def test_three_node_cycle(self) -> None:
        """A -> B -> C -> A cycle."""
        i1 = _make_intent("a1", "A", provides=["AService"], requires=["CService"])
        i2 = _make_intent("a2", "B", provides=["BService"], requires=["AService"])
        i3 = _make_intent("a3", "C", provides=["CService"], requires=["BService"])
        resolver = _make_resolver(i1, i2, i3)
        cycles = find_cycles(resolver)
        assert len(cycles) >= 1
        # Cycle should contain all three nodes
        all_ids = set()
        for c in cycles:
            all_ids.update(c.intent_ids)
        assert all_ids == {"A", "B", "C"}

    def test_cycle_str(self) -> None:
        cycle = DependencyCycle(
            intent_ids=("A", "B"),
            agent_ids=("a1", "a2"),
        )
        s = str(cycle)
        assert "A(a1)" in s
        assert "B(a2)" in s
        assert "->" in s

    def test_empty_graph(self) -> None:
        resolver = _make_resolver()
        assert find_cycles(resolver) == []

    def test_single_intent_no_cycle(self) -> None:
        i = _make_intent("a1", "solo", provides=["X"])
        resolver = _make_resolver(i)
        assert find_cycles(resolver) == []

    def test_diamond_no_cycle(self) -> None:
        """Diamond: A->B, A->C, B->D, C->D has no cycles."""
        d = _make_intent("a4", "D", provides=["DService"])
        b = _make_intent("a2", "B", provides=["BService"], requires=["DService"])
        c = _make_intent("a3", "C", provides=["CService"], requires=["DService"])
        a = _make_intent("a1", "A", requires=["BService", "CService"])
        resolver = _make_resolver(a, b, c, d)
        assert find_cycles(resolver) == []


class TestTopologicalOrder:
    """Tests for topological sort."""

    def test_linear_order(self) -> None:
        """A requires B requires C => order is [C, B, A]."""
        c = _make_intent("a3", "C", provides=["CService"])
        b = _make_intent("a2", "B", provides=["BService"], requires=["CService"])
        a = _make_intent("a1", "A", requires=["BService"])
        resolver = _make_resolver(a, b, c)

        order = topological_order(resolver)
        assert order.index("C") < order.index("B")
        assert order.index("B") < order.index("A")

    def test_raises_on_cycle(self) -> None:
        i1 = _make_intent("a1", "A", provides=["AService"], requires=["BService"])
        i2 = _make_intent("a2", "B", provides=["BService"], requires=["AService"])
        resolver = _make_resolver(i1, i2)

        with pytest.raises(ValueError, match="cycle"):
            topological_order(resolver)

    def test_empty_graph(self) -> None:
        resolver = _make_resolver()
        assert topological_order(resolver) == []

    def test_independent_intents(self) -> None:
        """Independent intents can be in any order."""
        i1 = _make_intent("a1", "A", provides=["X"])
        i2 = _make_intent("a2", "B", provides=["Y"])
        resolver = _make_resolver(i1, i2)
        order = topological_order(resolver)
        assert set(order) == {"A", "B"}

    def test_diamond_order(self) -> None:
        """Diamond: D before B and C, B and C before A."""
        d = _make_intent("a4", "D", provides=["DService"])
        b = _make_intent("a2", "B", provides=["BService"], requires=["DService"])
        c = _make_intent("a3", "C", provides=["CService"], requires=["DService"])
        a = _make_intent("a1", "A", requires=["BService", "CService"])
        resolver = _make_resolver(a, b, c, d)

        order = topological_order(resolver)
        assert order.index("D") < order.index("B")
        assert order.index("D") < order.index("C")
        assert order.index("B") < order.index("A")
        assert order.index("C") < order.index("A")
