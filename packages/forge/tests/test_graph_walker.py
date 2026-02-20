"""Tests for the graph walker module."""

import pytest

from animus_forge.workflow.graph_models import GraphEdge, GraphNode, WorkflowGraph
from animus_forge.workflow.graph_walker import GraphWalker

# ---------------------------------------------------------------------------
# Helpers - graph builders
# ---------------------------------------------------------------------------


def _make_graph(nodes, edges):
    """Build a WorkflowGraph from compact node/edge specs."""
    graph_nodes = [
        GraphNode(id=n["id"], type=n.get("type", "agent"), data=n.get("data", {})) for n in nodes
    ]
    graph_edges = [
        GraphEdge(
            id=e.get("id", f"{e['source']}->{e['target']}"),
            source=e["source"],
            target=e["target"],
            source_handle=e.get("source_handle"),
        )
        for e in edges
    ]
    return WorkflowGraph(id="test", name="Test Graph", nodes=graph_nodes, edges=graph_edges)


def _linear_graph():
    """A -> B -> C."""
    return _make_graph(
        [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ],
    )


def _diamond_graph():
    """A -> B, A -> C, B -> D, C -> D."""
    return _make_graph(
        [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}],
        [
            {"source": "A", "target": "B"},
            {"source": "A", "target": "C"},
            {"source": "B", "target": "D"},
            {"source": "C", "target": "D"},
        ],
    )


def _branch_graph():
    """A (branch) --true--> B, --false--> C."""
    return _make_graph(
        [
            {
                "id": "A",
                "type": "branch",
                "data": {
                    "condition": {
                        "field": "val",
                        "operator": "equals",
                        "value": 42,
                    },
                },
            },
            {"id": "B"},
            {"id": "C"},
        ],
        [
            {"source": "A", "target": "B", "source_handle": "true"},
            {"source": "A", "target": "C", "source_handle": "false"},
        ],
    )


def _loop_count_graph(count=3, max_iter=10):
    """A loop node with count-based iteration."""
    return _make_graph(
        [
            {
                "id": "loop1",
                "type": "loop",
                "data": {
                    "loop_type": "count",
                    "count": count,
                    "max_iterations": max_iter,
                },
            },
            {"id": "body"},
        ],
        [{"source": "loop1", "target": "body"}],
    )


def _loop_while_graph():
    """A loop node with while-condition."""
    return _make_graph(
        [
            {
                "id": "loop1",
                "type": "loop",
                "data": {
                    "loop_type": "while",
                    "max_iterations": 10,
                    "condition": {
                        "field": "running",
                        "operator": "equals",
                        "value": True,
                    },
                },
            },
            {"id": "body"},
        ],
        [{"source": "loop1", "target": "body"}],
    )


def _loop_foreach_graph():
    """A loop node with for_each iteration."""
    return _make_graph(
        [
            {
                "id": "loop1",
                "type": "loop",
                "data": {
                    "loop_type": "for_each",
                    "max_iterations": 100,
                    "collection": "items",
                },
            },
            {"id": "body"},
        ],
        [{"source": "loop1", "target": "body"}],
    )


def _cycle_graph():
    """A -> B -> C -> A (cycle)."""
    return _make_graph(
        [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "A"},
        ],
    )


# ---------------------------------------------------------------------------
# get_start_nodes
# ---------------------------------------------------------------------------


class TestGetStartNodes:
    """Tests for GraphWalker.get_start_nodes."""

    def test_linear(self):
        walker = GraphWalker(_linear_graph())
        assert walker.get_start_nodes() == ["A"]

    def test_diamond(self):
        walker = GraphWalker(_diamond_graph())
        assert walker.get_start_nodes() == ["A"]

    def test_multiple_start_nodes(self):
        graph = _make_graph(
            [{"id": "A"}, {"id": "B"}, {"id": "C"}],
            [{"source": "A", "target": "C"}, {"source": "B", "target": "C"}],
        )
        walker = GraphWalker(graph)
        starts = walker.get_start_nodes()
        assert set(starts) == {"A", "B"}

    def test_start_type_node(self):
        graph = _make_graph(
            [{"id": "s1", "type": "start"}, {"id": "A"}, {"id": "B"}],
            [
                {"source": "s1", "target": "A"},
                {"source": "A", "target": "B"},
            ],
        )
        walker = GraphWalker(graph)
        starts = walker.get_start_nodes()
        assert "s1" in starts

    def test_no_edges(self):
        """All nodes are start nodes if no edges."""
        graph = _make_graph(
            [{"id": "A"}, {"id": "B"}],
            [],
        )
        walker = GraphWalker(graph)
        assert set(walker.get_start_nodes()) == {"A", "B"}


# ---------------------------------------------------------------------------
# get_ready_nodes
# ---------------------------------------------------------------------------


class TestGetReadyNodes:
    """Tests for GraphWalker.get_ready_nodes."""

    def test_linear_nothing_completed(self):
        walker = GraphWalker(_linear_graph())
        ready = walker.get_ready_nodes(set())
        assert ready == ["A"]

    def test_linear_first_completed(self):
        walker = GraphWalker(_linear_graph())
        ready = walker.get_ready_nodes({"A"})
        assert ready == ["B"]

    def test_linear_all_completed(self):
        walker = GraphWalker(_linear_graph())
        ready = walker.get_ready_nodes({"A", "B", "C"})
        assert ready == []

    def test_diamond_parallel(self):
        walker = GraphWalker(_diamond_graph())
        ready = walker.get_ready_nodes({"A"})
        assert set(ready) == {"B", "C"}

    def test_diamond_converge(self):
        walker = GraphWalker(_diamond_graph())
        ready = walker.get_ready_nodes({"A", "B", "C"})
        assert ready == ["D"]

    def test_diamond_partial_converge(self):
        """D not ready until both B and C complete."""
        walker = GraphWalker(_diamond_graph())
        ready = walker.get_ready_nodes({"A", "B"})
        assert "D" not in ready

    def test_branch_true_path(self):
        walker = GraphWalker(_branch_graph())
        ready = walker.get_ready_nodes({"A"}, branch_decisions={"A": "true"})
        assert "B" in ready
        assert "C" not in ready

    def test_branch_false_path(self):
        walker = GraphWalker(_branch_graph())
        ready = walker.get_ready_nodes({"A"}, branch_decisions={"A": "false"})
        assert "C" in ready
        assert "B" not in ready

    def test_branch_no_decision(self):
        """Without branch decision, both paths are ready."""
        walker = GraphWalker(_branch_graph())
        ready = walker.get_ready_nodes({"A"}, branch_decisions={})
        assert set(ready) == {"B", "C"}


# ---------------------------------------------------------------------------
# get_downstream_nodes / get_all_downstream
# ---------------------------------------------------------------------------


class TestGetDownstream:
    """Tests for downstream node queries."""

    def test_immediate_downstream(self):
        walker = GraphWalker(_diamond_graph())
        downstream = walker.get_downstream_nodes("A")
        assert set(downstream) == {"B", "C"}

    def test_downstream_with_handle(self):
        walker = GraphWalker(_branch_graph())
        true_branch = walker.get_downstream_nodes("A", handle="true")
        assert true_branch == ["B"]
        false_branch = walker.get_downstream_nodes("A", handle="false")
        assert false_branch == ["C"]

    def test_no_downstream(self):
        walker = GraphWalker(_linear_graph())
        assert walker.get_downstream_nodes("C") == []

    def test_all_downstream_transitive(self):
        walker = GraphWalker(_linear_graph())
        all_down = walker.get_all_downstream("A")
        assert all_down == {"B", "C"}

    def test_all_downstream_with_handle(self):
        walker = GraphWalker(_branch_graph())
        all_down = walker.get_all_downstream("A", handle="true")
        assert "B" in all_down
        # C is only reachable via the "false" handle
        assert "C" not in all_down

    def test_all_downstream_diamond(self):
        walker = GraphWalker(_diamond_graph())
        all_down = walker.get_all_downstream("A")
        assert all_down == {"B", "C", "D"}


# ---------------------------------------------------------------------------
# evaluate_branch
# ---------------------------------------------------------------------------


class TestEvaluateBranch:
    """Tests for branch condition evaluation."""

    def test_equals_true(self):
        walker = GraphWalker(_branch_graph())
        result = walker.evaluate_branch("A", {"val": 42})
        assert result == "true"

    def test_equals_false(self):
        walker = GraphWalker(_branch_graph())
        result = walker.evaluate_branch("A", {"val": 99})
        assert result == "false"

    def test_non_branch_node_raises(self):
        walker = GraphWalker(_linear_graph())
        with pytest.raises(ValueError, match="not a branch"):
            walker.evaluate_branch("A", {})

    def test_missing_node_raises(self):
        walker = GraphWalker(_linear_graph())
        with pytest.raises(ValueError):
            walker.evaluate_branch("MISSING", {})


# ---------------------------------------------------------------------------
# _evaluate_condition - exhaustive operator coverage
# ---------------------------------------------------------------------------


class TestEvaluateCondition:
    """Tests for the _evaluate_condition method (all operators)."""

    @pytest.fixture
    def walker(self):
        return GraphWalker(_linear_graph())

    def test_none_actual_returns_false(self, walker):
        assert walker._evaluate_condition(None, "equals", 1) is False

    def test_equals(self, walker):
        assert walker._evaluate_condition(5, "equals", 5) is True
        assert walker._evaluate_condition(5, "equals", 6) is False

    def test_not_equals(self, walker):
        assert walker._evaluate_condition(5, "not_equals", 6) is True
        assert walker._evaluate_condition(5, "not_equals", 5) is False

    def test_contains_string(self, walker):
        assert walker._evaluate_condition("hello world", "contains", "world") is True
        assert walker._evaluate_condition("hello", "contains", "world") is False

    def test_contains_list(self, walker):
        assert walker._evaluate_condition([1, 2, 3], "contains", 2) is True
        assert walker._evaluate_condition([1, 2, 3], "contains", 4) is False

    def test_contains_non_iterable(self, walker):
        assert walker._evaluate_condition(42, "contains", 4) is False

    def test_greater_than(self, walker):
        assert walker._evaluate_condition(10, "greater_than", 5) is True
        assert walker._evaluate_condition(3, "greater_than", 5) is False

    def test_greater_than_non_numeric(self, walker):
        assert walker._evaluate_condition("abc", "greater_than", 5) is False

    def test_less_than(self, walker):
        assert walker._evaluate_condition(3, "less_than", 5) is True
        assert walker._evaluate_condition(10, "less_than", 5) is False

    def test_less_than_non_numeric(self, walker):
        assert walker._evaluate_condition("abc", "less_than", 5) is False

    def test_in_list(self, walker):
        assert walker._evaluate_condition("a", "in", ["a", "b", "c"]) is True
        assert walker._evaluate_condition("x", "in", ["a", "b", "c"]) is False

    def test_in_string(self, walker):
        assert walker._evaluate_condition("he", "in", "hello") is True
        assert walker._evaluate_condition("x", "in", "hello") is False

    def test_in_non_collection(self, walker):
        assert walker._evaluate_condition("a", "in", 42) is False

    def test_not_empty_truthy(self, walker):
        assert walker._evaluate_condition("hello", "not_empty", None) is True
        assert walker._evaluate_condition([1], "not_empty", None) is True
        assert walker._evaluate_condition(1, "not_empty", None) is True

    def test_not_empty_falsy(self, walker):
        assert walker._evaluate_condition("", "not_empty", None) is False
        assert walker._evaluate_condition([], "not_empty", None) is False
        assert walker._evaluate_condition(0, "not_empty", None) is False

    def test_is_true(self, walker):
        assert walker._evaluate_condition(True, "is_true", None) is True
        assert walker._evaluate_condition(1, "is_true", None) is True
        assert walker._evaluate_condition(False, "is_true", None) is False

    def test_is_false(self, walker):
        assert walker._evaluate_condition(False, "is_false", None) is True
        assert walker._evaluate_condition(0, "is_false", None) is True
        assert walker._evaluate_condition(True, "is_false", None) is False

    def test_unknown_operator(self, walker):
        assert walker._evaluate_condition(5, "unknown_op", 5) is False


# ---------------------------------------------------------------------------
# should_continue_loop
# ---------------------------------------------------------------------------


class TestShouldContinueLoop:
    """Tests for loop continuation logic."""

    def test_count_loop_under(self):
        walker = GraphWalker(_loop_count_graph(count=3))
        assert walker.should_continue_loop("loop1", {}, 0) is True
        assert walker.should_continue_loop("loop1", {}, 1) is True
        assert walker.should_continue_loop("loop1", {}, 2) is True

    def test_count_loop_at_limit(self):
        walker = GraphWalker(_loop_count_graph(count=3))
        assert walker.should_continue_loop("loop1", {}, 3) is False

    def test_count_loop_max_iterations(self):
        """Max iterations cap overrides count."""
        walker = GraphWalker(_loop_count_graph(count=100, max_iter=5))
        assert walker.should_continue_loop("loop1", {}, 5) is False

    def test_while_loop_true(self):
        walker = GraphWalker(_loop_while_graph())
        assert walker.should_continue_loop("loop1", {"running": True}, 0) is True

    def test_while_loop_false(self):
        walker = GraphWalker(_loop_while_graph())
        assert walker.should_continue_loop("loop1", {"running": False}, 0) is False

    def test_while_loop_max_iterations(self):
        walker = GraphWalker(_loop_while_graph())
        assert walker.should_continue_loop("loop1", {"running": True}, 10) is False

    def test_foreach_loop(self):
        walker = GraphWalker(_loop_foreach_graph())
        ctx = {"items": ["a", "b", "c"]}
        assert walker.should_continue_loop("loop1", ctx, 0) is True
        assert walker.should_continue_loop("loop1", ctx, 2) is True
        assert walker.should_continue_loop("loop1", ctx, 3) is False

    def test_foreach_non_iterable(self):
        walker = GraphWalker(_loop_foreach_graph())
        assert walker.should_continue_loop("loop1", {"items": 42}, 0) is False

    def test_foreach_empty_collection(self):
        walker = GraphWalker(_loop_foreach_graph())
        assert walker.should_continue_loop("loop1", {"items": []}, 0) is False

    def test_unknown_loop_type(self):
        graph = _make_graph(
            [
                {
                    "id": "loop1",
                    "type": "loop",
                    "data": {
                        "loop_type": "weird",
                        "max_iterations": 10,
                    },
                }
            ],
            [],
        )
        walker = GraphWalker(graph)
        assert walker.should_continue_loop("loop1", {}, 0) is False

    def test_non_loop_node_raises(self):
        walker = GraphWalker(_linear_graph())
        with pytest.raises(ValueError, match="not a loop"):
            walker.should_continue_loop("A", {}, 0)


# ---------------------------------------------------------------------------
# get_loop_item
# ---------------------------------------------------------------------------


class TestGetLoopItem:
    """Tests for for-each loop item retrieval."""

    def test_valid_item(self):
        walker = GraphWalker(_loop_foreach_graph())
        ctx = {"items": ["a", "b", "c"]}
        assert walker.get_loop_item("loop1", ctx, 0) == "a"
        assert walker.get_loop_item("loop1", ctx, 1) == "b"
        assert walker.get_loop_item("loop1", ctx, 2) == "c"

    def test_out_of_bounds(self):
        walker = GraphWalker(_loop_foreach_graph())
        ctx = {"items": ["a"]}
        assert walker.get_loop_item("loop1", ctx, 5) is None

    def test_non_loop_node(self):
        walker = GraphWalker(_linear_graph())
        assert walker.get_loop_item("A", {}, 0) is None

    def test_non_foreach_loop(self):
        walker = GraphWalker(_loop_count_graph())
        assert walker.get_loop_item("loop1", {}, 0) is None

    def test_non_iterable_collection(self):
        walker = GraphWalker(_loop_foreach_graph())
        assert walker.get_loop_item("loop1", {"items": 42}, 0) is None


# ---------------------------------------------------------------------------
# detect_cycles
# ---------------------------------------------------------------------------


class TestDetectCycles:
    """Tests for cycle detection."""

    def test_no_cycles(self):
        walker = GraphWalker(_linear_graph())
        cycles = walker.detect_cycles()
        assert cycles == []

    def test_diamond_no_cycles(self):
        walker = GraphWalker(_diamond_graph())
        assert walker.detect_cycles() == []

    def test_simple_cycle(self):
        walker = GraphWalker(_cycle_graph())
        cycles = walker.detect_cycles()
        assert len(cycles) >= 1
        # The cycle should contain A, B, C
        flat = set()
        for c in cycles:
            flat.update(c)
        assert {"A", "B", "C"}.issubset(flat)

    def test_self_loop(self):
        graph = _make_graph(
            [{"id": "A"}],
            [{"source": "A", "target": "A"}],
        )
        walker = GraphWalker(graph)
        cycles = walker.detect_cycles()
        assert len(cycles) >= 1


# ---------------------------------------------------------------------------
# topological_sort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    """Tests for topological sort."""

    def test_linear(self):
        walker = GraphWalker(_linear_graph())
        order = walker.topological_sort()
        assert order == ["A", "B", "C"]

    def test_diamond(self):
        walker = GraphWalker(_diamond_graph())
        order = walker.topological_sort()
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_independent_nodes(self):
        graph = _make_graph([{"id": "A"}, {"id": "B"}, {"id": "C"}], [])
        walker = GraphWalker(graph)
        order = walker.topological_sort()
        assert set(order) == {"A", "B", "C"}

    def test_cycle_raises(self):
        walker = GraphWalker(_cycle_graph())
        with pytest.raises(ValueError, match="[Cc]ycles"):
            walker.topological_sort()


# ---------------------------------------------------------------------------
# _build_adjacency (internal)
# ---------------------------------------------------------------------------


class TestBuildAdjacency:
    """Tests for adjacency list construction."""

    def test_outgoing(self):
        walker = GraphWalker(_linear_graph())
        assert len(walker._outgoing["A"]) == 1
        assert walker._outgoing["A"][0].target == "B"

    def test_incoming(self):
        walker = GraphWalker(_linear_graph())
        assert len(walker._incoming["B"]) == 1
        assert walker._incoming["B"][0].source == "A"

    def test_no_edges(self):
        graph = _make_graph([{"id": "A"}], [])
        walker = GraphWalker(graph)
        assert walker._outgoing["A"] == []
        assert walker._incoming["A"] == []
