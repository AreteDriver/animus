"""Tests for ReactFlow workflow graph models."""

import sys

sys.path.insert(0, "src")

from animus_forge.workflow.graph_models import (
    GraphEdge,
    GraphNode,
    NodePosition,
    WorkflowGraph,
)


class TestNodePosition:
    """Tests for NodePosition dataclass."""

    def test_creation(self):
        """NodePosition can be created with x and y."""
        pos = NodePosition(x=100.5, y=200.0)
        assert pos.x == 100.5
        assert pos.y == 200.0

    def test_default_values(self):
        """NodePosition defaults to origin."""
        pos = NodePosition(0, 0)
        assert pos.x == 0
        assert pos.y == 0


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_minimal_creation(self):
        """GraphNode can be created with minimal fields."""
        node = GraphNode(id="node-1", type="agent")

        assert node.id == "node-1"
        assert node.type == "agent"
        assert node.data == {}
        assert node.position.x == 0
        assert node.position.y == 0

    def test_full_creation(self):
        """GraphNode can be created with all fields."""
        node = GraphNode(
            id="node-2",
            type="shell",
            data={"command": "ls -la"},
            position=NodePosition(x=150, y=250),
        )

        assert node.id == "node-2"
        assert node.type == "shell"
        assert node.data == {"command": "ls -la"}
        assert node.position.x == 150
        assert node.position.y == 250

    def test_from_dict_minimal(self):
        """GraphNode.from_dict works with minimal data."""
        data = {"id": "n1", "type": "checkpoint"}
        node = GraphNode.from_dict(data)

        assert node.id == "n1"
        assert node.type == "checkpoint"

    def test_from_dict_full(self):
        """GraphNode.from_dict works with full ReactFlow format."""
        data = {
            "id": "node-123",
            "type": "branch",
            "data": {"condition": {"field": "status", "operator": "equals", "value": "ok"}},
            "position": {"x": 300, "y": 400},
        }
        node = GraphNode.from_dict(data)

        assert node.id == "node-123"
        assert node.type == "branch"
        assert node.data["condition"]["field"] == "status"
        assert node.position.x == 300
        assert node.position.y == 400

    def test_from_dict_missing_position(self):
        """GraphNode.from_dict handles missing position."""
        data = {"id": "n1", "type": "agent"}
        node = GraphNode.from_dict(data)

        assert node.position.x == 0
        assert node.position.y == 0

    def test_to_dict(self):
        """GraphNode.to_dict produces ReactFlow format."""
        node = GraphNode(
            id="node-1",
            type="loop",
            data={"max_iterations": 10},
            position=NodePosition(x=100, y=200),
        )
        result = node.to_dict()

        assert result == {
            "id": "node-1",
            "type": "loop",
            "data": {"max_iterations": 10},
            "position": {"x": 100, "y": 200},
        }

    def test_roundtrip(self):
        """GraphNode survives dict roundtrip."""
        original = GraphNode(
            id="test",
            type="parallel",
            data={"steps": [1, 2, 3]},
            position=NodePosition(50, 75),
        )

        reconstructed = GraphNode.from_dict(original.to_dict())

        assert reconstructed.id == original.id
        assert reconstructed.type == original.type
        assert reconstructed.data == original.data
        assert reconstructed.position.x == original.position.x
        assert reconstructed.position.y == original.position.y


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_minimal_creation(self):
        """GraphEdge can be created with minimal fields."""
        edge = GraphEdge(id="e1", source="n1", target="n2")

        assert edge.id == "e1"
        assert edge.source == "n1"
        assert edge.target == "n2"
        assert edge.source_handle is None
        assert edge.target_handle is None
        assert edge.label is None

    def test_full_creation(self):
        """GraphEdge can be created with all fields."""
        edge = GraphEdge(
            id="edge-1",
            source="branch-1",
            target="node-2",
            source_handle="true",
            target_handle="input",
            label="Yes",
        )

        assert edge.source_handle == "true"
        assert edge.target_handle == "input"
        assert edge.label == "Yes"

    def test_from_dict_minimal(self):
        """GraphEdge.from_dict works with minimal data."""
        data = {"id": "e1", "source": "a", "target": "b"}
        edge = GraphEdge.from_dict(data)

        assert edge.id == "e1"
        assert edge.source == "a"
        assert edge.target == "b"

    def test_from_dict_full(self):
        """GraphEdge.from_dict works with full ReactFlow format."""
        data = {
            "id": "edge-123",
            "source": "node-1",
            "target": "node-2",
            "sourceHandle": "false",
            "targetHandle": "in",
            "label": "No Branch",
        }
        edge = GraphEdge.from_dict(data)

        assert edge.id == "edge-123"
        assert edge.source_handle == "false"
        assert edge.target_handle == "in"
        assert edge.label == "No Branch"

    def test_to_dict_minimal(self):
        """GraphEdge.to_dict omits None fields."""
        edge = GraphEdge(id="e1", source="a", target="b")
        result = edge.to_dict()

        assert result == {"id": "e1", "source": "a", "target": "b"}
        assert "sourceHandle" not in result
        assert "targetHandle" not in result
        assert "label" not in result

    def test_to_dict_full(self):
        """GraphEdge.to_dict includes all fields when set."""
        edge = GraphEdge(
            id="e1",
            source="a",
            target="b",
            source_handle="out",
            target_handle="in",
            label="Connection",
        )
        result = edge.to_dict()

        assert result["sourceHandle"] == "out"
        assert result["targetHandle"] == "in"
        assert result["label"] == "Connection"

    def test_roundtrip(self):
        """GraphEdge survives dict roundtrip."""
        original = GraphEdge(
            id="test-edge",
            source="node-a",
            target="node-b",
            source_handle="true",
            label="Test",
        )

        reconstructed = GraphEdge.from_dict(original.to_dict())

        assert reconstructed.id == original.id
        assert reconstructed.source == original.source
        assert reconstructed.target == original.target
        assert reconstructed.source_handle == original.source_handle
        assert reconstructed.label == original.label


class TestWorkflowGraph:
    """Tests for WorkflowGraph dataclass."""

    def test_minimal_creation(self):
        """WorkflowGraph can be created with minimal fields."""
        graph = WorkflowGraph(id="wf-1", name="Test Workflow")

        assert graph.id == "wf-1"
        assert graph.name == "Test Workflow"
        assert graph.nodes == []
        assert graph.edges == []
        assert graph.variables == {}
        assert graph.description == ""
        assert graph.version == "1.0"

    def test_full_creation(self):
        """WorkflowGraph can be created with all fields."""
        nodes = [
            GraphNode(id="start", type="start"),
            GraphNode(id="step1", type="agent"),
            GraphNode(id="end", type="end"),
        ]
        edges = [
            GraphEdge(id="e1", source="start", target="step1"),
            GraphEdge(id="e2", source="step1", target="end"),
        ]
        graph = WorkflowGraph(
            id="full-wf",
            name="Full Workflow",
            nodes=nodes,
            edges=edges,
            variables={"input": "value"},
            description="A complete workflow",
            version="2.0",
        )

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.variables == {"input": "value"}
        assert graph.version == "2.0"

    def test_get_node(self):
        """WorkflowGraph.get_node finds node by ID."""
        graph = WorkflowGraph(
            id="wf",
            name="Test",
            nodes=[
                GraphNode(id="a", type="agent"),
                GraphNode(id="b", type="shell"),
            ],
        )

        assert graph.get_node("a").type == "agent"
        assert graph.get_node("b").type == "shell"
        assert graph.get_node("nonexistent") is None

    def test_get_incoming_edges(self):
        """WorkflowGraph.get_incoming_edges returns edges pointing to a node."""
        graph = WorkflowGraph(
            id="wf",
            name="Test",
            nodes=[
                GraphNode(id="a", type="start"),
                GraphNode(id="b", type="agent"),
                GraphNode(id="c", type="end"),
            ],
            edges=[
                GraphEdge(id="e1", source="a", target="b"),
                GraphEdge(id="e2", source="b", target="c"),
            ],
        )

        incoming_b = graph.get_incoming_edges("b")
        assert len(incoming_b) == 1
        assert incoming_b[0].source == "a"

        incoming_a = graph.get_incoming_edges("a")
        assert len(incoming_a) == 0

    def test_get_outgoing_edges(self):
        """WorkflowGraph.get_outgoing_edges returns edges from a node."""
        graph = WorkflowGraph(
            id="wf",
            name="Test",
            nodes=[
                GraphNode(id="branch", type="branch"),
                GraphNode(id="yes", type="agent"),
                GraphNode(id="no", type="agent"),
            ],
            edges=[
                GraphEdge(id="e1", source="branch", target="yes", source_handle="true"),
                GraphEdge(id="e2", source="branch", target="no", source_handle="false"),
            ],
        )

        outgoing = graph.get_outgoing_edges("branch")
        assert len(outgoing) == 2

        # Verify both handles are present
        handles = {e.source_handle for e in outgoing}
        assert handles == {"true", "false"}

    def test_get_start_nodes(self):
        """WorkflowGraph.get_start_nodes returns nodes without incoming edges."""
        graph = WorkflowGraph(
            id="wf",
            name="Test",
            nodes=[
                GraphNode(id="start1", type="start"),
                GraphNode(id="start2", type="agent"),  # Also no incoming
                GraphNode(id="middle", type="shell"),
            ],
            edges=[
                GraphEdge(id="e1", source="start1", target="middle"),
                GraphEdge(id="e2", source="start2", target="middle"),
            ],
        )

        starts = graph.get_start_nodes()
        start_ids = {n.id for n in starts}
        assert start_ids == {"start1", "start2"}

    def test_get_end_nodes(self):
        """WorkflowGraph.get_end_nodes returns nodes without outgoing edges."""
        graph = WorkflowGraph(
            id="wf",
            name="Test",
            nodes=[
                GraphNode(id="start", type="start"),
                GraphNode(id="end1", type="end"),
                GraphNode(id="end2", type="agent"),  # Also no outgoing
            ],
            edges=[
                GraphEdge(id="e1", source="start", target="end1"),
                GraphEdge(id="e2", source="start", target="end2"),
            ],
        )

        ends = graph.get_end_nodes()
        end_ids = {n.id for n in ends}
        assert end_ids == {"end1", "end2"}

    def test_from_dict(self):
        """WorkflowGraph.from_dict parses ReactFlow format."""
        data = {
            "id": "imported-wf",
            "name": "Imported Workflow",
            "description": "From JSON",
            "version": "1.5",
            "variables": {"key": "value"},
            "nodes": [
                {"id": "n1", "type": "start", "position": {"x": 0, "y": 0}},
                {
                    "id": "n2",
                    "type": "agent",
                    "data": {"prompt": "Hello"},
                    "position": {"x": 100, "y": 0},
                },
            ],
            "edges": [{"id": "e1", "source": "n1", "target": "n2"}],
        }

        graph = WorkflowGraph.from_dict(data)

        assert graph.id == "imported-wf"
        assert graph.name == "Imported Workflow"
        assert graph.description == "From JSON"
        assert graph.version == "1.5"
        assert graph.variables == {"key": "value"}
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.nodes[1].data == {"prompt": "Hello"}

    def test_from_dict_minimal(self):
        """WorkflowGraph.from_dict handles minimal data."""
        data = {"nodes": [], "edges": []}
        graph = WorkflowGraph.from_dict(data)

        assert graph.id == ""
        assert graph.name == "Untitled Workflow"
        assert graph.version == "1.0"

    def test_to_dict(self):
        """WorkflowGraph.to_dict produces ReactFlow format."""
        graph = WorkflowGraph(
            id="export-wf",
            name="Export Test",
            description="Testing export",
            version="2.0",
            variables={"output": "result"},
            nodes=[GraphNode(id="n1", type="agent")],
            edges=[GraphEdge(id="e1", source="n1", target="n1")],
        )

        result = graph.to_dict()

        assert result["id"] == "export-wf"
        assert result["name"] == "Export Test"
        assert result["description"] == "Testing export"
        assert result["version"] == "2.0"
        assert result["variables"] == {"output": "result"}
        assert len(result["nodes"]) == 1
        assert len(result["edges"]) == 1

    def test_roundtrip(self):
        """WorkflowGraph survives dict roundtrip."""
        original = WorkflowGraph(
            id="roundtrip-test",
            name="Roundtrip Test",
            description="Testing serialization",
            version="1.2.3",
            variables={"a": 1, "b": [1, 2, 3]},
            nodes=[
                GraphNode(
                    id="n1",
                    type="branch",
                    data={"condition": {"field": "x"}},
                    position=NodePosition(100, 200),
                ),
                GraphNode(id="n2", type="agent"),
            ],
            edges=[
                GraphEdge(id="e1", source="n1", target="n2", source_handle="true", label="Yes"),
            ],
        )

        reconstructed = WorkflowGraph.from_dict(original.to_dict())

        assert reconstructed.id == original.id
        assert reconstructed.name == original.name
        assert reconstructed.description == original.description
        assert reconstructed.version == original.version
        assert reconstructed.variables == original.variables
        assert len(reconstructed.nodes) == len(original.nodes)
        assert len(reconstructed.edges) == len(original.edges)
        assert reconstructed.nodes[0].data == original.nodes[0].data
        assert reconstructed.edges[0].source_handle == "true"


class TestComplexGraphs:
    """Tests for complex graph structures."""

    def test_diamond_pattern(self):
        """Graph with diamond pattern (branch then merge)."""
        graph = WorkflowGraph(
            id="diamond",
            name="Diamond",
            nodes=[
                GraphNode(id="start", type="start"),
                GraphNode(id="branch", type="branch"),
                GraphNode(id="left", type="agent"),
                GraphNode(id="right", type="agent"),
                GraphNode(id="merge", type="agent"),
                GraphNode(id="end", type="end"),
            ],
            edges=[
                GraphEdge(id="e1", source="start", target="branch"),
                GraphEdge(id="e2", source="branch", target="left", source_handle="true"),
                GraphEdge(id="e3", source="branch", target="right", source_handle="false"),
                GraphEdge(id="e4", source="left", target="merge"),
                GraphEdge(id="e5", source="right", target="merge"),
                GraphEdge(id="e6", source="merge", target="end"),
            ],
        )

        # Start node
        starts = graph.get_start_nodes()
        assert len(starts) == 1
        assert starts[0].id == "start"

        # End node
        ends = graph.get_end_nodes()
        assert len(ends) == 1
        assert ends[0].id == "end"

        # Branch has 2 outgoing
        branch_out = graph.get_outgoing_edges("branch")
        assert len(branch_out) == 2

        # Merge has 2 incoming
        merge_in = graph.get_incoming_edges("merge")
        assert len(merge_in) == 2

    def test_parallel_execution_pattern(self):
        """Graph with parallel execution nodes."""
        graph = WorkflowGraph(
            id="parallel",
            name="Parallel",
            nodes=[
                GraphNode(id="start", type="start"),
                GraphNode(id="fan_out", type="fan_out"),
                GraphNode(id="task1", type="agent"),
                GraphNode(id="task2", type="agent"),
                GraphNode(id="task3", type="agent"),
                GraphNode(id="fan_in", type="fan_in"),
                GraphNode(id="end", type="end"),
            ],
            edges=[
                GraphEdge(id="e1", source="start", target="fan_out"),
                GraphEdge(id="e2", source="fan_out", target="task1"),
                GraphEdge(id="e3", source="fan_out", target="task2"),
                GraphEdge(id="e4", source="fan_out", target="task3"),
                GraphEdge(id="e5", source="task1", target="fan_in"),
                GraphEdge(id="e6", source="task2", target="fan_in"),
                GraphEdge(id="e7", source="task3", target="fan_in"),
                GraphEdge(id="e8", source="fan_in", target="end"),
            ],
        )

        # Fan out has 3 outgoing
        fan_out = graph.get_outgoing_edges("fan_out")
        assert len(fan_out) == 3

        # Fan in has 3 incoming
        fan_in = graph.get_incoming_edges("fan_in")
        assert len(fan_in) == 3

    def test_loop_pattern(self):
        """Graph with a loop (cycle)."""
        graph = WorkflowGraph(
            id="loop",
            name="Loop",
            nodes=[
                GraphNode(id="start", type="start"),
                GraphNode(
                    id="loop",
                    type="loop",
                    data={"loop_type": "while", "max_iterations": 5},
                ),
                GraphNode(id="body", type="agent"),
                GraphNode(id="end", type="end"),
            ],
            edges=[
                GraphEdge(id="e1", source="start", target="loop"),
                GraphEdge(id="e2", source="loop", target="body", source_handle="body"),
                GraphEdge(id="e3", source="body", target="loop"),  # Back edge
                GraphEdge(id="e4", source="loop", target="end", source_handle="done"),
            ],
        )

        # Loop has multiple outgoing
        loop_out = graph.get_outgoing_edges("loop")
        assert len(loop_out) == 2

        # Body points back to loop
        body_out = graph.get_outgoing_edges("body")
        assert body_out[0].target == "loop"
