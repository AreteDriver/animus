"""Data models for ReactFlow-style workflow graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class NodePosition:
    """Position of a node in the graph canvas."""

    x: float
    y: float


@dataclass
class GraphNode:
    """A node in the workflow graph.

    Represents a single step/operation in the visual workflow builder.
    Node types correspond to step types in the workflow executor.
    """

    id: str
    type: Literal[
        "agent",  # AI agent step (claude_code, openai)
        "shell",  # Shell command execution
        "checkpoint",  # Resume point
        "branch",  # Conditional branch
        "loop",  # Loop iteration
        "parallel",  # Parallel execution group
        "fan_out",  # Fan out to multiple
        "fan_in",  # Fan in from multiple
        "map_reduce",  # Map-reduce pattern
        "start",  # Workflow start node
        "end",  # Workflow end node
    ]
    data: dict[str, Any] = field(default_factory=dict)
    position: NodePosition = field(default_factory=lambda: NodePosition(0, 0))

    @classmethod
    def from_dict(cls, data: dict) -> GraphNode:
        """Create a GraphNode from a dictionary (ReactFlow format)."""
        position = data.get("position", {})
        return cls(
            id=data["id"],
            type=data.get("type", "agent"),
            data=data.get("data", {}),
            position=NodePosition(
                x=position.get("x", 0),
                y=position.get("y", 0),
            ),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary (ReactFlow format)."""
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data,
            "position": {"x": self.position.x, "y": self.position.y},
        }


@dataclass
class GraphEdge:
    """An edge connecting two nodes in the workflow graph.

    Edges define the execution flow between nodes.
    source_handle is used for conditional branches (e.g., "true", "false").
    """

    id: str
    source: str  # Source node ID
    target: str  # Target node ID
    source_handle: str | None = None  # For branch nodes: "true" or "false"
    target_handle: str | None = None
    label: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> GraphEdge:
        """Create a GraphEdge from a dictionary (ReactFlow format)."""
        return cls(
            id=data["id"],
            source=data["source"],
            target=data["target"],
            source_handle=data.get("sourceHandle"),
            target_handle=data.get("targetHandle"),
            label=data.get("label"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary (ReactFlow format)."""
        result = {
            "id": self.id,
            "source": self.source,
            "target": self.target,
        }
        if self.source_handle:
            result["sourceHandle"] = self.source_handle
        if self.target_handle:
            result["targetHandle"] = self.target_handle
        if self.label:
            result["label"] = self.label
        return result


@dataclass
class WorkflowGraph:
    """A complete workflow graph from the visual builder.

    Contains nodes and edges defining the workflow structure,
    plus metadata about the workflow itself.
    """

    id: str
    name: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    version: str = "1.0"

    def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_incoming_edges(self, node_id: str) -> list[GraphEdge]:
        """Get all edges pointing to a node."""
        return [e for e in self.edges if e.target == node_id]

    def get_outgoing_edges(self, node_id: str) -> list[GraphEdge]:
        """Get all edges starting from a node."""
        return [e for e in self.edges if e.source == node_id]

    def get_start_nodes(self) -> list[GraphNode]:
        """Get nodes with no incoming edges (entry points)."""
        nodes_with_incoming = {e.target for e in self.edges}
        return [n for n in self.nodes if n.id not in nodes_with_incoming]

    def get_end_nodes(self) -> list[GraphNode]:
        """Get nodes with no outgoing edges (exit points)."""
        nodes_with_outgoing = {e.source for e in self.edges}
        return [n for n in self.nodes if n.id not in nodes_with_outgoing]

    @classmethod
    def from_dict(cls, data: dict) -> WorkflowGraph:
        """Create a WorkflowGraph from a dictionary (ReactFlow format)."""
        nodes = [GraphNode.from_dict(n) for n in data.get("nodes", [])]
        edges = [GraphEdge.from_dict(e) for e in data.get("edges", [])]

        return cls(
            id=data.get("id", ""),
            name=data.get("name", "Untitled Workflow"),
            nodes=nodes,
            edges=edges,
            variables=data.get("variables", {}),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary (ReactFlow format)."""
        return {
            "id": self.id,
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "variables": self.variables,
            "description": self.description,
            "version": self.version,
        }
