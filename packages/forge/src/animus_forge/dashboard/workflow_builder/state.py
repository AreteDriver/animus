"""Session state management for the visual workflow builder."""

from __future__ import annotations

import streamlit as st

from .constants import NODE_TYPE_CONFIG


def _init_session_state() -> None:
    """Initialize session state for the workflow builder."""
    if "builder_nodes" not in st.session_state:
        st.session_state.builder_nodes = []
    if "builder_edges" not in st.session_state:
        st.session_state.builder_edges = []
    if "builder_metadata" not in st.session_state:
        st.session_state.builder_metadata = {
            "name": "New Workflow",
            "version": "1.0",
            "description": "",
            "token_budget": 100000,
            "timeout_seconds": 3600,
        }
    if "builder_inputs" not in st.session_state:
        st.session_state.builder_inputs = {}
    if "builder_outputs" not in st.session_state:
        st.session_state.builder_outputs = []
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = None
    if "connection_mode" not in st.session_state:
        st.session_state.connection_mode = False
    if "connection_source" not in st.session_state:
        st.session_state.connection_source = None
    # Persistence state
    if "builder_current_file" not in st.session_state:
        st.session_state.builder_current_file = None  # Path to current workflow file
    if "builder_dirty" not in st.session_state:
        st.session_state.builder_dirty = False  # True if unsaved changes exist


def _new_workflow() -> None:
    """Reset to a new empty workflow."""
    st.session_state.builder_nodes = []
    st.session_state.builder_edges = []
    st.session_state.builder_metadata = {
        "name": "New Workflow",
        "version": "1.0",
        "description": "",
        "token_budget": 100000,
        "timeout_seconds": 3600,
    }
    st.session_state.builder_inputs = {}
    st.session_state.builder_outputs = []
    st.session_state.selected_node = None
    st.session_state.builder_current_file = None
    st.session_state.builder_dirty = False


def _mark_dirty() -> None:
    """Mark the current workflow as having unsaved changes."""
    st.session_state.builder_dirty = True


def _generate_node_id(step_type: str) -> str:
    """Generate a unique node ID."""
    base_id = step_type.replace("_", "-")
    existing_ids = {n["id"] for n in st.session_state.builder_nodes}
    counter = 1
    while f"{base_id}-{counter}" in existing_ids:
        counter += 1
    return f"{base_id}-{counter}"


def _add_node(step_type: str) -> None:
    """Add a new node to the canvas."""
    node_id = _generate_node_id(step_type)
    config = NODE_TYPE_CONFIG[step_type]

    # Calculate position (grid layout)
    num_nodes = len(st.session_state.builder_nodes)
    col = num_nodes % 3
    row = num_nodes // 3

    node = {
        "id": node_id,
        "type": step_type,
        "position": {"x": 100 + col * 250, "y": 100 + row * 180},
        "data": {
            "label": f"{config['icon']} {node_id}",
            "params": {},
            "on_failure": "abort",
            "max_retries": 3,
            "timeout_seconds": 300,
            "outputs": [],
            "depends_on": [],
            "condition": None,
        },
    }

    st.session_state.builder_nodes.append(node)
    st.session_state.selected_node = node_id
    _mark_dirty()


def _delete_node(node_id: str) -> None:
    """Delete a node and its connections."""
    st.session_state.builder_nodes = [
        n for n in st.session_state.builder_nodes if n["id"] != node_id
    ]
    st.session_state.builder_edges = [
        e
        for e in st.session_state.builder_edges
        if e["source"] != node_id and e["target"] != node_id
    ]
    if st.session_state.selected_node == node_id:
        st.session_state.selected_node = None
    _mark_dirty()


def _add_edge(source: str, target: str, label: str | None = None) -> None:
    """Add an edge between two nodes."""
    # Check if edge already exists
    for edge in st.session_state.builder_edges:
        if edge["source"] == source and edge["target"] == target:
            return  # Edge already exists

    edge_id = f"edge-{source}-{target}"
    edge = {
        "id": edge_id,
        "source": source,
        "target": target,
        "label": label,
    }
    st.session_state.builder_edges.append(edge)
    _mark_dirty()


def _delete_edge(edge_id: str) -> None:
    """Delete an edge."""
    st.session_state.builder_edges = [
        e for e in st.session_state.builder_edges if e["id"] != edge_id
    ]
    _mark_dirty()
