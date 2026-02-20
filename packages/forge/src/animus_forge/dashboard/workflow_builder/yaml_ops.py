"""YAML conversion operations for the visual workflow builder."""

from __future__ import annotations

import streamlit as st

from .constants import NODE_TYPE_CONFIG


def _build_yaml_from_state() -> dict:
    """Build YAML workflow dict from current session state."""
    meta = st.session_state.builder_metadata

    # Build steps from nodes
    steps = []
    for node in st.session_state.builder_nodes:
        step = {
            "id": node["id"],
            "type": node["type"],
        }

        # Add params if not empty
        params = node["data"].get("params", {})
        if params:
            step["params"] = params

        # Add outputs if defined
        outputs = node["data"].get("outputs", [])
        if outputs:
            step["outputs"] = outputs

        # Add depends_on if defined
        deps = node["data"].get("depends_on", [])
        if deps:
            step["depends_on"] = deps if len(deps) > 1 else deps[0]

        # Add condition if defined
        condition = node["data"].get("condition")
        if condition and condition.get("field"):
            step["condition"] = condition

        # Add failure handling
        on_failure = node["data"].get("on_failure", "abort")
        if on_failure != "abort":
            step["on_failure"] = on_failure

        max_retries = node["data"].get("max_retries", 3)
        if max_retries != 3:
            step["max_retries"] = max_retries

        timeout = node["data"].get("timeout_seconds", 300)
        if timeout != 300:
            step["timeout_seconds"] = timeout

        steps.append(step)

    # Build workflow dict
    workflow = {
        "name": meta["name"],
        "version": meta["version"],
        "description": meta["description"],
        "token_budget": meta["token_budget"],
        "timeout_seconds": meta["timeout_seconds"],
    }

    # Add inputs if defined
    if st.session_state.builder_inputs:
        workflow["inputs"] = st.session_state.builder_inputs

    # Add outputs if defined
    if st.session_state.builder_outputs:
        workflow["outputs"] = st.session_state.builder_outputs

    # Add steps
    workflow["steps"] = steps

    return workflow


def _load_yaml_to_state(workflow_data: dict) -> None:
    """Load a YAML workflow dict into session state."""
    # Load metadata
    st.session_state.builder_metadata = {
        "name": workflow_data.get("name", "Imported Workflow"),
        "version": workflow_data.get("version", "1.0"),
        "description": workflow_data.get("description", ""),
        "token_budget": workflow_data.get("token_budget", 100000),
        "timeout_seconds": workflow_data.get("timeout_seconds", 3600),
    }

    # Load inputs
    st.session_state.builder_inputs = workflow_data.get("inputs", {})

    # Load outputs
    st.session_state.builder_outputs = workflow_data.get("outputs", [])

    # Load steps as nodes
    nodes = []
    edges = []

    steps = workflow_data.get("steps", [])
    for i, step in enumerate(steps):
        # Calculate position
        col = i % 3
        row = i // 3

        node = {
            "id": step["id"],
            "type": step["type"],
            "position": {"x": 100 + col * 250, "y": 100 + row * 180},
            "data": {
                "label": f"{NODE_TYPE_CONFIG.get(step['type'], {}).get('icon', '\U0001f4e6')} {step['id']}",
                "params": step.get("params", {}),
                "on_failure": step.get("on_failure", "abort"),
                "max_retries": step.get("max_retries", 3),
                "timeout_seconds": step.get("timeout_seconds", 300),
                "outputs": step.get("outputs", []),
                "depends_on": [],
                "condition": step.get("condition"),
            },
        }

        # Handle depends_on
        deps = step.get("depends_on", [])
        if isinstance(deps, str):
            deps = [deps]
        node["data"]["depends_on"] = deps

        # Create edges from dependencies
        for dep in deps:
            edges.append(
                {
                    "id": f"edge-{dep}-{step['id']}",
                    "source": dep,
                    "target": step["id"],
                    "label": None,
                }
            )

        nodes.append(node)

    st.session_state.builder_nodes = nodes
    st.session_state.builder_edges = edges
    st.session_state.selected_node = None
    st.session_state.builder_dirty = False
