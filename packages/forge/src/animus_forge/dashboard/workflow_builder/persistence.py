"""File persistence operations for the visual workflow builder."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import streamlit as st
import yaml

# Reference the package module for names that tests patch via
# animus_forge.dashboard.workflow_builder.X (runtime attribute lookup).
import animus_forge.dashboard.workflow_builder as _pkg
from animus_forge.workflow.loader import validate_workflow

from .yaml_ops import _build_yaml_from_state, _load_yaml_to_state

logger = logging.getLogger(__name__)


def _get_workflows_dir() -> Path:
    """Get the workflows directory from settings."""
    try:
        return _pkg.get_settings().workflows_dir
    except Exception:
        # Fallback to local workflows directory
        return Path("workflows")


def _get_builder_state_path(workflow_name: str) -> Path:
    """Get path for builder state JSON (preserves node positions/metadata)."""
    workflows_dir = _pkg._get_workflows_dir()
    builder_dir = workflows_dir / ".builder_state"
    builder_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w\-]", "_", workflow_name.lower())
    safe_name = safe_name.strip("_")[:50] or "state"
    result = builder_dir / f"{safe_name}.json"
    # Guard against path traversal
    if not result.resolve().is_relative_to(builder_dir.resolve()):
        raise ValueError("Invalid workflow name")
    return result


def _save_builder_state(workflow_name: str) -> None:
    """Save the full builder state (nodes, edges, positions) to JSON."""
    state_path = _get_builder_state_path(workflow_name)
    state = {
        "nodes": st.session_state.builder_nodes,
        "edges": st.session_state.builder_edges,
        "metadata": st.session_state.builder_metadata,
        "inputs": st.session_state.builder_inputs,
        "outputs": st.session_state.builder_outputs,
    }
    try:
        with open(state_path, "w") as f:  # noqa: PTH123
            json.dump(state, f, indent=2)
        logger.debug("Saved builder state to %s", state_path.name)
    except Exception as e:
        logger.error("Failed to save builder state: %s", e)


def _load_builder_state(workflow_name: str) -> bool:
    """Load builder state from JSON if it exists. Returns True if loaded."""
    state_path = _get_builder_state_path(workflow_name)
    if not state_path.exists():
        return False

    try:
        with open(state_path) as f:  # noqa: PTH123
            state = json.load(f)
        st.session_state.builder_nodes = state.get("nodes", [])
        st.session_state.builder_edges = state.get("edges", [])
        st.session_state.builder_metadata = state.get("metadata", {})
        st.session_state.builder_inputs = state.get("inputs", {})
        st.session_state.builder_outputs = state.get("outputs", [])
        st.session_state.selected_node = None
        logger.debug("Loaded builder state from %s", state_path.name)
        return True
    except Exception as e:
        logger.error(f"Failed to load builder state: {e}")
        return False


def _list_saved_workflows() -> list[dict]:
    """List all saved workflows with metadata."""
    workflows_dir = _pkg._get_workflows_dir()
    workflows_dir.mkdir(parents=True, exist_ok=True)

    workflows = []
    for yaml_path in sorted(workflows_dir.glob("*.yaml")):
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            workflows.append(
                {
                    "path": yaml_path,
                    "name": data.get("name", yaml_path.stem),
                    "version": data.get("version", "?"),
                    "description": data.get("description", ""),
                    "steps": len(data.get("steps", [])),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to load workflow {yaml_path}: {e}")
            workflows.append(
                {
                    "path": yaml_path,
                    "name": yaml_path.stem,
                    "version": "?",
                    "description": f"Error: {e}",
                    "steps": 0,
                }
            )
    return workflows


def _save_workflow_yaml(filepath: Path | None = None) -> Path | None:
    """Save current workflow to YAML file. Returns path on success."""
    workflow = _build_yaml_from_state()
    errors = validate_workflow(workflow)
    if errors:
        return None

    if filepath is None:
        workflows_dir = _pkg._get_workflows_dir()
        workflows_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^\w\-]", "_", workflow["name"].lower())
        safe_name = safe_name.strip("_")[:50] or "workflow"
        filepath = workflows_dir / f"{safe_name}.yaml"

        # Ensure filepath stays within workflows_dir
        if not filepath.resolve().is_relative_to(workflows_dir.resolve()):
            logger.error("Invalid workflow name - path traversal attempt")
            return None

    try:
        with open(filepath, "w") as f:  # noqa: PTH123
            yaml.dump(workflow, f, default_flow_style=False, sort_keys=False)
        # Also save builder state for positions
        _save_builder_state(workflow["name"])
        st.session_state.builder_current_file = filepath
        st.session_state.builder_dirty = False
        logger.info("Saved workflow to %s", filepath.name)
        return filepath
    except Exception as e:
        logger.error("Failed to save workflow: %s", e)
        return None


def _delete_workflow(filepath: Path) -> bool:
    """Delete a workflow YAML and its builder state."""
    try:
        # Load to get name for state file
        with open(filepath) as f:
            data = yaml.safe_load(f)
        workflow_name = data.get("name", filepath.stem)

        # Delete YAML
        filepath.unlink()

        # Delete builder state if exists
        state_path = _get_builder_state_path(workflow_name)
        if state_path.exists():
            state_path.unlink()

        logger.info(f"Deleted workflow {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete workflow: {e}")
        return False


def _load_workflow_from_file(filepath: Path) -> bool:
    """Load a workflow from a YAML file, trying builder state first."""
    try:
        with open(filepath) as f:
            data = yaml.safe_load(f)
        workflow_name = data.get("name", filepath.stem)

        # Try to load builder state (preserves positions)
        if _load_builder_state(workflow_name):
            st.session_state.builder_current_file = filepath
            return True

        # Fall back to loading from YAML (recalculates positions)
        _load_yaml_to_state(data)
        st.session_state.builder_current_file = filepath
        return True
    except Exception as e:
        logger.error(f"Failed to load workflow from {filepath}: {e}")
        return False
