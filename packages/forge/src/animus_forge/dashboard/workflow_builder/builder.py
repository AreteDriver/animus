"""Main entry point for the visual workflow builder."""

from __future__ import annotations

import streamlit as st

from animus_forge.workflow.loader import validate_workflow

from .persistence import _save_workflow_yaml
from .renderers import (
    _render_canvas,
    _render_execute_preview,
    _render_import_section,
    _render_node_config,
    _render_node_palette,
    _render_saved_workflows,
    _render_templates_section,
    _render_visual_graph,
    _render_yaml_preview,
)
from .state import _init_session_state, _new_workflow
from .yaml_ops import _build_yaml_from_state


def render_workflow_builder() -> None:
    """Main entry point for the visual workflow builder."""
    st.title("\U0001f3a8 Visual Workflow Builder")

    _init_session_state()

    # Show current file status
    current_file = st.session_state.builder_current_file
    is_dirty = st.session_state.builder_dirty
    workflow_name = st.session_state.builder_metadata.get("name", "New Workflow")

    status_text = workflow_name
    if current_file:
        status_text = f"{workflow_name} ({current_file.name})"
    if is_dirty:
        status_text += " *"

    st.caption(f"Current: {status_text}")

    # Top action bar
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
        if st.button("New", use_container_width=True, help="Create new workflow"):
            _new_workflow()
            st.rerun()

    with col2:
        # Quick save button
        workflow = _build_yaml_from_state()
        errors = validate_workflow(workflow)
        save_disabled = bool(errors) or not is_dirty

        if st.button(
            "Save",
            use_container_width=True,
            disabled=save_disabled,
            help="Save workflow (Ctrl+S)",
        ):
            filepath = _save_workflow_yaml()
            if filepath:
                st.toast(f"Saved to {filepath.name}")
                st.rerun()

    with col3:
        node_count = len(st.session_state.builder_nodes)
        st.metric("Nodes", node_count)

    with col4:
        edge_count = len(st.session_state.builder_edges)
        st.metric("Connections", edge_count)

    with col5:
        if errors:
            st.error(f"{len(errors)} error(s)")
        else:
            st.success("Valid")

    st.divider()

    # Main layout: sidebar + main content
    sidebar, main = st.columns([1, 3])

    with sidebar:
        tab1, tab2, tab3, tab4 = st.tabs(["Nodes", "Settings", "Saved", "Import"])

        with tab1:
            _render_templates_section()

        with tab2:
            _render_node_palette()

        with tab3:
            _render_saved_workflows()

        with tab4:
            _render_import_section()

    with main:
        tab1, tab2, tab3, tab4 = st.tabs(["Canvas", "Visual", "YAML", "Execute"])

        with tab1:
            _render_canvas()
            st.divider()
            _render_node_config()

        with tab2:
            _render_visual_graph()

        with tab3:
            _render_yaml_preview()

        with tab4:
            _render_execute_preview()
