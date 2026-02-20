"""Execution preview and workflow settings rendering."""

from __future__ import annotations

import streamlit as st

from animus_forge.workflow.loader import validate_workflow

from ..yaml_ops import _build_yaml_from_state


def _render_workflow_settings() -> None:
    """Render workflow metadata settings."""
    st.markdown("### Workflow Settings")

    meta = st.session_state.builder_metadata

    meta["name"] = st.text_input("Workflow Name", value=meta["name"], key="wf_name")
    meta["version"] = st.text_input("Version", value=meta["version"], key="wf_version")
    meta["description"] = st.text_area(
        "Description", value=meta["description"], key="wf_desc", height=100
    )

    col1, col2 = st.columns(2)
    with col1:
        meta["token_budget"] = st.number_input(
            "Token Budget",
            min_value=1000,
            max_value=1000000,
            value=meta["token_budget"],
            key="wf_budget",
        )
    with col2:
        meta["timeout_seconds"] = st.number_input(
            "Timeout (seconds)",
            min_value=60,
            max_value=86400,
            value=meta["timeout_seconds"],
            key="wf_timeout",
        )

    # Inputs configuration
    st.markdown("#### Workflow Inputs")

    inputs = st.session_state.builder_inputs

    # Add new input
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        new_input_name = st.text_input("Input Name", key="new_input_name")
    with col2:
        new_input_type = st.selectbox("Type", ["string", "list", "object"], key="new_input_type")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Add Input", key="add_input_btn"):
            if new_input_name and new_input_name not in inputs:
                inputs[new_input_name] = {
                    "type": new_input_type,
                    "required": True,
                    "description": "",
                }
                st.rerun()

    # Show existing inputs
    for name, config in list(inputs.items()):
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.markdown(f"**{name}** ({config['type']})")
        with col2:
            config["required"] = st.checkbox(
                "Required",
                value=config.get("required", True),
                key=f"input_req_{name}",
            )
        with col3:
            if st.button("Remove", key=f"del_input_{name}"):
                del inputs[name]
                st.rerun()

    # Outputs configuration
    st.markdown("#### Workflow Outputs")
    outputs_str = st.text_input(
        "Output Variables (comma-separated)",
        value=", ".join(st.session_state.builder_outputs),
        key="wf_outputs",
    )
    st.session_state.builder_outputs = [o.strip() for o in outputs_str.split(",") if o.strip()]


def _render_execute_preview() -> None:
    """Render execution preview with validation and step summary."""
    workflow = _build_yaml_from_state()
    errors = validate_workflow(workflow)

    st.markdown("### Execution Preview")

    if errors:
        st.error(f"**{len(errors)} validation error(s) must be fixed before execution:**")
        for error in errors:
            st.markdown(f"- {error}")
        return

    st.success("Workflow is valid and ready to execute.")

    # Step execution order
    steps = workflow.get("steps", [])
    if steps:
        st.markdown("#### Execution Order")
        for i, step in enumerate(steps, 1):
            deps = step.get("depends_on", [])
            if isinstance(deps, str):
                deps = [deps]
            dep_str = f" (after: {', '.join(deps)})" if deps else ""
            condition = step.get("condition")
            cond_str = (
                f" | condition: `{condition['field']} {condition.get('operator', '==')} {condition.get('value', '')}`"
                if condition
                else ""
            )
            on_failure = step.get("on_failure", "abort")
            fail_str = f" | on_failure: {on_failure}" if on_failure != "abort" else ""
            st.markdown(f"**{i}.** `{step['id']}` ({step['type']}){dep_str}{cond_str}{fail_str}")

    # Variables
    variables = workflow.get("variables", {})
    if variables:
        st.markdown("#### Variables")
        st.json(variables)

    st.info("Save the workflow, then execute it from the **Execute** page.")
