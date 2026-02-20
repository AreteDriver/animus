"""Node palette and configuration panel rendering."""

from __future__ import annotations

import streamlit as st

from animus_forge.workflow.loader import VALID_ON_FAILURE, VALID_OPERATORS

from ..constants import AGENT_ROLES, NODE_TYPE_CONFIG
from ..state import _add_node


def _render_node_palette() -> None:
    """Render the node palette sidebar."""
    st.markdown("### Node Types")
    st.markdown("Click to add a node to the canvas")

    for step_type, config in NODE_TYPE_CONFIG.items():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(
                f"<span style='font-size: 24px;'>{config['icon']}</span>",
                unsafe_allow_html=True,
            )
        with col2:
            if st.button(config["label"], key=f"add_{step_type}", use_container_width=True):
                _add_node(step_type)
                st.rerun()

        st.caption(config["description"])
        st.divider()


def _render_node_config() -> None:
    """Render configuration panel for selected node."""
    selected_id = st.session_state.selected_node

    if not selected_id:
        st.info("Select a node to configure it")
        return

    # Find the selected node
    node = None
    node_idx = None
    for i, n in enumerate(st.session_state.builder_nodes):
        if n["id"] == selected_id:
            node = n
            node_idx = i
            break

    if not node:
        st.warning("Selected node not found")
        return

    node_type = node["type"]
    config = NODE_TYPE_CONFIG.get(node_type, {})

    st.markdown(f"### {config.get('icon', '\U0001f4e6')} Configure: {selected_id}")

    # Basic settings
    with st.expander("Basic Settings", expanded=True):
        new_id = st.text_input("Node ID", value=node["id"], key="node_id_input")
        if new_id != node["id"]:
            # Update ID in edges too
            for edge in st.session_state.builder_edges:
                if edge["source"] == node["id"]:
                    edge["source"] = new_id
                    edge["id"] = f"edge-{new_id}-{edge['target']}"
                if edge["target"] == node["id"]:
                    edge["target"] = new_id
                    edge["id"] = f"edge-{edge['source']}-{new_id}"
            # Update depends_on in other nodes
            for other_node in st.session_state.builder_nodes:
                deps = other_node["data"].get("depends_on", [])
                if node["id"] in deps:
                    deps[deps.index(node["id"])] = new_id
            node["id"] = new_id
            st.session_state.selected_node = new_id

        # Failure handling
        on_failure = st.selectbox(
            "On Failure",
            list(VALID_ON_FAILURE) + ["fallback", "continue_with_default"],
            index=0,
            key="on_failure_select",
        )
        node["data"]["on_failure"] = on_failure

        max_retries = st.number_input(
            "Max Retries",
            min_value=0,
            max_value=10,
            value=node["data"].get("max_retries", 3),
            key="max_retries_input",
        )
        node["data"]["max_retries"] = max_retries

        timeout = st.number_input(
            "Timeout (seconds)",
            min_value=1,
            max_value=3600,
            value=node["data"].get("timeout_seconds", 300),
            key="timeout_input",
        )
        node["data"]["timeout_seconds"] = timeout

    # Type-specific parameters
    with st.expander("Step Parameters", expanded=True):
        params = node["data"].get("params", {})

        if node_type in ("claude_code", "openai"):
            params["role"] = st.selectbox(
                "Agent Role",
                AGENT_ROLES,
                index=AGENT_ROLES.index(params.get("role", "builder"))
                if params.get("role") in AGENT_ROLES
                else 0,
                key="role_select",
            )

            params["prompt"] = st.text_area(
                "Prompt",
                value=params.get("prompt", ""),
                height=150,
                help="Use ${variable} for variable substitution",
                key="prompt_input",
            )

            params["estimated_tokens"] = st.number_input(
                "Estimated Tokens",
                min_value=100,
                max_value=100000,
                value=params.get("estimated_tokens", 5000),
                key="tokens_input",
            )

            if node_type == "openai":
                params["model"] = st.selectbox(
                    "Model",
                    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                    key="model_select",
                )
                params["temperature"] = st.slider(
                    "Temperature",
                    0.0,
                    2.0,
                    value=params.get("temperature", 0.7),
                    key="temp_slider",
                )

        elif node_type == "shell":
            params["command"] = st.text_input(
                "Command",
                value=params.get("command", ""),
                help="Shell command to execute. Use ${variable} for substitution.",
                key="command_input",
            )
            params["allow_failure"] = st.checkbox(
                "Allow Failure",
                value=params.get("allow_failure", False),
                key="allow_failure_check",
            )

        elif node_type == "checkpoint":
            params["message"] = st.text_input(
                "Checkpoint Message",
                value=params.get("message", ""),
                key="checkpoint_msg_input",
            )

        elif node_type == "fan_out":
            params["items_from"] = st.text_input(
                "Items From (variable)",
                value=params.get("items_from", ""),
                help="Variable containing list of items to fan out",
                key="fan_out_items_input",
            )

        elif node_type == "fan_in":
            params["aggregate_as"] = st.text_input(
                "Aggregate As",
                value=params.get("aggregate_as", "results"),
                help="Variable name for aggregated results",
                key="fan_in_agg_input",
            )

        elif node_type == "loop":
            params["max_iterations"] = st.number_input(
                "Max Iterations",
                min_value=1,
                max_value=100,
                value=params.get("max_iterations", 10),
                key="loop_max_input",
            )

        node["data"]["params"] = params

    # Condition configuration
    with st.expander("Condition (Optional)"):
        condition = node["data"].get("condition") or {}

        use_condition = st.checkbox(
            "Add Condition",
            value=bool(condition),
            key="use_condition_check",
        )

        if use_condition:
            cond_field = st.text_input(
                "Field",
                value=condition.get("field", ""),
                help="Context field to check",
                key="cond_field_input",
            )
            cond_operator = st.selectbox(
                "Operator",
                list(VALID_OPERATORS),
                key="cond_op_select",
            )
            cond_value = st.text_input(
                "Value",
                value=str(condition.get("value", "")),
                help="Value to compare against (not needed for not_empty)",
                key="cond_value_input",
            )

            node["data"]["condition"] = {
                "field": cond_field,
                "operator": cond_operator,
                "value": cond_value,
            }
        else:
            node["data"]["condition"] = None

    # Outputs configuration
    with st.expander("Outputs"):
        outputs_str = st.text_input(
            "Output Variables (comma-separated)",
            value=", ".join(node["data"].get("outputs", [])),
            help="Variables this step produces",
            key="outputs_input",
        )
        node["data"]["outputs"] = [o.strip() for o in outputs_str.split(",") if o.strip()]

    # Update the node in session state
    st.session_state.builder_nodes[node_idx] = node
