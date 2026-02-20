"""Canvas rendering — node cards and workflow canvas."""

from __future__ import annotations

import streamlit as st

from ..constants import NODE_TYPE_CONFIG
from ..state import _add_edge, _delete_edge, _delete_node
from ._helpers import _get_node_execution_status


def _render_node_card(node: dict) -> None:
    """Render a single node card with enhanced visuals."""
    node_id = node["id"]
    node_type = node["type"]
    config = NODE_TYPE_CONFIG.get(
        node_type, {"icon": "\U0001f4e6", "color": "#666", "label": node_type}
    )
    is_selected = st.session_state.selected_node == node_id

    # Get execution status
    status_color, status_icon = _get_node_execution_status(node_id)

    # Selection styling
    if is_selected:
        border_style = "3px solid #007bff"
        box_shadow = "0 4px 12px rgba(0, 123, 255, 0.3)"
        transform = "scale(1.02)"
    else:
        border_style = f"2px solid {config['color']}50"
        box_shadow = "0 2px 8px rgba(0, 0, 0, 0.1)"
        transform = "scale(1)"

    # Get node details
    deps = node["data"].get("depends_on", [])
    outputs = node["data"].get("outputs", [])
    params = node["data"].get("params", {})
    role = params.get("role", "")

    # Build dependency badge
    deps_html = ""
    if deps:
        deps_html = f"""
            <div style="
                display: inline-flex;
                align-items: center;
                gap: 4px;
                background: #f3f4f6;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 10px;
                color: #6b7280;
                margin-top: 6px;
            ">
                <span>\u2b05</span> {", ".join(deps)}
            </div>
        """

    # Build outputs badge
    outputs_html = ""
    if outputs:
        outputs_html = f"""
            <div style="
                display: inline-flex;
                align-items: center;
                gap: 4px;
                background: {config["color"]}15;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 10px;
                color: {config["color"]};
                margin-top: 6px;
                margin-left: 4px;
            ">
                <span>\U0001f4e4</span> {", ".join(outputs)}
            </div>
        """

    # Build role badge
    role_html = ""
    if role:
        role_html = f"""
            <span style="
                background: {config["color"]}20;
                color: {config["color"]};
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 10px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            ">{role}</span>
        """

    # Build status indicator
    status_html = ""
    if status_color:
        status_html = f"""
            <div style="
                position: absolute;
                top: -6px;
                right: -6px;
                background: {status_color};
                color: white;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            ">{status_icon}</div>
        """

    # Node card HTML with enhanced styling
    st.markdown(
        f"""
        <div style="
            position: relative;
            border: {border_style};
            border-radius: 16px;
            padding: 16px;
            margin: 10px 0;
            background: linear-gradient(145deg, white, {config["color"]}08);
            box-shadow: {box_shadow};
            transform: {transform};
            transition: all 0.2s ease;
        ">
            {status_html}
            <div style="display: flex; align-items: flex-start; gap: 12px;">
                <div style="
                    font-size: 32px;
                    background: {config["color"]}15;
                    padding: 8px;
                    border-radius: 12px;
                    line-height: 1;
                ">{config["icon"]}</div>
                <div style="flex: 1; min-width: 0;">
                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        <span style="font-weight: 700; font-size: 15px; color: #1f2937;">{node_id}</span>
                        {role_html}
                    </div>
                    <div style="font-size: 12px; color: #6b7280; margin-top: 2px;">{config["label"]}</div>
                    <div style="margin-top: 4px;">
                        {deps_html}{outputs_html}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Selection and delete buttons with better styling
    col1, col2 = st.columns(2)
    with col1:
        if st.button("\u270f\ufe0f Edit", key=f"select_{node_id}", use_container_width=True):
            st.session_state.selected_node = node_id
            st.rerun()
    with col2:
        if st.button("\U0001f5d1\ufe0f Delete", key=f"delete_{node_id}", use_container_width=True):
            _delete_node(node_id)
            st.rerun()


def _render_canvas() -> None:
    """Render the workflow canvas with nodes."""
    st.markdown("### \U0001f3af Workflow Canvas")

    nodes = st.session_state.builder_nodes

    if not nodes:
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 48px 24px;
                background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
                border-radius: 20px;
                border: 2px dashed #7dd3fc;
                margin: 16px 0;
            ">
                <div style="font-size: 56px; margin-bottom: 16px;">\U0001f680</div>
                <div style="font-size: 20px; font-weight: 700; color: #0369a1; margin-bottom: 8px;">
                    Start Building Your Workflow
                </div>
                <div style="font-size: 14px; color: #0284c7; max-width: 400px; margin: 0 auto;">
                    Choose a <strong>Template</strong> to get started quickly, or add individual <strong>Nodes</strong> from the sidebar
                </div>
                <div style="
                    display: flex;
                    justify-content: center;
                    gap: 24px;
                    margin-top: 24px;
                    font-size: 13px;
                    color: #6b7280;
                ">
                    <span>\U0001f4cb Templates</span>
                    <span>\u2022</span>
                    <span>\U0001f916 AI Agents</span>
                    <span>\u2022</span>
                    <span>\U0001f4bb Shell Commands</span>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )
        return

    # Render nodes in columns
    num_cols = min(len(nodes), 3)
    cols = st.columns(num_cols)

    for i, node in enumerate(nodes):
        with cols[i % num_cols]:
            _render_node_card(node)

    # Connection builder with improved styling
    st.markdown("---")
    st.markdown("#### \U0001f517 Connections")

    if len(nodes) >= 2:
        col1, col2, col3 = st.columns([2, 2, 1])

        node_ids = [n["id"] for n in nodes]

        with col1:
            source = st.selectbox("From", node_ids, key="conn_source")
        with col2:
            # Filter out source from targets
            target_options = [n for n in node_ids if n != source]
            target = st.selectbox("To", target_options, key="conn_target")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("\U0001f517 Connect", use_container_width=True, type="primary"):
                if source and target:
                    _add_edge(source, target)
                    # Also update depends_on
                    for node in st.session_state.builder_nodes:
                        if node["id"] == target:
                            deps = node["data"].get("depends_on", [])
                            if source not in deps:
                                deps.append(source)
                                node["data"]["depends_on"] = deps
                    st.rerun()

    # Show existing connections with improved styling
    edges = st.session_state.builder_edges
    if edges:
        st.markdown(
            """
            <div style="
                background: #f9fafb;
                border-radius: 12px;
                padding: 12px;
                margin-top: 12px;
            ">
                <div style="font-size: 12px; font-weight: 600; color: #6b7280; margin-bottom: 8px;">
                    Active Connections
                </div>
        """,
            unsafe_allow_html=True,
        )

        for edge in edges:
            col1, col2 = st.columns([4, 1])
            with col1:
                source_node = next((n for n in nodes if n["id"] == edge["source"]), None)
                target_node = next((n for n in nodes if n["id"] == edge["target"]), None)
                source_config = NODE_TYPE_CONFIG.get(source_node["type"], {}) if source_node else {}
                target_config = NODE_TYPE_CONFIG.get(target_node["type"], {}) if target_node else {}

                st.markdown(
                    f"""
                    <div style="
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        padding: 6px 0;
                    ">
                        <span style="font-size: 16px;">{source_config.get("icon", "\U0001f4e6")}</span>
                        <span style="font-weight: 500;">{edge["source"]}</span>
                        <span style="color: #9ca3af;">\u2192</span>
                        <span style="font-size: 16px;">{target_config.get("icon", "\U0001f4e6")}</span>
                        <span style="font-weight: 500;">{edge["target"]}</span>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("\u2715", key=f"del_edge_{edge['id']}", help="Remove connection"):
                    _delete_edge(edge["id"])
                    # Also update depends_on
                    for node in st.session_state.builder_nodes:
                        if node["id"] == edge["target"]:
                            deps = node["data"].get("depends_on", [])
                            if edge["source"] in deps:
                                deps.remove(edge["source"])
                                node["data"]["depends_on"] = deps
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
