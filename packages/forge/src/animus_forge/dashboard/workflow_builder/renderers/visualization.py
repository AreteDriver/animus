"""Visual graph rendering for workflow topology."""

from __future__ import annotations

import streamlit as st

from ..constants import NODE_TYPE_CONFIG
from ._helpers import _get_node_execution_status


def _render_visual_graph() -> None:
    """Render a visual representation of the workflow graph with enhanced styling."""
    nodes = st.session_state.builder_nodes
    edges = st.session_state.builder_edges

    if not nodes:
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 60px 20px;
                color: #9ca3af;
                background: linear-gradient(135deg, #f9fafb, #f3f4f6);
                border-radius: 16px;
                border: 2px dashed #d1d5db;
            ">
                <div style="font-size: 48px; margin-bottom: 16px;">\U0001f517</div>
                <div style="font-size: 18px; font-weight: 600;">No workflow steps yet</div>
                <div style="font-size: 14px; margin-top: 8px;">Add nodes from the Templates or Nodes tab</div>
            </div>
        """,
            unsafe_allow_html=True,
        )
        return

    st.markdown("### \U0001f500 Visual Flow")

    # Build adjacency for topological display
    node_map = {n["id"]: n for n in nodes}
    incoming = {n["id"]: set() for n in nodes}
    outgoing = {n["id"]: set() for n in nodes}

    for edge in edges:
        if edge["source"] in node_map and edge["target"] in node_map:
            incoming[edge["target"]].add(edge["source"])
            outgoing[edge["source"]].add(edge["target"])

    # Find roots (no incoming edges)
    roots = [n["id"] for n in nodes if not incoming[n["id"]]]
    if not roots:
        roots = [nodes[0]["id"]]  # Fallback to first node

    # Simple level assignment
    levels = {}
    queue = [(r, 0) for r in roots]
    visited = set()

    while queue:
        node_id, level = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        levels[node_id] = max(levels.get(node_id, 0), level)
        for target in outgoing.get(node_id, []):
            queue.append((target, level + 1))

    # Assign unvisited nodes
    for n in nodes:
        if n["id"] not in levels:
            levels[n["id"]] = 0

    # Group by level
    level_groups = {}
    for node_id, level in levels.items():
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(node_id)

    # Calculate if this is a parallel workflow
    max_parallel = max(len(group) for group in level_groups.values())
    is_parallel = max_parallel > 1

    # Render level by level with enhanced visuals
    total_levels = len(level_groups)
    for level in sorted(level_groups.keys()):
        node_ids = level_groups[level]
        is_parallel_level = len(node_ids) > 1
        cols = st.columns(max(len(node_ids), 1))

        # Show parallel indicator
        if is_parallel_level:
            st.markdown(
                """
                <div style="
                    text-align: center;
                    font-size: 11px;
                    color: #3b82f6;
                    background: #eff6ff;
                    padding: 4px 12px;
                    border-radius: 12px;
                    display: inline-block;
                    margin: 0 auto 8px auto;
                    width: fit-content;
                ">
                    \U0001f500 Parallel Execution
                </div>
            """,
                unsafe_allow_html=True,
            )

        for i, node_id in enumerate(node_ids):
            node = node_map[node_id]
            config = NODE_TYPE_CONFIG.get(node["type"], {"icon": "\U0001f4e6", "color": "#666"})

            # Get execution status for visual indicator
            status_color, status_icon = _get_node_execution_status(node_id)

            # Get role if available
            params = node["data"].get("params", {})
            role = params.get("role", "")
            role_badge = (
                f'<div style="font-size: 9px; color: {config["color"]}; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">{role}</div>'
                if role
                else ""
            )

            # Status ring styling
            status_ring = ""
            if status_color:
                status_ring = (
                    f"box-shadow: 0 0 0 3px {status_color}40, 0 4px 12px rgba(0,0,0,0.15);"
                )

            with cols[i]:
                st.markdown(
                    f"""
                    <div style="
                        text-align: center;
                        padding: 16px 12px;
                        border: 2px solid {config["color"]};
                        border-radius: 16px;
                        background: linear-gradient(145deg, white, {config["color"]}10);
                        margin: 4px;
                        {status_ring}
                        transition: all 0.2s ease;
                    ">
                        <div style="
                            font-size: 32px;
                            background: {config["color"]}15;
                            width: 56px;
                            height: 56px;
                            border-radius: 14px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin: 0 auto 8px auto;
                            position: relative;
                        ">
                            {config["icon"]}
                            {f'<span style="position: absolute; top: -4px; right: -4px; font-size: 14px;">{status_icon}</span>' if status_icon else ""}
                        </div>
                        <div style="font-size: 13px; font-weight: 700; color: #1f2937;">{node_id}</div>
                        <div style="font-size: 11px; color: #6b7280;">{config["label"]}</div>
                        {role_badge}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Draw connection arrows if not last level
        if level < max(level_groups.keys()):
            next_level_count = len(level_groups.get(level + 1, []))
            current_count = len(node_ids)

            # Choose arrow style based on branching
            if current_count == 1 and next_level_count > 1:
                # Fan out
                arrow_html = """
                    <div style="text-align: center; padding: 8px 0;">
                        <div style="font-size: 16px; color: #3b82f6;">\u2199\ufe0f \u2b07 \u2198\ufe0f</div>
                    </div>
                """
            elif current_count > 1 and next_level_count == 1:
                # Fan in
                arrow_html = """
                    <div style="text-align: center; padding: 8px 0;">
                        <div style="font-size: 16px; color: #8b5cf6;">\u2198\ufe0f \u2b07 \u2199\ufe0f</div>
                    </div>
                """
            else:
                # Regular flow
                arrow_html = """
                    <div style="text-align: center; padding: 8px 0;">
                        <div style="
                            width: 2px;
                            height: 20px;
                            background: linear-gradient(to bottom, #d1d5db, #9ca3af);
                            margin: 0 auto;
                            border-radius: 1px;
                        "></div>
                        <div style="
                            width: 0;
                            height: 0;
                            border-left: 6px solid transparent;
                            border-right: 6px solid transparent;
                            border-top: 8px solid #9ca3af;
                            margin: 0 auto;
                        "></div>
                    </div>
                """
            st.markdown(arrow_html, unsafe_allow_html=True)

    # Show workflow summary
    st.markdown(
        f"""
        <div style="
            margin-top: 20px;
            padding: 12px 16px;
            background: #f9fafb;
            border-radius: 12px;
            display: flex;
            justify-content: center;
            gap: 24px;
            font-size: 13px;
            color: #6b7280;
        ">
            <span>\U0001f4ca <strong>{len(nodes)}</strong> steps</span>
            <span>\U0001f517 <strong>{len(edges)}</strong> connections</span>
            <span>\U0001f4d0 <strong>{total_levels}</strong> levels</span>
            {"<span>\U0001f500 <strong>parallel</strong></span>" if is_parallel else "<span>\U0001f4cf <strong>sequential</strong></span>"}
        </div>
    """,
        unsafe_allow_html=True,
    )
