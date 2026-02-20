"""MCP server management dashboard page for Streamlit."""

from __future__ import annotations

import json

import streamlit as st

try:
    from animus_forge.mcp.manager import MCPConnectorManager
    from animus_forge.mcp.models import (
        MCPAuthType,
        MCPServerCreateInput,
        MCPServerType,
    )

    MCP_AVAILABLE = True
except ImportError:
    MCPConnectorManager = None  # type: ignore[assignment, misc]
    MCPServerCreateInput = None  # type: ignore[assignment, misc]
    MCPServerType = None  # type: ignore[assignment, misc]
    MCPAuthType = None  # type: ignore[assignment, misc]
    MCP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_manager() -> MCPConnectorManager | None:
    """Get cached MCPConnectorManager instance."""
    if not MCP_AVAILABLE:
        return None
    try:
        from animus_forge.state.database import get_database

        return MCPConnectorManager(backend=get_database())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main render entry-point
# ---------------------------------------------------------------------------


def render_mcp_page() -> None:
    """Render the MCP server management page."""
    st.title("MCP Servers")

    if not MCP_AVAILABLE:
        st.error("MCP module not available. Check your installation.")
        return

    manager = _get_manager()
    if manager is None:
        st.error("Failed to initialise MCP manager. Check database configuration.")
        return

    tab1, tab2, tab3 = st.tabs(
        [
            "Server Management",
            "Tool Discovery",
            "Tool Execution",
        ]
    )

    with tab1:
        _render_server_management(manager)

    with tab2:
        _render_tool_discovery(manager)

    with tab3:
        _render_tool_execution(manager)


# ---------------------------------------------------------------------------
# Tab 1 — Server Management
# ---------------------------------------------------------------------------


def _render_server_management(manager: MCPConnectorManager) -> None:
    """Render server list, add form, delete & test-connection controls."""
    servers = manager.list_servers()

    st.subheader("Registered Servers")

    if servers:
        for server in servers:
            _render_server_row(manager, server)
    else:
        st.info("No MCP servers registered. Add one below.")

    st.divider()
    _render_add_server_form(manager)


def _render_server_row(manager: MCPConnectorManager, server) -> None:
    """Render a single server row with status, test, and delete controls."""
    with st.expander(f"**{server.name}** — {server.url}"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f"**ID:** `{server.id}`")
            st.write(f"**Type:** {server.type}")
            st.write(f"**Status:** {server.status}")

        with col2:
            st.write(f"**Auth:** {server.authType}")
            st.write(f"**Tools:** {len(server.tools)}")
            if server.description:
                st.write(f"**Description:** {server.description}")

        with col3:
            if server.lastConnected:
                st.write(f"**Last connected:** {server.lastConnected:%Y-%m-%d %H:%M}")
            if server.error:
                st.warning(f"Error: {server.error}")

        bcol1, bcol2 = st.columns(2)

        with bcol1:
            if st.button("Test Connection", key=f"test_{server.id}"):
                with st.spinner("Testing connection..."):
                    result = manager.test_connection(server.id)
                if result.success:
                    st.success(
                        f"Connected — discovered {len(result.tools)} tool(s), "
                        f"{len(result.resources)} resource(s)"
                    )
                else:
                    st.error(f"Connection failed: {result.error}")

        with bcol2:
            if st.button("Delete", key=f"del_{server.id}"):
                if manager.delete_server(server.id):
                    st.success(f"Deleted server '{server.name}'")
                    st.rerun()
                else:
                    st.error("Failed to delete server")


def _render_add_server_form(manager: MCPConnectorManager) -> None:
    """Render the add-server form."""
    st.subheader("Add Server")

    with st.form("add_mcp_server", clear_on_submit=True):
        name = st.text_input("Name", placeholder="My MCP Server")
        url = st.text_input("URL", placeholder="http://localhost:8080/mcp")

        col1, col2 = st.columns(2)
        with col1:
            server_type = st.selectbox(
                "Type",
                [t.value for t in MCPServerType],
                index=0,
            )
        with col2:
            auth_type = st.selectbox(
                "Auth Type",
                [a.value for a in MCPAuthType],
                index=0,
            )

        description = st.text_area(
            "Description (optional)", placeholder="What does this server do?"
        )

        submitted = st.form_submit_button("Add Server", type="primary")

        if submitted:
            if not name or not url:
                st.warning("Name and URL are required.")
            else:
                try:
                    data = MCPServerCreateInput(
                        name=name,
                        url=url,
                        type=server_type,
                        authType=auth_type,
                        description=description or None,
                    )
                    server = manager.create_server(data)
                    st.success(f"Server '{server.name}' registered (ID: {server.id})")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add server: {e}")


# ---------------------------------------------------------------------------
# Tab 2 — Tool Discovery
# ---------------------------------------------------------------------------


def _render_tool_discovery(manager: MCPConnectorManager) -> None:
    """Render server selector and discovered tools/resources."""
    servers = manager.list_servers()

    if not servers:
        st.info("Register an MCP server first to discover tools.")
        return

    server_options = {s.id: s.name for s in servers}
    selected_id = st.selectbox(
        "Select Server",
        options=list(server_options.keys()),
        format_func=lambda x: server_options[x],
        key="discover_server",
    )

    if not selected_id:
        return

    if st.button("Discover Tools"):
        with st.spinner("Discovering tools and resources..."):
            result = manager.test_connection(selected_id)

        if result.success:
            st.success(
                f"Discovered {len(result.tools)} tool(s) and {len(result.resources)} resource(s)"
            )
        else:
            st.error(f"Discovery failed: {result.error}")

    # Always show currently known tools/resources for the selected server
    server = manager.get_server(selected_id)
    if server is None:
        return

    _render_discovered_tools(server.tools)
    _render_discovered_resources(server.resources)


def _render_discovered_tools(tools: list) -> None:
    """Render discovered tools as expandable cards."""
    if not tools:
        st.info("No tools discovered yet. Click 'Discover Tools' above.")
        return

    st.subheader(f"Tools ({len(tools)})")
    for tool in tools:
        with st.expander(f"**{tool.name}** — {tool.description}"):
            st.write(f"**Name:** {tool.name}")
            st.write(f"**Description:** {tool.description}")
            if tool.inputSchema:
                st.markdown("**Input Schema:**")
                st.json(tool.inputSchema)


def _render_discovered_resources(resources: list) -> None:
    """Render discovered resources."""
    if not resources:
        return

    st.subheader(f"Resources ({len(resources)})")
    for resource in resources:
        with st.expander(f"**{resource.name}** — {resource.uri}"):
            st.write(f"**URI:** {resource.uri}")
            if resource.description:
                st.write(f"**Description:** {resource.description}")
            if resource.mimeType:
                st.write(f"**MIME Type:** {resource.mimeType}")


# ---------------------------------------------------------------------------
# Tab 3 — Tool Execution
# ---------------------------------------------------------------------------


def _render_tool_execution(manager: MCPConnectorManager) -> None:
    """Render tool execution panel: server -> tool -> dynamic form -> run."""
    servers = manager.list_servers()

    if not servers:
        st.info("Register an MCP server first.")
        return

    # Only show servers that have discovered tools
    servers_with_tools = [s for s in servers if s.tools]
    if not servers_with_tools:
        st.info(
            "No servers have discovered tools yet. "
            "Use the Tool Discovery tab to discover tools first."
        )
        return

    server_options = {s.id: s.name for s in servers_with_tools}
    selected_server_id = st.selectbox(
        "Select Server",
        options=list(server_options.keys()),
        format_func=lambda x: server_options[x],
        key="exec_server",
    )

    if not selected_server_id:
        return

    server = manager.get_server(selected_server_id)
    if server is None or not server.tools:
        return

    # Tool selector
    tool_options = {t.name: t.name for t in server.tools}
    selected_tool_name = st.selectbox(
        "Select Tool",
        options=list(tool_options.keys()),
        key="exec_tool",
    )

    if not selected_tool_name:
        return

    selected_tool = next((t for t in server.tools if t.name == selected_tool_name), None)
    if selected_tool is None:
        return

    st.write(f"**Description:** {selected_tool.description}")

    # Dynamic form from inputSchema
    arguments = _render_dynamic_form(selected_tool)

    if st.button("Execute Tool", type="primary"):
        with st.spinner(f"Executing {selected_tool_name}..."):
            try:
                from animus_forge.mcp.client import HAS_MCP_SDK, call_mcp_tool

                if not HAS_MCP_SDK:
                    st.warning(
                        "MCP SDK not installed. Tool execution requires the `mcp` Python package."
                    )
                    st.json({"tool": selected_tool_name, "arguments": arguments})
                    return

                result = call_mcp_tool(
                    server_type=server.type,
                    server_url=server.url,
                    tool_name=selected_tool_name,
                    arguments=arguments,
                )
                st.success("Execution complete")
                st.json(result)
            except Exception as e:
                st.error(f"Execution failed: {e}")


def _render_dynamic_form(tool) -> dict:
    """Build a dynamic Streamlit form from a tool's inputSchema.

    Returns:
        Dictionary of argument name -> user-provided value.
    """
    schema = tool.inputSchema or {}
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    if not properties:
        st.info("This tool takes no input parameters.")
        return {}

    arguments: dict = {}

    for prop_name, prop_spec in properties.items():
        label = prop_name
        if prop_name in required_fields:
            label += " *"

        prop_type = prop_spec.get("type", "string")
        enum_values = prop_spec.get("enum")

        if enum_values:
            value = st.selectbox(
                label,
                options=enum_values,
                key=f"arg_{tool.name}_{prop_name}",
            )
        elif prop_type == "boolean":
            value = st.checkbox(
                label,
                key=f"arg_{tool.name}_{prop_name}",
            )
        elif prop_type == "integer":
            value = st.number_input(
                label,
                step=1,
                key=f"arg_{tool.name}_{prop_name}",
            )
        elif prop_type == "number":
            value = st.number_input(
                label,
                step=0.1,
                key=f"arg_{tool.name}_{prop_name}",
            )
        elif prop_type in ("object", "array"):
            raw = st.text_area(
                f"{label} (JSON)",
                value="{}",
                key=f"arg_{tool.name}_{prop_name}",
            )
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                st.warning(f"Invalid JSON for {prop_name}")
                value = raw
        else:
            # Default: string input
            value = st.text_input(
                label,
                key=f"arg_{tool.name}_{prop_name}",
            )

        if value is not None and value != "":
            arguments[prop_name] = value

    return arguments
