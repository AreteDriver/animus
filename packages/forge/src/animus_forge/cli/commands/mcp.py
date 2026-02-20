"""MCP commands — manage MCP server connections."""

from __future__ import annotations

import json

import typer
from rich.table import Table

from ..helpers import console

mcp_app = typer.Typer(help="Manage MCP server connections")


def _get_manager():
    """Lazy-import and build an MCPConnectorManager."""
    from animus_forge.mcp.manager import MCPConnectorManager
    from animus_forge.state.database import get_database

    return MCPConnectorManager(get_database())


@mcp_app.command("list")
def mcp_list(
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """List registered MCP servers."""
    try:
        manager = _get_manager()
        servers = manager.list_servers()
    except Exception as e:
        console.print(f"[red]Error listing servers:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps([s.model_dump() for s in servers], indent=2, default=str))
        return

    if not servers:
        console.print("[yellow]No MCP servers registered[/yellow]")
        return

    table = Table(title="MCP Servers")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("URL")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Tools", justify="right")

    for s in servers:
        status = s.status if isinstance(s.status, str) else s.status.value
        status_color = {
            "connected": "green",
            "disconnected": "yellow",
            "error": "red",
            "not_configured": "dim",
            "connecting": "cyan",
        }.get(status, "white")
        server_type = s.type if isinstance(s.type, str) else s.type.value
        table.add_row(
            s.id[:12],
            s.name,
            s.url,
            server_type,
            f"[{status_color}]{status}[/{status_color}]",
            str(len(s.tools)),
        )

    console.print(table)


@mcp_app.command("add")
def mcp_add(
    name: str = typer.Argument(..., help="Server name"),
    url: str = typer.Argument(..., help="Server URL or stdio command"),
    server_type: str = typer.Option("sse", "--type", "-t", help="Server type (stdio|sse)"),
    auth: str = typer.Option("none", "--auth", "-a", help="Auth type (none|bearer|api_key)"),
    description: str = typer.Option(None, "--description", "-d", help="Server description"),
):
    """Register a new MCP server."""
    try:
        from animus_forge.mcp.models import MCPServerCreateInput

        manager = _get_manager()
        data = MCPServerCreateInput(
            name=name,
            url=url,
            type=server_type,
            authType=auth,
            description=description,
        )
        server = manager.create_server(data)
        console.print(f"[green]Server registered:[/green] {server.name} ({server.id})")
    except Exception as e:
        console.print(f"[red]Error adding server:[/red] {e}")
        raise typer.Exit(1)


@mcp_app.command("remove")
def mcp_remove(
    server_id: str = typer.Argument(..., help="Server ID to remove"),
):
    """Remove an MCP server registration."""
    try:
        manager = _get_manager()
        if manager.delete_server(server_id):
            console.print(f"[green]Server removed:[/green] {server_id}")
        else:
            console.print(f"[red]Server not found:[/red] {server_id}")
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error removing server:[/red] {e}")
        raise typer.Exit(1)


@mcp_app.command("test")
def mcp_test(
    server_id: str = typer.Argument(..., help="Server ID to test"),
):
    """Test connection to an MCP server."""
    try:
        manager = _get_manager()
        result = manager.test_connection(server_id)

        if result.success:
            console.print("[green]Connection successful[/green]")
            console.print(f"  Tools discovered: {len(result.tools)}")
            console.print(f"  Resources discovered: {len(result.resources)}")
        else:
            console.print(f"[red]Connection failed:[/red] {result.error}")
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error testing connection:[/red] {e}")
        raise typer.Exit(1)


@mcp_app.command("discover")
def mcp_discover(
    server_id: str = typer.Argument(..., help="Server ID to discover"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Discover tools and resources on an MCP server."""
    try:
        manager = _get_manager()
        result = manager.test_connection(server_id)

        if not result.success:
            console.print(f"[red]Discovery failed:[/red] {result.error}")
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error discovering server:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        print(
            json.dumps(
                {
                    "tools": [t.model_dump() for t in result.tools],
                    "resources": [r.model_dump() for r in result.resources],
                },
                indent=2,
                default=str,
            )
        )
        return

    if result.tools:
        tools_table = Table(title="Discovered Tools")
        tools_table.add_column("Name", style="cyan")
        tools_table.add_column("Description")
        for t in result.tools:
            tools_table.add_row(t.name, t.description)
        console.print(tools_table)
    else:
        console.print("[yellow]No tools discovered[/yellow]")

    if result.resources:
        res_table = Table(title="Discovered Resources")
        res_table.add_column("URI", style="cyan")
        res_table.add_column("Name")
        res_table.add_column("Type")
        for r in result.resources:
            res_table.add_row(r.uri, r.name, r.mimeType or "-")
        console.print(res_table)


@mcp_app.command("tools")
def mcp_tools(
    server_id: str = typer.Argument(..., help="Server ID"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """List cached tools for an MCP server."""
    try:
        manager = _get_manager()
        tools = manager.get_tools(server_id)
    except Exception as e:
        console.print(f"[red]Error getting tools:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps([t.model_dump() for t in tools], indent=2, default=str))
        return

    if not tools:
        console.print("[yellow]No cached tools for this server[/yellow]")
        return

    table = Table(title="Cached Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    for t in tools:
        table.add_row(t.name, t.description)
    console.print(table)
