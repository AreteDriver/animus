"""Metrics commands — export and serve Prometheus metrics."""

from __future__ import annotations

from pathlib import Path

import typer

from ..helpers import console

metrics_app = typer.Typer(help="Export and view metrics")


@metrics_app.command("export")
def metrics_export(
    format: str = typer.Option(
        "prometheus",
        "--format",
        "-f",
        help="Export format (prometheus, json, text)",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file (stdout if not specified)"
    ),
):
    """Export workflow metrics."""
    try:
        from animus_forge.metrics import (
            JsonExporter,
            PrometheusExporter,
            get_collector,
        )

        collector = get_collector()

        if format == "prometheus":
            exporter = PrometheusExporter()
            content = exporter.export(collector)
        elif format == "json":
            exporter = JsonExporter()
            content = exporter.export(collector)
        else:
            # Text summary
            summary = collector.get_summary()
            lines = [
                "Gorgon Metrics Summary",
                "=" * 40,
                f"Workflows Total: {summary['workflows_total']}",
                f"Workflows Active: {summary['workflows_active']}",
                f"Workflows Completed: {summary['workflows_completed']}",
                f"Workflows Failed: {summary['workflows_failed']}",
                f"Success Rate: {summary['success_rate']:.1%}",
                f"Tokens Used: {summary['tokens_used']:,}",
            ]
            if summary.get("avg_duration_ms"):
                lines.append(f"Avg Duration: {summary['avg_duration_ms']:.0f}ms")
            content = "\n".join(lines)

        if output:
            output.write_text(content)
            console.print(f"[green]✓ Metrics exported to:[/green] {output}")
        else:
            print(content)

    except Exception as e:
        console.print(f"[red]Error exporting metrics:[/red] {e}")
        raise typer.Exit(1)


@metrics_app.command("serve")
def metrics_serve(
    port: int = typer.Option(9090, "--port", "-p", help="Port to serve metrics on"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
):
    """Start Prometheus metrics HTTP server."""
    try:
        from animus_forge.metrics import PrometheusMetricsServer, get_collector

        collector = get_collector()
        server = PrometheusMetricsServer(collector, host=host, port=port)

        console.print("[cyan]Starting Prometheus metrics server...[/cyan]")
        console.print(f"[bold]URL:[/bold] http://{host}:{port}/metrics")
        console.print(f"[bold]Health:[/bold] http://{host}:{port}/health")
        console.print("\n[dim]Press Ctrl+C to stop[/dim]")

        server.start()

        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            server.stop()
            console.print("[green]✓ Server stopped[/green]")

    except Exception as e:
        console.print(f"[red]Error starting server:[/red] {e}")
        raise typer.Exit(1)


@metrics_app.command("push")
def metrics_push(
    gateway_url: str = typer.Argument(..., help="Push gateway URL"),
    job: str = typer.Option("gorgon", "--job", "-j", help="Job name"),
    instance: str = typer.Option(None, "--instance", "-i", help="Instance name"),
):
    """Push metrics to Prometheus Push Gateway."""
    try:
        from animus_forge.metrics import PrometheusPushGateway, get_collector

        collector = get_collector()
        gateway = PrometheusPushGateway(
            url=gateway_url,
            job=job,
            instance=instance,
        )

        with console.status("[cyan]Pushing metrics...", spinner="dots"):
            success = gateway.push(collector)

        if success:
            console.print(f"[green]✓ Metrics pushed to:[/green] {gateway_url}")
        else:
            console.print("[red]Failed to push metrics[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error pushing metrics:[/red] {e}")
        raise typer.Exit(1)
