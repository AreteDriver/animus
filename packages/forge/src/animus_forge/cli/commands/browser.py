"""Browser automation commands — navigate, screenshot, extract."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..helpers import console

browser_app = typer.Typer(help="Browser automation with Playwright")


@browser_app.command("navigate")
def browser_navigate(
    url: str = typer.Argument(..., help="URL to navigate to"),
    screenshot: bool = typer.Option(False, "--screenshot", "-s", help="Take screenshot"),
    extract: bool = typer.Option(False, "--extract", "-e", help="Extract page content"),
    headless: bool = typer.Option(True, "--headless/--no-headless", help="Run headless"),
):
    """Navigate to a URL and optionally extract content.

    Example:
        gorgon browser navigate https://example.com
        gorgon browser navigate https://news.ycombinator.com --extract
        gorgon browser navigate https://example.com --screenshot --no-headless
    """
    import asyncio

    try:
        from animus_forge.browser import BrowserAutomation, BrowserConfig
    except ImportError as e:
        console.print(f"[red]Missing dependencies:[/red] {e}")
        console.print("\nInstall with: pip install 'gorgon[browser]'")
        console.print("Then run: playwright install chromium")
        raise typer.Exit(1)

    async def run():
        config = BrowserConfig(headless=headless)
        async with BrowserAutomation(config) as browser:
            console.print(f"[cyan]Navigating to:[/cyan] {url}")

            result = await browser.navigate(url)
            if not result.success:
                console.print(f"[red]Navigation failed:[/red] {result.error}")
                raise typer.Exit(1)

            console.print(f"[green]Title:[/green] {result.title}")
            console.print(f"[green]URL:[/green] {result.url}")

            if extract:
                console.print("\n[cyan]Extracting content...[/cyan]")
                extract_result = await browser.extract_content()
                if extract_result.success:
                    data = extract_result.data
                    console.print("\n[bold]Content Preview:[/bold]")
                    text = data.get("text", "")[:1000]
                    console.print(text)
                    if data.get("links"):
                        console.print(f"\n[bold]Links:[/bold] {len(data['links'])} found")

            if screenshot:
                console.print("\n[cyan]Taking screenshot...[/cyan]")
                ss_result = await browser.screenshot()
                if ss_result.success:
                    console.print(f"[green]Screenshot saved:[/green] {ss_result.screenshot_path}")

    try:
        asyncio.run(run())
    except Exception as e:
        console.print(f"[red]Browser error:[/red] {e}")
        raise typer.Exit(1)


@browser_app.command("screenshot")
def browser_screenshot(
    url: str = typer.Argument(..., help="URL to screenshot"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path"),
    full_page: bool = typer.Option(False, "--full", "-f", help="Capture full page"),
):
    """Take a screenshot of a URL.

    Example:
        gorgon browser screenshot https://example.com
        gorgon browser screenshot https://example.com -o screenshot.png --full
    """
    import asyncio

    try:
        from animus_forge.browser import BrowserAutomation
    except ImportError:
        console.print("[red]Playwright not installed.[/red]")
        console.print("\nInstall with: pip install 'gorgon[browser]'")
        raise typer.Exit(1)

    async def run():
        async with BrowserAutomation() as browser:
            console.print(f"[cyan]Loading:[/cyan] {url}")

            result = await browser.navigate(url)
            if not result.success:
                console.print(f"[red]Failed to load page:[/red] {result.error}")
                raise typer.Exit(1)

            console.print(f"[green]Page loaded:[/green] {result.title}")

            path = str(output) if output else None
            ss_result = await browser.screenshot(path=path, full_page=full_page)

            if ss_result.success:
                console.print(f"[green]Screenshot saved:[/green] {ss_result.screenshot_path}")
            else:
                console.print(f"[red]Screenshot failed:[/red] {ss_result.error}")
                raise typer.Exit(1)

    try:
        asyncio.run(run())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@browser_app.command("extract")
def browser_extract(
    url: str = typer.Argument(..., help="URL to extract content from"),
    selector: str = typer.Option(None, "--selector", "-s", help="CSS selector to extract"),
    links: bool = typer.Option(True, "--links/--no-links", help="Extract links"),
    tables: bool = typer.Option(True, "--tables/--no-tables", help="Extract tables"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Extract content from a URL.

    Example:
        gorgon browser extract https://example.com
        gorgon browser extract https://news.ycombinator.com --json
        gorgon browser extract https://example.com -s "main article"
    """
    import asyncio

    try:
        from animus_forge.browser import BrowserAutomation
    except ImportError:
        console.print("[red]Playwright not installed.[/red]")
        raise typer.Exit(1)

    async def run():
        async with BrowserAutomation() as browser:
            result = await browser.navigate(url)
            if not result.success:
                console.print(f"[red]Failed to load:[/red] {result.error}")
                raise typer.Exit(1)

            extract_result = await browser.extract_content(
                selector=selector,
                extract_links=links,
                extract_tables=tables,
            )

            if not extract_result.success:
                console.print(f"[red]Extraction failed:[/red] {extract_result.error}")
                raise typer.Exit(1)

            data = extract_result.data

            if json_output:
                print(json.dumps(data, indent=2))
            else:
                console.print(f"[bold]Title:[/bold] {data.get('title')}")
                console.print(f"[bold]URL:[/bold] {data.get('url')}\n")

                text = data.get("text", "")[:2000]
                console.print("[bold]Content:[/bold]")
                console.print(text)

                if data.get("links"):
                    console.print(f"\n[bold]Links ({len(data['links'])}):[/bold]")
                    for link in data["links"][:10]:
                        console.print(f"  • {link['text'][:50]}: {link['href']}")

                if data.get("tables"):
                    console.print(f"\n[bold]Tables:[/bold] {len(data['tables'])} found")

    try:
        asyncio.run(run())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
