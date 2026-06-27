"""CLI for the content module."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from .config import ContentConfig
from .generator import ContentGenerator
from .models import ContentPlatform, ContentStatus, ContentTopic
from .publisher import Publisher
from .store import ContentStore
from .tracker import EarningsTracker

app = typer.Typer(name="animus-content", help="Autonomous write-to-earn content module")
console = Console()


def _load_config() -> ContentConfig:
    config_path = ContentConfig().data_dir / "config.yaml"
    return ContentConfig.load(config_path)


@app.command()
def generate(
    topic: str = typer.Argument(..., help="Topic to write about"),
    platform: str = typer.Option("mirror", help="Target platform"),
    word_count: int = typer.Option(800, help="Target word count"),
) -> None:
    """Generate an article on a topic."""
    config = _load_config()
    store = ContentStore(config.data_dir)
    gen = ContentGenerator(config)

    content_topic = ContentTopic(
        topic=topic,
        target_platform=ContentPlatform(platform),
    )

    article = asyncio.run(gen.generate(content_topic, word_count=word_count))
    store.save_article(article)

    console.print(f"[green]Generated:[/green] {article.title}")
    console.print(f"  ID: {article.id}")
    console.print(f"  Words: {article.word_count}")
    console.print(f"  Platform: {article.platform.value}")


@app.command()
def publish(
    article_id: str = typer.Argument(..., help="Article ID to publish"),
    platform: str = typer.Option(None, help="Override target platform"),
) -> None:
    """Publish an article to a platform."""
    config = _load_config()
    store = ContentStore(config.data_dir)
    publisher = Publisher(store=store)

    article = store.get_article(article_id)
    if article is None:
        console.print(f"[red]Article {article_id} not found[/red]")
        raise typer.Exit(1)

    target = ContentPlatform(platform) if platform else None
    result = asyncio.run(publisher.publish(article, platform=target))

    if result.success:
        console.print(f"[green]Published:[/green] {result.published_url}")
    else:
        console.print(f"[red]Failed:[/red] {result.error}")


@app.command(name="list")
def list_articles(
    platform: str = typer.Option(None, help="Filter by platform"),
    status: str = typer.Option(None, help="Filter by status"),
) -> None:
    """List articles in the store."""
    config = _load_config()
    store = ContentStore(config.data_dir)

    plat = ContentPlatform(platform) if platform else None
    stat = ContentStatus(status) if status else None
    articles = store.list_articles(platform=plat, status=stat)

    table = Table(title="Articles")
    table.add_column("ID", style="cyan")
    table.add_column("Title")
    table.add_column("Platform")
    table.add_column("Status")
    table.add_column("Words", justify="right")

    for a in articles:
        table.add_row(a.id, a.title[:50], a.platform.value, a.status.value, str(a.word_count))

    console.print(table)


@app.command()
def earnings() -> None:
    """Show earnings report."""
    config = _load_config()
    store = ContentStore(config.data_dir)
    tracker = EarningsTracker(store=store)

    total = tracker.total_earnings_usd()
    by_platform = tracker.earnings_by_platform()

    console.print(f"\n[bold]Total Earnings:[/bold] ${total:.2f}")

    if by_platform:
        table = Table(title="Earnings by Platform")
        table.add_column("Platform")
        table.add_column("USD", justify="right")
        for plat, amount in by_platform.items():
            table.add_row(plat.value, f"${amount:.2f}")
        console.print(table)
    else:
        console.print("[dim]No earnings recorded yet.[/dim]")


@app.command()
def topics(
    platform: str = typer.Option("mirror", help="Target platform"),
    count: int = typer.Option(5, help="Number of topics"),
) -> None:
    """Suggest content topics."""
    config = _load_config()
    gen = ContentGenerator(config)

    suggested = asyncio.run(gen.suggest_topics(platform=ContentPlatform(platform), count=count))

    if not suggested:
        console.print("[dim]No topics suggested (LLM may be unavailable).[/dim]")
        return

    table = Table(title="Suggested Topics")
    table.add_column("Topic")
    table.add_column("Keywords")
    table.add_column("Engagement", justify="right")

    for t in suggested:
        table.add_row(t.topic, ", ".join(t.keywords), f"{t.estimated_engagement:.2f}")

    console.print(table)


@app.command()
def init() -> None:
    """Initialize content module configuration."""
    config = _load_config()
    config_path = config.data_dir / "config.yaml"
    if config_path.exists():
        console.print("[yellow]Config already exists.[/yellow]")
        return
    config.save(config_path)
    console.print(f"[green]Config created at {config_path}[/green]")


if __name__ == "__main__":
    app()
