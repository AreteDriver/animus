"""Memory step — select backend and storage configuration."""

from __future__ import annotations

from rich.console import Console
from rich.prompt import IntPrompt, Prompt

_BACKENDS = {
    1: ("sqlite", "SQLite", "Zero setup. Works immediately. Best for single machine."),
    2: ("chromadb", "ChromaDB", "Vector search. Better semantic memory retrieval."),
    3: ("weaviate", "Weaviate", "Enterprise. Requires separate Weaviate instance."),
}

_DEFAULT_PATH = "~/.local/share/animus/"
_DEFAULT_MAX_TOKENS = 100_000


def run_memory(console: Console) -> dict[str, str | int]:
    """Prompt the user to select a memory backend and storage settings.

    Presents three options (SQLite, ChromaDB, Weaviate) and collects the
    storage path and maximum context token limit.

    Args:
        console: Rich Console instance for terminal output.

    Returns:
        Dictionary with keys ``backend``, ``path``, and ``max_context_tokens``.
    """
    console.print()
    console.print("[bold cyan]Memory Backend[/bold cyan]")
    console.print()

    for num, (_, label, desc) in _BACKENDS.items():
        console.print(f"  [bold]{num}.[/bold] [cyan]{label}[/cyan] \u2014 {desc}")

    console.print()
    choice = IntPrompt.ask(
        "[bold]Select backend[/bold]",
        console=console,
        default=1,
        choices=["1", "2", "3"],
    )
    backend, label, _ = _BACKENDS[choice]
    console.print(f"  Selected: [cyan]{label}[/cyan]")

    console.print()
    path = Prompt.ask("[bold]Storage path[/bold]", console=console, default=_DEFAULT_PATH)

    max_tokens_str = Prompt.ask(
        "[bold]Max context tokens[/bold]",
        console=console,
        default=str(_DEFAULT_MAX_TOKENS),
    )
    try:
        max_context_tokens = int(max_tokens_str)
    except ValueError:
        console.print(
            f"  [yellow]Invalid value '{max_tokens_str}', using {_DEFAULT_MAX_TOKENS}[/yellow]"
        )
        max_context_tokens = _DEFAULT_MAX_TOKENS

    return {"backend": backend, "path": path, "max_context_tokens": max_context_tokens}
