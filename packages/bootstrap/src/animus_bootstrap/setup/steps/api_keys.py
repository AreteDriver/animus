"""API keys step — collect and validate provider credentials."""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Confirm, Prompt

from animus_bootstrap.setup.validators import test_anthropic_key

_MAX_RETRIES = 3


def run_api_keys(console: Console) -> dict[str, str]:
    """Prompt for API keys and validate them against live endpoints.

    The Anthropic key is required and validated with up to 3 retries.
    The OpenAI key is optional and follows the same validation pattern.

    Args:
        console: Rich Console instance for terminal output.

    Returns:
        Dictionary with keys ``anthropic_key`` and ``openai_key``.
    """
    console.print()
    console.print("[bold cyan]API Keys[/bold cyan]")
    console.print()

    # --- Anthropic (required) ---
    anthropic_key = _collect_anthropic_key(console)

    # --- OpenAI (optional) ---
    openai_key = ""
    console.print()
    add_openai = Confirm.ask(
        "[bold]Add an OpenAI API key?[/bold] (optional)", console=console, default=False
    )
    if add_openai:
        openai_key = Prompt.ask("  OpenAI API key", console=console, password=True)

    return {"anthropic_key": anthropic_key, "openai_key": openai_key}


def _collect_anthropic_key(console: Console) -> str:
    """Prompt and validate the Anthropic API key with retries.

    Args:
        console: Rich Console instance for terminal output.

    Returns:
        A validated Anthropic API key string.

    Raises:
        SystemExit: After exhausting all retries.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        key = Prompt.ask("[bold]Anthropic API key[/bold]", console=console, password=True)

        with console.status("[cyan]Validating key...[/cyan]"):
            valid = test_anthropic_key(key)

        if valid:
            console.print("  [green]\u2713[/green] Anthropic key validated")
            return key

        remaining = _MAX_RETRIES - attempt
        if remaining > 0:
            console.print(
                f"  [red]\u2717[/red] Invalid key. {remaining} "
                f"{'attempts' if remaining > 1 else 'attempt'} remaining."
            )
        else:
            console.print("  [red]\u2717[/red] Invalid key. No attempts remaining.")

    console.print()
    console.print("[bold red]Error:[/bold red] A valid Anthropic API key is required to continue.")
    console.print("  Get one at [link=https://console.anthropic.com/]console.anthropic.com[/link]")
    raise SystemExit(1)
