"""AI provider step — collect and validate provider credentials."""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Confirm, Prompt

from animus_bootstrap.setup.validators import test_anthropic_key

_MAX_RETRIES = 3


def run_api_keys(console: Console) -> dict[str, object]:
    """Prompt for AI provider selection and credentials.

    Presents three provider options with Ollama recommended as the
    local-first, private, free option.  Multiple providers can be
    configured simultaneously.

    Args:
        console: Rich Console instance for terminal output.

    Returns:
        Dictionary with provider config data.
    """
    console.print()
    console.print("[bold cyan]AI Provider[/bold cyan]")
    console.print()
    console.print("  Choose your AI backend. You can configure multiple providers.")
    console.print()
    console.print("  [bold green]A) Ollama (local, free, private) — RECOMMENDED[/bold green]")
    console.print("     Runs entirely on your machine. No data leaves your device.")
    console.print("  [bold]B) Anthropic[/bold] — Claude API (cloud, best quality)")
    console.print("  [bold]C) OpenAI[/bold] — GPT API (cloud, fallback)")
    console.print()

    result: dict[str, object] = {
        "anthropic_key": "",
        "openai_key": "",
        "ollama_enabled": False,
        "ollama_model": "llama3.2",
        "default_backend": "anthropic",
    }

    # --- Ollama ---
    use_ollama = Confirm.ask(
        "[bold]Configure Ollama (local LLM)?[/bold]", console=console, default=True
    )
    if use_ollama:
        ollama_data = _configure_ollama(console)
        result.update(ollama_data)

    # --- Anthropic ---
    console.print()
    add_anthropic = Confirm.ask(
        "[bold]Add an Anthropic API key?[/bold]",
        console=console,
        default=not use_ollama,
    )
    if add_anthropic:
        result["anthropic_key"] = _collect_anthropic_key(console)

    # --- OpenAI ---
    console.print()
    add_openai = Confirm.ask(
        "[bold]Add an OpenAI API key?[/bold] (optional)", console=console, default=False
    )
    if add_openai:
        result["openai_key"] = Prompt.ask("  OpenAI API key", console=console, password=True)

    # Determine default backend
    if result.get("ollama_enabled"):
        result["default_backend"] = "ollama"
    elif result.get("anthropic_key"):
        result["default_backend"] = "anthropic"

    # Must have at least one provider
    if not result.get("ollama_enabled") and not result.get("anthropic_key"):
        console.print("[bold red]Error:[/bold red] At least one AI provider is required.")
        console.print("  Configure Ollama or provide an Anthropic API key.")
        raise SystemExit(1)

    return result


def _configure_ollama(console: Console) -> dict[str, object]:
    """Detect and configure Ollama.

    Checks if Ollama is running, lists available models, and lets the
    user select one.

    Returns:
        Dict with ollama_enabled, ollama_model, ollama_host, ollama_port.
    """
    import httpx

    host = "localhost"
    port = 11434
    base_url = f"http://{host}:{port}"

    # Auto-detect
    console.print()
    with console.status("[cyan]Checking for Ollama...[/cyan]"):
        available = False
        models: list[str] = []
        try:
            resp = httpx.get(f"{base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                available = True
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            available = False

    if not available:
        console.print("  [yellow]Ollama not detected[/yellow] at localhost:11434")
        console.print("  Install from: [link=https://ollama.com]ollama.com[/link]")
        console.print("  Then run: [bold]ollama serve[/bold]")
        return {"ollama_enabled": False}

    console.print(f"  [green]Ollama detected[/green] at {base_url}")

    if models:
        console.print(f"  Available models: {', '.join(models[:10])}")
        default_model = models[0] if models else "llama3.2"
        model = Prompt.ask(
            "  [bold]Model to use[/bold]",
            console=console,
            default=default_model,
        )
    else:
        console.print("  No models pulled yet.")
        console.print("  Run: [bold]ollama pull llama3.2[/bold]")
        model = Prompt.ask(
            "  [bold]Model name[/bold]",
            console=console,
            default="llama3.2",
        )

    return {
        "ollama_enabled": True,
        "ollama_model": model,
        "ollama_host": host,
        "ollama_port": port,
    }


def _collect_anthropic_key(console: Console) -> str:
    """Prompt and validate the Anthropic API key with retries."""
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
    console.print("[bold red]Error:[/bold red] Could not validate Anthropic API key.")
    console.print("  Get one at [link=https://console.anthropic.com/]console.anthropic.com[/link]")
    return ""
