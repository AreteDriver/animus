"""Entry point for running animus as a module."""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory

from animus import CognitiveLayer, MemoryLayer, ModelConfig, ReasoningMode

console = Console()


def main():
    """Main entry point for Animus CLI."""
    data_dir = Path.home() / ".animus"
    data_dir.mkdir(exist_ok=True)

    console.print(Panel.fit(
        "[bold cyan]Animus[/bold cyan]\n"
        "[dim]Personal cognitive sovereignty[/dim]",
        border_style="cyan"
    ))

    # Initialize layers
    memory = MemoryLayer(data_dir)
    cognitive = CognitiveLayer(
        primary_config=ModelConfig.ollama(),
    )

    history = FileHistory(str(data_dir / "history"))

    console.print("\n[dim]Type 'exit' to quit, 'help' for commands[/dim]\n")

    while True:
        try:
            user_input = prompt(">>> ", history=history).strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                console.print("[dim]Goodbye.[/dim]")
                break

            if user_input.lower() == "help":
                console.print("""
[bold]Commands:[/bold]
  exit     - Quit Animus
  help     - Show this help
  /deep    - Use deep reasoning mode for next query
  /recall  - Search memories

[bold]Otherwise:[/bold]
  Just type naturally. Animus will respond.
""")
                continue

            # Check for command prefixes
            mode = ReasoningMode.QUICK
            if user_input.startswith("/deep "):
                mode = ReasoningMode.DEEP
                user_input = user_input[6:]
            elif user_input.startswith("/recall "):
                query = user_input[8:]
                memories = memory.recall(query)
                if memories:
                    for m in memories:
                        console.print(f"[dim]{m.id[:8]}[/dim] {m.content[:100]}...")
                else:
                    console.print("[dim]No memories found.[/dim]")
                continue

            # Get relevant context from memory
            context_memories = memory.recall(user_input, limit=3)
            context = "\n".join(m.content for m in context_memories) if context_memories else None

            # Generate response
            console.print()
            response = cognitive.think(user_input, context=context, mode=mode)
            console.print(response)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Use 'exit' to quit.[/dim]")
        except EOFError:
            break


if __name__ == "__main__":
    main()
