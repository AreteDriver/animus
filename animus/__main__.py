"""Entry point for running animus as a module."""

import atexit

from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from animus.cognitive import CognitiveLayer, ModelConfig, ReasoningMode
from animus.config import AnimusConfig
from animus.logging import get_logger, setup_logging
from animus.memory import Conversation, MemoryLayer, MemoryType

console = Console()
logger = get_logger("cli")


def show_help():
    """Display help information."""
    console.print(
        """
[bold]Commands:[/bold]
  exit      - Quit Animus
  help      - Show this help
  /deep     - Use deep reasoning mode for next query
  /recall   - Search memories (e.g., /recall python)
  /history  - Show recent conversations
  /remember - Store a fact (e.g., /remember I prefer dark mode)
  /forget   - Delete a memory by ID (e.g., /forget abc123)
  /status   - Show system status

[bold]Otherwise:[/bold]
  Just type naturally. Animus will respond.
"""
    )


def show_status(config: AnimusConfig, memory: MemoryLayer):
    """Show system status."""
    table = Table(title="Animus Status")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Data Directory", str(config.data_dir))
    table.add_row("Model Provider", config.model.provider)
    table.add_row("Model Name", config.model.name)
    table.add_row("Memory Backend", memory.backend_type)
    table.add_row("Log Level", config.log_level)

    memories = memory.store.list_all()
    table.add_row("Total Memories", str(len(memories)))

    episodic = len([m for m in memories if m.memory_type == MemoryType.EPISODIC])
    semantic = len([m for m in memories if m.memory_type == MemoryType.SEMANTIC])
    table.add_row("Episodic Memories", str(episodic))
    table.add_row("Semantic Memories", str(semantic))

    console.print(table)


def show_history(memory: MemoryLayer, limit: int = 5):
    """Show recent conversations."""
    conversations = memory.recall("Conversation from", MemoryType.EPISODIC, limit=limit)

    if not conversations:
        console.print("[dim]No conversation history found.[/dim]")
        return

    console.print(f"\n[bold]Recent Conversations ({len(conversations)}):[/bold]\n")
    for conv in conversations:
        # Extract first line as title
        lines = conv.content.split("\n")
        title = lines[0] if lines else "Untitled"
        preview = lines[1][:60] + "..." if len(lines) > 1 else ""

        console.print(f"[cyan]{conv.id[:8]}[/cyan] {title}")
        if preview:
            console.print(f"  [dim]{preview}[/dim]")
    console.print()


def main():
    """Main entry point for Animus CLI."""
    # Load configuration
    config = AnimusConfig.load()
    config.ensure_dirs()

    # Save default config if it doesn't exist
    if not config.config_file.exists():
        config.save()
        logger.info(f"Created default config at {config.config_file}")

    # Setup logging
    setup_logging(
        log_file=config.log_file if config.log_to_file else None,
        level=config.log_level,
        log_to_file=config.log_to_file,
    )
    logger.info("Animus starting up")

    console.print(
        Panel.fit(
            "[bold cyan]Animus[/bold cyan]\n[dim]Personal cognitive sovereignty[/dim]",
            border_style="cyan",
        )
    )

    # Initialize layers
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend)

    # Build model config from settings
    if config.model.provider == "ollama":
        model_config = ModelConfig.ollama(config.model.name)
        model_config.base_url = config.model.ollama_url
    elif config.model.provider == "anthropic":
        model_config = ModelConfig.anthropic(config.model.name)
    else:
        model_config = ModelConfig.ollama(config.model.name)

    cognitive = CognitiveLayer(primary_config=model_config)

    history = FileHistory(str(config.history_file))

    # Start a new conversation
    conversation = Conversation.new()
    logger.info(f"Started conversation {conversation.id[:8]}")

    # Save conversation on exit
    def save_on_exit():
        if conversation.messages:
            memory.save_conversation(conversation)
            logger.info(
                f"Saved conversation {conversation.id[:8]} with {len(conversation.messages)} messages"
            )

    atexit.register(save_on_exit)

    console.print("\n[dim]Type 'help' for commands, 'exit' to quit[/dim]\n")

    while True:
        try:
            user_input = prompt(">>> ", history=history).strip()

            if not user_input:
                continue

            # Commands
            if user_input.lower() == "exit":
                console.print("[dim]Goodbye.[/dim]")
                break

            if user_input.lower() == "help":
                show_help()
                continue

            if user_input.lower() == "/status":
                show_status(config, memory)
                continue

            if user_input.lower() == "/history":
                show_history(memory)
                continue

            if user_input.startswith("/recall "):
                query = user_input[8:]
                memories = memory.recall(query)
                if memories:
                    for m in memories:
                        type_label = f"[{m.memory_type.value}]"
                        console.print(f"[dim]{m.id[:8]}[/dim] {type_label} {m.content[:80]}...")
                else:
                    console.print("[dim]No memories found.[/dim]")
                continue

            if user_input.startswith("/remember "):
                fact = user_input[10:]
                mem = memory.remember(fact, MemoryType.SEMANTIC)
                console.print(f"[green]Remembered:[/green] {mem.id[:8]}")
                continue

            if user_input.startswith("/forget "):
                mem_id = user_input[8:]
                # Try to find memory by partial ID
                all_mems = memory.store.list_all()
                matches = [m for m in all_mems if m.id.startswith(mem_id)]
                if len(matches) == 1:
                    memory.forget(matches[0].id)
                    console.print(f"[yellow]Forgotten:[/yellow] {matches[0].id[:8]}")
                elif len(matches) > 1:
                    console.print(f"[red]Ambiguous ID, {len(matches)} matches[/red]")
                else:
                    console.print("[red]Memory not found[/red]")
                continue

            # Check for mode prefixes
            mode = ReasoningMode.QUICK
            if user_input.startswith("/deep "):
                mode = ReasoningMode.DEEP
                user_input = user_input[6:]

            # Record user message
            conversation.add_message("user", user_input)

            # Get relevant context from memory
            context_memories = memory.recall(user_input, limit=3)
            context = "\n".join(m.content for m in context_memories) if context_memories else None

            # Generate response
            console.print()
            logger.debug(f"Generating response with mode={mode.value}")
            response = cognitive.think(user_input, context=context, mode=mode)

            # Record assistant response
            conversation.add_message("assistant", response)

            console.print(response)
            console.print()

            # Auto-save every 10 messages
            if len(conversation.messages) % 10 == 0:
                memory.save_conversation(conversation)
                conversation = Conversation.new()
                logger.debug("Auto-saved conversation, started new one")

        except KeyboardInterrupt:
            console.print("\n[dim]Use 'exit' to quit.[/dim]")
        except EOFError:
            break
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
