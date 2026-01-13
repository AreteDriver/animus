"""Entry point for running animus as a module."""

import atexit
from pathlib import Path

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
[bold]Basic Commands:[/bold]
  exit       - Quit Animus
  help       - Show this help
  /status    - Show system status
  /stats     - Show memory statistics

[bold]Memory Commands:[/bold]
  /remember <text>       - Store a semantic memory
  /recall <query>        - Search memories
  /forget <id>           - Delete a memory by ID
  /review <id>           - Show full memory details
  /list [type]           - List memories (type: semantic, episodic, procedural)
  /history               - Show recent conversations

[bold]Tagging:[/bold]
  /tag <id> <tag1> [tag2...]  - Add tags to a memory
  /untag <id> <tag>           - Remove a tag
  /tags                       - List all tags with counts
  /search-tags <tag1> [tag2...] - Find memories by tags

[bold]Structured Memory:[/bold]
  /fact <subject> | <predicate> | <object>  - Store structured fact
  /procedure <name> | <trigger> | <step1>; <step2>...  - Store procedure

[bold]Export/Import:[/bold]
  /export [path]   - Export memories to JSON (default: ~/animus-export.json)
  /import <path>   - Import memories from file
  /backup [path]   - Create full backup

[bold]Reasoning:[/bold]
  /deep <query>    - Use deep reasoning mode

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

    by_type = {}
    for m in memories:
        by_type[m.memory_type.value] = by_type.get(m.memory_type.value, 0) + 1
    for t, count in sorted(by_type.items()):
        table.add_row(f"  {t.capitalize()}", str(count))

    console.print(table)


def show_stats(memory: MemoryLayer):
    """Show detailed memory statistics."""
    stats = memory.get_statistics()

    table = Table(title="Memory Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Memories", str(stats["total"]))
    table.add_row("Average Confidence", f"{stats['avg_confidence']:.2f}")
    table.add_row("Unique Tags", str(stats["unique_tags"]))

    console.print(table)

    if stats["by_type"]:
        console.print("\n[bold]By Type:[/bold]")
        for t, count in sorted(stats["by_type"].items()):
            console.print(f"  {t}: {count}")

    if stats["by_source"]:
        console.print("\n[bold]By Source:[/bold]")
        for s, count in sorted(stats["by_source"].items()):
            console.print(f"  {s}: {count}")

    if stats["by_subtype"]:
        console.print("\n[bold]By Subtype:[/bold]")
        for st, count in sorted(stats["by_subtype"].items()):
            console.print(f"  {st}: {count}")

    if stats["top_tags"]:
        console.print("\n[bold]Top Tags:[/bold]")
        for tag, count in stats["top_tags"]:
            console.print(f"  {tag}: {count}")


def show_history(memory: MemoryLayer, limit: int = 5):
    """Show recent conversations."""
    conversations = memory.recall("Conversation from", MemoryType.EPISODIC, limit=limit)

    if not conversations:
        console.print("[dim]No conversation history found.[/dim]")
        return

    console.print(f"\n[bold]Recent Conversations ({len(conversations)}):[/bold]\n")
    for conv in conversations:
        lines = conv.content.split("\n")
        title = lines[0] if lines else "Untitled"
        preview = lines[1][:60] + "..." if len(lines) > 1 else ""
        tags_str = f" [dim]tags: {', '.join(conv.tags)}[/dim]" if conv.tags else ""

        console.print(f"[cyan]{conv.id[:8]}[/cyan] {title}{tags_str}")
        if preview:
            console.print(f"  [dim]{preview}[/dim]")
    console.print()


def show_review(memory: MemoryLayer, mem_id: str):
    """Show full details of a memory."""
    mem = memory.get_memory(mem_id)
    if not mem:
        console.print("[red]Memory not found[/red]")
        return

    table = Table(title=f"Memory {mem.id[:8]}", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("ID", mem.id)
    table.add_row("Type", mem.memory_type.value)
    table.add_row("Subtype", mem.subtype or "-")
    table.add_row("Source", mem.source)
    table.add_row("Confidence", f"{mem.confidence:.2f}")
    table.add_row("Tags", ", ".join(mem.tags) if mem.tags else "-")
    table.add_row("Created", mem.created_at.strftime("%Y-%m-%d %H:%M:%S"))
    table.add_row("Updated", mem.updated_at.strftime("%Y-%m-%d %H:%M:%S"))

    console.print(table)
    console.print("\n[bold]Content:[/bold]")
    console.print(mem.content)

    if mem.metadata:
        console.print("\n[bold]Metadata:[/bold]")
        for k, v in mem.metadata.items():
            console.print(f"  {k}: {v}")


def show_list(memory: MemoryLayer, type_filter: str | None = None, limit: int = 20):
    """List memories, optionally filtered by type."""
    mem_type = None
    if type_filter:
        try:
            mem_type = MemoryType(type_filter.lower())
        except ValueError:
            console.print(f"[red]Unknown type: {type_filter}[/red]")
            console.print("Valid types: semantic, episodic, procedural, active")
            return

    memories = memory.store.list_all(mem_type)

    if not memories:
        console.print("[dim]No memories found.[/dim]")
        return

    # Sort by updated_at descending
    memories = sorted(memories, key=lambda m: m.updated_at, reverse=True)[:limit]

    console.print(f"\n[bold]Memories ({len(memories)}):[/bold]\n")
    for mem in memories:
        type_label = f"[{mem.memory_type.value}]"
        tags_str = f" [{', '.join(mem.tags)}]" if mem.tags else ""
        content_preview = mem.content[:60].replace("\n", " ")
        if len(mem.content) > 60:
            content_preview += "..."

        console.print(f"[cyan]{mem.id[:8]}[/cyan] {type_label}{tags_str} {content_preview}")


def show_tags(memory: MemoryLayer):
    """Show all tags with counts."""
    tags = memory.get_all_tags()

    if not tags:
        console.print("[dim]No tags found.[/dim]")
        return

    console.print(f"\n[bold]Tags ({len(tags)}):[/bold]\n")
    for tag, count in sorted(tags.items(), key=lambda x: (-x[1], x[0])):
        console.print(f"  [cyan]{tag}[/cyan]: {count}")


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

            # Basic commands
            if user_input.lower() == "exit":
                console.print("[dim]Goodbye.[/dim]")
                break

            if user_input.lower() == "help":
                show_help()
                continue

            if user_input.lower() == "/status":
                show_status(config, memory)
                continue

            if user_input.lower() == "/stats":
                show_stats(memory)
                continue

            if user_input.lower() == "/history":
                show_history(memory)
                continue

            if user_input.lower() == "/tags":
                show_tags(memory)
                continue

            # Commands with arguments
            if user_input.startswith("/recall "):
                query = user_input[8:]
                memories = memory.recall(query)
                if memories:
                    for m in memories:
                        type_label = f"[{m.memory_type.value}]"
                        tags_str = f" [{', '.join(m.tags)}]" if m.tags else ""
                        console.print(
                            f"[dim]{m.id[:8]}[/dim] {type_label}{tags_str} {m.content[:70]}..."
                        )
                else:
                    console.print("[dim]No memories found.[/dim]")
                continue

            if user_input.startswith("/remember "):
                fact = user_input[10:]
                mem = memory.remember(fact, MemoryType.SEMANTIC)
                console.print(f"[green]Remembered:[/green] {mem.id[:8]}")
                continue

            if user_input.startswith("/forget "):
                mem_id = user_input[8:].strip()
                if memory.forget(mem_id):
                    console.print(f"[yellow]Forgotten:[/yellow] {mem_id[:8]}")
                else:
                    console.print("[red]Memory not found[/red]")
                continue

            if user_input.startswith("/review "):
                mem_id = user_input[8:].strip()
                show_review(memory, mem_id)
                continue

            if user_input.startswith("/list"):
                parts = user_input.split()
                type_filter = parts[1] if len(parts) > 1 else None
                show_list(memory, type_filter)
                continue

            # Tagging commands
            if user_input.startswith("/tag "):
                parts = user_input[5:].split()
                if len(parts) < 2:
                    console.print("[red]Usage: /tag <id> <tag1> [tag2...][/red]")
                    continue
                mem_id = parts[0]
                tags = parts[1:]
                success = all(memory.add_tag(mem_id, tag) for tag in tags)
                if success:
                    console.print(f"[green]Tagged {mem_id[:8]} with: {', '.join(tags)}[/green]")
                else:
                    console.print("[red]Memory not found[/red]")
                continue

            if user_input.startswith("/untag "):
                parts = user_input[7:].split()
                if len(parts) < 2:
                    console.print("[red]Usage: /untag <id> <tag>[/red]")
                    continue
                mem_id, tag = parts[0], parts[1]
                if memory.remove_tag(mem_id, tag):
                    console.print(f"[yellow]Removed tag '{tag}' from {mem_id[:8]}[/yellow]")
                else:
                    console.print("[red]Memory not found or tag not present[/red]")
                continue

            if user_input.startswith("/search-tags "):
                tags = user_input[13:].split()
                if not tags:
                    console.print("[red]Usage: /search-tags <tag1> [tag2...][/red]")
                    continue
                memories = memory.recall_by_tags(tags)
                if memories:
                    console.print(f"\n[bold]Memories with tags {tags}:[/bold]\n")
                    for m in memories:
                        console.print(f"[dim]{m.id[:8]}[/dim] {m.content[:70]}...")
                else:
                    console.print("[dim]No memories found with those tags.[/dim]")
                continue

            # Structured memory commands
            if user_input.startswith("/fact "):
                # Format: /fact subject | predicate | object
                fact_str = user_input[6:]
                parts = [p.strip() for p in fact_str.split("|")]
                if len(parts) != 3:
                    console.print("[red]Usage: /fact subject | predicate | object[/red]")
                    console.print("[dim]Example: /fact User | prefers | dark mode[/dim]")
                    continue
                subject, predicate, obj = parts
                mem = memory.remember_fact(subject, predicate, obj, category="fact")
                console.print(f"[green]Fact stored:[/green] {mem.id[:8]}")
                continue

            if user_input.startswith("/procedure "):
                # Format: /procedure name | trigger | step1; step2; step3
                proc_str = user_input[11:]
                parts = [p.strip() for p in proc_str.split("|")]
                if len(parts) != 3:
                    console.print("[red]Usage: /procedure name | trigger | step1; step2...[/red]")
                    console.print(
                        "[dim]Example: /procedure morning | waking up | check email; review calendar[/dim]"
                    )
                    continue
                name, trigger, steps_str = parts
                steps = [s.strip() for s in steps_str.split(";") if s.strip()]
                mem = memory.remember_procedure(name, trigger, steps)
                console.print(f"[green]Procedure stored:[/green] {mem.id[:8]}")
                continue

            # Export/Import commands
            if user_input.startswith("/export"):
                parts = user_input.split()
                if len(parts) > 1:
                    export_path = Path(parts[1]).expanduser()
                else:
                    export_path = Path.home() / "animus-export.json"

                data = memory.export_memories()
                export_path.write_text(data)
                console.print(f"[green]Exported to:[/green] {export_path}")
                continue

            if user_input.startswith("/import "):
                import_path = Path(user_input[8:].strip()).expanduser()
                if not import_path.exists():
                    console.print(f"[red]File not found: {import_path}[/red]")
                    continue
                data = import_path.read_text()
                count = memory.import_memories(data)
                console.print(f"[green]Imported {count} memories[/green]")
                continue

            if user_input.startswith("/backup"):
                parts = user_input.split()
                if len(parts) > 1:
                    backup_path = Path(parts[1]).expanduser()
                else:
                    backup_path = Path.home() / "animus-backup.zip"

                memory.backup(backup_path)
                console.print(f"[green]Backup created:[/green] {backup_path}")
                continue

            # Reasoning mode
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
