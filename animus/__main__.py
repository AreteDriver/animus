"""Entry point for running animus as a module."""

import asyncio
import atexit
from pathlib import Path

from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from animus.api import APIServer
from animus.cognitive import CognitiveLayer, ModelConfig, ReasoningMode, detect_mode
from animus.config import AnimusConfig
from animus.decision import DecisionFramework
from animus.entities import EntityMemory, EntityType
from animus.integrations import (
    FilesystemIntegration,
    IntegrationManager,
    TodoistIntegration,
    WebhookIntegration,
)
from animus.learning import LearningLayer
from animus.logging import get_logger, setup_logging
from animus.memory import Conversation, MemoryLayer, MemoryType
from animus.proactive import ProactiveEngine
from animus.tasks import TaskTracker
from animus.tools import create_default_registry, create_memory_tools
from animus.voice import VoiceInterface

# Optional sync module
try:
    from animus.protocols.sync import SyncProvider
    from animus.sync import DeviceDiscovery, SyncableState, SyncClient, SyncServer

    SYNC_AVAILABLE = True
except ImportError:
    SYNC_AVAILABLE = False
    DeviceDiscovery = None
    SyncClient = None
    SyncServer = None
    SyncableState = None

# Optional Google integrations
try:
    from animus.integrations.google import GoogleCalendarIntegration
    from animus.integrations.google.gmail import GmailIntegration

    GOOGLE_INTEGRATIONS_AVAILABLE = True
except ImportError:
    GOOGLE_INTEGRATIONS_AVAILABLE = False
    GoogleCalendarIntegration = None
    GmailIntegration = None

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

[bold]Tools (Phase 2):[/bold]
  /tools             - List available tools
  /tool <name> [args] - Execute a tool directly
                       Examples: /tool get_datetime
                                 /tool read_file path=/etc/hostname

[bold]Reasoning:[/bold]
  /deep <query>      - Use deep reasoning mode
  /research <query>  - Research mode with web search
  /brief [topic]     - Generate situation briefing from memory

[bold]Decision Support:[/bold]
  /decide <question> - Start structured decision analysis

[bold]Tasks:[/bold]
  /task add <desc>   - Add a new task
  /task list         - List pending tasks
  /task done <id>    - Mark task complete
  /task start <id>   - Mark task in progress
  /task delete <id>  - Delete a task

[bold]API Server (Phase 3):[/bold]
  /server start [port]  - Start API server (default: 8420)
  /server stop          - Stop API server
  /server status        - Show server status

[bold]Voice (Phase 3):[/bold]
  /voice on             - Enable voice input mode
  /voice off            - Disable voice input
  /speak <text>         - Speak text aloud
  /speak-toggle         - Toggle TTS for responses

[bold]Integrations (Phase 4):[/bold]
  /integrations           - List all integrations with status
  /integrate <service>    - Connect an integration
  /disconnect <service>   - Disconnect an integration
  Services: filesystem, todoist, google_calendar, gmail, webhooks

[bold]Self-Learning (Phase 5):[/bold]
  /learning               - Show learning dashboard
  /learning scan          - Trigger pattern detection
  /learning approve <id>  - Approve pending learning
  /learning reject <id>   - Reject pending learning
  /learning history       - Show learning event log
  /unlearn <id>           - Remove a learned item
  /guardrails             - List all guardrails
  /guardrail add <rule>   - Add a user-defined guardrail
  /learning rollback      - List rollback checkpoints
  /learning rollback <id> - Rollback to checkpoint

[bold]Proactive Intelligence:[/bold]
  /briefing               - Generate morning briefing
  /nudges                 - Show active nudges
  /nudges dismiss [id]    - Dismiss a nudge (or all)
  /meeting-prep <topic>   - Prepare context for a meeting

[bold]Entities & Relationships:[/bold]
  /entities               - List tracked entities
  /entity add <name> <type>   - Add an entity (types: person, project, organization, etc.)
  /entity <name>          - Show entity details and context
  /entity delete <name>   - Delete an entity

[bold]Cross-Device Sync (Phase 6):[/bold]
  /sync start             - Start sync server (enables discovery)
  /sync stop              - Stop sync server
  /sync status            - Show sync status and connected peers
  /sync discover          - List discovered devices on network
  /sync connect <addr>    - Connect to device (ws://host:port or device_id)
  /sync disconnect        - Disconnect from current peer
  /sync now               - Trigger manual sync with peer
  /sync pair              - Show pairing code (shared secret)

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

    # Phase 5: Learning Layer (initialized before cognitive to pass reference)
    learning: LearningLayer | None = None
    if config.learning.enabled:
        learning = LearningLayer(
            memory=memory,
            data_dir=config.data_dir,
            min_pattern_occurrences=config.learning.min_pattern_occurrences,
            min_pattern_confidence=config.learning.min_pattern_confidence,
            lookback_days=config.learning.lookback_days,
        )
        logger.info("Learning layer initialized")

    # Entity Memory
    entity_memory: EntityMemory | None = None
    if config.entities.enabled:
        entity_memory = EntityMemory(config.data_dir / "entities")
        memory.entity_memory = entity_memory
        logger.info("Entity memory initialized")

    # Proactive Engine
    proactive: ProactiveEngine | None = None
    if config.proactive.enabled:
        proactive = ProactiveEngine(config.data_dir, memory)
        logger.info("Proactive engine initialized")

    cognitive = CognitiveLayer(
        primary_config=model_config,
        learning=learning,
        entity_memory=entity_memory if config.entities.auto_extract else None,
        proactive=proactive,
    )

    # Initialize Phase 2 components
    tools = create_default_registry(security_config=config.tools_security)
    for tool in create_memory_tools(memory):
        tools.register(tool)

    tasks = TaskTracker(config.data_dir)
    decisions = DecisionFramework(cognitive)

    # Phase 3: API Server and Voice Interface
    api_server: APIServer | None = None
    voice: VoiceInterface | None = None

    # Phase 6: Cross-device Sync
    sync_state: SyncableState | None = None
    sync_server: SyncServer | None = None
    sync_client: SyncProvider | None = None
    discovery: DeviceDiscovery | None = None

    # Phase 4: Integration Manager
    integrations = IntegrationManager(config.data_dir / "integrations")

    # Register available integrations
    integrations.register(FilesystemIntegration(config.data_dir / "integrations"))
    integrations.register(TodoistIntegration())
    integrations.register(WebhookIntegration())

    if GOOGLE_INTEGRATIONS_AVAILABLE:
        integrations.register(GoogleCalendarIntegration(config.data_dir / "integrations"))
        integrations.register(GmailIntegration(config.data_dir / "integrations"))

    # Reconnect integrations from stored credentials
    asyncio.run(integrations.reconnect_from_stored())

    # Register integration tools with main registry
    for tool in integrations.get_all_tools():
        tools.register(tool)

    logger.info(f"Integration manager initialized, {len(integrations.list_connected())} connected")

    # Initialize voice if configured
    if config.voice.input_enabled or config.voice.output_enabled:
        try:
            voice = VoiceInterface(
                whisper_model=config.voice.whisper_model,
                tts_engine=config.voice.tts_engine,
                tts_rate=config.voice.tts_rate,
            )
            voice.response_tts_enabled = config.voice.output_enabled
            logger.info("Voice interface initialized")
        except ImportError as e:
            logger.warning(f"Voice dependencies not available: {e}")
            voice = None

    # Start proactive background scanning if configured
    if proactive and config.proactive.background_enabled:
        proactive.start_background(config.proactive.background_interval_seconds)
        logger.info("Proactive background scanning started")

    # Auto-start API server if configured
    if config.api.enabled:
        try:
            api_server = APIServer(
                memory=memory,
                cognitive=cognitive,
                tools=tools,
                tasks=tasks,
                decisions=decisions,
                host=config.api.host,
                port=config.api.port,
                api_key=config.api.api_key,
                integrations=integrations,
                entity_memory=entity_memory,
                proactive=proactive,
            )
            api_server.start()
            console.print(f"[dim]API server started on {config.api.host}:{config.api.port}[/dim]")
        except ImportError as e:
            logger.warning(f"API dependencies not available: {e}")
            api_server = None

    history = FileHistory(str(config.history_file))

    # Start a new conversation
    conversation = Conversation.new()
    logger.info(f"Started conversation {conversation.id[:8]}")

    # Cleanup on exit
    def cleanup_on_exit():
        if conversation.messages:
            memory.save_conversation(conversation)
            logger.info(
                f"Saved conversation {conversation.id[:8]} with {len(conversation.messages)} messages"
            )
        if proactive and proactive.is_running:
            proactive.stop_background()
            logger.info("Proactive background stopped")
        if api_server and api_server.is_running:
            api_server.stop()
            logger.info("API server stopped")
        if voice and voice.input.is_listening:
            voice.stop_listening()
            logger.info("Voice listening stopped")
        # Cleanup sync components
        if sync_client and sync_client.is_connected:
            asyncio.run(sync_client.disconnect())
            logger.info("Sync client disconnected")
        if discovery and discovery.is_running:
            discovery.stop()
            logger.info("Discovery stopped")
        if sync_server and sync_server.is_running:
            asyncio.run(sync_server.stop())
            logger.info("Sync server stopped")

    atexit.register(cleanup_on_exit)

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

            # =========================================================
            # Phase 2: Tools, Tasks, Decisions, Research, Briefing
            # =========================================================

            # Tools commands
            if user_input.lower() == "/tools":
                console.print("\n[bold]Available Tools:[/bold]\n")
                for tool in tools.list_tools():
                    approval_marker = " [requires approval]" if tool.requires_approval else ""
                    console.print(f"  [cyan]{tool.name}[/cyan]{approval_marker}")
                    console.print(f"    {tool.description}")
                    if tool.parameters.get("properties"):
                        for param, spec in tool.parameters["properties"].items():
                            req = (
                                " (required)"
                                if param in tool.parameters.get("required", [])
                                else ""
                            )
                            console.print(f"      - {param}: {spec.get('description', '')}{req}")
                console.print()
                continue

            if user_input.startswith("/tool "):
                # Parse: /tool name param1=value1 param2=value2
                parts = user_input[6:].split()
                if not parts:
                    console.print("[red]Usage: /tool <name> [param=value ...][/red]")
                    continue

                tool_name = parts[0]
                params = {}
                for part in parts[1:]:
                    if "=" in part:
                        key, val = part.split("=", 1)
                        params[key] = val
                    else:
                        # Positional argument for simple tools
                        params["query"] = part

                tool = tools.get(tool_name)
                if not tool:
                    console.print(f"[red]Unknown tool: {tool_name}[/red]")
                    continue

                # Approval for sensitive tools
                if tool.requires_approval:
                    confirm = prompt(f"Execute '{tool_name}'? (y/n): ").strip().lower()
                    if confirm != "y":
                        console.print("[yellow]Tool execution cancelled.[/yellow]")
                        continue

                result = tools.execute(tool_name, params)
                if result.success:
                    console.print(f"[green]Tool output:[/green]\n{result.output}")
                else:
                    console.print(f"[red]Tool error:[/red] {result.error}")
                continue

            # Task commands
            if user_input.startswith("/task "):
                task_cmd = user_input[6:].strip()

                if task_cmd.startswith("add "):
                    desc = task_cmd[4:].strip()
                    if not desc:
                        console.print("[red]Usage: /task add <description>[/red]")
                        continue
                    task = tasks.add(desc)
                    console.print(f"[green]Task added:[/green] {task.id[:8]} - {desc}")
                    continue

                if task_cmd == "list" or task_cmd.startswith("list"):
                    parts = task_cmd.split()
                    include_completed = "--all" in parts
                    task_list = tasks.list(include_completed=include_completed)

                    if not task_list:
                        console.print("[dim]No tasks found.[/dim]")
                        continue

                    console.print("\n[bold]Tasks:[/bold]\n")
                    for t in task_list:
                        status_color = {
                            "pending": "yellow",
                            "in_progress": "cyan",
                            "completed": "green",
                            "blocked": "red",
                        }.get(t.status.value, "white")
                        console.print(
                            f"  [{status_color}]{t.status.value:12}[/{status_color}] "
                            f"[dim]{t.id[:8]}[/dim] {t.description}"
                        )
                    continue

                if task_cmd.startswith("done "):
                    task_id = task_cmd[5:].strip()
                    if tasks.complete(task_id):
                        console.print(f"[green]Task completed:[/green] {task_id[:8]}")
                    else:
                        console.print("[red]Task not found[/red]")
                    continue

                if task_cmd.startswith("start "):
                    task_id = task_cmd[6:].strip()
                    if tasks.start(task_id):
                        console.print(f"[cyan]Task started:[/cyan] {task_id[:8]}")
                    else:
                        console.print("[red]Task not found[/red]")
                    continue

                if task_cmd.startswith("delete "):
                    task_id = task_cmd[7:].strip()
                    if tasks.delete(task_id):
                        console.print(f"[yellow]Task deleted:[/yellow] {task_id[:8]}")
                    else:
                        console.print("[red]Task not found[/red]")
                    continue

                console.print(
                    "[red]Unknown task command. Use: add, list, done, start, delete[/red]"
                )
                continue

            # Decision support
            if user_input.startswith("/decide "):
                question = user_input[8:].strip()
                if not question:
                    console.print("[red]Usage: /decide <question>[/red]")
                    continue

                console.print("\n[dim]Analyzing decision...[/dim]\n")
                decision = decisions.analyze(question)
                console.print(decision.format_analysis())
                continue

            # Research mode
            if user_input.startswith("/research "):
                query = user_input[10:].strip()
                if not query:
                    console.print("[red]Usage: /research <query>[/red]")
                    continue

                console.print("\n[dim]Researching...[/dim]\n")

                # Use web search tool
                search_result = tools.execute("web_search", {"query": query})
                context = search_result.output if search_result.success else None

                # Generate response with research context
                response = cognitive.think(
                    f"Based on this research, answer: {query}",
                    context=context,
                    mode=ReasoningMode.RESEARCH,
                )
                console.print(response)
                console.print()
                continue

            # Briefing
            if user_input.startswith("/brief"):
                parts = user_input.split(maxsplit=1)
                topic = parts[1] if len(parts) > 1 else None

                console.print("\n[dim]Generating briefing...[/dim]\n")
                briefing = cognitive.brief(memory, topic=topic)
                console.print(briefing)
                console.print()
                continue

            # =========================================================
            # End Phase 2 commands
            # =========================================================

            # =========================================================
            # Phase 3: API Server and Voice commands
            # =========================================================

            # Server commands
            if user_input.startswith("/server "):
                server_cmd = user_input[8:].strip()

                if server_cmd.startswith("start"):
                    parts = server_cmd.split()
                    port = int(parts[1]) if len(parts) > 1 else config.api.port

                    if api_server and api_server.is_running:
                        console.print(
                            f"[yellow]Server already running on {api_server.host}:{api_server.port}[/yellow]"
                        )
                        continue

                    try:
                        api_server = APIServer(
                            memory=memory,
                            cognitive=cognitive,
                            tools=tools,
                            tasks=tasks,
                            decisions=decisions,
                            host=config.api.host,
                            port=port,
                            api_key=config.api.api_key,
                            integrations=integrations,
                            entity_memory=entity_memory,
                            proactive=proactive,
                        )
                        api_server.start()
                        console.print(
                            f"[green]API server started on {config.api.host}:{port}[/green]"
                        )
                    except ImportError:
                        console.print(
                            "[red]FastAPI not installed. Install with: pip install 'animus[api]'[/red]"
                        )
                    except Exception as e:
                        console.print(f"[red]Failed to start server: {e}[/red]")
                    continue

                if server_cmd == "stop":
                    if api_server and api_server.is_running:
                        api_server.stop()
                        console.print("[yellow]API server stopped[/yellow]")
                    else:
                        console.print("[dim]Server not running[/dim]")
                    continue

                if server_cmd == "status":
                    if api_server and api_server.is_running:
                        console.print(
                            f"[green]Server running on {api_server.host}:{api_server.port}[/green]"
                        )
                    else:
                        console.print("[dim]Server not running[/dim]")
                    continue

                console.print("[red]Unknown server command. Use: start, stop, status[/red]")
                continue

            # Voice commands
            if user_input.startswith("/voice "):
                voice_cmd = user_input[7:].strip()

                if voice_cmd == "on":
                    if not voice:
                        try:
                            voice = VoiceInterface(
                                whisper_model=config.voice.whisper_model,
                                tts_engine=config.voice.tts_engine,
                                tts_rate=config.voice.tts_rate,
                            )
                        except ImportError:
                            console.print(
                                "[red]Voice dependencies not installed. Install with: pip install 'animus[voice]'[/red]"
                            )
                            continue

                    def on_speech(text: str):
                        console.print(f"\n[cyan]You said:[/cyan] {text}")

                    voice.start_listening(on_speech)
                    console.print("[green]Voice input enabled. Listening...[/green]")
                    continue

                if voice_cmd == "off":
                    if voice and voice.input.is_listening:
                        voice.stop_listening()
                        console.print("[yellow]Voice input disabled[/yellow]")
                    else:
                        console.print("[dim]Voice input not active[/dim]")
                    continue

                console.print("[red]Unknown voice command. Use: on, off[/red]")
                continue

            if user_input.startswith("/speak "):
                text = user_input[7:].strip()
                if not text:
                    console.print("[red]Usage: /speak <text>[/red]")
                    continue

                if not voice:
                    try:
                        voice = VoiceInterface(
                            whisper_model=config.voice.whisper_model,
                            tts_engine=config.voice.tts_engine,
                            tts_rate=config.voice.tts_rate,
                        )
                    except ImportError:
                        console.print(
                            "[red]Voice dependencies not installed. Install with: pip install 'animus[voice]'[/red]"
                        )
                        continue

                voice.speak(text)
                continue

            if user_input == "/speak-toggle":
                if not voice:
                    try:
                        voice = VoiceInterface(
                            whisper_model=config.voice.whisper_model,
                            tts_engine=config.voice.tts_engine,
                            tts_rate=config.voice.tts_rate,
                        )
                    except ImportError:
                        console.print(
                            "[red]Voice dependencies not installed. Install with: pip install 'animus[voice]'[/red]"
                        )
                        continue

                voice.response_tts_enabled = not voice.response_tts_enabled
                status = "enabled" if voice.response_tts_enabled else "disabled"
                console.print(f"[green]Response TTS {status}[/green]")
                continue

            # =========================================================
            # End Phase 3 commands
            # =========================================================

            # =========================================================
            # Phase 4: Integration commands
            # =========================================================

            if user_input == "/integrations":
                all_integrations = integrations.list_all()
                if not all_integrations:
                    console.print("[dim]No integrations registered.[/dim]")
                    continue

                console.print("\n[bold]Integrations:[/bold]\n")
                for info in all_integrations:
                    status_color = {
                        "connected": "green",
                        "disconnected": "dim",
                        "error": "red",
                        "expired": "yellow",
                    }.get(info.status.value, "white")

                    console.print(
                        f"  [{status_color}]{info.status.value:12}[/{status_color}] "
                        f"[cyan]{info.name:16}[/cyan] {info.display_name}"
                    )
                    if info.error_message:
                        console.print(f"    [red]Error: {info.error_message}[/red]")
                    if info.capabilities:
                        console.print(f"    [dim]Tools: {', '.join(info.capabilities[:5])}[/dim]")
                console.print()
                continue

            if user_input.startswith("/integrate "):
                service = user_input[11:].strip()

                integration = integrations.get(service)
                if not integration:
                    available = [i.name for i in integrations.list_all()]
                    console.print(f"[red]Unknown service: {service}[/red]")
                    console.print(f"[dim]Available: {', '.join(available)}[/dim]")
                    continue

                if integration.is_connected:
                    console.print(f"[yellow]{service} is already connected[/yellow]")
                    continue

                # Build credentials based on service type
                credentials: dict = {}

                if service == "filesystem":
                    path = prompt("Path to index (or leave empty for current): ").strip()
                    if path:
                        credentials["paths"] = [path]
                    else:
                        credentials["paths"] = [str(Path.cwd())]

                elif service == "todoist":
                    api_key = config.integrations.todoist.api_key
                    if not api_key:
                        api_key = prompt("Todoist API key: ").strip()
                    if not api_key:
                        console.print("[red]API key required[/red]")
                        continue
                    credentials["api_key"] = api_key

                elif service == "webhooks":
                    port = config.integrations.webhooks.port
                    secret = config.integrations.webhooks.secret
                    credentials = {"port": port}
                    if secret:
                        credentials["secret"] = secret

                elif service in ["google_calendar", "gmail"]:
                    if not GOOGLE_INTEGRATIONS_AVAILABLE:
                        console.print("[red]Google integration libraries not installed.[/red]")
                        console.print("[dim]Install with: pip install 'animus[integrations]'[/dim]")
                        continue

                    client_id = config.integrations.google.client_id
                    client_secret = config.integrations.google.client_secret

                    if not client_id:
                        client_id = prompt("Google Client ID: ").strip()
                    if not client_secret:
                        client_secret = prompt("Google Client Secret: ").strip()

                    if not client_id or not client_secret:
                        console.print("[red]Client ID and Secret required[/red]")
                        continue

                    credentials = {
                        "client_id": client_id,
                        "client_secret": client_secret,
                    }

                # Attempt connection
                console.print(f"[dim]Connecting to {service}...[/dim]")
                success = asyncio.run(integrations.connect(service, credentials))

                if success:
                    console.print(f"[green]Connected to {service}[/green]")
                    # Re-register tools
                    for tool in integration.get_tools():
                        tools.register(tool)
                else:
                    info = integration.get_info()
                    console.print(f"[red]Failed to connect: {info.error_message}[/red]")
                continue

            if user_input.startswith("/disconnect "):
                service = user_input[12:].strip()

                integration = integrations.get(service)
                if not integration:
                    console.print(f"[red]Unknown service: {service}[/red]")
                    continue

                if not integration.is_connected:
                    console.print(f"[dim]{service} is not connected[/dim]")
                    continue

                success = asyncio.run(integrations.disconnect(service))

                if success:
                    console.print(f"[yellow]Disconnected from {service}[/yellow]")
                else:
                    console.print(f"[red]Failed to disconnect from {service}[/red]")
                continue

            # =========================================================
            # End Phase 4 commands
            # =========================================================

            # =========================================================
            # Phase 5: Learning commands
            # =========================================================

            if user_input == "/learning":
                if not learning:
                    console.print("[dim]Learning is disabled in configuration[/dim]")
                    continue

                dashboard = learning.get_dashboard_data()
                console.print("\n[bold]Learning Dashboard[/bold]\n")
                console.print(f"  Total learned:     {dashboard.total_learned}")
                console.print(f"  Pending approval:  {dashboard.pending_approval}")
                console.print(f"  Events today:      {dashboard.events_today}")
                console.print(f"  Guardrail blocks:  {dashboard.guardrail_violations}")

                if dashboard.by_category:
                    console.print("\n  [dim]By category:[/dim]")
                    for cat, count in dashboard.by_category.items():
                        console.print(f"    {cat}: {count}")

                if dashboard.recently_applied:
                    console.print("\n  [dim]Recently applied:[/dim]")
                    for item in dashboard.recently_applied[:5]:
                        console.print(f"    [{item.id[:8]}] {item.content[:50]}...")

                pending = learning.get_pending_learnings()
                if pending:
                    console.print(f"\n  [yellow]Pending approval ({len(pending)}):[/yellow]")
                    for item in pending[:5]:
                        console.print(
                            f"    [{item.id[:8]}] {item.category.value}: {item.content[:40]}..."
                        )
                console.print()
                continue

            if user_input == "/learning scan":
                if not learning:
                    console.print("[dim]Learning is disabled in configuration[/dim]")
                    continue

                console.print("[dim]Scanning for patterns...[/dim]")
                patterns = learning.scan_and_learn()
                console.print(f"[green]Detected {len(patterns)} patterns[/green]")
                for p in patterns[:5]:
                    console.print(
                        f"  [{p.pattern_type.value}] {p.description[:50]}... "
                        f"(confidence: {p.confidence:.0%})"
                    )
                continue

            if user_input.startswith("/learning approve "):
                if not learning:
                    console.print("[dim]Learning is disabled in configuration[/dim]")
                    continue

                item_id = user_input[18:].strip()
                # Find item by full or partial ID
                found = None
                for item in learning.get_pending_learnings():
                    if item.id == item_id or item.id.startswith(item_id):
                        found = item
                        break

                if not found:
                    console.print(f"[red]Pending learning not found: {item_id}[/red]")
                    continue

                if learning.approve_learning(found.id):
                    console.print(f"[green]Approved: {found.content[:50]}...[/green]")
                else:
                    console.print("[red]Failed to approve learning[/red]")
                continue

            if user_input.startswith("/learning reject "):
                if not learning:
                    console.print("[dim]Learning is disabled in configuration[/dim]")
                    continue

                item_id = user_input[17:].strip()
                found = None
                for item in learning.get_pending_learnings():
                    if item.id == item_id or item.id.startswith(item_id):
                        found = item
                        break

                if not found:
                    console.print(f"[red]Pending learning not found: {item_id}[/red]")
                    continue

                if learning.reject_learning(found.id):
                    console.print(f"[yellow]Rejected: {found.content[:50]}...[/yellow]")
                else:
                    console.print("[red]Failed to reject learning[/red]")
                continue

            if user_input == "/learning history":
                if not learning:
                    console.print("[dim]Learning is disabled in configuration[/dim]")
                    continue

                events = learning.transparency.get_history(limit=20)
                if not events:
                    console.print("[dim]No learning events yet[/dim]")
                    continue

                console.print("\n[bold]Learning History[/bold]\n")
                for event in events:
                    time_str = event.timestamp.strftime("%Y-%m-%d %H:%M")
                    console.print(
                        f"  {time_str} [{event.event_type:12}] {event.learned_item_id[:8]}"
                    )
                console.print()
                continue

            if user_input.startswith("/learning rollback "):
                if not learning:
                    console.print("[dim]Learning is disabled in configuration[/dim]")
                    continue

                point_id = user_input[19:].strip()
                success, unlearned = learning.rollback_to(point_id)
                if success:
                    console.print(f"[yellow]Rolled back, unlearned {len(unlearned)} items[/yellow]")
                else:
                    console.print(f"[red]Rollback point not found: {point_id}[/red]")
                continue

            if user_input == "/learning rollback":
                if not learning:
                    console.print("[dim]Learning is disabled in configuration[/dim]")
                    continue

                points = learning.rollback.get_rollback_points()
                if not points:
                    console.print("[dim]No rollback points available[/dim]")
                    continue

                console.print("\n[bold]Rollback Points[/bold]\n")
                for point in points:
                    time_str = point.timestamp.strftime("%Y-%m-%d %H:%M")
                    console.print(
                        f"  [{point.id[:8]}] {time_str} - {point.description} "
                        f"({len(point.learned_item_ids)} items)"
                    )
                console.print()
                continue

            if user_input.startswith("/unlearn "):
                if not learning:
                    console.print("[dim]Learning is disabled in configuration[/dim]")
                    continue

                item_id = user_input[9:].strip()
                found = None
                for item in learning.get_all_learnings():
                    if item.id == item_id or item.id.startswith(item_id):
                        found = item
                        break

                if not found:
                    console.print(f"[red]Learning not found: {item_id}[/red]")
                    continue

                if learning.unlearn(found.id):
                    console.print(f"[yellow]Unlearned: {found.content[:50]}...[/yellow]")
                else:
                    console.print("[red]Failed to unlearn[/red]")
                continue

            if user_input == "/guardrails":
                if not learning:
                    console.print("[dim]Learning is disabled in configuration[/dim]")
                    continue

                guardrails = learning.guardrails.get_all_guardrails()
                console.print("\n[bold]Guardrails[/bold]\n")
                for g in guardrails:
                    lock = "[red]IMMUTABLE[/red]" if g.immutable else "[dim]user[/dim]"
                    console.print(f"  [{g.id[:12]:12}] {lock} {g.rule}")
                console.print()
                continue

            if user_input.startswith("/guardrail add "):
                if not learning:
                    console.print("[dim]Learning is disabled in configuration[/dim]")
                    continue

                rule = user_input[15:].strip()
                if not rule:
                    console.print("[red]Please provide a rule[/red]")
                    continue

                guardrail = learning.add_user_guardrail(rule, f"User-defined: {rule}")
                console.print(f"[green]Added guardrail: {guardrail.id}[/green]")
                continue

            # =========================================================
            # End Phase 5 commands
            # =========================================================

            # =========================================================
            # Proactive Intelligence commands
            # =========================================================

            if user_input == "/briefing":
                if not proactive:
                    console.print("[dim]Proactive engine is disabled in configuration[/dim]")
                    continue

                console.print("[dim]Generating briefing...[/dim]\n")
                nudge = proactive.generate_morning_brief()
                console.print(Panel(nudge.content, title="Morning Briefing", border_style="cyan"))
                continue

            if user_input == "/nudges":
                if not proactive:
                    console.print("[dim]Proactive engine is disabled in configuration[/dim]")
                    continue

                active = proactive.get_active_nudges()
                if not active:
                    console.print("[dim]No active nudges[/dim]")
                    continue

                console.print(f"\n[bold]Active Nudges ({len(active)}):[/bold]\n")
                for n in active:
                    priority_color = {
                        "urgent": "red",
                        "high": "yellow",
                        "medium": "cyan",
                        "low": "dim",
                    }.get(n.priority.value, "white")
                    console.print(
                        f"  [{priority_color}]{n.priority.value:6}[/{priority_color}] "
                        f"[dim]{n.id[:8]}[/dim] {n.title}"
                    )
                    console.print(f"         {n.content[:100]}")
                console.print()
                continue

            if user_input.startswith("/nudges dismiss"):
                if not proactive:
                    console.print("[dim]Proactive engine is disabled in configuration[/dim]")
                    continue

                parts = user_input.split()
                if len(parts) > 2:
                    nudge_id = parts[2]
                    # Try partial match
                    matched = False
                    for n in proactive.get_active_nudges():
                        if n.id.startswith(nudge_id) or n.id == nudge_id:
                            proactive.dismiss_nudge(n.id)
                            console.print(f"[yellow]Dismissed: {n.title}[/yellow]")
                            matched = True
                            break
                    if not matched:
                        console.print(f"[red]Nudge not found: {nudge_id}[/red]")
                else:
                    count = proactive.dismiss_all()
                    console.print(f"[yellow]Dismissed {count} nudges[/yellow]")
                continue

            if user_input.startswith("/meeting-prep "):
                if not proactive:
                    console.print("[dim]Proactive engine is disabled in configuration[/dim]")
                    continue

                topic = user_input[14:].strip()
                if not topic:
                    console.print("[red]Usage: /meeting-prep <person or topic>[/red]")
                    continue

                console.print(f"[dim]Preparing context for '{topic}'...[/dim]\n")
                nudge = proactive.prepare_meeting_context(topic)
                console.print(
                    Panel(nudge.content, title=f"Meeting Prep: {topic}", border_style="cyan")
                )
                continue

            # =========================================================
            # Entity/Relationship commands
            # =========================================================

            if user_input == "/entities":
                if not entity_memory:
                    console.print("[dim]Entity memory is disabled in configuration[/dim]")
                    continue

                entities = entity_memory.list_entities(limit=30)
                if not entities:
                    console.print("[dim]No entities tracked yet[/dim]")
                    continue

                console.print(f"\n[bold]Tracked Entities ({len(entities)}):[/bold]\n")
                for e in entities:
                    aliases = f" ({', '.join(e.aliases)})" if e.aliases else ""
                    last = e.last_mentioned.strftime("%b %d") if e.last_mentioned else "never"
                    console.print(
                        f"  [cyan]{e.entity_type.value:12}[/cyan] "
                        f"{e.name}{aliases} "
                        f"[dim]mentions: {e.mention_count}, last: {last}[/dim]"
                    )
                console.print()
                continue

            if user_input.startswith("/entity add "):
                if not entity_memory:
                    console.print("[dim]Entity memory is disabled in configuration[/dim]")
                    continue

                parts = user_input[12:].split()
                if len(parts) < 2:
                    console.print("[red]Usage: /entity add <name> <type> [alias1,alias2][/red]")
                    console.print(
                        "[dim]Types: person, project, organization, place, topic, event, tool[/dim]"
                    )
                    continue

                name = parts[0]
                try:
                    etype = EntityType(parts[1].lower())
                except ValueError:
                    console.print(f"[red]Unknown entity type: {parts[1]}[/red]")
                    continue

                aliases = parts[2].split(",") if len(parts) > 2 else []
                entity = entity_memory.add_entity(name, etype, aliases=aliases)
                console.print(f"[green]Entity added:[/green] {entity.name} ({etype.value})")
                continue

            if user_input.startswith("/entity delete "):
                if not entity_memory:
                    console.print("[dim]Entity memory is disabled in configuration[/dim]")
                    continue

                name = user_input[15:].strip()
                found = entity_memory.find_entity(name)
                if not found:
                    console.print(f"[red]Entity not found: {name}[/red]")
                    continue

                entity_memory.delete_entity(found.id)
                console.print(f"[yellow]Deleted entity: {found.name}[/yellow]")
                continue

            if (
                user_input.startswith("/entity ")
                and not user_input.startswith("/entity add ")
                and not user_input.startswith("/entity delete ")
            ):
                if not entity_memory:
                    console.print("[dim]Entity memory is disabled in configuration[/dim]")
                    continue

                name = user_input[8:].strip()
                found = entity_memory.find_entity(name)
                if not found:
                    console.print(f"[red]Entity not found: {name}[/red]")
                    continue

                context = entity_memory.generate_entity_context(found.id)
                console.print(Panel(context, title=found.name, border_style="cyan"))
                continue

            # =========================================================
            # End Proactive & Entity commands
            # =========================================================

            # =========================================================
            # Phase 6: Cross-device Sync commands
            # =========================================================

            if user_input.startswith("/sync ") or user_input == "/sync":
                if not SYNC_AVAILABLE:
                    console.print(
                        "[red]Sync dependencies not installed. Install with: pip install websockets zeroconf[/red]"
                    )
                    continue

                sync_cmd = user_input[5:].strip() if len(user_input) > 5 else ""

                if sync_cmd == "start":
                    # Initialize sync components if not already done
                    if not sync_state:
                        sync_state = SyncableState(config.data_dir)

                    if sync_server and sync_server.is_running:
                        console.print(
                            f"[yellow]Sync server already running on port {sync_server.port}[/yellow]"
                        )
                        continue

                    # Start sync server
                    sync_server = SyncServer(
                        state=sync_state,
                        port=8422,
                    )

                    success = asyncio.run(sync_server.start())
                    if not success:
                        console.print("[red]Failed to start sync server[/red]")
                        continue

                    # Start device discovery
                    if not discovery:
                        discovery = DeviceDiscovery(
                            device_id=sync_state.device_id,
                            device_name=f"animus-{sync_state.device_id[:8]}",
                            port=8422,
                        )

                    if discovery.start():
                        console.print(
                            f"[green]Sync server started on port 8422[/green]\n"
                            f"  Device ID: {sync_state.device_id[:8]}...\n"
                            f"  Shared secret: {sync_server.shared_secret[:8]}..."
                        )
                    else:
                        console.print(
                            "[yellow]Sync server started but discovery failed[/yellow]\n"
                            "  Other devices can connect via: ws://<your-ip>:8422"
                        )
                    continue

                if sync_cmd == "stop":
                    stopped_something = False

                    if sync_client and sync_client.is_connected:
                        asyncio.run(sync_client.disconnect())
                        sync_client = None
                        stopped_something = True

                    if discovery and discovery.is_running:
                        discovery.stop()
                        stopped_something = True

                    if sync_server and sync_server.is_running:
                        asyncio.run(sync_server.stop())
                        stopped_something = True

                    if stopped_something:
                        console.print("[yellow]Sync services stopped[/yellow]")
                    else:
                        console.print("[dim]Sync services not running[/dim]")
                    continue

                if sync_cmd == "status":
                    console.print("\n[bold]Sync Status[/bold]\n")

                    # Server status
                    if sync_server and sync_server.is_running:
                        console.print(
                            f"  [green]Server:[/green] Running on port {sync_server.port}"
                        )
                        peers = sync_server.get_peers()
                        if peers:
                            console.print(f"  [green]Connected peers:[/green] {len(peers)}")
                            for peer in peers:
                                console.print(f"    - {peer.device_name} ({peer.device_id[:8]}...)")
                        else:
                            console.print("  [dim]No peers connected[/dim]")
                    else:
                        console.print("  [dim]Server: Not running[/dim]")

                    # Client status
                    if sync_client and sync_client.is_connected:
                        console.print(
                            f"  [green]Client:[/green] Connected to {sync_client.peer_device_name}"
                        )
                    else:
                        console.print("  [dim]Client: Not connected[/dim]")

                    # Discovery status
                    if discovery and discovery.is_running:
                        devices = discovery.get_devices()
                        console.print(
                            f"  [green]Discovery:[/green] Active ({len(devices)} devices)"
                        )
                    else:
                        console.print("  [dim]Discovery: Not running[/dim]")

                    # State info
                    if sync_state:
                        console.print(f"\n  State version: {sync_state.version}")
                        console.print(f"  Device ID: {sync_state.device_id[:8]}...")
                    console.print()
                    continue

                if sync_cmd == "discover":
                    if not discovery or not discovery.is_running:
                        console.print("[dim]Discovery not running. Start with /sync start[/dim]")
                        continue

                    devices = discovery.get_devices()
                    if not devices:
                        console.print("[dim]No devices discovered yet[/dim]")
                        continue

                    console.print("\n[bold]Discovered Devices[/bold]\n")
                    for device in devices:
                        console.print(
                            f"  [{device.device_id[:8]}] {device.name}\n"
                            f"    Address: {device.address}\n"
                            f"    Version: {device.version}"
                        )
                    console.print()
                    continue

                if sync_cmd.startswith("connect "):
                    target = sync_cmd[8:].strip()
                    if not target:
                        console.print(
                            "[red]Usage: /sync connect <ws://host:port or device_id>[/red]"
                        )
                        continue

                    if not sync_state:
                        sync_state = SyncableState(config.data_dir)

                    # Determine if target is device_id or address
                    address = target
                    if not target.startswith("ws://"):
                        # Look up device by ID
                        if discovery and discovery.is_running:
                            for device in discovery.get_devices():
                                if (
                                    device.device_id.startswith(target)
                                    or device.device_id == target
                                ):
                                    address = device.address
                                    break
                            else:
                                console.print(f"[red]Device not found: {target}[/red]")
                                continue
                        else:
                            console.print(
                                "[red]Provide full address (ws://host:port) or start discovery first[/red]"
                            )
                            continue

                    # Prompt for shared secret
                    secret = prompt("Shared secret from target device: ").strip()
                    if not secret:
                        console.print("[red]Shared secret required for authentication[/red]")
                        continue

                    sync_client = SyncClient(state=sync_state, shared_secret=secret)

                    console.print(f"[dim]Connecting to {address}...[/dim]")
                    success = asyncio.run(sync_client.connect(address))

                    if success:
                        console.print(f"[green]Connected to {sync_client.peer_device_name}[/green]")
                    else:
                        console.print("[red]Connection failed[/red]")
                        sync_client = None
                    continue

                if sync_cmd == "disconnect":
                    if not sync_client or not sync_client.is_connected:
                        console.print("[dim]Not connected to any peer[/dim]")
                        continue

                    peer_name = sync_client.peer_device_name
                    asyncio.run(sync_client.disconnect())
                    sync_client = None
                    console.print(f"[yellow]Disconnected from {peer_name}[/yellow]")
                    continue

                if sync_cmd == "now":
                    if not sync_client or not sync_client.is_connected:
                        console.print(
                            "[dim]Not connected to any peer. Use /sync connect first[/dim]"
                        )
                        continue

                    console.print("[dim]Syncing...[/dim]")
                    result = asyncio.run(sync_client.sync())

                    if result.success:
                        console.print(
                            f"[green]Sync complete[/green]\n"
                            f"  Sent: {result.changes_sent} changes\n"
                            f"  Received: {result.changes_received} changes\n"
                            f"  Duration: {result.duration_ms}ms"
                        )
                    else:
                        console.print(f"[red]Sync failed: {result.error}[/red]")
                    continue

                if sync_cmd == "pair":
                    if sync_server and sync_server.is_running:
                        console.print(
                            f"\n[bold]Pairing Information[/bold]\n\n"
                            f"  Share this secret with the device you want to sync:\n"
                            f"  [cyan]{sync_server.shared_secret}[/cyan]\n"
                        )
                    else:
                        console.print("[dim]Start sync server first with /sync start[/dim]")
                    continue

                # Unknown sync command
                console.print(
                    "[red]Unknown sync command. Use: start, stop, status, discover, connect, disconnect, now, pair[/red]"
                )
                continue

            # =========================================================
            # End Phase 6 commands
            # =========================================================

            # Reasoning mode with auto-detection
            mode = detect_mode(user_input)
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

            # Speak response if TTS enabled
            if voice and voice.response_tts_enabled:
                voice.speak(response, async_=True)

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
