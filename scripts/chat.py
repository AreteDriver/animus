#!/usr/bin/env python3
"""Animus chat agent — thin wrapper over Animus Core."""

from __future__ import annotations

import os
import sys

# Ensure animus package is importable from monorepo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "core"))

from animus.cognitive import (
    CognitiveLayer,
    ModelConfig,
    detect_mode,
)
from animus.config import AnimusConfig
from animus.logging import setup_logging
from animus.memory import Conversation, MemoryLayer
from animus.tools import create_default_registry, create_memory_tools

# ANSI colors
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
BOLD = "\033[1m"
NC = "\033[0m"

ANIMUS_ROOT = os.path.expanduser("~/projects/animus")
MAX_AGENT_LOOPS = 8
AUTO_SAVE_INTERVAL = 10  # Save conversation every N messages

AGENT_CONTEXT = """\
You are Animus, an AI agent. You ACT on code — do not explain or advise.

CRITICAL: When asked to do something, MUST use tools. Never describe what you "would" do.
If you catch yourself writing "you could" or "I would suggest" — STOP. Use a tool instead.

Rules:
- ALWAYS read a file before editing it
- ONE change at a time, then verify
- Paths are relative to ~/projects/animus/ or absolute
- After edits, run: ruff check packages/ --fix && ruff format packages/
- Never push to git, never delete files without asking"""


def build_file_tree() -> str:
    """Build a real file tree of the project for context."""
    tree_lines = ["~/projects/animus/"]
    for pkg_name, pkg_import, pkg_path in [
        ("core", "animus", "packages/core/animus"),
        ("forge", "animus_forge", "packages/forge/src/animus_forge"),
        ("quorum", "convergent", "packages/quorum/python/convergent"),
    ]:
        full = os.path.join(ANIMUS_ROOT, pkg_path)
        if not os.path.isdir(full):
            continue
        tree_lines.append(f"  packages/{pkg_name}/ (import {pkg_import})")
        try:
            entries = sorted(os.listdir(full))
            dirs = [
                e
                for e in entries
                if os.path.isdir(os.path.join(full, e)) and not e.startswith(("__", "."))
            ]
            files = [e for e in entries if e.endswith(".py") and e != "__init__.py"]
            for d in dirs:
                sub_files = [
                    f
                    for f in os.listdir(os.path.join(full, d))
                    if f.endswith(".py") and f != "__init__.py"
                ]
                tree_lines.append(f"    {d}/ ({len(sub_files)} modules)")
            for f in files[:15]:
                tree_lines.append(f"    {f}")
            if len(files) > 15:
                tree_lines.append(f"    ... +{len(files) - 15} more")
        except OSError:
            pass
    return "\n".join(tree_lines)


def build_model_configs() -> tuple[ModelConfig, ModelConfig | None]:
    """Build primary + fallback model configs from env vars."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    ollama_model = os.environ.get("OLLAMA_MODEL", "deepseek-coder-v2")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    ollama_config = ModelConfig.ollama(ollama_model)
    ollama_config.base_url = ollama_host

    if anthropic_key:
        primary = ModelConfig.anthropic("claude-sonnet-4-20250514")
        primary.api_key = anthropic_key
        return primary, ollama_config
    return ollama_config, None


def approval_callback(tool_name: str, params: dict) -> bool:
    """Terminal approval for sensitive tools."""
    if os.environ.get("ANIMUS_AUTO_APPROVE", "").lower() in ("1", "true", "yes"):
        return True
    print(f"  {YELLOW}Tool: {tool_name}{NC}")
    for k, v in params.items():
        preview = str(v)[:100]
        if len(str(v)) > 100:
            preview += "..."
        print(f"    {DIM}{k}: {preview}{NC}")
    try:
        resp = input(f"  {YELLOW}Execute? {NC}{DIM}[Y/n]{NC} ").strip().lower()
        return resp in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def handle_slash(
    cmd: str,
    memory: MemoryLayer,
    cognitive: CognitiveLayer,
    conversation: Conversation,
) -> str | None:
    """Handle slash commands. Returns 'quit', 'clear', or None if not handled."""
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()

    if command in ("/quit", "/exit", "/q"):
        return "quit"

    if command == "/clear":
        print(f"{DIM}Conversation saved and cleared.{NC}")
        return "clear"

    if command == "/status":
        stats = memory.get_statistics()
        print(f"  Memories: {stats.get('total', 0)}  |  Tags: {stats.get('unique_tags', 0)}")
        provider = cognitive.primary_config.provider.value
        model = cognitive.primary_config.model_name
        print(f"  Provider: {provider}  |  Model: {model}")
        if cognitive.fallback_config:
            fb = cognitive.fallback_config
            print(f"  Fallback: {fb.provider.value} / {fb.model_name}")
        auto = os.environ.get("ANIMUS_AUTO_APPROVE", "false")
        print(f"  Auto-approve: {auto}")
        return ""

    if command == "/auto":
        current = os.environ.get("ANIMUS_AUTO_APPROVE", "false")
        new_val = "false" if current.lower() in ("1", "true", "yes") else "true"
        os.environ["ANIMUS_AUTO_APPROVE"] = new_val
        color = GREEN if new_val == "true" else YELLOW
        print(f"Auto-approve: {color}{new_val.upper()}{NC}")
        if new_val == "true":
            print(f"{YELLOW}  Tools will execute without confirmation.{NC}")
        return ""

    if command in ("/help", "/?"):
        print(f"""
{BOLD}Talk naturally. Animus acts as an agent with tools.{NC}

  It can read files, edit code, write files, and run commands.
  You approve each action before it executes (unless /auto is on).

{BOLD}Examples:{NC}
  {CYAN}review cognitive.py for bugs and fix them{NC}
  {CYAN}run the core tests and fix any failures{NC}
  {CYAN}add type hints to packages/core/animus/memory.py{NC}

{BOLD}Slash commands:{NC}
  {CYAN}/status{NC}   Provider, model, and memory info
  {CYAN}/auto{NC}     Toggle auto-approve for tool execution
  {CYAN}/clear{NC}    Save and reset conversation
  {CYAN}/help{NC}     This message
  {CYAN}/quit{NC}     Save and exit
""")
        return ""

    return None  # Not a recognized slash command


def save_conversation(memory: MemoryLayer, conversation: Conversation) -> None:
    """Save conversation to memory if it has messages."""
    if conversation.messages:
        try:
            memory.save_conversation(conversation)
        except Exception as e:
            print(f"{DIM}(Could not save conversation: {e}){NC}")


def main() -> None:
    config = AnimusConfig.load()
    config.ensure_dirs()
    setup_logging(
        log_file=config.log_file if config.log_to_file else None,
        level=config.log_level,
        log_to_file=config.log_to_file,
    )

    # Initialize memory
    memory = MemoryLayer(config.data_dir, backend=config.memory.backend)

    # Build model configs: Claude primary if key available, Ollama fallback
    primary, fallback = build_model_configs()
    cognitive = CognitiveLayer(primary_config=primary, fallback_config=fallback)

    # Build tool registry with all built-in tools + memory tools
    tools = create_default_registry(security_config=config.tools_security)
    for tool in create_memory_tools(memory):
        tools.register(tool)

    # Banner
    print(f"{CYAN}{BOLD}Animus{NC} — {primary.provider.value}/{primary.model_name}")
    if fallback:
        print(f"  Fallback: {fallback.provider.value}/{fallback.model_name}")
    stats = memory.get_statistics()
    print(f"  {stats.get('total', 0)} memories | /help for commands\n")

    conversation = Conversation.new()
    file_tree = build_file_tree()

    while True:
        try:
            user_input = input(f"{CYAN}>{NC} ").strip()
        except (EOFError, KeyboardInterrupt):
            save_conversation(memory, conversation)
            print(f"\n{DIM}Goodbye.{NC}")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            result = handle_slash(user_input, memory, cognitive, conversation)
            if result == "quit":
                save_conversation(memory, conversation)
                print(f"\n{DIM}Goodbye.{NC}")
                break
            if result == "clear":
                save_conversation(memory, conversation)
                conversation = Conversation.new()
                continue
            if result is not None:
                continue

        # Build context: agent personality + file tree + memory recall
        context_parts = [AGENT_CONTEXT, f"\nProject layout:\n{file_tree}"]
        recalled = memory.recall(user_input, limit=3)
        if recalled:
            context_parts.append(
                "\nRelevant memories:\n" + "\n".join(f"- {m.content}" for m in recalled)
            )
        context = "\n".join(context_parts)

        # Track in conversation
        conversation.add_message("user", user_input)

        # Detect reasoning mode and run agent loop
        mode = detect_mode(user_input)
        print()
        response = cognitive.think_with_tools(
            prompt=user_input,
            context=context,
            mode=mode,
            tools=tools,
            max_iterations=MAX_AGENT_LOOPS,
            approval_callback=approval_callback,
        )

        print(response)
        print()

        conversation.add_message("assistant", response)

        # Auto-save periodically
        if len(conversation.messages) >= AUTO_SAVE_INTERVAL:
            save_conversation(memory, conversation)
            conversation = Conversation.new()


if __name__ == "__main__":
    main()
