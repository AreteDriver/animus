#!/usr/bin/env python3
"""Animus agent interface — local LLM with tools (read, write, run)."""

from __future__ import annotations

import json
import os
import re
import readline  # noqa: F401 — enables arrow keys, history in input()
import subprocess
import sys
import urllib.request

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
ANIMUS_ROOT = os.path.expanduser("~/projects/animus")
AUTO_APPROVE = False  # When True, executes tool calls without asking

# Ollama native tool calling schemas
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to ~/projects/animus/",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates or overwrites)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace specific text in a file (find and replace)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old_text": {"type": "string", "description": "Exact text to find"},
                    "new_text": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]

SYSTEM_PROMPT_TEMPLATE = """\
You are Animus, an AI agent. You ACT on code — do not explain or advise.

CRITICAL: When asked to do something, MUST use tools. Never describe what you "would" do.
If you catch yourself writing "you could" or "I would suggest" — STOP. Use a tool.

You have these tools: read_file, run_command, write_file, edit_file.
Use them via function calls. The system will execute them and return results.

## Correct behavior examples

User: "add type hints to memory.py"
WRONG: "Here's how you could add type hints..."
RIGHT: Call read_file("packages/core/animus/memory.py"), then edit_file to add hints.

User: "run the tests"
WRONG: "You can run tests with pytest..."
RIGHT: Call run_command with "cd packages/core && source .venv/bin/activate && pytest tests/ -x -q"

User: "what does cognitive.py do?"
RIGHT: Call read_file with "packages/core/animus/cognitive.py", then explain based on what you read.

## Rules
- ALWAYS read before editing
- ONE change at a time, then test
- Paths are relative to ~/projects/animus/
- After edits, run: ruff check packages/ --fix && ruff format packages/
- Never push to git, never delete files, never change public API signatures

## Project
{file_tree}

Forge API: localhost:8000. Python 3.10+. Ruff linting. pytest testing.
Each package has its own .venv in packages/<pkg>/.venv/

Test commands:
- Quorum: cd packages/quorum && source .venv/bin/activate && pytest tests/ -x -q
- Core: cd packages/core && source .venv/bin/activate && pytest tests/ -x -q
- Forge: cd packages/forge && source .venv/bin/activate && pytest tests/ -x -q
"""


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
        # List top-level .py files and subdirs
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
            for f in files[:15]:  # Cap at 15 to save context
                tree_lines.append(f"    {f}")
            if len(files) > 15:
                tree_lines.append(f"    ... +{len(files) - 15} more")
        except OSError:
            pass
    return "\n".join(tree_lines)


def get_system_prompt() -> str:
    """Build system prompt with live file tree."""
    return SYSTEM_PROMPT_TEMPLATE.format(file_tree=build_file_tree())


CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
BOLD = "\033[1m"
NC = "\033[0m"

# Tool call patterns
RE_READ = re.compile(r"<tool:read>(.*?)</tool:read>", re.DOTALL)
RE_RUN = re.compile(r"<tool:run>(.*?)</tool:run>", re.DOTALL)
RE_WRITE = re.compile(r'<tool:write\s+path="(.*?)">(.*?)</tool:write>', re.DOTALL)
RE_EDIT = re.compile(
    r'<tool:edit\s+path="(.*?)">\s*<<<< OLD\n(.*?)\n====\n(.*?)\n>>>> NEW\s*</tool:edit>',
    re.DOTALL,
)

history: list[dict] = []
MAX_AGENT_LOOPS = 8  # Max tool-use rounds per user message


def resolve_path(path: str) -> str:
    """Resolve a path relative to ANIMUS_ROOT."""
    path = path.strip()
    if path.startswith("/"):
        return path
    if path.startswith("~/"):
        return os.path.expanduser(path)
    return os.path.join(ANIMUS_ROOT, path)


def confirm(action: str) -> bool:
    """Ask user to confirm an action. Returns True if approved."""
    if AUTO_APPROVE:
        return True
    try:
        resp = input(f"  {YELLOW}Execute? {NC}{DIM}[Y/n]{NC} ").strip().lower()
        return resp in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def exec_read(path: str) -> str:
    """Read a file and return contents."""
    full_path = resolve_path(path)
    print(f"  {DIM}READ {full_path}{NC}")
    try:
        with open(full_path) as f:
            content = f.read()
        lines = content.splitlines()
        # Truncate very large files
        if len(lines) > 300:
            content = "\n".join(lines[:300]) + f"\n\n... ({len(lines) - 300} more lines truncated)"
        return f"Contents of {path} ({len(lines)} lines):\n```\n{content}\n```"
    except FileNotFoundError:
        return f"ERROR: File not found: {full_path}"
    except PermissionError:
        return f"ERROR: Permission denied: {full_path}"


def exec_run(command: str) -> str:
    """Run a shell command and return output."""
    command = command.strip()
    print(f"  {DIM}RUN {command}{NC}")
    if not confirm(command):
        return "SKIPPED: User declined to run this command."
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=ANIMUS_ROOT,
        timeout=120,
    )
    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += result.stderr
    if not output:
        output = "(no output)"
    # Truncate massive output
    if len(output) > 5000:
        output = output[:5000] + "\n... (truncated)"
    status = "OK" if result.returncode == 0 else f"EXIT CODE {result.returncode}"
    return f"[{status}]\n{output}"


def exec_write(path: str, content: str) -> str:
    """Write content to a file."""
    full_path = resolve_path(path)
    content = content.strip("\n") + "\n"
    lines = content.splitlines()
    print(f"  {DIM}WRITE {full_path} ({len(lines)} lines){NC}")
    if not confirm(f"Write {len(lines)} lines to {path}"):
        return "SKIPPED: User declined to write this file."
    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        return f"OK: Wrote {len(lines)} lines to {path}"
    except PermissionError:
        return f"ERROR: Permission denied: {full_path}"


def exec_edit(path: str, old: str, new: str) -> str:
    """Edit a file by replacing old text with new text."""
    full_path = resolve_path(path)
    print(f"  {DIM}EDIT {full_path}{NC}")
    print(f"  {RED}- {old.splitlines()[0]}{NC}")
    print(f"  {GREEN}+ {new.splitlines()[0]}{NC}")
    if len(old.splitlines()) > 1:
        print(f"  {DIM}  ({len(old.splitlines())} lines -> {len(new.splitlines())} lines){NC}")
    if not confirm(f"Edit {path}"):
        return "SKIPPED: User declined this edit."
    try:
        with open(full_path) as f:
            content = f.read()
        if old not in content:
            return (
                f"ERROR: Could not find the OLD text in {path}. "
                "Read the file first to get exact content."
            )
        count = content.count(old)
        if count > 1:
            return (
                f"ERROR: OLD text matches {count} locations in {path}. "
                "Provide more context to make it unique."
            )
        content = content.replace(old, new, 1)
        with open(full_path, "w") as f:
            f.write(content)
        return f"OK: Edited {path}"
    except FileNotFoundError:
        return f"ERROR: File not found: {full_path}"


def extract_and_execute_tools(response: str) -> str | None:
    """Find tool calls in the response, execute them, return combined results."""
    results = []

    for match in RE_READ.finditer(response):
        results.append(exec_read(match.group(1)))

    for match in RE_RUN.finditer(response):
        results.append(exec_run(match.group(1)))

    for match in RE_WRITE.finditer(response):
        results.append(exec_write(match.group(1), match.group(2)))

    for match in RE_EDIT.finditer(response):
        results.append(exec_edit(match.group(1), match.group(2), match.group(3)))

    if results:
        return "\n\n".join(results)
    return None


VALID_TOOL_NAMES = {"read_file", "run_command", "write_file", "edit_file"}

# Matches JSON objects with "name" and "arguments" keys embedded in text
RE_INLINE_JSON = re.compile(
    r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})\s*\}',
)


def extract_inline_json_tool_calls(response: str) -> list[dict] | None:
    """Parse tool calls written as JSON in text (models that understand
    the schema but output calls as text instead of native tool_calls)."""
    calls = []
    for match in RE_INLINE_JSON.finditer(response):
        name = match.group(1)
        if name not in VALID_TOOL_NAMES:
            continue
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        calls.append({"function": {"name": name, "arguments": args}})
    return calls or None


def execute_native_tool_calls(tool_calls: list[dict]) -> str | None:
    """Execute native Ollama tool calls, return combined results."""
    results = []
    for call in tool_calls:
        fn = call.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        # Arguments may arrive as a JSON string from some models
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                results.append(f"ERROR: Could not parse arguments for {name}: {args}")
                continue

        if name == "read_file":
            results.append(exec_read(args.get("path", "")))
        elif name == "run_command":
            results.append(exec_run(args.get("command", "")))
        elif name == "write_file":
            results.append(exec_write(args.get("path", ""), args.get("content", "")))
        elif name == "edit_file":
            results.append(
                exec_edit(
                    args.get("path", ""),
                    args.get("old_text", ""),
                    args.get("new_text", ""),
                )
            )
        else:
            results.append(f"ERROR: Unknown tool '{name}'")

    return "\n\n".join(results) if results else None


def ollama_chat(user_message: str, role: str = "user") -> dict:
    """Send a message to Ollama and stream the response.

    Returns {"content": str, "tool_calls": list[dict] | None}.
    """
    history.append({"role": role, "content": user_message})

    messages = [{"role": "system", "content": get_system_prompt()}, *history[-20:]]

    payload: dict = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "tools": TOOLS,
    }

    data = json.dumps(payload).encode()

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    full_response: list[str] = []
    tool_calls: list[dict] = []
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                # Stream text content
                token = msg.get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full_response.append(token)
                # Accumulate tool calls from chunks
                chunk_tools = msg.get("tool_calls")
                if chunk_tools:
                    tool_calls.extend(chunk_tools)
                if chunk.get("done"):
                    break
    except Exception as e:
        error_msg = f"\n{YELLOW}[Ollama error: {e}]{NC}"
        print(error_msg)
        return {"content": "", "tool_calls": None}

    print()
    response_text = "".join(full_response)
    # Store assistant message in history (include tool_calls metadata if present)
    assistant_msg: dict = {"role": "assistant", "content": response_text}
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    history.append(assistant_msg)
    return {"content": response_text, "tool_calls": tool_calls or None}


WAFFLE_PHRASES = [
    "here's how you could",
    "you can do this by",
    "i would suggest",
    "i would recommend",
    "here are the steps",
    "you could try",
    "you should consider",
    "the approach would be",
    "to accomplish this",
    "follow these steps",
    "here is a general",
    "you'll need to",
    "first, you need to",
    "let me outline",
    "i'll walk you through",
    # Models that describe tool calls instead of making them
    "this could involve",
    "we would need to",
    "we can break down",
    "here's an example workflow",
    "each of these steps",
    "implement a script that",
    "set up monitoring",
    "develop a mechanism",
    "given these steps",
]


def is_waffle(response: str) -> bool:
    """Detect if the model is explaining instead of acting."""
    lower = response.lower()
    # If it used tools, it's not waffling
    if "<tool:" in response:
        return False
    # Check for waffle phrases
    return any(phrase in lower for phrase in WAFFLE_PHRASES)


def agent_loop(user_message: str) -> None:
    """Run the agent loop: send message, execute tools, feed results back."""
    response = ollama_chat(user_message)
    content = response["content"]

    # Anti-waffle: if the model explains instead of acting (text only, no tool calls)
    if not response.get("tool_calls") and is_waffle(content):
        print(f"\n{YELLOW}[Nudging — model explained instead of acting]{NC}\n")
        response = ollama_chat(
            "STOP. You just explained what to do instead of doing it. "
            "Use your tools NOW. Call read_file or run_command. "
            "Do not explain. Act."
        )

    for _ in range(MAX_AGENT_LOOPS):
        # Try native tool calls first
        if response.get("tool_calls"):
            tool_results = execute_native_tool_calls(response["tool_calls"])
        else:
            # Try inline JSON (models that output tool calls as text)
            inline = extract_inline_json_tool_calls(response["content"])
            if inline:
                tool_results = execute_native_tool_calls(inline)
            else:
                # Fall back to XML parsing
                tool_results = extract_and_execute_tools(response["content"])

        if tool_results is None:
            break  # No tool calls — agent is done

        print(f"\n{DIM}--- tool results ---{NC}")
        # Feed results back as a tool role message
        response = ollama_chat(
            f"Tool results:\n\n{tool_results}\n\n"
            "Continue with the task. Use more tools if needed, or summarize what you did.",
            role="tool",
        )

    print()


def handle_slash_command(cmd: str) -> bool:
    """Handle special /commands. Returns True if handled."""
    global OLLAMA_MODEL, AUTO_APPROVE
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("/quit", "/exit", "/q"):
        print(f"\n{DIM}Goodbye.{NC}")
        sys.exit(0)

    elif command == "/status":
        print(f"\n{BOLD}Forge API:{NC}")
        try:
            with urllib.request.urlopen("http://localhost:8000/health", timeout=3) as r:
                health = json.loads(r.read())
                print(f"  {GREEN}healthy{NC} — {health.get('timestamp', '?')}")
        except Exception:
            print(f"  {YELLOW}not responding{NC}")

        print(f"\n{BOLD}Systemd:{NC}")
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "animus-forge"],
            capture_output=True,
            text=True,
        )
        svc_status = result.stdout.strip()
        color = GREEN if svc_status == "active" else YELLOW
        print(f"  {color}{svc_status}{NC}")

        print(f"\n{BOLD}Ollama ({OLLAMA_MODEL}):{NC}")
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        for line in result.stdout.strip().split("\n"):
            print(f"  {line}")
        print(f"\n{BOLD}Agent mode:{NC} auto-approve={'ON' if AUTO_APPROVE else 'OFF'}")
        print()
        return True

    elif command == "/logs":
        print(f"{DIM}Tailing Forge logs (Ctrl+C to stop)...{NC}\n")
        try:
            subprocess.run(
                ["journalctl", "--user", "-u", "animus-forge", "-f", "--no-pager", "-n", "20"],
            )
        except KeyboardInterrupt:
            print()
        return True

    elif command == "/model":
        if arg:
            OLLAMA_MODEL = arg
            print(f"Switched to {CYAN}{OLLAMA_MODEL}{NC}")
        else:
            print(f"Current model: {CYAN}{OLLAMA_MODEL}{NC}")
            print(f"Usage: {DIM}/model llama3.1:8b{NC}")
        return True

    elif command == "/auto":
        AUTO_APPROVE = not AUTO_APPROVE
        state = "ON" if AUTO_APPROVE else "OFF"
        color = GREEN if AUTO_APPROVE else YELLOW
        print(f"Auto-approve: {color}{state}{NC}")
        if AUTO_APPROVE:
            print(f"{YELLOW}  Tools will execute without confirmation. Use with care.{NC}")
        return True

    elif command == "/clear":
        history.clear()
        print(f"{DIM}Conversation cleared.{NC}")
        return True

    elif command in ("/review", "/read"):
        if not arg:
            print(f"Usage: {DIM}/review <filepath>{NC}")
            return True
        filepath = resolve_path(arg)
        try:
            with open(filepath) as f:
                code = f.read()
            print(f"{DIM}Reading {filepath} ({len(code.splitlines())} lines)...{NC}\n")
            agent_loop(
                f"Review this file for bugs, missing error handling, and missing type hints. "
                f"For each issue: line number, what's wrong, exact fix. No style opinions.\n\n"
                f"File: {arg}\n\n```python\n{code}\n```"
            )
        except FileNotFoundError:
            print(f"{YELLOW}File not found: {filepath}{NC}")
        return True

    elif command == "/test":
        if not arg:
            print(f"Usage: {DIM}/test <filepath>{NC}")
            return True
        filepath = resolve_path(arg)
        try:
            with open(filepath) as f:
                code = f.read()
            print(f"{DIM}Generating tests for {filepath}...{NC}\n")
            agent_loop(
                f"Write pytest tests for this module. Mock all external calls. "
                f"Use asyncio.run() for async (no pytest-asyncio). "
                f"Include happy path, edge cases, errors. Output only code.\n\n"
                f"Module: {arg}\n\n```python\n{code}\n```"
            )
        except FileNotFoundError:
            print(f"{YELLOW}File not found: {filepath}{NC}")
        return True

    elif command == "/run":
        if not arg:
            print(f"Usage: {DIM}/run <shell command>{NC}")
            return True
        print(f"{DIM}$ {arg}{NC}")
        result = subprocess.run(arg, shell=True, capture_output=True, text=True, cwd=ANIMUS_ROOT)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"{YELLOW}{result.stderr}{NC}")
        return True

    elif command in ("/help", "/?"):
        print(f"""
{BOLD}Talk naturally — Ollama acts as an agent with tools.{NC}

  It can read files, edit code, and run commands.
  You approve each action before it executes (unless /auto is on).

{BOLD}Examples:{NC}
  {CYAN}review cognitive.py for bugs and fix them{NC}
  {CYAN}run the forge tests and fix any failures{NC}
  {CYAN}add type hints to packages/core/animus/memory.py{NC}
  {CYAN}what does the supervisor agent do?{NC}

{BOLD}Slash commands:{NC}
  {CYAN}/status{NC}           Service + Ollama health
  {CYAN}/review <file>{NC}    AI code review (reads file automatically)
  {CYAN}/test <file>{NC}      Generate tests for a module
  {CYAN}/model <name>{NC}     Switch Ollama model
  {CYAN}/auto{NC}             Toggle auto-approve (skip confirmations)
  {CYAN}/run <cmd>{NC}        Run a shell command directly
  {CYAN}/logs{NC}             Tail Forge service logs
  {CYAN}/clear{NC}            Reset conversation history
  {CYAN}/help{NC}             This message
  {CYAN}/quit{NC}             Exit
""")
        return True

    return False


def main() -> None:
    print(f"""{CYAN}{BOLD}
    ___          _
   / _ \\        (_)
  / /_\\ \\ _ __   _ _ __ ___  _   _ ___
  |  _  || '_ \\ | | '_ ` _ \\| | | / __|
  | | | || | | || | | | | | | |_| \\__ \\
  \\_| |_/|_| |_||_|_| |_| |_|\\__,_|___/
{NC}""")

    # Status check
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=2) as _:
            print(f"  Forge API: {GREEN}running{NC}  |  Model: {CYAN}{OLLAMA_MODEL}{NC}")
    except Exception:
        print(f"  Forge API: {YELLOW}offline{NC}  |  Model: {CYAN}{OLLAMA_MODEL}{NC}")

    print("  Talk naturally. I can read files, edit code, and run commands.")
    print(f"  Type {DIM}/help{NC} for commands, {DIM}/quit{NC} to exit.\n")

    while True:
        try:
            user_input = input(f"{CYAN}>{NC} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye.{NC}")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if handle_slash_command(user_input):
                continue

        # Agent loop: send to Ollama, execute tools, feed back results
        print()
        agent_loop(user_input)


if __name__ == "__main__":
    main()
