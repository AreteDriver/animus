"""Slash command registry and handlers."""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from collections.abc import Callable, Coroutine
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from animus_forge.tui.app import GorgonApp

logger = logging.getLogger(__name__)

# File-name patterns that likely contain secrets — checked before attaching.
_SENSITIVE_PATTERNS: list[str] = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*credentials*",
    "*secret*",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "known_hosts",
    "authorized_keys",
]

_SENSITIVE_DIRS: set[str] = {".ssh", ".aws", ".azure", ".gcp", ".gnupg"}


def _is_sensitive_path(path: Path) -> bool:
    """Return True if *path* looks like it contains secrets."""
    name_lower = path.name.lower()
    for pattern in _SENSITIVE_PATTERNS:
        if PurePath(name_lower).match(pattern):
            return True
    for part in path.parts:
        if part.lower() in _SENSITIVE_DIRS:
            return True
    return False


def _clipboard_copy(text: str) -> str:
    """Copy *text* to the system clipboard.  Cross-platform."""
    system = platform.system()
    if system == "Darwin":
        cmd = ["pbcopy"]
    elif system == "Windows":
        cmd = ["clip"]
    else:
        # Linux / BSD — prefer xclip, fall back to xsel
        xclip = shutil.which("xclip")
        if xclip:
            cmd = [xclip, "-selection", "clipboard"]
        else:
            xsel = shutil.which("xsel")
            if xsel:
                cmd = [xsel, "--clipboard", "--input"]
            else:
                return "No clipboard utility found. Install xclip (apt install xclip) or xsel."

    try:
        proc = subprocess.run(
            cmd,
            input=text.encode(),
            timeout=5,
            capture_output=True,
        )
        if proc.returncode == 0:
            return "Last response copied to clipboard."
        logger.debug("Clipboard tool stderr: %s", proc.stderr.decode(errors="replace"))
        return f"Copy failed ({cmd[0]} returned {proc.returncode})."
    except subprocess.TimeoutExpired:
        return f"Copy failed ({cmd[0]} timed out)."
    except FileNotFoundError:
        return f"{cmd[0]} not found."
    except Exception as e:
        logger.error("Clipboard copy failed: %s", e)
        return f"Copy failed: {type(e).__name__}"


class CommandRegistry:
    """Registry for slash commands with async handlers."""

    def __init__(self, app: GorgonApp):
        self.app = app
        self._commands: dict[str, tuple[Callable, str]] = {}

    def register(
        self,
        name: str,
        handler: Callable[[str], Coroutine[Any, Any, str | None]],
        help_text: str = "",
    ) -> None:
        self._commands[name] = (handler, help_text)

    async def execute(self, name: str, args: str = "") -> str | None:
        if name not in self._commands:
            return f"Unknown command: /{name}. Type /help for available commands."
        handler, _ = self._commands[name]
        try:
            return await handler(args)
        except Exception as e:
            logger.error(f"Command /{name} failed: {e}")
            return f"Error: {e}"

    def get_completions(self, prefix: str) -> list[str]:
        return [f"/{name}" for name in self._commands if name.startswith(prefix)]

    def get_help(self) -> str:
        lines = ["Available commands:"]
        for name, (_, help_text) in sorted(self._commands.items()):
            lines.append(f"  /{name:<12s} {help_text}")
        return "\n".join(lines)


def create_command_registry(app: GorgonApp) -> CommandRegistry:
    """Create and populate the command registry with all handlers."""
    registry = CommandRegistry(app)

    # Keep a set of paths the user has explicitly confirmed as OK.
    _confirmed_sensitive: set[str] = set()

    async def cmd_help(_args: str) -> str:
        return registry.get_help()

    async def cmd_quit(_args: str) -> str | None:
        app.exit()
        return None

    async def cmd_clear(_args: str) -> str | None:
        app.action_clear_chat()
        return None

    async def cmd_switch(args: str) -> str:
        if not args:
            providers = app._provider_manager.list_providers() if app._provider_manager else []
            return f"Usage: /switch <provider>\nAvailable: {', '.join(providers)}"
        if not app._provider_manager:
            return "No provider manager available."
        try:
            app._provider_manager.set_default(args.strip())
            provider = app._provider_manager.get_default()
            if provider:
                cs = app._get_chat_screen()
                if cs:
                    cs.sidebar.provider_name = provider.name
                    cs.sidebar.model_name = provider.default_model
                    cs.status_bar.provider_name = provider.name
                    cs.status_bar.model_name = provider.default_model
                return f"Switched to {provider.name} ({provider.default_model})"
            return "Switch failed."
        except Exception as e:
            return f"Switch failed: {e}"

    async def cmd_model(args: str) -> str:
        if not args:
            return "Usage: /model <name>"
        if not app._provider_manager:
            return "No provider manager available."
        provider = app._provider_manager.get_default()
        if not provider:
            return "No default provider."
        provider.config.default_model = args.strip()
        cs = app._get_chat_screen()
        if cs:
            cs.sidebar.model_name = args.strip()
            cs.status_bar.model_name = args.strip()
        return f"Model set to {args.strip()}"

    async def cmd_models(_args: str) -> str:
        if not app._provider_manager:
            return "No provider manager available."
        provider = app._provider_manager.get_default()
        if not provider:
            return "No default provider."
        models = provider.list_models()
        return f"Models for {provider.name}:\n" + "\n".join(f"  {m}" for m in models)

    async def cmd_providers(_args: str) -> str:
        if not app._provider_manager:
            return "No provider manager available."
        names = app._provider_manager.list_providers()
        default = app._provider_manager._default_provider
        lines = []
        for name in names:
            marker = " *" if name == default else ""
            lines.append(f"  {name}{marker}")
        return "Registered providers (* = active):\n" + "\n".join(lines)

    async def cmd_system(args: str) -> str:
        if not args:
            current = app._system_prompt or "(none)"
            return f"System prompt: {current}"
        app._system_prompt = args
        return f"System prompt set ({len(args)} chars)"

    async def cmd_tokens(_args: str) -> str:
        cs = app._get_chat_screen()
        if cs:
            sb = cs.sidebar
            return (
                f"Token usage:\n"
                f"  Input:  {sb.input_tokens:,}\n"
                f"  Output: {sb.output_tokens:,}\n"
                f"  Total:  {sb.input_tokens + sb.output_tokens:,}"
            )
        return "Token data unavailable."

    async def cmd_file(args: str) -> str:
        if not args:
            return "Usage: /file <path>"

        raw = args.strip()

        # Support "confirm:<path>" to bypass sensitive-file warning
        force = False
        if raw.startswith("confirm:"):
            raw = raw[len("confirm:") :]
            force = True

        path = Path(raw).expanduser().resolve()
        if not path.exists():
            return f"File not found: {path}"
        if not path.is_file():
            return f"Not a file: {path}"

        # Warn about sensitive files (keys, .env, credentials, etc.)
        if not force and str(path) not in _confirmed_sensitive:
            if _is_sensitive_path(path):
                _confirmed_sensitive.add(str(path))
                return (
                    f"Warning: '{path.name}' looks like a sensitive file "
                    f"(credentials, keys, env vars).\n"
                    f"Its contents will be sent to an external AI API.\n"
                    f"To proceed anyway: /file confirm:{path}"
                )

        cs = app._get_chat_screen()
        if cs:
            cs.sidebar.add_file(str(path))
        return f"Added: {path}"

    async def cmd_files(_args: str) -> str:
        cs = app._get_chat_screen()
        if cs:
            files = cs.sidebar.files
            if not files:
                return "No files attached."
            return "Attached files:\n" + "\n".join(f"  {f}" for f in files)
        return "No sidebar."

    async def cmd_unfile(args: str) -> str:
        if not args:
            return "Usage: /unfile <path|all>"
        cs = app._get_chat_screen()
        if not cs:
            return "No sidebar."
        if args.strip() == "all":
            cs.sidebar.clear_files()
            return "All files removed."
        cs.sidebar.remove_file(args.strip())
        return f"Removed: {args.strip()}"

    async def cmd_agent(args: str) -> str:
        if not args:
            return (
                f"Agent mode: {app._agent_mode}\n"
                "Usage: /agent <off|auto|planner|builder|tester|reviewer|architect|documenter|analyst>"
            )
        mode = args.strip().lower()
        valid = {
            "off",
            "auto",
            "planner",
            "builder",
            "tester",
            "reviewer",
            "architect",
            "documenter",
            "analyst",
        }
        if mode not in valid:
            return f"Invalid mode: {mode}. Options: {', '.join(sorted(valid))}"
        app._agent_mode = mode
        app._supervisor = None  # Reset supervisor for mode change
        cs = app._get_chat_screen()
        if cs:
            cs.sidebar.agent_mode = mode
        return f"Agent mode: {mode}"

    async def cmd_save(args: str) -> str:
        from animus_forge.tui.session import TUISession

        title = args.strip() if args.strip() else None
        provider = ""
        model = ""
        if app._provider_manager:
            p = app._provider_manager.get_default()
            if p:
                provider = p.name
                model = p.default_model

        session = TUISession.create(
            provider=provider,
            model=model,
            system_prompt=app._system_prompt,
            messages=app._messages,
        )
        if title:
            session.title = title
        session.save()
        app._session = session
        return f"Session saved: {session.filepath.name}"

    async def cmd_load(_args: str) -> str:
        from animus_forge.tui.session import TUISession

        sessions = TUISession.list_sessions()
        if not sessions:
            return "No saved sessions."

        # If a numeric index was given, load that session directly.
        if _args.strip().isdigit():
            idx = int(_args.strip())
            recent = sessions[-10:]
            if 0 <= idx < len(recent):
                session = TUISession.load(recent[idx])
                app._session = session
                app._messages = session.messages
                app._system_prompt = session.system_prompt
                cs = app._get_chat_screen()
                if cs:
                    cs.chat_display.clear()
                    cs.chat_display._message_buffer.clear()
                    for msg in session.messages:
                        if msg["role"] == "user":
                            cs.chat_display.add_user_message(msg["content"])
                        elif msg["role"] == "assistant":
                            cs.chat_display.add_assistant_message(msg["content"])
                return f"Loaded: {recent[idx].name}"
            return f"Invalid index: {idx}"

        # Otherwise list recent sessions.
        lines = ["Saved sessions:"]
        for i, s in enumerate(sessions[-10:]):
            lines.append(f"  {i}: {s.name}")
        lines.append("\nUse /load <number> to load a session.")
        return "\n".join(lines)

    async def cmd_history(_args: str) -> str:
        from animus_forge.tui.session import TUISession

        sessions = TUISession.list_sessions()
        if not sessions:
            return "No saved sessions."
        lines = ["Session history:"]
        for s in sessions[-20:]:
            lines.append(f"  {s.name}")
        return "\n".join(lines)

    async def cmd_title(args: str) -> str:
        if not args:
            return "Usage: /title <name>"
        if app._session:
            app._session.title = args.strip()
            app._session.save()
            return f"Title set: {args.strip()}"
        return "No active session. Use /save first."

    async def cmd_copy(_args: str) -> str:
        if not app._messages:
            return "No messages to copy."
        last_assistant = None
        for msg in reversed(app._messages):
            if msg["role"] == "assistant":
                last_assistant = msg["content"]
                break
        if not last_assistant:
            return "No assistant response to copy."
        return _clipboard_copy(last_assistant)

    # Register all commands
    registry.register("help", cmd_help, "Show available commands")
    registry.register("quit", cmd_quit, "Exit Gorgon")
    registry.register("q", cmd_quit, "Exit Gorgon (alias)")
    registry.register("clear", cmd_clear, "Clear chat history")
    registry.register("switch", cmd_switch, "Switch provider: /switch <name>")
    registry.register("model", cmd_model, "Set model: /model <name>")
    registry.register("models", cmd_models, "List available models")
    registry.register("providers", cmd_providers, "List registered providers")
    registry.register("system", cmd_system, "Set system prompt: /system <text>")
    registry.register("tokens", cmd_tokens, "Show token usage")
    registry.register("file", cmd_file, "Attach file: /file <path>")
    registry.register("files", cmd_files, "List attached files")
    registry.register("unfile", cmd_unfile, "Remove file: /unfile <path|all>")
    registry.register("agent", cmd_agent, "Set agent mode: /agent <off|auto|role>")
    registry.register("save", cmd_save, "Save session: /save [name]")
    registry.register("load", cmd_load, "Load session: /load [number]")
    registry.register("history", cmd_history, "Show session history")
    registry.register("title", cmd_title, "Set session title: /title <name>")
    registry.register("copy", cmd_copy, "Copy last response to clipboard")

    return registry
