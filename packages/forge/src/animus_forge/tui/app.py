"""Main Textual App for the Gorgon TUI."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from textual.app import App
from textual.binding import Binding

from animus_forge.tui.chat_screen import ChatScreen
from animus_forge.tui.widgets.input_bar import InputBar

logger = logging.getLogger(__name__)

CSS_PATH = Path(__file__).parent / "styles.tcss"

# Patterns that look like API keys / secrets — used to scrub error messages
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9\-_]{20,}"),  # OpenAI / Anthropic
    re.compile(r"ghp_[A-Za-z0-9]{36,}"),  # GitHub PAT
    re.compile(r"gho_[A-Za-z0-9]{36,}"),  # GitHub OAuth
    re.compile(r"secret_[A-Za-z0-9]{20,}"),  # Notion
    re.compile(r"Bearer\s+[A-Za-z0-9\-_.]{20,}", re.IGNORECASE),
]

# Maximum file size to read into memory for context (1 MB)
_MAX_FILE_READ_BYTES = 1_000_000
# Maximum characters to include per file in the prompt
_MAX_FILE_CONTEXT_CHARS = 10_000


def _sanitize_error(msg: str) -> str:
    """Strip potential secrets from an error string before displaying."""
    for pat in _SECRET_PATTERNS:
        msg = pat.sub("***", msg)
    return msg


class GorgonApp(App):
    """Gorgon TUI - Unified AI terminal interface.

    Unifies Claude, OpenAI, Ollama into a single interactive terminal app
    with streaming, file context, multi-agent orchestration, and session persistence.
    """

    TITLE = "Gorgon"
    SUB_TITLE = "AI Terminal Interface"
    CSS_PATH = CSS_PATH

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+b", "toggle_sidebar", "Toggle Sidebar", show=True),
        Binding("ctrl+l", "clear_chat", "Clear Chat", show=True),
        Binding("ctrl+n", "new_session", "New Session", show=True),
        Binding("ctrl+c", "cancel_generation", "Cancel", show=True, priority=True),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._provider_manager = None
        self._messages: list[dict] = []
        self._system_prompt: str | None = None
        self._cancel_event: asyncio.Event = asyncio.Event()
        self._is_streaming: bool = False
        self._command_registry = None
        self._session = None
        self._agent_mode: str = "off"
        self._supervisor = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_chat_screen(self) -> ChatScreen | None:
        """Return current screen if it is the ChatScreen, else None."""
        screen = self.screen
        return screen if isinstance(screen, ChatScreen) else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self.push_screen(ChatScreen(name="chat"))
        self._init_providers()

    def _init_providers(self) -> None:
        """Initialize provider manager from settings."""
        try:
            from animus_forge.tui.providers import create_provider_manager

            self._provider_manager = create_provider_manager()
            provider = self._provider_manager.get_default()
            if provider:
                cs = self._get_chat_screen()
                if cs:
                    cs.sidebar.provider_name = provider.name
                    cs.sidebar.model_name = provider.default_model
                    cs.status_bar.provider_name = provider.name
                    cs.status_bar.model_name = provider.default_model
                    cs.chat_display.add_system_message(
                        f"Connected to {provider.name} ({provider.default_model}). Type a message or /help."
                    )
            else:
                cs = self._get_chat_screen()
                if cs:
                    cs.chat_display.add_error_message(
                        "No providers configured. Set API keys in .env"
                    )
        except Exception as e:
            logger.error(f"Failed to init providers: {e}")
            cs = self._get_chat_screen()
            if cs:
                cs.chat_display.add_error_message(
                    f"Provider init failed: {_sanitize_error(str(e))}"
                )

    def _init_commands(self) -> None:
        """Lazy-init the command registry."""
        if self._command_registry is None:
            from animus_forge.tui.commands import create_command_registry

            self._command_registry = create_command_registry(self)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    async def on_input_bar_submitted(self, event: InputBar.Submitted) -> None:
        """Handle user input submission."""
        value = event.value

        if event.is_command:
            await self._handle_command(value)
        else:
            await self._handle_chat_message(value)

    async def _handle_command(self, text: str) -> None:
        """Parse and execute a slash command."""
        self._init_commands()
        cs = self._get_chat_screen()
        if not cs:
            return

        parts = text[1:].split(None, 1)
        cmd_name = parts[0].lower() if parts else ""
        cmd_args = parts[1] if len(parts) > 1 else ""

        result = await self._command_registry.execute(cmd_name, cmd_args)
        if result:
            cs.chat_display.add_system_message(result)

    async def _handle_chat_message(self, content: str) -> None:
        """Send a chat message and stream the response."""
        cs = self._get_chat_screen()
        if not cs:
            return

        if not self._provider_manager:
            cs.chat_display.add_error_message("No providers configured.")
            return

        provider = self._provider_manager.get_default()
        if not provider:
            cs.chat_display.add_error_message("No default provider set.")
            return

        # Show user message
        cs.chat_display.add_user_message(content)
        self._messages.append(
            {
                "role": "user",
                "content": content,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        # Stream response
        self._cancel_event.clear()
        self._is_streaming = True
        cs.status_bar.is_streaming = True
        cs.chat_display.begin_assistant_stream()

        full_response = ""
        try:
            from animus_forge.tui.streaming import StreamResult, stream_completion

            # Build a single system prompt (avoids injecting multiple system messages)
            system_parts: list[str] = []
            if self._system_prompt:
                system_parts.append(self._system_prompt)

            file_context = self._build_file_context()
            if file_context:
                system_parts.append(file_context)

            combined_system = "\n\n".join(system_parts) if system_parts else None

            # Build conversation messages (user/assistant only)
            api_messages = [
                {"role": msg["role"], "content": msg["content"]} for msg in self._messages
            ]

            result = StreamResult()
            async for chunk in stream_completion(
                provider, api_messages, system_prompt=combined_system, result=result
            ):
                if self._cancel_event.is_set():
                    cs.chat_display.append_stream_chunk("\n[cancelled]")
                    break
                full_response += chunk
                cs.chat_display.append_stream_chunk(chunk)

            # Update token counts in sidebar
            if result.input_tokens or result.output_tokens:
                cs.sidebar.add_tokens(result.input_tokens, result.output_tokens)

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            cs.chat_display.add_error_message(_sanitize_error(str(e)))
        finally:
            self._is_streaming = False
            cs.status_bar.is_streaming = False
            cs.chat_display.end_assistant_stream(full_response)

        if full_response:
            self._messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

            # Auto-save session if active
            if self._session:
                try:
                    self._session.messages = self._messages
                    self._session.save()
                except Exception as e:
                    logger.debug(f"Auto-save failed: {e}")

    def _build_file_context(self) -> str:
        """Build file context string from attached files.

        Checks file size *before* reading to avoid loading huge files
        into memory.
        """
        cs = self._get_chat_screen()
        if not cs:
            return ""

        files = cs.sidebar.files
        if not files:
            return ""

        parts: list[str] = []
        for fpath in files:
            try:
                p = Path(fpath)
                # Pre-check size to avoid OOM on huge files
                size = p.stat().st_size
                if size > _MAX_FILE_READ_BYTES:
                    parts.append(
                        f"--- File: {fpath} ---\n"
                        f"[Skipped: file too large ({size:,} bytes, limit {_MAX_FILE_READ_BYTES:,})]"
                    )
                    continue

                content = p.read_text(errors="replace")
                if len(content) > _MAX_FILE_CONTEXT_CHARS:
                    content = content[:_MAX_FILE_CONTEXT_CHARS] + "\n... (truncated)"
                parts.append(f"--- File: {fpath} ---\n{content}")
            except Exception as e:
                parts.append(f"--- File: {fpath} ---\n[Error reading: {e}]")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        cs = self._get_chat_screen()
        if cs:
            cs.sidebar.display = not cs.sidebar.display

    def action_clear_chat(self) -> None:
        """Clear chat display and messages."""
        cs = self._get_chat_screen()
        if cs:
            cs.chat_display.clear()
            cs.chat_display._message_buffer.clear()
            self._messages.clear()
            cs.sidebar.input_tokens = 0
            cs.sidebar.output_tokens = 0
            cs.chat_display.add_system_message("Chat cleared.")

    def action_new_session(self) -> None:
        """Start a new session."""
        self.action_clear_chat()
        self._session = None
        self._system_prompt = None
        cs = self._get_chat_screen()
        if cs:
            cs.sidebar.clear_files()
            cs.chat_display.add_system_message("New session started.")

    def action_cancel_generation(self) -> None:
        """Cancel ongoing generation."""
        if self._is_streaming:
            self._cancel_event.set()
