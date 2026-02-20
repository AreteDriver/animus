"""Scrollable message display with markdown and syntax highlighting."""

from __future__ import annotations

from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import RichLog


class ChatDisplay(RichLog):
    """Scrollable chat message display.

    Uses RichLog for efficient append-only rendering with
    Rich markdown and syntax highlighting support.
    """

    DEFAULT_CSS = """
    ChatDisplay {
        height: 1fr;
        border: solid $surface-lighten-2;
        padding: 0 1;
        scrollbar-size: 1 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._message_buffer: list[dict[str, str]] = []
        self._is_streaming: bool = False

    def add_user_message(self, content: str) -> None:
        """Append a user message to the display."""
        self._message_buffer.append({"role": "user", "content": content})
        label = Text("\n> You", style="bold cyan")
        self.write(label)
        self.write(Text(content))

    def add_assistant_message(self, content: str) -> None:
        """Append a complete assistant message (markdown rendered)."""
        self._message_buffer.append({"role": "assistant", "content": content})
        label = Text("\n< Assistant", style="bold green")
        self.write(label)
        self.write(Markdown(content))

    def add_system_message(self, content: str) -> None:
        """Append a system/info message."""
        self._message_buffer.append({"role": "system", "content": content})
        self.write(Text(f"\n[system] {content}", style="dim yellow"))

    def add_error_message(self, content: str) -> None:
        """Append an error message."""
        self._message_buffer.append({"role": "error", "content": content})
        self.write(Text(f"\n[error] {content}", style="bold red"))

    def begin_assistant_stream(self) -> None:
        """Start a new assistant streaming response."""
        self._is_streaming = True
        label = Text("\n< Assistant", style="bold green")
        self.write(label)

    def append_stream_chunk(self, text: str) -> None:
        """Append a chunk of streaming text."""
        self.write(Text(text, end=""))

    def end_assistant_stream(self, full_content: str) -> None:
        """Finalize a streamed response by re-rendering with markdown.

        Clears the raw streamed chunks and replays the full message
        buffer so that the completed response is markdown-formatted.
        """
        self._is_streaming = False
        if full_content:
            self._message_buffer.append({"role": "assistant", "content": full_content})
        self._rebuild_display()

    def _rebuild_display(self) -> None:
        """Clear and re-render all buffered messages."""
        self.clear()
        for msg in self._message_buffer:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                self.write(Text("\n> You", style="bold cyan"))
                self.write(Text(content))
            elif role == "assistant":
                self.write(Text("\n< Assistant", style="bold green"))
                self.write(Markdown(content))
            elif role == "system":
                self.write(Text(f"\n[system] {content}", style="dim yellow"))
            elif role == "error":
                self.write(Text(f"\n[error] {content}", style="bold red"))
            elif role == "agent":
                agent_name = msg.get("agent", "agent")
                self.write(Text(f"\n[{agent_name}]", style="bold magenta"))
                self.write(Markdown(content))

    def add_agent_message(self, role: str, content: str) -> None:
        """Append a message from a named agent role."""
        self._message_buffer.append({"role": "agent", "content": content, "agent": role})
        label = Text(f"\n[{role}]", style="bold magenta")
        self.write(label)
        self.write(Markdown(content))
