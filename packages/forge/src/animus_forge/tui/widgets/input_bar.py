"""Multi-line input bar with slash command detection."""

from __future__ import annotations

from textual import on
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class InputBar(TextArea):
    """Multi-line input with Enter=send, Shift+Enter=newline.

    Emits InputBar.Submitted when user presses Enter to send.
    Detects slash commands (input starting with /).
    """

    DEFAULT_CSS = """
    InputBar {
        height: auto;
        min-height: 3;
        max-height: 10;
        border: solid $surface-lighten-2;
        padding: 0 1;
    }
    InputBar:focus {
        border: solid $accent;
    }
    """

    BINDINGS = [
        Binding("enter", "submit", "Send", show=False),
    ]

    class Submitted(Message):
        """Fired when user submits input."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

        @property
        def is_command(self) -> bool:
            return self.value.startswith("/")

    def __init__(self, **kwargs) -> None:
        super().__init__(
            language=None,
            theme="monokai",
            show_line_numbers=False,
            **kwargs,
        )

    def action_submit(self) -> None:
        """Send the current input."""
        value = self.text.strip()
        if value:
            self.post_message(self.Submitted(value))
            self.clear()

    @on(TextArea.Changed)
    def _on_change(self, event: TextArea.Changed) -> None:
        """Auto-resize height based on content."""
        lines = self.text.count("\n") + 1
        self.styles.height = min(max(3, lines + 2), 10)
