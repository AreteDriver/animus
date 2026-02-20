"""Bottom status bar showing provider, model, and streaming state."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """Footer status bar.

    Shows: provider | model | streaming indicator | help hint.
    """

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    """

    provider_name: reactive[str] = reactive("none")
    model_name: reactive[str] = reactive("none")
    is_streaming: reactive[bool] = reactive(False)

    def render(self) -> str:
        streaming = " [streaming...]" if self.is_streaming else ""
        return (
            f" {self.provider_name} | {self.model_name}{streaming}"
            f"  |  Ctrl+Q quit  Ctrl+B sidebar  /help commands"
        )
