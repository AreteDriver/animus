"""Primary chat screen composing all widgets."""

from __future__ import annotations

import logging

from textual.containers import Horizontal, Vertical
from textual.screen import Screen

from animus_forge.tui.widgets.chat_display import ChatDisplay
from animus_forge.tui.widgets.input_bar import InputBar
from animus_forge.tui.widgets.sidebar import Sidebar
from animus_forge.tui.widgets.status_bar import StatusBar

logger = logging.getLogger(__name__)


class ChatScreen(Screen):
    """Main chat screen with sidebar, chat display, input bar, and status bar."""

    def compose(self):
        with Horizontal(id="chat-container"):
            yield Sidebar(id="sidebar")
            with Vertical(id="main-content"):
                yield ChatDisplay(id="chat-display")
                yield InputBar(id="input-bar")
        yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        self.query_one(InputBar).focus()

    @property
    def chat_display(self) -> ChatDisplay:
        return self.query_one("#chat-display", ChatDisplay)

    @property
    def input_bar(self) -> InputBar:
        return self.query_one("#input-bar", InputBar)

    @property
    def sidebar(self) -> Sidebar:
        return self.query_one("#sidebar", Sidebar)

    @property
    def status_bar(self) -> StatusBar:
        return self.query_one("#status-bar", StatusBar)
