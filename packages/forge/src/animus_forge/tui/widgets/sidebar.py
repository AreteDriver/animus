"""Sidebar panel showing provider, model, files, and agent info."""

from __future__ import annotations

from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static


class Sidebar(Vertical):
    """Collapsible sidebar showing session state.

    Displays: provider name, model, token counts, file context list,
    and active agent mode.
    """

    DEFAULT_CSS = """
    Sidebar {
        width: 28;
        dock: left;
        border-right: solid $surface-lighten-2;
        padding: 1;
        background: $surface;
    }
    Sidebar .sidebar-header {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    Sidebar .sidebar-section {
        padding-bottom: 1;
    }
    Sidebar .sidebar-label {
        color: $text-muted;
    }
    Sidebar .sidebar-value {
        color: $text;
        padding-left: 1;
    }
    """

    provider_name: reactive[str] = reactive("none")
    model_name: reactive[str] = reactive("none")
    input_tokens: reactive[int] = reactive(0)
    output_tokens: reactive[int] = reactive(0)
    agent_mode: reactive[str] = reactive("off")
    files: reactive[list] = reactive(list, init=False)

    def compose(self):
        yield Static("GORGON", classes="sidebar-header")
        yield Static("", id="sidebar-provider", classes="sidebar-section")
        yield Static("", id="sidebar-tokens", classes="sidebar-section")
        yield Static("", id="sidebar-agent", classes="sidebar-section")
        yield Static("", id="sidebar-files", classes="sidebar-section")

    def on_mount(self) -> None:
        self.files = []
        self._refresh_all()

    def watch_provider_name(self) -> None:
        self._refresh_provider()

    def watch_model_name(self) -> None:
        self._refresh_provider()

    def watch_input_tokens(self) -> None:
        self._refresh_tokens()

    def watch_output_tokens(self) -> None:
        self._refresh_tokens()

    def watch_agent_mode(self) -> None:
        self._refresh_agent()

    def _refresh_all(self) -> None:
        self._refresh_provider()
        self._refresh_tokens()
        self._refresh_agent()
        self._refresh_files()

    def _refresh_provider(self) -> None:
        widget = self.query_one("#sidebar-provider", Static)
        widget.update(f"Provider: {self.provider_name}\nModel: {self.model_name}")

    def _refresh_tokens(self) -> None:
        widget = self.query_one("#sidebar-tokens", Static)
        total = self.input_tokens + self.output_tokens
        widget.update(
            f"Tokens: {total:,}\n  In: {self.input_tokens:,}\n  Out: {self.output_tokens:,}"
        )

    def _refresh_agent(self) -> None:
        widget = self.query_one("#sidebar-agent", Static)
        widget.update(f"Agent: {self.agent_mode}")

    def _refresh_files(self) -> None:
        widget = self.query_one("#sidebar-files", Static)
        if not self.files:
            widget.update("Files: (none)")
        else:
            lines = ["Files:"]
            for f in self.files:
                name = f if len(f) < 22 else f"...{f[-19:]}"
                lines.append(f"  {name}")
            widget.update("\n".join(lines))

    def add_file(self, path: str) -> None:
        if path not in self.files:
            self.files = [*self.files, path]
            self._refresh_files()

    def remove_file(self, path: str) -> None:
        self.files = [f for f in self.files if f != path]
        self._refresh_files()

    def clear_files(self) -> None:
        self.files = []
        self._refresh_files()

    def add_tokens(self, input_t: int, output_t: int) -> None:
        self.input_tokens += input_t
        self.output_tokens += output_t
