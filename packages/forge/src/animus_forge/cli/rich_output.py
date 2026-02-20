"""Rich CLI output utilities with colors, progress bars, and formatting."""

from __future__ import annotations

__all__ = [
    "Layout",
    "Live",
    "OutputStyle",
    "RichOutput",
    "StepProgress",
    "Style",
    "Text",
    "get_output",
    "print_error",
    "print_header",
    "print_info",
    "print_success",
    "print_table",
    "print_warning",
]

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

try:
    from rich.console import Console
    from rich.layout import Layout  # noqa: F401
    from rich.live import Live  # noqa: F401
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.style import Style  # noqa: F401
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text  # noqa: F401
    from rich.tree import Tree

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None


class OutputStyle(Enum):
    """Output styling options."""

    SUCCESS = "green"
    ERROR = "red"
    WARNING = "yellow"
    INFO = "blue"
    MUTED = "dim"
    HIGHLIGHT = "bold cyan"


@dataclass
class StepProgress:
    """Progress information for a workflow step."""

    step_id: str
    step_name: str
    status: str  # pending, running, completed, failed
    message: str | None = None
    duration_ms: int | None = None


class RichOutput:
    """Rich output handler for CLI with fallback to plain text."""

    def __init__(self, force_plain: bool = False):
        """Initialize rich output.

        Args:
            force_plain: Force plain text output even if Rich is available
        """
        self.use_rich = RICH_AVAILABLE and not force_plain
        self.console = Console() if self.use_rich else None

    def print(
        self,
        message: str,
        style: OutputStyle | None = None,
        prefix: str | None = None,
    ) -> None:
        """Print a styled message.

        Args:
            message: Message to print
            style: Optional style to apply
            prefix: Optional prefix (emoji/icon)
        """
        if prefix:
            message = f"{prefix} {message}"

        if self.use_rich and self.console:
            style_str = style.value if style else None
            self.console.print(message, style=style_str)
        else:
            print(message)

    def success(self, message: str) -> None:
        """Print success message."""
        self.print(message, OutputStyle.SUCCESS, "✅")

    def error(self, message: str) -> None:
        """Print error message."""
        self.print(message, OutputStyle.ERROR, "❌")

    def warning(self, message: str) -> None:
        """Print warning message."""
        self.print(message, OutputStyle.WARNING, "⚠️")

    def info(self, message: str) -> None:
        """Print info message."""
        self.print(message, OutputStyle.INFO, "ℹ️")

    def header(self, title: str, subtitle: str | None = None) -> None:
        """Print a header section.

        Args:
            title: Header title
            subtitle: Optional subtitle
        """
        if self.use_rich and self.console:
            content = title
            if subtitle:
                content += f"\n[dim]{subtitle}[/dim]"
            panel = Panel(content, title="🐍 Gorgon", border_style="cyan")
            self.console.print(panel)
        else:
            print(f"\n{'=' * 50}")
            print(f"  {title}")
            if subtitle:
                print(f"  {subtitle}")
            print(f"{'=' * 50}\n")

    def table(
        self,
        headers: list[str],
        rows: list[list[str]],
        title: str | None = None,
    ) -> None:
        """Print a formatted table.

        Args:
            headers: Column headers
            rows: Table rows
            title: Optional table title
        """
        if self.use_rich and self.console:
            table = Table(title=title, show_header=True, header_style="bold cyan")
            for header in headers:
                table.add_column(header)
            for row in rows:
                table.add_row(*row)
            self.console.print(table)
        else:
            if title:
                print(f"\n{title}")
                print("-" * 40)
            # Simple plain text table
            col_widths = [
                max(len(str(h)), max(len(str(row[i])) for row in rows) if rows else 0)
                for i, h in enumerate(headers)
            ]
            header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
            print(header_line)
            print("-" * len(header_line))
            for row in rows:
                print(" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))

    def code(self, code: str, language: str = "python") -> None:
        """Print syntax-highlighted code.

        Args:
            code: Code to display
            language: Programming language
        """
        if self.use_rich and self.console:
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            print(f"\n```{language}")
            print(code)
            print("```\n")

    def markdown(self, content: str) -> None:
        """Print rendered markdown.

        Args:
            content: Markdown content
        """
        if self.use_rich and self.console:
            md = Markdown(content)
            self.console.print(md)
        else:
            print(content)

    def tree(self, title: str, items: dict) -> None:
        """Print a tree structure.

        Args:
            title: Root title
            items: Nested dictionary of items
        """
        if self.use_rich and self.console:
            tree = Tree(f"[bold]{title}[/bold]")
            self._add_tree_items(tree, items)
            self.console.print(tree)
        else:
            print(f"\n{title}")
            self._print_plain_tree(items, indent=2)

    def _add_tree_items(self, tree: Tree, items: dict) -> None:
        """Recursively add items to rich tree."""
        for key, value in items.items():
            if isinstance(value, dict):
                branch = tree.add(f"[cyan]{key}[/cyan]")
                self._add_tree_items(branch, value)
            else:
                tree.add(f"[dim]{key}:[/dim] {value}")

    def _print_plain_tree(self, items: dict, indent: int = 0) -> None:
        """Print plain text tree."""
        for key, value in items.items():
            if isinstance(value, dict):
                print(" " * indent + f"├── {key}")
                self._print_plain_tree(value, indent + 4)
            else:
                print(" " * indent + f"├── {key}: {value}")

    @contextmanager
    def progress(
        self,
        description: str = "Processing",
        total: int | None = None,
    ) -> Generator[Callable[[int], None], None, None]:
        """Context manager for progress display.

        Args:
            description: Progress description
            total: Total steps (None for indeterminate)

        Yields:
            Function to update progress
        """
        if self.use_rich:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task(description, total=total or 100)

                def update(advance: int = 1) -> None:
                    progress.update(task, advance=advance)

                yield update
        else:
            print(f"{description}...")

            def update(advance: int = 1) -> None:
                print(".", end="", flush=True)

            yield update
            print(" Done!")

    @contextmanager
    def spinner(self, message: str = "Processing...") -> Generator[None, None, None]:
        """Context manager for spinner display.

        Args:
            message: Spinner message

        Yields:
            None
        """
        if self.use_rich and self.console:
            with self.console.status(message, spinner="dots"):
                yield
        else:
            print(message, end="", flush=True)
            yield
            print(" Done!")

    def workflow_progress(
        self,
        workflow_name: str,
        steps: list[StepProgress],
    ) -> None:
        """Display workflow execution progress.

        Args:
            workflow_name: Name of the workflow
            steps: List of step progress info
        """
        if self.use_rich and self.console:
            self.console.print(f"\n[bold cyan]Workflow:[/bold cyan] {workflow_name}\n")

            for step in steps:
                status_styles = {
                    "pending": ("⏳", "dim"),
                    "running": ("🔄", "yellow"),
                    "completed": ("✅", "green"),
                    "failed": ("❌", "red"),
                }
                icon, style = status_styles.get(step.status, ("•", "white"))

                line = f"  {icon} [{style}]{step.step_id}[/{style}]"
                if step.message:
                    line += f" - {step.message}"
                if step.duration_ms:
                    line += f" [dim]({step.duration_ms}ms)[/dim]"

                self.console.print(line)
        else:
            print(f"\nWorkflow: {workflow_name}")
            print("-" * 40)
            for step in steps:
                status_icons = {
                    "pending": "[ ]",
                    "running": "[~]",
                    "completed": "[x]",
                    "failed": "[!]",
                }
                icon = status_icons.get(step.status, "[ ]")
                line = f"  {icon} {step.step_id}"
                if step.message:
                    line += f" - {step.message}"
                if step.duration_ms:
                    line += f" ({step.duration_ms}ms)"
                print(line)

    def agent_status(
        self,
        role: str,
        status: str,
        message: str | None = None,
    ) -> None:
        """Display agent status update.

        Args:
            role: Agent role
            status: Current status
            message: Optional status message
        """
        role_icons = {
            "planner": "📋",
            "builder": "🔨",
            "tester": "🧪",
            "reviewer": "👁️",
            "model_builder": "🎮",
            "data_analyst": "📊",
            "devops": "🔧",
            "security_auditor": "🔒",
            "migrator": "🔄",
        }
        icon = role_icons.get(role, "🤖")

        if self.use_rich and self.console:
            status_color = {
                "starting": "yellow",
                "running": "blue",
                "completed": "green",
                "failed": "red",
            }.get(status, "white")

            text = f"{icon} [bold]{role}[/bold] [{status_color}]{status}[/{status_color}]"
            if message:
                text += f" - {message}"
            self.console.print(text)
        else:
            text = f"{icon} {role}: {status}"
            if message:
                text += f" - {message}"
            print(text)

    def divider(self) -> None:
        """Print a visual divider."""
        if self.use_rich and self.console:
            self.console.print("─" * 50, style="dim")
        else:
            print("-" * 50)

    def newline(self) -> None:
        """Print a newline."""
        print()


# Global instance for convenience
_output: RichOutput | None = None


def get_output(force_plain: bool = False) -> RichOutput:
    """Get the global rich output instance.

    Args:
        force_plain: Force plain text output

    Returns:
        RichOutput instance
    """
    global _output
    if _output is None or force_plain:
        _output = RichOutput(force_plain=force_plain)
    return _output


# Convenience functions
def print_success(message: str) -> None:
    """Print success message."""
    get_output().success(message)


def print_error(message: str) -> None:
    """Print error message."""
    get_output().error(message)


def print_warning(message: str) -> None:
    """Print warning message."""
    get_output().warning(message)


def print_info(message: str) -> None:
    """Print info message."""
    get_output().info(message)


def print_header(title: str, subtitle: str | None = None) -> None:
    """Print header."""
    get_output().header(title, subtitle)


def print_table(headers: list[str], rows: list[list[str]], title: str | None = None) -> None:
    """Print table."""
    get_output().table(headers, rows, title)
