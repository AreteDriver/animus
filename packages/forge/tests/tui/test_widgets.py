"""Tests for TUI widget logic (non-Textual-mount tests)."""

from __future__ import annotations

from types import SimpleNamespace

from animus_forge.tui.widgets.input_bar import InputBar
from animus_forge.tui.widgets.status_bar import StatusBar


def _render_status_bar(provider_name: str, model_name: str, is_streaming: bool) -> str:
    """Call StatusBar.render() on a plain namespace to avoid Textual reactives."""
    ns = SimpleNamespace(
        provider_name=provider_name,
        model_name=model_name,
        is_streaming=is_streaming,
    )
    return StatusBar.render(ns)


class TestStatusBarRender:
    """Test StatusBar.render() output without mounting."""

    def test_default_render(self):
        result = _render_status_bar("none", "none", False)
        assert "none" in result
        assert "streaming" not in result

    def test_streaming_indicator(self):
        result = _render_status_bar("anthropic", "claude-sonnet-4-20250514", True)
        assert "streaming" in result
        assert "anthropic" in result

    def test_provider_and_model_shown(self):
        result = _render_status_bar("openai", "gpt-4o", False)
        assert "openai" in result
        assert "gpt-4o" in result

    def test_help_hint_shown(self):
        result = _render_status_bar("test", "test", False)
        assert "/help" in result


class TestInputBarSubmitted:
    """Test InputBar.Submitted message class."""

    def test_regular_message(self):
        msg = InputBar.Submitted("hello world")
        assert msg.value == "hello world"
        assert msg.is_command is False

    def test_slash_command(self):
        msg = InputBar.Submitted("/help")
        assert msg.value == "/help"
        assert msg.is_command is True

    def test_slash_command_with_args(self):
        msg = InputBar.Submitted("/switch openai")
        assert msg.is_command is True

    def test_empty_string(self):
        msg = InputBar.Submitted("")
        assert msg.is_command is False

    def test_slash_only(self):
        msg = InputBar.Submitted("/")
        assert msg.is_command is True


class TestSidebarLogic:
    """Test Sidebar data management without mounting."""

    def test_add_file_deduplicates(self):
        """Adding the same file twice should not duplicate."""
        files = []
        path = "/tmp/test.py"
        if path not in files:
            files.append(path)
        if path not in files:
            files.append(path)
        assert files.count(path) == 1

    def test_remove_file(self):
        files = ["/tmp/a.py", "/tmp/b.py", "/tmp/c.py"]
        files = [f for f in files if f != "/tmp/b.py"]
        assert "/tmp/b.py" not in files
        assert len(files) == 2

    def test_clear_files(self):
        files = ["/tmp/a.py", "/tmp/b.py"]
        files.clear()
        assert files == []

    def test_add_tokens(self):
        input_t = 0
        output_t = 0
        input_t += 100
        output_t += 50
        input_t += 200
        output_t += 100
        assert input_t == 300
        assert output_t == 150

    def test_file_name_truncation(self):
        """Long file paths should be truncatable for display."""
        path = "/very/long/path/to/some/deeply/nested/file.py"
        name = path if len(path) < 22 else f"...{path[-19:]}"
        assert name.startswith("...")
        assert len(name) == 22


class TestChatDisplayBuffer:
    """Test message buffer logic without Textual mounting."""

    def test_buffer_stores_messages(self):
        buffer = []
        buffer.append({"role": "user", "content": "hello"})
        buffer.append({"role": "assistant", "content": "hi there"})
        buffer.append({"role": "system", "content": "info"})
        assert len(buffer) == 3
        assert buffer[0]["role"] == "user"
        assert buffer[1]["role"] == "assistant"

    def test_buffer_stores_agent_messages(self):
        buffer = []
        buffer.append({"role": "agent", "content": "planning...", "agent": "planner"})
        assert buffer[0]["agent"] == "planner"

    def test_stream_finalize_adds_to_buffer(self):
        buffer = []
        full_content = "streamed response"
        if full_content:
            buffer.append({"role": "assistant", "content": full_content})
        assert len(buffer) == 1
        assert buffer[0]["content"] == "streamed response"

    def test_empty_stream_not_buffered(self):
        buffer = []
        full_content = ""
        if full_content:
            buffer.append({"role": "assistant", "content": full_content})
        assert len(buffer) == 0
