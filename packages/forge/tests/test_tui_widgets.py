"""Tests for TUI widget modules — ChatDisplay, Sidebar, and streaming internals.

Targets:
  - src/animus_forge/tui/widgets/chat_display.py (26% → 90%+)
  - src/animus_forge/tui/widgets/sidebar.py (41% → 90%+)
  - src/animus_forge/tui/streaming.py (64% → 90%+)

Uses SimpleNamespace trick for Textual reactives and MagicMock for widgets.
Uses asyncio.run() wrapper pattern (NOT pytest-asyncio).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.markdown import Markdown
from rich.text import Text

from animus_forge.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ProviderError,
    ProviderType,
    StreamChunk,
)
from animus_forge.tui.streaming import (
    StreamResult,
    _stream_anthropic,
    _stream_fallback,
    _stream_ollama,
    _stream_openai,
    stream_completion,
)
from animus_forge.tui.widgets.chat_display import ChatDisplay
from animus_forge.tui.widgets.sidebar import Sidebar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_display() -> ChatDisplay:
    """Create a ChatDisplay instance with mocked Textual internals.

    We bypass the RichLog.__init__ (which needs a Textual app context)
    by calling object.__init__ and setting up the attributes manually.
    """
    cd = object.__new__(ChatDisplay)
    cd._message_buffer = []
    cd._is_streaming = False
    cd.write = MagicMock()
    cd.clear = MagicMock()
    return cd


def _make_sidebar() -> SimpleNamespace:
    """Create a namespace that mimics Sidebar's attributes for logic testing.

    This avoids mounting a Textual widget while still exercising
    the actual Sidebar methods via unbound calls.
    Binds all Sidebar instance methods so internal self.method() calls work.
    """
    mock_widgets = {
        "#sidebar-provider": MagicMock(),
        "#sidebar-tokens": MagicMock(),
        "#sidebar-agent": MagicMock(),
        "#sidebar-files": MagicMock(),
    }

    def query_one(selector, cls=None):
        return mock_widgets[selector]

    ns = SimpleNamespace(
        provider_name="none",
        model_name="none",
        input_tokens=0,
        output_tokens=0,
        agent_mode="off",
        files=[],
        query_one=query_one,
        _mock_widgets=mock_widgets,
    )
    # Bind Sidebar instance methods so self.method() calls work within methods
    import types

    for method_name in (
        "_refresh_all",
        "_refresh_provider",
        "_refresh_tokens",
        "_refresh_agent",
        "_refresh_files",
        "on_mount",
        "watch_provider_name",
        "watch_model_name",
        "watch_input_tokens",
        "watch_output_tokens",
        "watch_agent_mode",
        "add_file",
        "remove_file",
        "clear_files",
        "add_tokens",
    ):
        method = getattr(Sidebar, method_name)
        setattr(ns, method_name, types.MethodType(method, ns))
    return ns


def _make_provider(provider_type: ProviderType, initialized: bool = True):
    """Create a mock provider with the given type."""
    provider = MagicMock()
    provider.provider_type = provider_type
    provider._initialized = initialized
    provider.default_model = "test-model"
    provider.initialize = MagicMock()
    return provider


# ===========================================================================
# ChatDisplay Tests
# ===========================================================================


class TestChatDisplayInit:
    """Test ChatDisplay.__init__ sets up internal state."""

    def test_init_creates_empty_buffer(self):
        cd = _make_chat_display()
        assert cd._message_buffer == []

    def test_init_not_streaming(self):
        cd = _make_chat_display()
        assert cd._is_streaming is False


class TestChatDisplayAddUserMessage:
    """Test ChatDisplay.add_user_message()."""

    def test_appends_to_buffer(self):
        cd = _make_chat_display()
        cd.add_user_message("hello world")
        assert len(cd._message_buffer) == 1
        assert cd._message_buffer[0] == {"role": "user", "content": "hello world"}

    def test_writes_label_and_content(self):
        cd = _make_chat_display()
        cd.add_user_message("test input")
        assert cd.write.call_count == 2
        # First call: label
        label_arg = cd.write.call_args_list[0][0][0]
        assert isinstance(label_arg, Text)
        assert "> You" in str(label_arg)
        # Second call: content
        content_arg = cd.write.call_args_list[1][0][0]
        assert isinstance(content_arg, Text)

    def test_multiple_messages(self):
        cd = _make_chat_display()
        cd.add_user_message("first")
        cd.add_user_message("second")
        assert len(cd._message_buffer) == 2
        assert cd._message_buffer[0]["content"] == "first"
        assert cd._message_buffer[1]["content"] == "second"


class TestChatDisplayAddAssistantMessage:
    """Test ChatDisplay.add_assistant_message()."""

    def test_appends_to_buffer(self):
        cd = _make_chat_display()
        cd.add_assistant_message("response text")
        assert len(cd._message_buffer) == 1
        assert cd._message_buffer[0] == {
            "role": "assistant",
            "content": "response text",
        }

    def test_writes_label_and_markdown(self):
        cd = _make_chat_display()
        cd.add_assistant_message("# Hello")
        assert cd.write.call_count == 2
        label_arg = cd.write.call_args_list[0][0][0]
        assert isinstance(label_arg, Text)
        assert "< Assistant" in str(label_arg)
        # Markdown rendered content
        content_arg = cd.write.call_args_list[1][0][0]
        assert isinstance(content_arg, Markdown)


class TestChatDisplayAddSystemMessage:
    """Test ChatDisplay.add_system_message()."""

    def test_appends_to_buffer(self):
        cd = _make_chat_display()
        cd.add_system_message("system info")
        assert cd._message_buffer[0] == {"role": "system", "content": "system info"}

    def test_writes_formatted_text(self):
        cd = _make_chat_display()
        cd.add_system_message("connected")
        cd.write.assert_called_once()
        text_arg = cd.write.call_args[0][0]
        assert isinstance(text_arg, Text)
        assert "[system] connected" in str(text_arg)


class TestChatDisplayAddErrorMessage:
    """Test ChatDisplay.add_error_message()."""

    def test_appends_to_buffer(self):
        cd = _make_chat_display()
        cd.add_error_message("something broke")
        assert cd._message_buffer[0] == {
            "role": "error",
            "content": "something broke",
        }

    def test_writes_error_formatted_text(self):
        cd = _make_chat_display()
        cd.add_error_message("timeout")
        cd.write.assert_called_once()
        text_arg = cd.write.call_args[0][0]
        assert isinstance(text_arg, Text)
        assert "[error] timeout" in str(text_arg)


class TestChatDisplayStreaming:
    """Test begin_assistant_stream, append_stream_chunk, end_assistant_stream."""

    def test_begin_sets_streaming_flag(self):
        cd = _make_chat_display()
        cd.begin_assistant_stream()
        assert cd._is_streaming is True

    def test_begin_writes_label(self):
        cd = _make_chat_display()
        cd.begin_assistant_stream()
        cd.write.assert_called_once()
        label = cd.write.call_args[0][0]
        assert isinstance(label, Text)
        assert "< Assistant" in str(label)

    def test_append_chunk_writes_text(self):
        cd = _make_chat_display()
        cd.begin_assistant_stream()
        cd.write.reset_mock()
        cd.append_stream_chunk("hello ")
        cd.write.assert_called_once()
        text_arg = cd.write.call_args[0][0]
        assert isinstance(text_arg, Text)

    def test_append_multiple_chunks(self):
        cd = _make_chat_display()
        cd.begin_assistant_stream()
        cd.write.reset_mock()
        cd.append_stream_chunk("hello ")
        cd.append_stream_chunk("world")
        assert cd.write.call_count == 2

    def test_end_stream_clears_flag(self):
        cd = _make_chat_display()
        cd.begin_assistant_stream()
        cd.end_assistant_stream("full content")
        assert cd._is_streaming is False

    def test_end_stream_adds_to_buffer(self):
        cd = _make_chat_display()
        cd.begin_assistant_stream()
        cd.end_assistant_stream("complete response")
        assert len(cd._message_buffer) == 1
        assert cd._message_buffer[0] == {
            "role": "assistant",
            "content": "complete response",
        }

    def test_end_stream_empty_content_not_buffered(self):
        cd = _make_chat_display()
        cd.begin_assistant_stream()
        cd.end_assistant_stream("")
        assert len(cd._message_buffer) == 0

    def test_end_stream_calls_rebuild(self):
        cd = _make_chat_display()
        cd.begin_assistant_stream()
        cd.end_assistant_stream("test")
        # _rebuild_display clears and re-renders
        cd.clear.assert_called_once()


class TestChatDisplayRebuildDisplay:
    """Test ChatDisplay._rebuild_display()."""

    def test_clears_display(self):
        cd = _make_chat_display()
        cd._message_buffer = [{"role": "user", "content": "hi"}]
        cd._rebuild_display()
        cd.clear.assert_called_once()

    def test_rebuilds_user_messages(self):
        cd = _make_chat_display()
        cd._message_buffer = [{"role": "user", "content": "hello"}]
        cd._rebuild_display()
        # clear + 2 writes (label + content)
        assert cd.write.call_count == 2

    def test_rebuilds_assistant_messages(self):
        cd = _make_chat_display()
        cd._message_buffer = [{"role": "assistant", "content": "response"}]
        cd._rebuild_display()
        assert cd.write.call_count == 2
        # Second write is Markdown
        content_arg = cd.write.call_args_list[1][0][0]
        assert isinstance(content_arg, Markdown)

    def test_rebuilds_system_messages(self):
        cd = _make_chat_display()
        cd._message_buffer = [{"role": "system", "content": "info"}]
        cd._rebuild_display()
        assert cd.write.call_count == 1
        text_arg = cd.write.call_args[0][0]
        assert "[system] info" in str(text_arg)

    def test_rebuilds_error_messages(self):
        cd = _make_chat_display()
        cd._message_buffer = [{"role": "error", "content": "oops"}]
        cd._rebuild_display()
        assert cd.write.call_count == 1
        text_arg = cd.write.call_args[0][0]
        assert "[error] oops" in str(text_arg)

    def test_rebuilds_agent_messages(self):
        cd = _make_chat_display()
        cd._message_buffer = [{"role": "agent", "content": "planning...", "agent": "planner"}]
        cd._rebuild_display()
        assert cd.write.call_count == 2
        label_arg = cd.write.call_args_list[0][0][0]
        assert "[planner]" in str(label_arg)
        content_arg = cd.write.call_args_list[1][0][0]
        assert isinstance(content_arg, Markdown)

    def test_rebuilds_agent_message_default_name(self):
        cd = _make_chat_display()
        cd._message_buffer = [{"role": "agent", "content": "working..."}]
        cd._rebuild_display()
        label_arg = cd.write.call_args_list[0][0][0]
        assert "[agent]" in str(label_arg)

    def test_rebuilds_mixed_messages(self):
        cd = _make_chat_display()
        cd._message_buffer = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "connected"},
            {"role": "error", "content": "timeout"},
            {"role": "agent", "content": "done", "agent": "builder"},
        ]
        cd._rebuild_display()
        # user: 2, assistant: 2, system: 1, error: 1, agent: 2 = 8
        assert cd.write.call_count == 8


class TestChatDisplayAddAgentMessage:
    """Test ChatDisplay.add_agent_message()."""

    def test_appends_to_buffer_with_agent_name(self):
        cd = _make_chat_display()
        cd.add_agent_message("reviewer", "looks good")
        assert cd._message_buffer[0] == {
            "role": "agent",
            "content": "looks good",
            "agent": "reviewer",
        }

    def test_writes_label_and_markdown(self):
        cd = _make_chat_display()
        cd.add_agent_message("planner", "# Step 1\nDo this")
        assert cd.write.call_count == 2
        label_arg = cd.write.call_args_list[0][0][0]
        assert isinstance(label_arg, Text)
        assert "[planner]" in str(label_arg)
        content_arg = cd.write.call_args_list[1][0][0]
        assert isinstance(content_arg, Markdown)


# ===========================================================================
# Sidebar Tests
# ===========================================================================


class TestSidebarOnMount:
    """Test Sidebar.on_mount() initializes state."""

    def test_on_mount_sets_empty_files(self):
        ns = _make_sidebar()
        ns.on_mount()
        assert ns.files == []

    def test_on_mount_calls_refresh_all(self):
        """on_mount should call _refresh_all which refreshes all widgets."""
        ns = _make_sidebar()
        ns.provider_name = "anthropic"
        ns.model_name = "claude-sonnet"
        ns.on_mount()
        # _refresh_all -> _refresh_provider updates sidebar-provider
        provider_widget = ns._mock_widgets["#sidebar-provider"]
        provider_widget.update.assert_called()
        tokens_widget = ns._mock_widgets["#sidebar-tokens"]
        tokens_widget.update.assert_called()
        agent_widget = ns._mock_widgets["#sidebar-agent"]
        agent_widget.update.assert_called()
        files_widget = ns._mock_widgets["#sidebar-files"]
        files_widget.update.assert_called()


class TestSidebarWatchMethods:
    """Test Sidebar watch_* reactive watchers."""

    def test_watch_provider_name(self):
        ns = _make_sidebar()
        ns.provider_name = "openai"
        ns.model_name = "gpt-4o"
        ns.watch_provider_name()
        widget = ns._mock_widgets["#sidebar-provider"]
        widget.update.assert_called_once()
        update_text = widget.update.call_args[0][0]
        assert "openai" in update_text
        assert "gpt-4o" in update_text

    def test_watch_model_name(self):
        ns = _make_sidebar()
        ns.provider_name = "anthropic"
        ns.model_name = "claude-sonnet-4-20250514"
        ns.watch_model_name()
        widget = ns._mock_widgets["#sidebar-provider"]
        widget.update.assert_called_once()
        update_text = widget.update.call_args[0][0]
        assert "claude-sonnet-4-20250514" in update_text

    def test_watch_input_tokens(self):
        ns = _make_sidebar()
        ns.input_tokens = 500
        ns.output_tokens = 200
        ns.watch_input_tokens()
        widget = ns._mock_widgets["#sidebar-tokens"]
        widget.update.assert_called_once()
        update_text = widget.update.call_args[0][0]
        assert "700" in update_text  # total
        assert "500" in update_text  # input
        assert "200" in update_text  # output

    def test_watch_output_tokens(self):
        ns = _make_sidebar()
        ns.input_tokens = 1000
        ns.output_tokens = 500
        ns.watch_output_tokens()
        widget = ns._mock_widgets["#sidebar-tokens"]
        widget.update.assert_called_once()
        update_text = widget.update.call_args[0][0]
        assert "1,500" in update_text  # total

    def test_watch_agent_mode(self):
        ns = _make_sidebar()
        ns.agent_mode = "planner"
        ns.watch_agent_mode()
        widget = ns._mock_widgets["#sidebar-agent"]
        widget.update.assert_called_once()
        update_text = widget.update.call_args[0][0]
        assert "planner" in update_text


class TestSidebarRefreshMethods:
    """Test Sidebar._refresh_* methods."""

    def test_refresh_provider(self):
        ns = _make_sidebar()
        ns.provider_name = "ollama"
        ns.model_name = "llama3.2"
        ns._refresh_provider()
        widget = ns._mock_widgets["#sidebar-provider"]
        text = widget.update.call_args[0][0]
        assert "Provider: ollama" in text
        assert "Model: llama3.2" in text

    def test_refresh_tokens_formatting(self):
        ns = _make_sidebar()
        ns.input_tokens = 12345
        ns.output_tokens = 6789
        ns._refresh_tokens()
        widget = ns._mock_widgets["#sidebar-tokens"]
        text = widget.update.call_args[0][0]
        assert "19,134" in text  # total formatted
        assert "12,345" in text  # input formatted
        assert "6,789" in text  # output formatted

    def test_refresh_tokens_zero(self):
        ns = _make_sidebar()
        ns.input_tokens = 0
        ns.output_tokens = 0
        ns._refresh_tokens()
        widget = ns._mock_widgets["#sidebar-tokens"]
        text = widget.update.call_args[0][0]
        assert "Tokens: 0" in text

    def test_refresh_agent(self):
        ns = _make_sidebar()
        ns.agent_mode = "auto"
        ns._refresh_agent()
        widget = ns._mock_widgets["#sidebar-agent"]
        text = widget.update.call_args[0][0]
        assert "Agent: auto" in text

    def test_refresh_files_empty(self):
        ns = _make_sidebar()
        ns.files = []
        ns._refresh_files()
        widget = ns._mock_widgets["#sidebar-files"]
        text = widget.update.call_args[0][0]
        assert "Files: (none)" in text

    def test_refresh_files_with_items(self):
        ns = _make_sidebar()
        ns.files = ["/tmp/a.py", "/tmp/b.py"]
        ns._refresh_files()
        widget = ns._mock_widgets["#sidebar-files"]
        text = widget.update.call_args[0][0]
        assert "Files:" in text
        assert "/tmp/a.py" in text
        assert "/tmp/b.py" in text

    def test_refresh_files_truncates_long_names(self):
        ns = _make_sidebar()
        long_path = "/very/long/path/to/some/deeply/nested/module/file.py"
        ns.files = [long_path]
        ns._refresh_files()
        widget = ns._mock_widgets["#sidebar-files"]
        text = widget.update.call_args[0][0]
        assert "..." in text
        # Should end with last 19 chars of the path
        assert long_path[-19:] in text

    def test_refresh_files_short_names_not_truncated(self):
        ns = _make_sidebar()
        short_path = "/tmp/short.py"
        ns.files = [short_path]
        ns._refresh_files()
        widget = ns._mock_widgets["#sidebar-files"]
        text = widget.update.call_args[0][0]
        assert "..." not in text
        assert "/tmp/short.py" in text

    def test_refresh_all(self):
        ns = _make_sidebar()
        ns._refresh_all()
        # All 4 widgets should have been updated
        for key in (
            "#sidebar-provider",
            "#sidebar-tokens",
            "#sidebar-agent",
            "#sidebar-files",
        ):
            ns._mock_widgets[key].update.assert_called()


class TestSidebarAddFile:
    """Test Sidebar.add_file()."""

    def test_adds_new_file(self):
        ns = _make_sidebar()
        ns.files = []
        ns.add_file("/tmp/test.py")
        assert "/tmp/test.py" in ns.files

    def test_deduplicates(self):
        ns = _make_sidebar()
        ns.files = ["/tmp/test.py"]
        ns.add_file("/tmp/test.py")
        assert ns.files.count("/tmp/test.py") == 1

    def test_adds_multiple_distinct(self):
        ns = _make_sidebar()
        ns.files = []
        ns.add_file("/tmp/a.py")
        ns.add_file("/tmp/b.py")
        assert len(ns.files) == 2

    def test_refreshes_files_display(self):
        ns = _make_sidebar()
        ns.files = []
        ns.add_file("/tmp/test.py")
        widget = ns._mock_widgets["#sidebar-files"]
        widget.update.assert_called()


class TestSidebarRemoveFile:
    """Test Sidebar.remove_file()."""

    def test_removes_existing_file(self):
        ns = _make_sidebar()
        ns.files = ["/tmp/a.py", "/tmp/b.py"]
        ns.remove_file("/tmp/a.py")
        assert "/tmp/a.py" not in ns.files
        assert "/tmp/b.py" in ns.files

    def test_removes_nonexistent_is_noop(self):
        ns = _make_sidebar()
        ns.files = ["/tmp/a.py"]
        ns.remove_file("/tmp/nonexistent.py")
        assert ns.files == ["/tmp/a.py"]

    def test_refreshes_files_display(self):
        ns = _make_sidebar()
        ns.files = ["/tmp/a.py"]
        ns.remove_file("/tmp/a.py")
        widget = ns._mock_widgets["#sidebar-files"]
        widget.update.assert_called()


class TestSidebarClearFiles:
    """Test Sidebar.clear_files()."""

    def test_clears_all_files(self):
        ns = _make_sidebar()
        ns.files = ["/tmp/a.py", "/tmp/b.py", "/tmp/c.py"]
        ns.clear_files()
        assert ns.files == []

    def test_refreshes_files_display(self):
        ns = _make_sidebar()
        ns.files = ["/tmp/a.py"]
        ns.clear_files()
        widget = ns._mock_widgets["#sidebar-files"]
        widget.update.assert_called()


class TestSidebarAddTokens:
    """Test Sidebar.add_tokens()."""

    def test_adds_to_zero(self):
        ns = _make_sidebar()
        ns.input_tokens = 0
        ns.output_tokens = 0
        ns.add_tokens(100, 50)
        assert ns.input_tokens == 100
        assert ns.output_tokens == 50

    def test_accumulates(self):
        ns = _make_sidebar()
        ns.input_tokens = 100
        ns.output_tokens = 50
        ns.add_tokens(200, 100)
        assert ns.input_tokens == 300
        assert ns.output_tokens == 150

    def test_zero_addition(self):
        ns = _make_sidebar()
        ns.input_tokens = 500
        ns.output_tokens = 250
        ns.add_tokens(0, 0)
        assert ns.input_tokens == 500
        assert ns.output_tokens == 250


# ===========================================================================
# Streaming Tests
# ===========================================================================


class TestStreamAnthropicDirect:
    """Test _stream_anthropic() with mocked Anthropic client."""

    def test_streams_text_and_captures_usage(self):
        async def _run():
            provider = _make_provider(ProviderType.ANTHROPIC)

            # Build mock async context manager for client.messages.stream()
            mock_usage = SimpleNamespace(input_tokens=42, output_tokens=18)
            mock_response = SimpleNamespace(usage=mock_usage)

            async def mock_text_stream():
                yield "Hello "
                yield "world"

            mock_stream_ctx = AsyncMock()
            mock_stream_ctx.text_stream = mock_text_stream()
            mock_stream_ctx.get_final_message = AsyncMock(return_value=mock_response)

            # async context manager protocol
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
            mock_cm.__aexit__ = AsyncMock(return_value=False)

            mock_client = MagicMock()
            mock_client.messages.stream = MagicMock(return_value=mock_cm)
            provider._async_client = mock_client

            result = StreamResult()
            chunks = []
            async for chunk in _stream_anthropic(
                provider,
                [{"role": "user", "content": "hi"}],
                "Be helpful.",
                None,
                result,
            ):
                chunks.append(chunk)

            assert chunks == ["Hello ", "world"]
            assert result.input_tokens == 42
            assert result.output_tokens == 18

        asyncio.run(_run())

    def test_no_async_client_raises(self):
        async def _run():
            provider = _make_provider(ProviderType.ANTHROPIC)
            provider._async_client = None

            result = StreamResult()
            with pytest.raises(ProviderError, match="async client not available"):
                async for _ in _stream_anthropic(
                    provider, [{"role": "user", "content": "hi"}], None, None, result
                ):
                    pass

        asyncio.run(_run())

    def test_uses_default_model_when_none(self):
        async def _run():
            provider = _make_provider(ProviderType.ANTHROPIC)
            provider.default_model = "claude-sonnet-4-20250514"

            mock_stream_ctx = AsyncMock()

            async def empty_stream():
                return
                yield  # make it a generator

            mock_stream_ctx.text_stream = empty_stream()
            mock_response = SimpleNamespace(usage=None)
            mock_stream_ctx.get_final_message = AsyncMock(return_value=mock_response)

            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
            mock_cm.__aexit__ = AsyncMock(return_value=False)

            mock_client = MagicMock()
            mock_client.messages.stream = MagicMock(return_value=mock_cm)
            provider._async_client = mock_client

            result = StreamResult()
            async for _ in _stream_anthropic(
                provider, [{"role": "user", "content": "hi"}], None, None, result
            ):
                pass

            call_kwargs = mock_client.messages.stream.call_args[1]
            assert call_kwargs["model"] == "claude-sonnet-4-20250514"

        asyncio.run(_run())

    def test_uses_override_model(self):
        async def _run():
            provider = _make_provider(ProviderType.ANTHROPIC)

            mock_stream_ctx = AsyncMock()

            async def empty_stream():
                return
                yield

            mock_stream_ctx.text_stream = empty_stream()
            mock_response = SimpleNamespace(usage=None)
            mock_stream_ctx.get_final_message = AsyncMock(return_value=mock_response)

            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
            mock_cm.__aexit__ = AsyncMock(return_value=False)

            mock_client = MagicMock()
            mock_client.messages.stream = MagicMock(return_value=mock_cm)
            provider._async_client = mock_client

            result = StreamResult()
            async for _ in _stream_anthropic(
                provider,
                [{"role": "user", "content": "hi"}],
                None,
                "claude-opus-4-20250514",
                result,
            ):
                pass

            call_kwargs = mock_client.messages.stream.call_args[1]
            assert call_kwargs["model"] == "claude-opus-4-20250514"

        asyncio.run(_run())

    def test_no_usage_in_response(self):
        """When response.usage is None, tokens stay at 0."""

        async def _run():
            provider = _make_provider(ProviderType.ANTHROPIC)

            mock_stream_ctx = AsyncMock()

            async def text_gen():
                yield "hi"

            mock_stream_ctx.text_stream = text_gen()
            mock_response = SimpleNamespace(usage=None)
            mock_stream_ctx.get_final_message = AsyncMock(return_value=mock_response)

            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
            mock_cm.__aexit__ = AsyncMock(return_value=False)

            mock_client = MagicMock()
            mock_client.messages.stream = MagicMock(return_value=mock_cm)
            provider._async_client = mock_client

            result = StreamResult()
            async for _ in _stream_anthropic(
                provider, [{"role": "user", "content": "hi"}], None, None, result
            ):
                pass

            assert result.input_tokens == 0
            assert result.output_tokens == 0

        asyncio.run(_run())

    def test_none_response(self):
        """When get_final_message returns None, tokens stay at 0."""

        async def _run():
            provider = _make_provider(ProviderType.ANTHROPIC)

            mock_stream_ctx = AsyncMock()

            async def text_gen():
                yield "hi"

            mock_stream_ctx.text_stream = text_gen()
            mock_stream_ctx.get_final_message = AsyncMock(return_value=None)

            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
            mock_cm.__aexit__ = AsyncMock(return_value=False)

            mock_client = MagicMock()
            mock_client.messages.stream = MagicMock(return_value=mock_cm)
            provider._async_client = mock_client

            result = StreamResult()
            async for _ in _stream_anthropic(
                provider, [{"role": "user", "content": "hi"}], None, None, result
            ):
                pass

            assert result.input_tokens == 0
            assert result.output_tokens == 0

        asyncio.run(_run())


class TestStreamOpenAIDirect:
    """Test _stream_openai() with mocked OpenAI client."""

    def test_streams_text_and_captures_usage(self):
        async def _run():
            provider = _make_provider(ProviderType.OPENAI)

            # Build mock chunks
            chunk1 = SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))],
                usage=None,
            )
            chunk2 = SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="world"))],
                usage=None,
            )
            # Final chunk with usage
            chunk3 = SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=30, completion_tokens=10),
            )

            async def mock_stream_gen():
                yield chunk1
                yield chunk2
                yield chunk3

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_stream_gen())
            provider._async_client = mock_client

            result = StreamResult()
            chunks = []
            async for chunk in _stream_openai(
                provider,
                [{"role": "user", "content": "hi"}],
                "Be helpful.",
                None,
                result,
            ):
                chunks.append(chunk)

            assert chunks == ["Hello ", "world"]
            assert result.input_tokens == 30
            assert result.output_tokens == 10

        asyncio.run(_run())

    def test_no_async_client_raises(self):
        async def _run():
            provider = _make_provider(ProviderType.OPENAI)
            provider._async_client = None

            result = StreamResult()
            with pytest.raises(ProviderError, match="async client not available"):
                async for _ in _stream_openai(
                    provider, [{"role": "user", "content": "hi"}], None, None, result
                ):
                    pass

        asyncio.run(_run())

    def test_prepends_system_prompt(self):
        async def _run():
            provider = _make_provider(ProviderType.OPENAI)

            async def empty_gen():
                return
                yield

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=empty_gen())
            provider._async_client = mock_client

            result = StreamResult()
            async for _ in _stream_openai(
                provider,
                [{"role": "user", "content": "hi"}],
                "System instruction",
                None,
                result,
            ):
                pass

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            messages = call_kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "System instruction"

        asyncio.run(_run())

    def test_no_system_prompt(self):
        async def _run():
            provider = _make_provider(ProviderType.OPENAI)

            async def empty_gen():
                return
                yield

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=empty_gen())
            provider._async_client = mock_client

            result = StreamResult()
            async for _ in _stream_openai(
                provider,
                [{"role": "user", "content": "hi"}],
                None,
                None,
                result,
            ):
                pass

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            messages = call_kwargs["messages"]
            # No system message prepended
            assert all(m["role"] != "system" for m in messages)

        asyncio.run(_run())

    def test_chunk_without_choices_skipped(self):
        """Chunks with empty choices should not yield content."""

        async def _run():
            provider = _make_provider(ProviderType.OPENAI)

            empty_chunk = SimpleNamespace(choices=[], usage=None)

            async def mock_gen():
                yield empty_chunk

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_gen())
            provider._async_client = mock_client

            result = StreamResult()
            chunks = []
            async for chunk in _stream_openai(
                provider, [{"role": "user", "content": "hi"}], None, None, result
            ):
                chunks.append(chunk)

            assert chunks == []

        asyncio.run(_run())

    def test_chunk_with_none_content_skipped(self):
        """Chunks where delta.content is None should not yield."""

        async def _run():
            provider = _make_provider(ProviderType.OPENAI)

            none_content_chunk = SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None))],
                usage=None,
            )

            async def mock_gen():
                yield none_content_chunk

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_gen())
            provider._async_client = mock_client

            result = StreamResult()
            chunks = []
            async for chunk in _stream_openai(
                provider, [{"role": "user", "content": "hi"}], None, None, result
            ):
                chunks.append(chunk)

            assert chunks == []

        asyncio.run(_run())


class TestStreamOllamaDirect:
    """Test _stream_ollama() usage capture and edge cases."""

    def test_yields_content_chunks(self):
        async def _run():
            provider = MagicMock()

            async def mock_stream(request):
                yield StreamChunk(content="hello ", model="llama", provider="ollama")
                yield StreamChunk(content="world", model="llama", provider="ollama", is_final=True)

            provider.complete_stream_async = mock_stream
            result = StreamResult()

            chunks = []
            async for chunk in _stream_ollama(
                provider,
                [{"role": "user", "content": "hi"}],
                "Be helpful",
                None,
                result,
            ):
                chunks.append(chunk)

            assert chunks == ["hello ", "world"]

        asyncio.run(_run())

    def test_skips_empty_content(self):
        async def _run():
            provider = MagicMock()

            async def mock_stream(request):
                yield StreamChunk(content="", model="llama", provider="ollama")
                yield StreamChunk(content="data", model="llama", provider="ollama")

            provider.complete_stream_async = mock_stream
            result = StreamResult()

            chunks = []
            async for chunk in _stream_ollama(
                provider, [{"role": "user", "content": "hi"}], None, None, result
            ):
                chunks.append(chunk)

            assert chunks == ["data"]

        asyncio.run(_run())

    def test_captures_usage_from_chunk(self):
        async def _run():
            provider = MagicMock()

            chunk_with_usage = StreamChunk(content="done", model="llama", provider="ollama")
            chunk_with_usage.usage = SimpleNamespace(input_tokens=25, output_tokens=15)

            async def mock_stream(request):
                yield chunk_with_usage

            provider.complete_stream_async = mock_stream
            result = StreamResult()

            async for _ in _stream_ollama(
                provider, [{"role": "user", "content": "hi"}], None, None, result
            ):
                pass

            assert result.input_tokens == 25
            assert result.output_tokens == 15

        asyncio.run(_run())

    def test_empty_messages_uses_empty_prompt(self):
        async def _run():
            provider = MagicMock()
            captured_request = None

            async def mock_stream(request):
                nonlocal captured_request
                captured_request = request
                return
                yield

            provider.complete_stream_async = mock_stream
            result = StreamResult()

            async for _ in _stream_ollama(provider, [], "system", "model-x", result):
                pass

            assert captured_request is not None
            assert captured_request.prompt == ""

        asyncio.run(_run())

    def test_passes_model_and_system_prompt(self):
        async def _run():
            provider = MagicMock()
            captured_request = None

            async def mock_stream(request):
                nonlocal captured_request
                captured_request = request
                return
                yield

            provider.complete_stream_async = mock_stream
            result = StreamResult()

            async for _ in _stream_ollama(
                provider,
                [{"role": "user", "content": "hi"}],
                "Be a pirate",
                "llama3.2:3b",
                result,
            ):
                pass

            assert captured_request.system_prompt == "Be a pirate"
            assert captured_request.model == "llama3.2:3b"

        asyncio.run(_run())


class TestStreamFallbackDirect:
    """Test _stream_fallback() usage capture."""

    def test_yields_single_chunk(self):
        async def _run():
            provider = MagicMock()
            provider.complete_async = AsyncMock(
                return_value=CompletionResponse(
                    content="full response",
                    model="test",
                    provider="test",
                    input_tokens=5,
                    output_tokens=10,
                )
            )
            result = StreamResult()
            chunks = []
            async for chunk in _stream_fallback(
                provider,
                [{"role": "user", "content": "hi"}],
                "sys prompt",
                None,
                result,
            ):
                chunks.append(chunk)
            assert chunks == ["full response"]

        asyncio.run(_run())

    def test_captures_usage_via_response_attributes(self):
        async def _run():
            provider = MagicMock()
            response = CompletionResponse(
                content="ok",
                model="test",
                provider="test",
                input_tokens=100,
                output_tokens=50,
            )
            # CompletionResponse has input_tokens/output_tokens directly,
            # but the code checks for response.usage. Add a usage attr.
            response.usage = SimpleNamespace(input_tokens=100, output_tokens=50)
            provider.complete_async = AsyncMock(return_value=response)

            result = StreamResult()
            async for _ in _stream_fallback(
                provider, [{"role": "user", "content": "hi"}], None, None, result
            ):
                pass

            assert result.input_tokens == 100
            assert result.output_tokens == 50

        asyncio.run(_run())

    def test_no_usage_attribute(self):
        """When response has no usage attribute, tokens stay at 0."""

        async def _run():
            provider = MagicMock()
            response = MagicMock()
            response.content = "done"
            # Explicitly remove usage
            del response.usage
            provider.complete_async = AsyncMock(return_value=response)

            result = StreamResult()
            async for _ in _stream_fallback(
                provider, [{"role": "user", "content": "hi"}], None, None, result
            ):
                pass

            assert result.input_tokens == 0
            assert result.output_tokens == 0

        asyncio.run(_run())

    def test_empty_messages_uses_empty_prompt(self):
        async def _run():
            provider = MagicMock()
            response = MagicMock()
            response.content = "result"
            del response.usage
            provider.complete_async = AsyncMock(return_value=response)

            result = StreamResult()
            async for _ in _stream_fallback(provider, [], None, "model-x", result):
                pass

            call_args = provider.complete_async.call_args[0][0]
            assert isinstance(call_args, CompletionRequest)
            assert call_args.prompt == ""
            assert call_args.model == "model-x"

        asyncio.run(_run())


class TestStreamCompletionIntegration:
    """Integration-level tests for stream_completion routing + error handling."""

    def test_initialize_exception_is_swallowed(self):
        """Provider.initialize() raising should not break streaming."""

        async def _run():
            provider = _make_provider(ProviderType.ANTHROPIC)
            provider.initialize.side_effect = RuntimeError("already init")

            with patch("animus_forge.tui.streaming._stream_anthropic") as mock_stream:

                async def gen(*args, **kwargs):
                    yield "ok"

                mock_stream.return_value = gen()

                chunks = []
                async for chunk in stream_completion(provider, [{"role": "user", "content": "hi"}]):
                    chunks.append(chunk)
                assert chunks == ["ok"]

        asyncio.run(_run())

    def test_multiple_system_messages_concatenated(self):
        """Multiple system messages should be joined with double newline."""

        async def _run():
            provider = _make_provider(ProviderType.ANTHROPIC)
            messages = [
                {"role": "system", "content": "Rule 1"},
                {"role": "system", "content": "Rule 2"},
                {"role": "user", "content": "hi"},
            ]

            with patch("animus_forge.tui.streaming._stream_anthropic") as mock_stream:

                async def gen(*args, **kwargs):
                    yield "ok"

                mock_stream.return_value = gen()

                async for _ in stream_completion(provider, messages):
                    pass

                call_args = mock_stream.call_args[0]
                system_prompt = call_args[2]
                assert "Rule 1" in system_prompt
                assert "Rule 2" in system_prompt
                assert "\n\n" in system_prompt

        asyncio.run(_run())

    def test_creates_default_result_when_none(self):
        """When result=None, a fresh StreamResult is created internally."""

        async def _run():
            provider = _make_provider(ProviderType.ANTHROPIC)

            with patch("animus_forge.tui.streaming._stream_anthropic") as mock_stream:

                async def gen(prov, msgs, sys, model, res):
                    assert isinstance(res, StreamResult)
                    yield "ok"

                mock_stream.side_effect = gen

                async for _ in stream_completion(
                    provider, [{"role": "user", "content": "hi"}], result=None
                ):
                    pass

        asyncio.run(_run())

    def test_no_system_messages_leaves_none(self):
        """When no system messages in list and no explicit prompt, system_prompt stays None."""

        async def _run():
            provider = _make_provider(ProviderType.ANTHROPIC)
            messages = [{"role": "user", "content": "hi"}]

            with patch("animus_forge.tui.streaming._stream_anthropic") as mock_stream:

                async def gen(*args, **kwargs):
                    yield "ok"

                mock_stream.return_value = gen()

                async for _ in stream_completion(provider, messages):
                    pass

                call_args = mock_stream.call_args[0]
                system_prompt = call_args[2]
                assert system_prompt is None

        asyncio.run(_run())

    def test_provider_error_not_caught(self):
        """ProviderError should propagate without fallback."""

        async def _run():
            provider = _make_provider(ProviderType.OPENAI)

            with patch(
                "animus_forge.tui.streaming._stream_openai",
                side_effect=ProviderError("invalid key"),
            ):
                with pytest.raises(ProviderError, match="invalid key"):
                    async for _ in stream_completion(provider, [{"role": "user", "content": "hi"}]):
                        pass

        asyncio.run(_run())

    def test_generic_error_triggers_fallback(self):
        """Non-ProviderError exceptions should trigger fallback."""

        async def _run():
            provider = _make_provider(ProviderType.OPENAI)

            with (
                patch(
                    "animus_forge.tui.streaming._stream_openai",
                    side_effect=ConnectionError("network"),
                ),
                patch("animus_forge.tui.streaming._stream_fallback") as mock_fb,
            ):

                async def gen(*args, **kwargs):
                    yield "fallback response"

                mock_fb.return_value = gen()

                chunks = []
                async for chunk in stream_completion(provider, [{"role": "user", "content": "hi"}]):
                    chunks.append(chunk)
                assert chunks == ["fallback response"]

        asyncio.run(_run())

    def test_ollama_error_triggers_fallback(self):
        """Ollama streaming error should trigger fallback."""

        async def _run():
            provider = _make_provider(ProviderType.OLLAMA)

            with (
                patch(
                    "animus_forge.tui.streaming._stream_ollama",
                    side_effect=OSError("connection refused"),
                ),
                patch("animus_forge.tui.streaming._stream_fallback") as mock_fb,
            ):

                async def gen(*args, **kwargs):
                    yield "recovered"

                mock_fb.return_value = gen()

                chunks = []
                async for chunk in stream_completion(provider, [{"role": "user", "content": "hi"}]):
                    chunks.append(chunk)
                assert chunks == ["recovered"]

        asyncio.run(_run())
