"""Supplementary tests for animus_forge.tui.commands — targets uncovered branches.

Covers: _clipboard_copy(), cmd_switch edge cases, cmd_model edge cases,
cmd_models edge cases, cmd_providers edge cases, cmd_tokens no-screen,
cmd_file confirm/directory, cmd_files with files, cmd_unfile branches,
cmd_load (list + numeric), cmd_history with sessions, cmd_title empty args,
cmd_copy with assistant response.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from animus_forge.tui.commands import (
    CommandRegistry,
    _clipboard_copy,
    _is_sensitive_path,
    create_command_registry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(
    *,
    provider_manager: MagicMock | None = "default",
    with_screen: bool = True,
) -> MagicMock:
    """Build a mock GorgonApp suitable for command tests.

    Args:
        provider_manager: Set to None to simulate missing provider_manager.
            "default" creates a standard MagicMock.
        with_screen: If True, wire _get_chat_screen to return a mock ChatScreen.
    """
    app = MagicMock()
    app._agent_mode = "off"
    app._system_prompt = None
    app._messages = []
    app._session = None
    app._supervisor = None

    if provider_manager == "default":
        app._provider_manager = MagicMock()
    else:
        app._provider_manager = provider_manager

    if with_screen:
        screen = MagicMock()
        screen.sidebar = MagicMock()
        screen.sidebar.files = []
        screen.sidebar.input_tokens = 0
        screen.sidebar.output_tokens = 0
        screen.status_bar = MagicMock()
        screen.chat_display = MagicMock()
        app.screen = screen
        app._get_chat_screen = MagicMock(return_value=screen)
    else:
        app._get_chat_screen = MagicMock(return_value=None)

    return app


def _run(coro):
    """Synchronous wrapper for async coroutines (no pytest-asyncio)."""
    return asyncio.run(coro)


# ===========================================================================
# _clipboard_copy — cross-platform clipboard
# ===========================================================================


class TestClipboardCopy:
    """Tests for _clipboard_copy() — all subprocess calls mocked."""

    def test_darwin_pbcopy_success(self):
        with (
            patch("animus_forge.tui.commands.platform") as mock_platform,
            patch("animus_forge.tui.commands.subprocess") as mock_sub,
        ):
            mock_platform.system.return_value = "Darwin"
            proc = MagicMock()
            proc.returncode = 0
            mock_sub.run.return_value = proc
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired

            result = _clipboard_copy("hello world")
            assert "copied" in result.lower()
            mock_sub.run.assert_called_once()
            call_args = mock_sub.run.call_args
            assert call_args[0][0] == ["pbcopy"]
            assert call_args[1]["input"] == b"hello world"

    def test_windows_clip_success(self):
        with (
            patch("animus_forge.tui.commands.platform") as mock_platform,
            patch("animus_forge.tui.commands.subprocess") as mock_sub,
        ):
            mock_platform.system.return_value = "Windows"
            proc = MagicMock()
            proc.returncode = 0
            mock_sub.run.return_value = proc
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired

            result = _clipboard_copy("data")
            assert "copied" in result.lower()
            call_args = mock_sub.run.call_args
            assert call_args[0][0] == ["clip"]

    def test_linux_xclip_success(self):
        with (
            patch("animus_forge.tui.commands.platform") as mock_platform,
            patch("animus_forge.tui.commands.shutil") as mock_shutil,
            patch("animus_forge.tui.commands.subprocess") as mock_sub,
        ):
            mock_platform.system.return_value = "Linux"
            mock_shutil.which.side_effect = lambda name: (
                "/usr/bin/xclip" if name == "xclip" else None
            )
            proc = MagicMock()
            proc.returncode = 0
            mock_sub.run.return_value = proc
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired

            result = _clipboard_copy("test")
            assert "copied" in result.lower()
            call_args = mock_sub.run.call_args
            assert call_args[0][0] == ["/usr/bin/xclip", "-selection", "clipboard"]

    def test_linux_xsel_fallback(self):
        with (
            patch("animus_forge.tui.commands.platform") as mock_platform,
            patch("animus_forge.tui.commands.shutil") as mock_shutil,
            patch("animus_forge.tui.commands.subprocess") as mock_sub,
        ):
            mock_platform.system.return_value = "Linux"
            mock_shutil.which.side_effect = lambda name: "/usr/bin/xsel" if name == "xsel" else None
            proc = MagicMock()
            proc.returncode = 0
            mock_sub.run.return_value = proc
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired

            result = _clipboard_copy("test")
            assert "copied" in result.lower()
            call_args = mock_sub.run.call_args
            assert call_args[0][0] == ["/usr/bin/xsel", "--clipboard", "--input"]

    def test_linux_no_clipboard_utility(self):
        with (
            patch("animus_forge.tui.commands.platform") as mock_platform,
            patch("animus_forge.tui.commands.shutil") as mock_shutil,
        ):
            mock_platform.system.return_value = "Linux"
            mock_shutil.which.return_value = None

            result = _clipboard_copy("test")
            assert "No clipboard utility" in result
            assert "xclip" in result

    def test_nonzero_returncode(self):
        with (
            patch("animus_forge.tui.commands.platform") as mock_platform,
            patch("animus_forge.tui.commands.subprocess") as mock_sub,
        ):
            mock_platform.system.return_value = "Darwin"
            proc = MagicMock()
            proc.returncode = 1
            proc.stderr = b"some error"
            mock_sub.run.return_value = proc
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired

            result = _clipboard_copy("test")
            assert "failed" in result.lower()
            assert "pbcopy" in result

    def test_timeout_expired(self):
        with (
            patch("animus_forge.tui.commands.platform") as mock_platform,
            patch("animus_forge.tui.commands.subprocess") as mock_sub,
        ):
            mock_platform.system.return_value = "Darwin"
            mock_sub.run.side_effect = subprocess.TimeoutExpired(cmd="pbcopy", timeout=5)
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired

            result = _clipboard_copy("test")
            assert "timed out" in result.lower()

    def test_file_not_found(self):
        with (
            patch("animus_forge.tui.commands.platform") as mock_platform,
            patch("animus_forge.tui.commands.subprocess") as mock_sub,
        ):
            mock_platform.system.return_value = "Darwin"
            mock_sub.run.side_effect = FileNotFoundError("pbcopy not found")
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired

            result = _clipboard_copy("test")
            assert "not found" in result.lower()

    def test_generic_exception(self):
        with (
            patch("animus_forge.tui.commands.platform") as mock_platform,
            patch("animus_forge.tui.commands.subprocess") as mock_sub,
        ):
            mock_platform.system.return_value = "Darwin"
            mock_sub.run.side_effect = OSError("broken")
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired

            result = _clipboard_copy("test")
            assert "Copy failed" in result
            assert "OSError" in result


# ===========================================================================
# cmd_switch edge cases
# ===========================================================================


class TestCmdSwitch:
    def test_switch_no_provider_manager(self):
        app = _make_app(provider_manager=None)
        registry = create_command_registry(app)
        result = _run(registry.execute("switch", "openai"))
        assert "No provider manager" in result

    def test_switch_provider_returns_none(self):
        app = _make_app()
        app._provider_manager.get_default.return_value = None
        registry = create_command_registry(app)
        result = _run(registry.execute("switch", "openai"))
        assert "failed" in result.lower()

    def test_switch_exception(self):
        app = _make_app()
        app._provider_manager.set_default.side_effect = ValueError("bad provider")
        registry = create_command_registry(app)
        result = _run(registry.execute("switch", "bad"))
        assert "failed" in result.lower()
        assert "bad provider" in result

    def test_switch_no_args_no_provider_manager(self):
        app = _make_app(provider_manager=None)
        registry = create_command_registry(app)
        result = _run(registry.execute("switch", ""))
        assert "Usage" in result


# ===========================================================================
# cmd_model edge cases
# ===========================================================================


class TestCmdModel:
    def test_model_no_args(self):
        app = _make_app()
        registry = create_command_registry(app)
        result = _run(registry.execute("model", ""))
        assert "Usage" in result

    def test_model_no_provider_manager(self):
        app = _make_app(provider_manager=None)
        registry = create_command_registry(app)
        result = _run(registry.execute("model", "gpt-4o"))
        assert "No provider manager" in result

    def test_model_no_default_provider(self):
        app = _make_app()
        app._provider_manager.get_default.return_value = None
        registry = create_command_registry(app)
        result = _run(registry.execute("model", "gpt-4o"))
        assert "No default provider" in result

    def test_model_no_chat_screen(self):
        app = _make_app(with_screen=False)
        provider = MagicMock()
        app._provider_manager.get_default.return_value = provider
        registry = create_command_registry(app)
        result = _run(registry.execute("model", "gpt-4o"))
        assert "gpt-4o" in result
        assert provider.config.default_model == "gpt-4o"


# ===========================================================================
# cmd_models edge cases
# ===========================================================================


class TestCmdModels:
    def test_models_no_provider_manager(self):
        app = _make_app(provider_manager=None)
        registry = create_command_registry(app)
        result = _run(registry.execute("models", ""))
        assert "No provider manager" in result

    def test_models_no_default_provider(self):
        app = _make_app()
        app._provider_manager.get_default.return_value = None
        registry = create_command_registry(app)
        result = _run(registry.execute("models", ""))
        assert "No default provider" in result


# ===========================================================================
# cmd_providers edge cases
# ===========================================================================


class TestCmdProviders:
    def test_providers_no_provider_manager(self):
        app = _make_app(provider_manager=None)
        registry = create_command_registry(app)
        result = _run(registry.execute("providers", ""))
        assert "No provider manager" in result


# ===========================================================================
# cmd_tokens edge cases
# ===========================================================================


class TestCmdTokens:
    def test_tokens_no_chat_screen(self):
        app = _make_app(with_screen=False)
        registry = create_command_registry(app)
        result = _run(registry.execute("tokens", ""))
        assert "unavailable" in result.lower()


# ===========================================================================
# cmd_file edge cases
# ===========================================================================


class TestCmdFile:
    def test_file_no_args(self):
        app = _make_app()
        registry = create_command_registry(app)
        result = _run(registry.execute("file", ""))
        assert "Usage" in result

    def test_file_is_directory(self, tmp_path):
        app = _make_app()
        registry = create_command_registry(app)
        result = _run(registry.execute("file", str(tmp_path)))
        assert "Not a file" in result

    def test_file_sensitive_confirm(self, tmp_path):
        """Confirm flag bypasses the sensitive file warning."""
        app = _make_app()
        f = tmp_path / ".env"
        f.write_text("SECRET=abc")
        registry = create_command_registry(app)
        result = _run(registry.execute("file", f"confirm:{f}"))
        assert "Added" in result

    def test_file_sensitive_second_attempt(self, tmp_path):
        """Second attempt at same sensitive file proceeds (already confirmed)."""
        app = _make_app()
        f = tmp_path / ".env"
        f.write_text("SECRET=abc")
        registry = create_command_registry(app)
        # First attempt triggers warning
        result1 = _run(registry.execute("file", str(f)))
        assert "warning" in result1.lower() or "sensitive" in result1.lower()
        # Second attempt (same file, same registry) — confirmed set allows through
        result2 = _run(registry.execute("file", str(f)))
        assert "Added" in result2

    def test_file_no_chat_screen(self, tmp_path):
        """Adding a file when there's no chat screen still returns Added."""
        app = _make_app(with_screen=False)
        f = tmp_path / "readme.txt"
        f.write_text("hello")
        registry = create_command_registry(app)
        result = _run(registry.execute("file", str(f)))
        assert "Added" in result


# ===========================================================================
# cmd_files edge cases
# ===========================================================================


class TestCmdFiles:
    def test_files_with_attachments(self):
        app = _make_app()
        cs = app._get_chat_screen()
        cs.sidebar.files = ["/tmp/a.py", "/tmp/b.py"]
        registry = create_command_registry(app)
        result = _run(registry.execute("files", ""))
        assert "/tmp/a.py" in result
        assert "/tmp/b.py" in result
        assert "Attached files" in result

    def test_files_no_sidebar(self):
        app = _make_app(with_screen=False)
        registry = create_command_registry(app)
        result = _run(registry.execute("files", ""))
        assert "No sidebar" in result


# ===========================================================================
# cmd_unfile edge cases
# ===========================================================================


class TestCmdUnfile:
    def test_unfile_no_args(self):
        app = _make_app()
        registry = create_command_registry(app)
        result = _run(registry.execute("unfile", ""))
        assert "Usage" in result

    def test_unfile_no_sidebar(self):
        app = _make_app(with_screen=False)
        registry = create_command_registry(app)
        result = _run(registry.execute("unfile", "test.py"))
        assert "No sidebar" in result

    def test_unfile_specific_file(self):
        app = _make_app()
        registry = create_command_registry(app)
        result = _run(registry.execute("unfile", "/tmp/test.py"))
        assert "Removed" in result
        cs = app._get_chat_screen()
        cs.sidebar.remove_file.assert_called_once_with("/tmp/test.py")


# ===========================================================================
# cmd_save edge cases
# ===========================================================================


class TestCmdSave:
    def test_save_no_title(self):
        """Save without explicit title uses auto-generated title."""
        app = _make_app()
        app._messages = [{"role": "user", "content": "hello"}]
        provider = MagicMock()
        provider.name = "openai"
        provider.default_model = "gpt-4o"
        app._provider_manager.get_default.return_value = provider
        registry = create_command_registry(app)

        with patch("animus_forge.tui.session.TUISession.save"):
            result = _run(registry.execute("save", ""))
            assert "saved" in result.lower() or "Session" in result

    def test_save_no_provider(self):
        """Save when no provider is configured still works (empty provider info)."""
        app = _make_app()
        app._messages = []
        app._provider_manager.get_default.return_value = None
        registry = create_command_registry(app)

        with patch("animus_forge.tui.session.TUISession.save"):
            result = _run(registry.execute("save", "test"))
            assert "saved" in result.lower() or "Session" in result

    def test_save_no_provider_manager(self):
        """Save when provider_manager is None."""
        app = _make_app(provider_manager=None)
        app._messages = []
        registry = create_command_registry(app)

        with patch("animus_forge.tui.session.TUISession.save"):
            result = _run(registry.execute("save", "test"))
            assert "saved" in result.lower() or "Session" in result


# ===========================================================================
# cmd_load
# ===========================================================================


class TestCmdLoad:
    def test_load_no_sessions(self):
        app = _make_app()
        registry = create_command_registry(app)
        with patch("animus_forge.tui.session.TUISession.list_sessions", return_value=[]):
            result = _run(registry.execute("load", ""))
            assert "No saved" in result

    def test_load_list_sessions(self):
        """Calling /load without numeric arg lists available sessions."""
        app = _make_app()
        registry = create_command_registry(app)
        fake_sessions = [
            Path("/tmp/2026-01-01_session1.json"),
            Path("/tmp/2026-01-02_session2.json"),
        ]
        with patch(
            "animus_forge.tui.session.TUISession.list_sessions",
            return_value=fake_sessions,
        ):
            result = _run(registry.execute("load", ""))
            assert "Saved sessions" in result
            assert "session1" in result
            assert "session2" in result
            assert "/load <number>" in result

    def test_load_by_index(self):
        """Calling /load <number> loads the corresponding session."""
        app = _make_app()
        registry = create_command_registry(app)
        fake_sessions = [
            Path("/tmp/2026-01-01_session1.json"),
            Path("/tmp/2026-01-02_session2.json"),
        ]
        mock_session = MagicMock()
        mock_session.messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        mock_session.system_prompt = "Be helpful"

        with (
            patch(
                "animus_forge.tui.session.TUISession.list_sessions",
                return_value=fake_sessions,
            ),
            patch(
                "animus_forge.tui.session.TUISession.load",
                return_value=mock_session,
            ) as mock_load,
        ):
            result = _run(registry.execute("load", "1"))
            assert "Loaded" in result
            assert "session2" in result
            mock_load.assert_called_once_with(fake_sessions[1])
            assert app._session is mock_session
            assert app._messages is mock_session.messages
            assert app._system_prompt == "Be helpful"

    def test_load_by_index_with_chat_screen(self):
        """Loading a session replays messages into chat display."""
        app = _make_app()
        cs = app._get_chat_screen()
        registry = create_command_registry(app)
        fake_sessions = [Path("/tmp/2026-01-01_test.json")]
        mock_session = MagicMock()
        mock_session.messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        mock_session.system_prompt = None

        with (
            patch(
                "animus_forge.tui.session.TUISession.list_sessions",
                return_value=fake_sessions,
            ),
            patch(
                "animus_forge.tui.session.TUISession.load",
                return_value=mock_session,
            ),
        ):
            result = _run(registry.execute("load", "0"))
            assert "Loaded" in result
            cs.chat_display.clear.assert_called_once()
            cs.chat_display.add_user_message.assert_called_once_with("question")
            cs.chat_display.add_assistant_message.assert_called_once_with("answer")

    def test_load_invalid_index(self):
        """Invalid numeric index returns error."""
        app = _make_app()
        registry = create_command_registry(app)
        fake_sessions = [Path("/tmp/2026-01-01_test.json")]

        with patch(
            "animus_forge.tui.session.TUISession.list_sessions",
            return_value=fake_sessions,
        ):
            result = _run(registry.execute("load", "99"))
            assert "Invalid index" in result

    def test_load_no_chat_screen(self):
        """Loading a session when there's no chat screen still works."""
        app = _make_app(with_screen=False)
        registry = create_command_registry(app)
        fake_sessions = [Path("/tmp/2026-01-01_test.json")]
        mock_session = MagicMock()
        mock_session.messages = [{"role": "user", "content": "hi"}]
        mock_session.system_prompt = None

        with (
            patch(
                "animus_forge.tui.session.TUISession.list_sessions",
                return_value=fake_sessions,
            ),
            patch(
                "animus_forge.tui.session.TUISession.load",
                return_value=mock_session,
            ),
        ):
            result = _run(registry.execute("load", "0"))
            assert "Loaded" in result


# ===========================================================================
# cmd_history
# ===========================================================================


class TestCmdHistory:
    def test_history_with_sessions(self):
        app = _make_app()
        registry = create_command_registry(app)
        fake_sessions = [
            Path("/tmp/2026-01-01_chat-about-code.json"),
            Path("/tmp/2026-01-02_debugging-session.json"),
        ]
        with patch(
            "animus_forge.tui.session.TUISession.list_sessions",
            return_value=fake_sessions,
        ):
            result = _run(registry.execute("history", ""))
            assert "Session history" in result
            assert "chat-about-code" in result
            assert "debugging-session" in result

    def test_history_empty(self):
        app = _make_app()
        registry = create_command_registry(app)
        with patch(
            "animus_forge.tui.session.TUISession.list_sessions",
            return_value=[],
        ):
            result = _run(registry.execute("history", ""))
            assert "No saved" in result


# ===========================================================================
# cmd_title edge cases
# ===========================================================================


class TestCmdTitle:
    def test_title_no_args(self):
        app = _make_app()
        registry = create_command_registry(app)
        result = _run(registry.execute("title", ""))
        assert "Usage" in result

    def test_title_with_active_session(self):
        app = _make_app()
        app._session = MagicMock()
        registry = create_command_registry(app)
        result = _run(registry.execute("title", "My Title"))
        assert app._session.title == "My Title"
        app._session.save.assert_called_once()
        assert "My Title" in result

    def test_title_no_session(self):
        app = _make_app()
        app._session = None
        registry = create_command_registry(app)
        result = _run(registry.execute("title", "Test"))
        assert "No active session" in result


# ===========================================================================
# cmd_copy — with actual clipboard call
# ===========================================================================


class TestCmdCopy:
    def test_copy_success(self):
        """Copy finds the last assistant message and calls _clipboard_copy."""
        app = _make_app()
        app._messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]
        registry = create_command_registry(app)

        with patch(
            "animus_forge.tui.commands._clipboard_copy",
            return_value="Last response copied to clipboard.",
        ) as mock_clip:
            result = _run(registry.execute("copy", ""))
            assert "copied" in result.lower()
            mock_clip.assert_called_once_with("Hello! How can I help?")

    def test_copy_multiple_assistant_messages(self):
        """Copy uses the LAST assistant message."""
        app = _make_app()
        app._messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "response two"},
        ]
        registry = create_command_registry(app)

        with patch(
            "animus_forge.tui.commands._clipboard_copy",
            return_value="Last response copied to clipboard.",
        ) as mock_clip:
            _run(registry.execute("copy", ""))
            mock_clip.assert_called_once_with("response two")

    def test_copy_no_messages(self):
        app = _make_app()
        app._messages = []
        registry = create_command_registry(app)
        result = _run(registry.execute("copy", ""))
        assert "No messages" in result

    def test_copy_only_user_messages(self):
        app = _make_app()
        app._messages = [{"role": "user", "content": "hi"}]
        registry = create_command_registry(app)
        result = _run(registry.execute("copy", ""))
        assert "No assistant" in result


# ===========================================================================
# cmd_system edge cases
# ===========================================================================


class TestCmdSystem:
    def test_system_show_when_none(self):
        app = _make_app()
        app._system_prompt = None
        registry = create_command_registry(app)
        result = _run(registry.execute("system", ""))
        assert "(none)" in result

    def test_system_set_shows_char_count(self):
        app = _make_app()
        registry = create_command_registry(app)
        result = _run(registry.execute("system", "Be concise and direct."))
        assert "22 chars" in result
        assert app._system_prompt == "Be concise and direct."


# ===========================================================================
# cmd_agent edge cases
# ===========================================================================


class TestCmdAgent:
    def test_agent_resets_supervisor(self):
        """Switching agent mode sets _supervisor to None."""
        app = _make_app()
        app._supervisor = MagicMock()
        registry = create_command_registry(app)
        _run(registry.execute("agent", "builder"))
        assert app._agent_mode == "builder"
        assert app._supervisor is None

    def test_agent_no_chat_screen(self):
        """Setting agent mode without chat screen still works."""
        app = _make_app(with_screen=False)
        registry = create_command_registry(app)
        result = _run(registry.execute("agent", "tester"))
        assert app._agent_mode == "tester"
        assert "tester" in result

    def test_agent_all_valid_modes(self):
        """All documented agent modes are accepted."""
        valid_modes = [
            "off",
            "auto",
            "planner",
            "builder",
            "tester",
            "reviewer",
            "architect",
            "documenter",
            "analyst",
        ]
        for mode in valid_modes:
            app = _make_app()
            registry = create_command_registry(app)
            result = _run(registry.execute("agent", mode))
            assert app._agent_mode == mode
            assert mode in result


# ===========================================================================
# _is_sensitive_path — additional patterns
# ===========================================================================


class TestIsSensitivePathExtended:
    def test_p12_file(self):
        assert _is_sensitive_path(Path("/certs/cert.p12")) is True

    def test_pfx_file(self):
        assert _is_sensitive_path(Path("/certs/cert.pfx")) is True

    def test_jks_file(self):
        assert _is_sensitive_path(Path("/certs/keystore.jks")) is True

    def test_id_ed25519(self):
        assert _is_sensitive_path(Path("/home/user/.ssh/id_ed25519")) is True

    def test_id_ecdsa(self):
        assert _is_sensitive_path(Path("/home/user/.ssh/id_ecdsa")) is True

    def test_id_dsa(self):
        assert _is_sensitive_path(Path("/home/user/.ssh/id_dsa")) is True

    def test_known_hosts(self):
        assert _is_sensitive_path(Path("/home/user/.ssh/known_hosts")) is True

    def test_authorized_keys(self):
        assert _is_sensitive_path(Path("/home/user/.ssh/authorized_keys")) is True

    def test_azure_dir(self):
        assert _is_sensitive_path(Path("/home/user/.azure/config")) is True

    def test_gcp_dir(self):
        assert _is_sensitive_path(Path("/home/user/.gcp/credentials.json")) is True

    def test_gnupg_dir(self):
        assert _is_sensitive_path(Path("/home/user/.gnupg/pubring.kbx")) is True

    def test_regular_python_file(self):
        assert _is_sensitive_path(Path("/project/src/main.py")) is False

    def test_regular_yaml(self):
        assert _is_sensitive_path(Path("/project/config.yaml")) is False


# ===========================================================================
# CommandRegistry — additional edge cases
# ===========================================================================


class TestCommandRegistryExtended:
    def test_completions_empty_prefix(self):
        """Empty prefix returns all commands."""
        app = _make_app()
        registry = CommandRegistry(app)

        async def noop(args):
            return None

        registry.register("help", noop, "Help")
        registry.register("quit", noop, "Quit")

        completions = registry.get_completions("")
        assert "/help" in completions
        assert "/quit" in completions

    def test_completions_no_match(self):
        """Prefix that matches nothing returns empty list."""
        app = _make_app()
        registry = CommandRegistry(app)

        async def noop(args):
            return None

        registry.register("help", noop, "Help")

        completions = registry.get_completions("z")
        assert completions == []

    def test_help_sorted(self):
        """Help output is sorted alphabetically by command name."""
        app = _make_app()
        registry = CommandRegistry(app)

        async def noop(args):
            return None

        registry.register("zebra", noop, "Z command")
        registry.register("alpha", noop, "A command")
        registry.register("mid", noop, "M command")

        help_text = registry.get_help()
        lines = help_text.strip().split("\n")
        # First line is "Available commands:"
        cmd_lines = [line.strip() for line in lines[1:]]
        assert cmd_lines[0].startswith("/alpha")
        assert cmd_lines[1].startswith("/mid")
        assert cmd_lines[2].startswith("/zebra")
