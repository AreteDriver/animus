"""Tests for the slash command registry and handlers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from animus_forge.tui.commands import (
    CommandRegistry,
    _is_sensitive_path,
    create_command_registry,
)

# ---------------------------------------------------------------------------
# _is_sensitive_path
# ---------------------------------------------------------------------------


class TestIsSensitivePath:
    def test_env_file(self):
        assert _is_sensitive_path(Path("/project/.env")) is True

    def test_env_local(self):
        assert _is_sensitive_path(Path("/project/.env.local")) is True

    def test_pem_file(self):
        assert _is_sensitive_path(Path("/certs/server.pem")) is True

    def test_key_file(self):
        assert _is_sensitive_path(Path("/certs/private.key")) is True

    def test_credentials_file(self):
        assert _is_sensitive_path(Path("/config/credentials.json")) is True

    def test_secret_in_name(self):
        assert _is_sensitive_path(Path("/config/client_secret.json")) is True

    def test_ssh_dir(self):
        assert _is_sensitive_path(Path("/home/user/.ssh/config")) is True

    def test_id_rsa(self):
        assert _is_sensitive_path(Path("/home/user/.ssh/id_rsa")) is True

    def test_aws_dir(self):
        assert _is_sensitive_path(Path("/home/user/.aws/credentials")) is True

    def test_normal_file(self):
        assert _is_sensitive_path(Path("/src/main.py")) is False

    def test_readme(self):
        assert _is_sensitive_path(Path("/project/README.md")) is False

    def test_pyproject(self):
        assert _is_sensitive_path(Path("/project/pyproject.toml")) is False


# ---------------------------------------------------------------------------
# CommandRegistry
# ---------------------------------------------------------------------------


class TestCommandRegistry:
    def _make_app(self):
        app = MagicMock()
        app._provider_manager = None
        app._agent_mode = "off"
        app._system_prompt = None
        app._messages = []
        app._session = None
        app.screen = MagicMock()
        return app

    def test_register_and_execute(self):
        app = self._make_app()
        registry = CommandRegistry(app)

        async def handler(args):
            return f"got: {args}"

        registry.register("test", handler, "A test command")

        result = asyncio.run(registry.execute("test", "hello"))
        assert result == "got: hello"

    def test_unknown_command(self):
        app = self._make_app()
        registry = CommandRegistry(app)
        result = asyncio.run(registry.execute("nonexistent", ""))
        assert "Unknown command" in result

    def test_get_completions(self):
        app = self._make_app()
        registry = CommandRegistry(app)

        async def noop(args):
            return None

        registry.register("switch", noop, "")
        registry.register("system", noop, "")
        registry.register("save", noop, "")
        registry.register("help", noop, "")

        completions = registry.get_completions("s")
        assert "/switch" in completions
        assert "/system" in completions
        assert "/save" in completions
        assert "/help" not in completions

    def test_get_help(self):
        app = self._make_app()
        registry = CommandRegistry(app)

        async def noop(args):
            return None

        registry.register("help", noop, "Show help")
        registry.register("quit", noop, "Exit app")

        help_text = registry.get_help()
        assert "/help" in help_text
        assert "/quit" in help_text
        assert "Show help" in help_text

    def test_handler_exception_returns_error(self):
        app = self._make_app()
        registry = CommandRegistry(app)

        async def bad_handler(args):
            raise ValueError("something broke")

        registry.register("bad", bad_handler, "")

        result = asyncio.run(registry.execute("bad", ""))
        assert "Error:" in result
        assert "something broke" in result


# ---------------------------------------------------------------------------
# Command Handlers (via create_command_registry)
# ---------------------------------------------------------------------------


class TestCommandHandlers:
    def _make_app(self):
        app = MagicMock()
        app._provider_manager = MagicMock()
        app._agent_mode = "off"
        app._system_prompt = None
        app._messages = []
        app._session = None
        app._supervisor = None

        # Mock screen with sidebar
        screen = MagicMock()
        screen.sidebar = MagicMock()
        screen.sidebar.files = []
        screen.sidebar.input_tokens = 0
        screen.sidebar.output_tokens = 0
        screen.status_bar = MagicMock()
        screen.chat_display = MagicMock()
        app.screen = screen
        app._get_chat_screen = MagicMock(return_value=screen)

        return app

    def _run(self, coro):
        return asyncio.run(coro)

    def test_help_lists_commands(self):
        app = self._make_app()
        registry = create_command_registry(app)
        result = self._run(registry.execute("help", ""))
        assert "/help" in result
        assert "/quit" in result
        assert "/switch" in result

    def test_quit_exits(self):
        app = self._make_app()
        registry = create_command_registry(app)
        self._run(registry.execute("quit", ""))
        app.exit.assert_called_once()

    def test_q_alias(self):
        app = self._make_app()
        registry = create_command_registry(app)
        self._run(registry.execute("q", ""))
        app.exit.assert_called_once()

    def test_clear(self):
        app = self._make_app()
        registry = create_command_registry(app)
        self._run(registry.execute("clear", ""))
        app.action_clear_chat.assert_called_once()

    def test_system_set(self):
        app = self._make_app()
        registry = create_command_registry(app)
        result = self._run(registry.execute("system", "Be concise."))
        assert app._system_prompt == "Be concise."
        assert "set" in result.lower()

    def test_system_show(self):
        app = self._make_app()
        app._system_prompt = "Current prompt"
        registry = create_command_registry(app)
        result = self._run(registry.execute("system", ""))
        assert "Current prompt" in result

    def test_switch_no_args(self):
        app = self._make_app()
        app._provider_manager.list_providers.return_value = ["anthropic", "openai"]
        registry = create_command_registry(app)
        result = self._run(registry.execute("switch", ""))
        assert "Usage" in result
        assert "anthropic" in result

    def test_switch_success(self):
        app = self._make_app()
        provider = MagicMock()
        provider.name = "openai"
        provider.default_model = "gpt-4o"
        app._provider_manager.get_default.return_value = provider
        registry = create_command_registry(app)
        result = self._run(registry.execute("switch", "openai"))
        assert "openai" in result
        app._provider_manager.set_default.assert_called_with("openai")

    def test_model_set(self):
        app = self._make_app()
        provider = MagicMock()
        app._provider_manager.get_default.return_value = provider
        registry = create_command_registry(app)
        result = self._run(registry.execute("model", "gpt-4o-mini"))
        assert "gpt-4o-mini" in result
        assert provider.config.default_model == "gpt-4o-mini"

    def test_models_list(self):
        app = self._make_app()
        provider = MagicMock()
        provider.name = "openai"
        provider.list_models.return_value = ["gpt-4o", "gpt-4o-mini"]
        app._provider_manager.get_default.return_value = provider
        registry = create_command_registry(app)
        result = self._run(registry.execute("models", ""))
        assert "gpt-4o" in result
        assert "gpt-4o-mini" in result

    def test_providers_list(self):
        app = self._make_app()
        app._provider_manager.list_providers.return_value = ["anthropic", "openai"]
        app._provider_manager._default_provider = "anthropic"
        registry = create_command_registry(app)
        result = self._run(registry.execute("providers", ""))
        assert "anthropic" in result
        assert "*" in result  # default marker

    def test_tokens(self):
        app = self._make_app()
        app.screen.sidebar.input_tokens = 1000
        app.screen.sidebar.output_tokens = 500
        registry = create_command_registry(app)
        result = self._run(registry.execute("tokens", ""))
        assert "1,000" in result
        assert "500" in result

    def test_agent_set(self):
        app = self._make_app()
        registry = create_command_registry(app)
        result = self._run(registry.execute("agent", "auto"))
        assert app._agent_mode == "auto"
        assert "auto" in result

    def test_agent_invalid(self):
        app = self._make_app()
        registry = create_command_registry(app)
        result = self._run(registry.execute("agent", "invalid_mode"))
        assert "Invalid" in result

    def test_agent_show_current(self):
        app = self._make_app()
        app._agent_mode = "planner"
        registry = create_command_registry(app)
        result = self._run(registry.execute("agent", ""))
        assert "planner" in result

    def test_file_nonexistent(self):
        app = self._make_app()
        registry = create_command_registry(app)
        result = self._run(registry.execute("file", "/nonexistent/path/to/file.py"))
        assert "not found" in result.lower()

    def test_file_attach(self, tmp_path):
        app = self._make_app()
        f = tmp_path / "test.py"
        f.write_text("print('hello')")
        registry = create_command_registry(app)
        result = self._run(registry.execute("file", str(f)))
        assert "Added" in result

    def test_file_sensitive_warning(self, tmp_path):
        app = self._make_app()
        f = tmp_path / ".env"
        f.write_text("SECRET=oops")
        registry = create_command_registry(app)
        result = self._run(registry.execute("file", str(f)))
        # Should warn about sensitive file
        assert "sensitive" in result.lower() or "warning" in result.lower()

    def test_files_empty(self):
        app = self._make_app()
        registry = create_command_registry(app)
        result = self._run(registry.execute("files", ""))
        assert "No files" in result

    def test_unfile_all(self):
        app = self._make_app()
        registry = create_command_registry(app)
        result = self._run(registry.execute("unfile", "all"))
        assert "removed" in result.lower()
        app.screen.sidebar.clear_files.assert_called_once()

    def test_save(self):
        app = self._make_app()
        app._messages = [{"role": "user", "content": "hi"}]
        provider = MagicMock()
        provider.name = "anthropic"
        provider.default_model = "claude-sonnet-4-20250514"
        app._provider_manager.get_default.return_value = provider
        registry = create_command_registry(app)

        with patch("animus_forge.tui.session.HISTORY_DIR") as mock_dir:
            mock_dir.mkdir = MagicMock()
            mock_dir.exists.return_value = True
            with patch("animus_forge.tui.session.TUISession.save") as mock_save:
                mock_save.return_value = Path("/tmp/test.json")
                result = self._run(registry.execute("save", "test session"))
                assert "saved" in result.lower() or "Session" in result

    def test_history_empty(self):
        app = self._make_app()
        registry = create_command_registry(app)
        with patch("animus_forge.tui.session.TUISession.list_sessions", return_value=[]):
            result = self._run(registry.execute("history", ""))
            assert "No saved" in result

    def test_title_no_session(self):
        app = self._make_app()
        app._session = None
        registry = create_command_registry(app)
        result = self._run(registry.execute("title", "new title"))
        assert "No active session" in result

    def test_title_with_session(self):
        app = self._make_app()
        app._session = MagicMock()
        registry = create_command_registry(app)
        result = self._run(registry.execute("title", "my chat"))
        assert app._session.title == "my chat"
        assert "my chat" in result

    def test_copy_no_messages(self):
        app = self._make_app()
        app._messages = []
        registry = create_command_registry(app)
        result = self._run(registry.execute("copy", ""))
        assert "No messages" in result

    def test_copy_no_assistant_response(self):
        app = self._make_app()
        app._messages = [{"role": "user", "content": "hi"}]
        registry = create_command_registry(app)
        result = self._run(registry.execute("copy", ""))
        assert "No assistant" in result
