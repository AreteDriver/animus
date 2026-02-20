"""Tests for the TUI provider manager factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from animus_forge.tui.providers import create_provider_manager


class TestCreateProviderManager:
    @patch("animus_forge.tui.providers.get_settings")
    def test_registers_anthropic_when_key_set(self, mock_settings):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="sk-ant-test",
            openai_api_key="",
        )
        with patch("animus_forge.tui.providers.ProviderManager") as MockManager:
            mgr = MockManager.return_value
            mgr.list_providers.return_value = ["anthropic"]
            mgr._default_provider = "anthropic"

            create_provider_manager()

            # Should have called register for anthropic
            calls = mgr.register.call_args_list
            names = [c[0][0] if c[0] else None for c in calls]
            assert "anthropic" in names

    @patch("animus_forge.tui.providers.get_settings")
    def test_registers_openai_when_key_set(self, mock_settings):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="",
            openai_api_key="sk-test-openai",
        )
        with patch("animus_forge.tui.providers.ProviderManager") as MockManager:
            mgr = MockManager.return_value
            mgr.list_providers.return_value = ["openai"]
            mgr._default_provider = "openai"

            create_provider_manager()

            names = [c[0][0] for c in mgr.register.call_args_list if c[0]]
            assert "openai" in names

    @patch("animus_forge.tui.providers.get_settings")
    def test_anthropic_preferred_over_openai(self, mock_settings):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="sk-ant-test",
            openai_api_key="sk-openai-test",
        )
        with patch("animus_forge.tui.providers.ProviderManager") as MockManager:
            mgr = MockManager.return_value
            mgr._default_provider = None

            create_provider_manager()

            # set_default should be called with "anthropic"
            mgr.set_default.assert_called_with("anthropic")

    @patch("animus_forge.tui.providers.get_settings")
    def test_no_keys_logs_warning(self, mock_settings):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="",
            openai_api_key="",
        )
        with (
            patch("animus_forge.tui.providers.ProviderManager") as MockManager,
            patch("animus_forge.tui.providers.logger") as mock_logger,
        ):
            mgr = MockManager.return_value
            # Ollama register also fails
            mgr.register.side_effect = Exception("no ollama")
            mgr._default_provider = None

            create_provider_manager()

            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "No AI providers" in warning_msg

    @patch("animus_forge.tui.providers.get_settings")
    def test_ollama_always_attempted(self, mock_settings):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="",
            openai_api_key="",
        )
        with patch("animus_forge.tui.providers.ProviderManager") as MockManager:
            mgr = MockManager.return_value
            mgr._default_provider = None

            create_provider_manager()

            names = [c[0][0] for c in mgr.register.call_args_list if c[0]]
            assert "ollama" in names

    @patch("animus_forge.tui.providers.get_settings")
    def test_failed_registration_continues(self, mock_settings):
        mock_settings.return_value = MagicMock(
            anthropic_api_key="sk-ant-test",
            openai_api_key="sk-openai-test",
        )
        with patch("animus_forge.tui.providers.ProviderManager") as MockManager:
            mgr = MockManager.return_value
            # Anthropic fails, OpenAI succeeds
            mgr.register.side_effect = [
                Exception("anthropic broken"),
                MagicMock(),  # openai ok
                Exception("no ollama"),  # ollama fails
            ]
            mgr._default_provider = None

            create_provider_manager()

            # Should still attempt set_default with openai
            mgr.set_default.assert_called_with("openai")
