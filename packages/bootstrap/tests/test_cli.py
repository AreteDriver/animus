"""Tests for the Animus Bootstrap CLI entry points."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from animus_bootstrap.cli import app
from animus_bootstrap.config.schema import (
    AnimusConfig,
    AnimusSection,
    ApiSection,
    ForgeSection,
    IdentitySection,
    MemorySection,
    ServicesSection,
)

runner = CliRunner()


def _make_config(**overrides: object) -> AnimusConfig:
    """Build an AnimusConfig with sensible test defaults, accepting overrides by section."""
    kwargs: dict[str, object] = {
        "animus": AnimusSection(version="0.1.0", first_run=False, data_dir="/tmp/animus-test"),
        "api": ApiSection(anthropic_key="sk-ant-xxxx1234abcd5678", openai_key="sk-oai-yyyy"),
        "forge": ForgeSection(enabled=False, host="localhost", port=8000),
        "memory": MemorySection(backend="sqlite", path="/tmp/animus-test/memory.db"),
        "identity": IdentitySection(name="Tester", timezone="US/Eastern", locale="en_US"),
        "services": ServicesSection(autostart=True, port=7700, log_level="info"),
    }
    kwargs.update(overrides)
    return AnimusConfig(**kwargs)


def _mock_config_manager(config: AnimusConfig | None = None) -> MagicMock:
    """Create a mock ConfigManager instance with sensible defaults."""
    if config is None:
        config = _make_config()
    mock = MagicMock()
    mock.load.return_value = config
    mock.exists.return_value = True
    mock.get_config_path.return_value = Path("/tmp/animus-test/config.toml")
    return mock


def _mock_installer(*, running: bool = True, os_name: str = "linux") -> MagicMock:
    """Create a mock AnimusInstaller instance."""
    mock = MagicMock()
    mock.is_service_running.return_value = running
    mock.detect_os.return_value = os_name
    mock.start_service.return_value = True
    mock.stop_service.return_value = True
    return mock


# ------------------------------------------------------------------
# No args — should show help
# ------------------------------------------------------------------


class TestNoArgs:
    def test_no_args_shows_help(self) -> None:
        """Typer's no_args_is_help=True causes exit code 0 with usage info."""
        result = runner.invoke(app, [])
        # Typer no_args_is_help exits with code 0 on modern versions, 2 on some
        assert result.exit_code in (0, 2)
        # Should contain usage/help text regardless
        output_lower = result.output.lower()
        assert "usage" in output_lower or "commands" in output_lower or "help" in output_lower


# ------------------------------------------------------------------
# status command
# ------------------------------------------------------------------


class TestStatusCommand:
    @patch("animus_bootstrap.daemon.installer.AnimusInstaller")
    @patch("animus_bootstrap.config.ConfigManager")
    def test_status_command(
        self,
        mock_config_cls: MagicMock,
        mock_installer_cls: MagicMock,
    ) -> None:
        config = _make_config()
        mock_config_cls.return_value = _mock_config_manager(config)
        mock_installer_cls.return_value = _mock_installer(running=True)

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Animus Status" in result.output
        assert "Version" in result.output

    @patch("animus_bootstrap.daemon.installer.AnimusInstaller")
    @patch("animus_bootstrap.config.ConfigManager")
    def test_status_shows_identity(
        self,
        mock_config_cls: MagicMock,
        mock_installer_cls: MagicMock,
    ) -> None:
        config = _make_config()
        mock_config_cls.return_value = _mock_config_manager(config)
        mock_installer_cls.return_value = _mock_installer(running=False)

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Tester" in result.output


# ------------------------------------------------------------------
# start command
# ------------------------------------------------------------------


class TestStartCommand:
    @patch("animus_bootstrap.daemon.installer.AnimusInstaller")
    def test_start_success(self, mock_installer_cls: MagicMock) -> None:
        mock_inst = _mock_installer()
        mock_inst.start_service.return_value = True
        mock_installer_cls.return_value = mock_inst

        result = runner.invoke(app, ["start"])
        assert result.exit_code == 0
        assert "started" in result.output.lower()
        mock_inst.start_service.assert_called_once()

    @patch("animus_bootstrap.daemon.installer.AnimusInstaller")
    def test_start_failure(self, mock_installer_cls: MagicMock) -> None:
        mock_inst = _mock_installer()
        mock_inst.start_service.return_value = False
        mock_installer_cls.return_value = mock_inst

        result = runner.invoke(app, ["start"])
        assert result.exit_code == 1
        assert "failed" in result.output.lower()


# ------------------------------------------------------------------
# stop command
# ------------------------------------------------------------------


class TestStopCommand:
    @patch("animus_bootstrap.daemon.installer.AnimusInstaller")
    def test_stop_success(self, mock_installer_cls: MagicMock) -> None:
        mock_inst = _mock_installer()
        mock_inst.stop_service.return_value = True
        mock_installer_cls.return_value = mock_inst

        result = runner.invoke(app, ["stop"])
        assert result.exit_code == 0
        assert "stopped" in result.output.lower()
        mock_inst.stop_service.assert_called_once()

    @patch("animus_bootstrap.daemon.installer.AnimusInstaller")
    def test_stop_failure(self, mock_installer_cls: MagicMock) -> None:
        mock_inst = _mock_installer()
        mock_inst.stop_service.return_value = False
        mock_installer_cls.return_value = mock_inst

        result = runner.invoke(app, ["stop"])
        assert result.exit_code == 1
        assert "failed" in result.output.lower()


# ------------------------------------------------------------------
# update command
# ------------------------------------------------------------------


class TestUpdateCommand:
    @patch("animus_bootstrap.daemon.updater.AnimusUpdater")
    def test_update_already_current(self, mock_updater_cls: MagicMock) -> None:
        mock_inst = MagicMock()
        mock_inst.get_current_version.return_value = "0.1.0"
        mock_inst.is_update_available.return_value = False
        mock_updater_cls.return_value = mock_inst

        result = runner.invoke(app, ["update"])
        assert result.exit_code == 0
        assert "up to date" in result.output.lower()

    @patch("animus_bootstrap.daemon.updater.AnimusUpdater")
    def test_update_available_accept(self, mock_updater_cls: MagicMock) -> None:
        mock_inst = MagicMock()
        mock_inst.get_current_version.return_value = "0.1.0"
        mock_inst.is_update_available.return_value = True
        mock_inst.get_latest_version.return_value = "0.2.0"
        mock_inst.apply_update.return_value = True
        mock_updater_cls.return_value = mock_inst

        result = runner.invoke(app, ["update"], input="y\n")
        assert result.exit_code == 0
        assert "0.2.0" in result.output
        mock_inst.apply_update.assert_called_once()

    @patch("animus_bootstrap.daemon.updater.AnimusUpdater")
    def test_update_available_decline(self, mock_updater_cls: MagicMock) -> None:
        mock_inst = MagicMock()
        mock_inst.get_current_version.return_value = "0.1.0"
        mock_inst.is_update_available.return_value = True
        mock_inst.get_latest_version.return_value = "0.2.0"
        mock_updater_cls.return_value = mock_inst

        result = runner.invoke(app, ["update"], input="n\n")
        assert result.exit_code == 0
        mock_inst.apply_update.assert_not_called()

    @patch("animus_bootstrap.daemon.updater.AnimusUpdater")
    def test_update_apply_fails(self, mock_updater_cls: MagicMock) -> None:
        mock_inst = MagicMock()
        mock_inst.get_current_version.return_value = "0.1.0"
        mock_inst.is_update_available.return_value = True
        mock_inst.get_latest_version.return_value = "0.2.0"
        mock_inst.apply_update.return_value = False
        mock_updater_cls.return_value = mock_inst

        result = runner.invoke(app, ["update"], input="y\n")
        assert result.exit_code == 1
        assert "failed" in result.output.lower()


# ------------------------------------------------------------------
# config get command
# ------------------------------------------------------------------


class TestConfigGet:
    @patch("animus_bootstrap.config.ConfigManager")
    def test_config_get_valid_key(self, mock_config_cls: MagicMock) -> None:
        config = _make_config()
        mock_config_cls.return_value = _mock_config_manager(config)

        result = runner.invoke(app, ["config", "get", "identity.name"])
        assert result.exit_code == 0
        assert "Tester" in result.output

    @patch("animus_bootstrap.config.ConfigManager")
    def test_config_get_invalid_key(self, mock_config_cls: MagicMock) -> None:
        config = _make_config()
        mock_config_cls.return_value = _mock_config_manager(config)

        result = runner.invoke(app, ["config", "get", "nonexistent.key"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    @patch("animus_bootstrap.config.ConfigManager")
    def test_config_get_masks_api_keys(self, mock_config_cls: MagicMock) -> None:
        config = _make_config()
        mock_config_cls.return_value = _mock_config_manager(config)

        result = runner.invoke(app, ["config", "get", "api.anthropic_key"])
        assert result.exit_code == 0
        # The full key should NOT appear in output
        assert "sk-ant-xxxx1234abcd5678" not in result.output
        # Masked form: first 4 + ... + last 4
        assert "sk-a" in result.output
        assert "5678" in result.output
        assert "..." in result.output

    @patch("animus_bootstrap.config.ConfigManager")
    def test_config_get_nested_section(self, mock_config_cls: MagicMock) -> None:
        config = _make_config()
        mock_config_cls.return_value = _mock_config_manager(config)

        result = runner.invoke(app, ["config", "get", "services.port"])
        assert result.exit_code == 0
        assert "7700" in result.output

    @patch("animus_bootstrap.config.ConfigManager")
    def test_config_get_deep_invalid_path(self, mock_config_cls: MagicMock) -> None:
        """Requesting a key deeper than section.field on a non-dict value fails."""
        config = _make_config()
        mock_config_cls.return_value = _mock_config_manager(config)

        result = runner.invoke(app, ["config", "get", "identity.name.extra"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()


# ------------------------------------------------------------------
# config set command
# ------------------------------------------------------------------


class TestConfigSet:
    @patch("animus_bootstrap.config.ConfigManager")
    def test_config_set_string(self, mock_config_cls: MagicMock) -> None:
        config = _make_config()
        mock_mgr = _mock_config_manager(config)
        mock_config_cls.return_value = mock_mgr

        result = runner.invoke(app, ["config", "set", "identity.name", "TestUser"])
        assert result.exit_code == 0
        assert "identity.name" in result.output
        assert "TestUser" in result.output
        mock_mgr.save.assert_called_once()

    @patch("animus_bootstrap.config.ConfigManager")
    def test_config_set_bool(self, mock_config_cls: MagicMock) -> None:
        config = _make_config()
        mock_mgr = _mock_config_manager(config)
        mock_config_cls.return_value = mock_mgr

        result = runner.invoke(app, ["config", "set", "forge.enabled", "true"])
        assert result.exit_code == 0
        assert "forge.enabled" in result.output
        assert "True" in result.output
        mock_mgr.save.assert_called_once()

    @patch("animus_bootstrap.config.ConfigManager")
    def test_config_set_int(self, mock_config_cls: MagicMock) -> None:
        config = _make_config()
        mock_mgr = _mock_config_manager(config)
        mock_config_cls.return_value = mock_mgr

        result = runner.invoke(app, ["config", "set", "services.port", "8080"])
        assert result.exit_code == 0
        assert "services.port" in result.output
        assert "8080" in result.output
        mock_mgr.save.assert_called_once()

    @patch("animus_bootstrap.config.ConfigManager")
    def test_config_set_invalid_section(self, mock_config_cls: MagicMock) -> None:
        config = _make_config()
        mock_config_cls.return_value = _mock_config_manager(config)

        result = runner.invoke(app, ["config", "set", "bad.key", "value"])
        assert result.exit_code == 1
        assert "unknown section" in result.output.lower()

    @patch("animus_bootstrap.config.ConfigManager")
    def test_config_set_invalid_field(self, mock_config_cls: MagicMock) -> None:
        config = _make_config()
        mock_config_cls.return_value = _mock_config_manager(config)

        result = runner.invoke(app, ["config", "set", "identity.nonexistent", "value"])
        assert result.exit_code == 1
        assert "unknown field" in result.output.lower()

    @patch("animus_bootstrap.config.ConfigManager")
    def test_config_set_bad_format(self, mock_config_cls: MagicMock) -> None:
        """Key without section.field format is rejected."""
        config = _make_config()
        mock_config_cls.return_value = _mock_config_manager(config)

        result = runner.invoke(app, ["config", "set", "justonepart", "value"])
        assert result.exit_code == 1
        assert "section.field" in result.output.lower()

    @patch("animus_bootstrap.config.ConfigManager")
    def test_config_set_int_invalid_value(self, mock_config_cls: MagicMock) -> None:
        """Setting an int field to a non-numeric value fails."""
        config = _make_config()
        mock_config_cls.return_value = _mock_config_manager(config)

        result = runner.invoke(app, ["config", "set", "services.port", "notanumber"])
        assert result.exit_code == 1
        assert "integer" in result.output.lower()


# ------------------------------------------------------------------
# setup command
# ------------------------------------------------------------------


class TestSetupCommand:
    @patch("animus_bootstrap.setup.wizard.AnimusWizard")
    @patch("animus_bootstrap.config.ConfigManager")
    def test_setup_runs_wizard(
        self, mock_config_cls: MagicMock, mock_wizard_cls: MagicMock
    ) -> None:
        mock_config_inst = MagicMock()
        mock_config_cls.return_value = mock_config_inst

        mock_wizard_inst = MagicMock()
        mock_wizard_inst.run.return_value = _make_config()
        mock_wizard_cls.return_value = mock_wizard_inst

        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        mock_wizard_cls.assert_called_once()
        mock_wizard_inst.run.assert_called_once()


# ------------------------------------------------------------------
# tools list command
# ------------------------------------------------------------------


class TestToolsListCommand:
    def test_tools_list_command(self) -> None:
        """tools list shows the empty tools table."""
        result = runner.invoke(app, ["tools", "list"])
        assert result.exit_code == 0
        assert "Registered Tools" in result.output
        assert "No tools loaded" in result.output


# ------------------------------------------------------------------
# proactive status command
# ------------------------------------------------------------------


class TestProactiveStatusCommand:
    @patch("animus_bootstrap.config.ConfigManager")
    def test_proactive_status_command(self, mock_config_cls: MagicMock) -> None:
        """proactive status shows engine settings."""
        config = _make_config()
        mock_config_cls.return_value = _mock_config_manager(config)

        result = runner.invoke(app, ["proactive", "status"])
        assert result.exit_code == 0
        assert "Proactive Engine" in result.output
        assert "22:00" in result.output
        assert "07:00" in result.output
        assert "UTC" in result.output


# ------------------------------------------------------------------
# automations list command
# ------------------------------------------------------------------


class TestAutomationsListCommand:
    def test_automations_list_command(self) -> None:
        """automations list shows the empty rules table."""
        result = runner.invoke(app, ["automations", "list"])
        assert result.exit_code == 0
        assert "Automation Rules" in result.output
        assert "No automation rules configured" in result.output


# ------------------------------------------------------------------
# personas list command
# ------------------------------------------------------------------


class TestPersonasListCommand:
    @patch("animus_bootstrap.config.ConfigManager")
    def test_personas_list_command(self, mock_config_cls: MagicMock) -> None:
        """personas list shows the default persona."""
        config = _make_config()
        mock_config_cls.return_value = _mock_config_manager(config)

        result = runner.invoke(app, ["personas", "list"])
        assert result.exit_code == 0
        assert "Persona Profiles" in result.output
        assert "Animus" in result.output
        assert "balanced" in result.output
        assert "General" in result.output
