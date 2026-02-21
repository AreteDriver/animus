"""Tests for the Animus Bootstrap wizard, steps, and validators."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from animus_bootstrap.cli import app
from animus_bootstrap.config import ConfigManager
from animus_bootstrap.setup.steps.api_keys import run_api_keys
from animus_bootstrap.setup.steps.channels import run_channels_step
from animus_bootstrap.setup.steps.device import run_device
from animus_bootstrap.setup.steps.forge import run_forge
from animus_bootstrap.setup.steps.identity import run_identity
from animus_bootstrap.setup.steps.memory import run_memory
from animus_bootstrap.setup.steps.sovereignty import run_sovereignty
from animus_bootstrap.setup.steps.welcome import run_welcome
from animus_bootstrap.setup.validators import (
    test_anthropic_key as validate_anthropic_key,
)
from animus_bootstrap.setup.validators import (
    test_forge_connection as validate_forge_connection,
)
from animus_bootstrap.setup.wizard import AnimusWizard, _mask_key

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_console() -> Console:
    """Return a Console that writes to a StringIO (no terminal output)."""
    return Console(file=io.StringIO(), force_terminal=True)


# ------------------------------------------------------------------
# Validators — test_anthropic_key
# ------------------------------------------------------------------


class TestAnthropicKeyValid:
    """Mocked 200 response means valid key."""

    def test_returns_true_on_200(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        with patch("animus_bootstrap.setup.validators.httpx.post", return_value=mock_resp):
            assert validate_anthropic_key("sk-ant-valid") is True


class TestAnthropicKeyInvalid:
    """Mocked 401 response means invalid key."""

    def test_returns_false_on_401(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 401
        with patch("animus_bootstrap.setup.validators.httpx.post", return_value=mock_resp):
            assert validate_anthropic_key("sk-ant-bad") is False


class TestAnthropicKeyNetworkError:
    """Network errors are caught and return False."""

    def test_returns_false_on_request_error(self) -> None:
        with patch(
            "animus_bootstrap.setup.validators.httpx.post",
            side_effect=httpx.RequestError("connection failed"),
        ):
            assert validate_anthropic_key("sk-ant-whatever") is False

    def test_returns_false_on_timeout(self) -> None:
        with patch(
            "animus_bootstrap.setup.validators.httpx.post",
            side_effect=httpx.TimeoutException("timed out"),
        ):
            assert validate_anthropic_key("sk-ant-whatever") is False


# ------------------------------------------------------------------
# Validators — test_forge_connection
# ------------------------------------------------------------------


class TestForgeConnectionSuccess:
    """Forge health endpoint returns 200."""

    def test_returns_true_on_200(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        with patch("animus_bootstrap.setup.validators.httpx.get", return_value=mock_resp):
            assert validate_forge_connection("localhost", 8000) is True


class TestForgeConnectionFailure:
    """Forge is unreachable (timeout or error)."""

    def test_returns_false_on_timeout(self) -> None:
        with patch(
            "animus_bootstrap.setup.validators.httpx.get",
            side_effect=httpx.TimeoutException("timed out"),
        ):
            assert validate_forge_connection("localhost", 8000) is False

    def test_returns_false_on_connect_error(self) -> None:
        with patch(
            "animus_bootstrap.setup.validators.httpx.get",
            side_effect=httpx.ConnectError("refused"),
        ):
            assert validate_forge_connection("localhost", 8000) is False

    def test_returns_false_on_500(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 500
        with patch("animus_bootstrap.setup.validators.httpx.get", return_value=mock_resp):
            assert validate_forge_connection("localhost", 8000) is False


# ------------------------------------------------------------------
# Steps — welcome
# ------------------------------------------------------------------


class TestWelcomeAccept:
    """User confirms at the welcome screen."""

    def test_returns_true_when_confirmed(self) -> None:
        console = _make_console()
        with patch("animus_bootstrap.setup.steps.welcome.Confirm.ask", return_value=True):
            result = run_welcome(console)
        assert result is True


class TestWelcomeDecline:
    """User declines at the welcome screen."""

    def test_returns_false_when_declined(self) -> None:
        console = _make_console()
        with patch("animus_bootstrap.setup.steps.welcome.Confirm.ask", return_value=False):
            result = run_welcome(console)
        assert result is False


# ------------------------------------------------------------------
# Steps — identity
# ------------------------------------------------------------------


class TestIdentityStep:
    """Identity step collects name, timezone, and locale."""

    def test_collects_all_fields(self) -> None:
        console = _make_console()
        with (
            patch(
                "animus_bootstrap.setup.steps.identity.Prompt.ask",
                side_effect=["Arete", "en_US"],
            ),
            patch(
                "animus_bootstrap.setup.steps.identity.Confirm.ask",
                return_value=True,
            ),
            patch(
                "animus_bootstrap.setup.steps.identity._detect_timezone",
                return_value="America/New_York",
            ),
        ):
            result = run_identity(console)

        assert result["name"] == "Arete"
        assert result["timezone"] == "America/New_York"
        assert result["locale"] == "en_US"

    def test_custom_timezone_when_declined(self) -> None:
        console = _make_console()
        with (
            patch(
                "animus_bootstrap.setup.steps.identity.Prompt.ask",
                side_effect=["Arete", "Europe/London", "en_GB"],
            ),
            patch(
                "animus_bootstrap.setup.steps.identity.Confirm.ask",
                return_value=False,
            ),
            patch(
                "animus_bootstrap.setup.steps.identity._detect_timezone",
                return_value="America/New_York",
            ),
        ):
            result = run_identity(console)

        assert result["name"] == "Arete"
        assert result["timezone"] == "Europe/London"
        assert result["locale"] == "en_GB"


# ------------------------------------------------------------------
# Steps — api_keys
# ------------------------------------------------------------------


class TestApiKeysValidFirstTry:
    """Key is valid on the first attempt (skip Ollama, add Anthropic)."""

    def test_valid_key_returned(self) -> None:
        console = _make_console()
        # Confirm.ask calls: Ollama? No, Anthropic? Yes, OpenAI? No
        with (
            patch(
                "animus_bootstrap.setup.steps.api_keys.Prompt.ask",
                return_value="sk-ant-valid",
            ),
            patch(
                "animus_bootstrap.setup.steps.api_keys.Confirm.ask",
                side_effect=[False, True, False],
            ),
            patch(
                "animus_bootstrap.setup.steps.api_keys.test_anthropic_key",
                return_value=True,
            ),
        ):
            result = run_api_keys(console)

        assert result["anthropic_key"] == "sk-ant-valid"
        assert result["openai_key"] == ""


class TestApiKeysRetryThenSucceed:
    """Key fails twice then succeeds on third attempt."""

    def test_retry_logic(self) -> None:
        console = _make_console()
        # Confirm: Ollama? No, Anthropic? Yes, OpenAI? No
        with (
            patch(
                "animus_bootstrap.setup.steps.api_keys.Prompt.ask",
                side_effect=["bad-key-1", "bad-key-2", "sk-ant-valid"],
            ),
            patch(
                "animus_bootstrap.setup.steps.api_keys.Confirm.ask",
                side_effect=[False, True, False],
            ),
            patch(
                "animus_bootstrap.setup.steps.api_keys.test_anthropic_key",
                side_effect=[False, False, True],
            ),
        ):
            result = run_api_keys(console)

        assert result["anthropic_key"] == "sk-ant-valid"


class TestApiKeysExhaustedRetries:
    """All 3 retries fail with no Ollama — raises SystemExit."""

    def test_raises_system_exit(self) -> None:
        console = _make_console()
        # Confirm: Ollama? No, Anthropic? Yes, OpenAI? No
        # _collect_anthropic_key returns "" after 3 failures
        # Then: no ollama + no anthropic key → SystemExit
        with (
            patch(
                "animus_bootstrap.setup.steps.api_keys.Prompt.ask",
                return_value="bad-key",
            ),
            patch(
                "animus_bootstrap.setup.steps.api_keys.Confirm.ask",
                side_effect=[False, True, False],
            ),
            patch(
                "animus_bootstrap.setup.steps.api_keys.test_anthropic_key",
                return_value=False,
            ),
            pytest.raises(SystemExit),
        ):
            run_api_keys(console)


class TestApiKeysWithOpenAI:
    """User opts to add an OpenAI key."""

    def test_collects_openai_key(self) -> None:
        console = _make_console()
        # Confirm: Ollama? No, Anthropic? Yes, OpenAI? Yes
        with (
            patch(
                "animus_bootstrap.setup.steps.api_keys.Prompt.ask",
                side_effect=["sk-ant-valid", "sk-oai-key"],
            ),
            patch(
                "animus_bootstrap.setup.steps.api_keys.Confirm.ask",
                side_effect=[False, True, True],
            ),
            patch(
                "animus_bootstrap.setup.steps.api_keys.test_anthropic_key",
                return_value=True,
            ),
        ):
            result = run_api_keys(console)

        assert result["anthropic_key"] == "sk-ant-valid"
        assert result["openai_key"] == "sk-oai-key"


# ------------------------------------------------------------------
# Steps — forge
# ------------------------------------------------------------------


class TestForgeAutoDetect:
    """Forge is detected at localhost:8000."""

    def test_returns_enabled_on_detect(self) -> None:
        console = _make_console()
        with (
            patch(
                "animus_bootstrap.setup.steps.forge.test_forge_connection",
                return_value=True,
            ),
            patch(
                "animus_bootstrap.setup.steps.forge.Confirm.ask",
                return_value=True,
            ),
            patch(
                "animus_bootstrap.setup.steps.forge.Prompt.ask",
                return_value="",
            ),
        ):
            result = run_forge(console)

        assert result["enabled"] is True
        assert result["host"] == "localhost"
        assert result["port"] == 8000


class TestForgeNotFoundSkip:
    """Forge not detected, user chooses to skip."""

    def test_returns_disabled_on_skip(self) -> None:
        console = _make_console()
        with (
            patch(
                "animus_bootstrap.setup.steps.forge.test_forge_connection",
                return_value=False,
            ),
            patch(
                "animus_bootstrap.setup.steps.forge.Confirm.ask",
                return_value=False,
            ),
        ):
            result = run_forge(console)

        assert result["enabled"] is False


class TestForgeManualConfig:
    """Forge not detected, user enters manual configuration."""

    def test_returns_manual_config(self) -> None:
        console = _make_console()
        with (
            patch(
                "animus_bootstrap.setup.steps.forge.test_forge_connection",
                return_value=False,
            ),
            patch(
                "animus_bootstrap.setup.steps.forge.Confirm.ask",
                return_value=True,
            ),
            patch(
                "animus_bootstrap.setup.steps.forge.Prompt.ask",
                side_effect=["10.0.0.5", "9000", "my-forge-key"],
            ),
        ):
            result = run_forge(console)

        assert result["enabled"] is True
        assert result["host"] == "10.0.0.5"
        assert result["port"] == 9000
        assert result["api_key"] == "my-forge-key"

    def test_invalid_port_uses_default(self) -> None:
        console = _make_console()
        with (
            patch(
                "animus_bootstrap.setup.steps.forge.test_forge_connection",
                return_value=False,
            ),
            patch(
                "animus_bootstrap.setup.steps.forge.Confirm.ask",
                return_value=True,
            ),
            patch(
                "animus_bootstrap.setup.steps.forge.Prompt.ask",
                side_effect=["localhost", "not-a-number", ""],
            ),
        ):
            result = run_forge(console)

        assert result["port"] == 8000


# ------------------------------------------------------------------
# Steps — memory
# ------------------------------------------------------------------


class TestMemorySqliteDefault:
    """User selects SQLite (option 1) with defaults."""

    def test_sqlite_selected(self) -> None:
        console = _make_console()
        with (
            patch(
                "animus_bootstrap.setup.steps.memory.IntPrompt.ask",
                return_value=1,
            ),
            patch(
                "animus_bootstrap.setup.steps.memory.Prompt.ask",
                side_effect=["~/.local/share/animus/", "100000"],
            ),
        ):
            result = run_memory(console)

        assert result["backend"] == "sqlite"
        assert result["max_context_tokens"] == 100_000


class TestMemoryChromaDB:
    """User selects ChromaDB (option 2)."""

    def test_chromadb_selected(self) -> None:
        console = _make_console()
        with (
            patch(
                "animus_bootstrap.setup.steps.memory.IntPrompt.ask",
                return_value=2,
            ),
            patch(
                "animus_bootstrap.setup.steps.memory.Prompt.ask",
                side_effect=["/opt/animus/chroma/", "200000"],
            ),
        ):
            result = run_memory(console)

        assert result["backend"] == "chromadb"
        assert result["path"] == "/opt/animus/chroma/"
        assert result["max_context_tokens"] == 200_000


class TestMemoryWeaviate:
    """User selects Weaviate (option 3)."""

    def test_weaviate_selected(self) -> None:
        console = _make_console()
        with (
            patch(
                "animus_bootstrap.setup.steps.memory.IntPrompt.ask",
                return_value=3,
            ),
            patch(
                "animus_bootstrap.setup.steps.memory.Prompt.ask",
                side_effect=["/data/weaviate/", "50000"],
            ),
        ):
            result = run_memory(console)

        assert result["backend"] == "weaviate"


class TestMemoryInvalidMaxTokens:
    """Invalid max_context_tokens falls back to default."""

    def test_fallback_on_invalid_tokens(self) -> None:
        console = _make_console()
        with (
            patch(
                "animus_bootstrap.setup.steps.memory.IntPrompt.ask",
                return_value=1,
            ),
            patch(
                "animus_bootstrap.setup.steps.memory.Prompt.ask",
                side_effect=["~/.local/share/animus/", "not-a-number"],
            ),
        ):
            result = run_memory(console)

        assert result["max_context_tokens"] == 100_000


# ------------------------------------------------------------------
# Steps — device
# ------------------------------------------------------------------


class TestDeviceDefaults:
    """Device step uses detected hostname and user-selected role."""

    def test_primary_role_with_hostname(self) -> None:
        console = _make_console()
        with (
            patch("animus_bootstrap.setup.steps.device.socket.gethostname", return_value="medusa"),
            patch(
                "animus_bootstrap.setup.steps.device.Prompt.ask",
                return_value="medusa",
            ),
            patch(
                "animus_bootstrap.setup.steps.device.IntPrompt.ask",
                return_value=1,
            ),
        ):
            result = run_device(console)

        assert result["machine_name"] == "medusa"
        assert result["role"] == "primary"

    def test_secondary_role(self) -> None:
        console = _make_console()
        with (
            patch("animus_bootstrap.setup.steps.device.socket.gethostname", return_value="stheno"),
            patch(
                "animus_bootstrap.setup.steps.device.Prompt.ask",
                return_value="stheno",
            ),
            patch(
                "animus_bootstrap.setup.steps.device.IntPrompt.ask",
                return_value=2,
            ),
        ):
            result = run_device(console)

        assert result["machine_name"] == "stheno"
        assert result["role"] == "secondary"

    def test_mobile_role(self) -> None:
        console = _make_console()
        with (
            patch("animus_bootstrap.setup.steps.device.socket.gethostname", return_value="phone"),
            patch(
                "animus_bootstrap.setup.steps.device.Prompt.ask",
                return_value="phone",
            ),
            patch(
                "animus_bootstrap.setup.steps.device.IntPrompt.ask",
                return_value=3,
            ),
        ):
            result = run_device(console)

        assert result["role"] == "mobile"


# ------------------------------------------------------------------
# Steps — sovereignty
# ------------------------------------------------------------------


class TestSovereigntyTelemetryOff:
    """User declines telemetry."""

    def test_telemetry_disabled(self) -> None:
        console = _make_console()
        with patch(
            "animus_bootstrap.setup.steps.sovereignty.Confirm.ask",
            side_effect=[False, True],
        ):
            result = run_sovereignty(console)

        assert result["telemetry"] is False
        assert result["data_local_only"] is True


class TestSovereigntyTelemetryOn:
    """User enables telemetry."""

    def test_telemetry_enabled(self) -> None:
        console = _make_console()
        with patch(
            "animus_bootstrap.setup.steps.sovereignty.Confirm.ask",
            side_effect=[True, True],
        ):
            result = run_sovereignty(console)

        assert result["telemetry"] is True
        assert result["data_local_only"] is True


class TestSovereigntyDataNotLocalOnly:
    """User declines data locality."""

    def test_data_local_only_false(self) -> None:
        console = _make_console()
        with patch(
            "animus_bootstrap.setup.steps.sovereignty.Confirm.ask",
            side_effect=[False, False],
        ):
            result = run_sovereignty(console)

        assert result["data_local_only"] is False


# ------------------------------------------------------------------
# Wizard — full run
# ------------------------------------------------------------------


class TestWizardFullRun:
    """Wizard orchestrates all steps and writes config."""

    def test_full_run_produces_config(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        wizard = AnimusWizard(mgr)
        wizard._console = _make_console()

        with (
            patch(
                "animus_bootstrap.setup.wizard.run_welcome",
                return_value=True,
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_identity",
                return_value={"name": "Arete", "timezone": "US/Eastern", "locale": "en_US"},
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_identity_files",
                return_value={"about": "", "generate_identity_files": False},
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_api_keys",
                return_value={"anthropic_key": "sk-ant-xxx", "openai_key": ""},
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_forge",
                return_value={"enabled": False, "host": "localhost", "port": 8000, "api_key": ""},
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_memory",
                return_value={
                    "backend": "sqlite",
                    "path": "~/.local/share/animus/memory.db",
                    "max_context_tokens": 100_000,
                },
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_device",
                return_value={"machine_name": "medusa", "role": "primary"},
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_sovereignty",
                return_value={"telemetry": False, "data_local_only": True},
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_channels_step",
                return_value={"channels": {}},
            ),
        ):
            config = wizard.run()

        # Verify config fields
        assert config.identity.name == "Arete"
        assert config.identity.timezone == "US/Eastern"
        assert config.identity.locale == "en_US"
        assert config.api.anthropic_key == "sk-ant-xxx"
        assert config.api.openai_key == ""
        assert config.forge.enabled is False
        assert config.memory.backend == "sqlite"
        assert config.animus.first_run is False

        # Verify config was persisted to disk
        assert mgr.exists() is True
        reloaded = mgr.load()
        assert reloaded.identity.name == "Arete"
        assert reloaded.api.anthropic_key == "sk-ant-xxx"

    def test_full_run_with_forge_enabled(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        wizard = AnimusWizard(mgr)
        wizard._console = _make_console()

        with (
            patch("animus_bootstrap.setup.wizard.run_welcome", return_value=True),
            patch(
                "animus_bootstrap.setup.wizard.run_identity",
                return_value={"name": "Arete", "timezone": "UTC", "locale": "en_US"},
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_identity_files",
                return_value={"about": "", "generate_identity_files": False},
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_api_keys",
                return_value={"anthropic_key": "sk-ant-xxx", "openai_key": "sk-oai-yyy"},
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_forge",
                return_value={
                    "enabled": True,
                    "host": "10.0.0.5",
                    "port": 9000,
                    "api_key": "forge-secret",
                },
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_memory",
                return_value={
                    "backend": "chromadb",
                    "path": "/data/chroma",
                    "max_context_tokens": 200_000,
                },
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_device",
                return_value={"machine_name": "stheno", "role": "secondary"},
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_sovereignty",
                return_value={"telemetry": True, "data_local_only": True},
            ),
            patch(
                "animus_bootstrap.setup.wizard.run_channels_step",
                return_value={"channels": {}},
            ),
        ):
            config = wizard.run()

        assert config.forge.enabled is True
        assert config.forge.host == "10.0.0.5"
        assert config.forge.port == 9000
        assert config.forge.api_key == "forge-secret"
        assert config.api.openai_key == "sk-oai-yyy"
        assert config.memory.backend == "chromadb"
        assert config.memory.max_context_tokens == 200_000


# ------------------------------------------------------------------
# Wizard — cancel at welcome
# ------------------------------------------------------------------


class TestWizardCancelAtWelcome:
    """User declines at the welcome screen — wizard exits cleanly."""

    def test_raises_system_exit(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        wizard = AnimusWizard(mgr)
        wizard._console = _make_console()

        with (
            patch("animus_bootstrap.setup.wizard.run_welcome", return_value=False),
            pytest.raises(SystemExit) as exc_info,
        ):
            wizard.run()

        assert exc_info.value.code == 0

    def test_no_config_written(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        wizard = AnimusWizard(mgr)
        wizard._console = _make_console()

        with (
            patch("animus_bootstrap.setup.wizard.run_welcome", return_value=False),
            pytest.raises(SystemExit),
        ):
            wizard.run()

        assert mgr.exists() is False


# ------------------------------------------------------------------
# Wizard — _run_step error handling
# ------------------------------------------------------------------


class TestWizardRunStepRetry:
    """Step failure triggers retry/skip logic."""

    def test_skippable_step_returns_empty_on_skip(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        wizard = AnimusWizard(mgr)
        wizard._console = _make_console()

        def failing_step(console: Console) -> dict:
            raise RuntimeError("step failed")

        with patch("rich.prompt.Confirm.ask", return_value=False):
            result = wizard._run_step(1, "Test", failing_step, skippable=True)

        assert result == {}

    def test_skippable_step_retries_on_confirm(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        wizard = AnimusWizard(mgr)
        wizard._console = _make_console()

        call_count = 0

        def step_fails_then_succeeds(console: Console) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first call fails")
            return {"ok": True}

        with patch("rich.prompt.Confirm.ask", return_value=True):
            result = wizard._run_step(1, "Test", step_fails_then_succeeds, skippable=True)

        assert result == {"ok": True}
        assert call_count == 2

    def test_non_skippable_step_reraises_on_decline(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        wizard = AnimusWizard(mgr)
        wizard._console = _make_console()

        def failing_step(console: Console) -> dict:
            raise RuntimeError("cannot skip this")

        with (
            patch("rich.prompt.Confirm.ask", return_value=False),
            pytest.raises(RuntimeError, match="cannot skip this"),
        ):
            wizard._run_step(1, "Test", failing_step, skippable=False)

    def test_keyboard_interrupt_raises_system_exit_130(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        wizard = AnimusWizard(mgr)
        wizard._console = _make_console()

        def interrupting_step(console: Console) -> dict:
            raise KeyboardInterrupt

        with pytest.raises(SystemExit) as exc_info:
            wizard._run_step(1, "Test", interrupting_step, skippable=True)

        assert exc_info.value.code == 130

    def test_system_exit_propagates(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        wizard = AnimusWizard(mgr)
        wizard._console = _make_console()

        def exiting_step(console: Console) -> dict:
            raise SystemExit(42)

        with pytest.raises(SystemExit) as exc_info:
            wizard._run_step(1, "Test", exiting_step, skippable=True)

        assert exc_info.value.code == 42


# ------------------------------------------------------------------
# Wizard — _build_config
# ------------------------------------------------------------------


class TestBuildConfig:
    """Verify _build_config assembles the AnimusConfig correctly."""

    def test_empty_data_preserves_defaults(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        wizard = AnimusWizard(mgr)
        config = wizard._build_config(
            identity_data={},
            api_data={},
            forge_data={},
            memory_data={},
            device_data={},
            sovereignty_data={},
        )
        assert config.identity.name == ""
        assert config.api.anthropic_key == ""
        assert config.forge.enabled is False
        assert config.memory.backend == "sqlite"

    def test_partial_data_merges(self, tmp_path: Path) -> None:
        mgr = ConfigManager(config_dir=tmp_path)
        wizard = AnimusWizard(mgr)
        config = wizard._build_config(
            identity_data={"name": "Test", "timezone": "UTC", "locale": "en"},
            api_data={"anthropic_key": "sk-ant-xxx"},
            forge_data={},
            memory_data={},
            device_data={},
            sovereignty_data={"telemetry": False},
        )
        assert config.identity.name == "Test"
        assert config.api.anthropic_key == "sk-ant-xxx"
        assert config.animus.first_run is False


# ------------------------------------------------------------------
# _mask_key helper
# ------------------------------------------------------------------


class TestMaskKey:
    """Verify API key masking for summary display."""

    def test_empty_key(self) -> None:
        assert _mask_key("") == "[dim]not set[/dim]"

    def test_short_key(self) -> None:
        # <= 8 chars: show "****" + last 4
        assert _mask_key("12345678") == "****5678"

    def test_very_short_key(self) -> None:
        assert _mask_key("abcd") == "****abcd"

    def test_long_key(self) -> None:
        # > 8 chars: show first 4 + "..." + last 4
        assert _mask_key("sk-ant-api0123456789") == "sk-a...6789"


# ------------------------------------------------------------------
# Steps — channels
# ------------------------------------------------------------------


class TestChannelsStepSkip:
    """User skips channel setup (no channels selected)."""

    def test_no_channels_selected(self) -> None:
        console = _make_console()
        with patch(
            "animus_bootstrap.setup.steps.channels.Prompt.ask",
            return_value="",
        ):
            result = run_channels_step(console)

        assert result == {"channels": {}}


class TestChannelsStepTelegram:
    """User selects and configures telegram."""

    def test_telegram_configured(self) -> None:
        console = _make_console()
        with patch(
            "animus_bootstrap.setup.steps.channels.Prompt.ask",
            side_effect=["telegram", "my-bot-token-123"],
        ):
            result = run_channels_step(console)

        assert "telegram" in result["channels"]
        assert result["channels"]["telegram"]["enabled"] is True
        assert result["channels"]["telegram"]["bot_token"] == "my-bot-token-123"


class TestChannelsStepWebchat:
    """Webchat requires no extra prompts — just enabled=True."""

    def test_webchat_no_extra_prompts(self) -> None:
        console = _make_console()
        with patch(
            "animus_bootstrap.setup.steps.channels.Prompt.ask",
            side_effect=["webchat"],
        ):
            result = run_channels_step(console)

        assert "webchat" in result["channels"]
        assert result["channels"]["webchat"]["enabled"] is True
        # No extra fields beyond 'enabled'
        assert len(result["channels"]["webchat"]) == 1


class TestChannelsStepMultiple:
    """User selects multiple channels."""

    def test_multiple_channels(self) -> None:
        console = _make_console()
        # Select telegram and webchat by number (1 and 8)
        with patch(
            "animus_bootstrap.setup.steps.channels.Prompt.ask",
            side_effect=[
                "1, 8",  # select telegram (1) and webchat (8)
                "tg-token",  # telegram bot_token
            ],
        ):
            result = run_channels_step(console)

        channels = result["channels"]
        assert "telegram" in channels
        assert channels["telegram"]["bot_token"] == "tg-token"
        assert "webchat" in channels
        assert channels["webchat"]["enabled"] is True


class TestChannelsStepEmail:
    """Email channel prompts for all SMTP/IMAP fields."""

    def test_email_all_fields(self) -> None:
        console = _make_console()
        with patch(
            "animus_bootstrap.setup.steps.channels.Prompt.ask",
            side_effect=[
                "email",  # selection
                "smtp.example.com",  # smtp_host
                "imap.example.com",  # imap_host
                "user@example.com",  # username
                "secret-pass",  # password
            ],
        ):
            result = run_channels_step(console)

        email = result["channels"]["email"]
        assert email["enabled"] is True
        assert email["smtp_host"] == "smtp.example.com"
        assert email["imap_host"] == "imap.example.com"
        assert email["username"] == "user@example.com"
        assert email["password"] == "secret-pass"


class TestChannelsStepSlack:
    """Slack channel prompts for both app_token and bot_token."""

    def test_slack_two_tokens(self) -> None:
        console = _make_console()
        with patch(
            "animus_bootstrap.setup.steps.channels.Prompt.ask",
            side_effect=[
                "slack",  # selection
                "xapp-slack-app",  # app_token
                "xoxb-slack-bot",  # bot_token
            ],
        ):
            result = run_channels_step(console)

        slack = result["channels"]["slack"]
        assert slack["enabled"] is True
        assert slack["app_token"] == "xapp-slack-app"
        assert slack["bot_token"] == "xoxb-slack-bot"


# ------------------------------------------------------------------
# CLI — channels subcommand
# ------------------------------------------------------------------


class TestChannelsCLI:
    """Tests for the ``animus-bootstrap channels`` CLI subcommand."""

    def test_channels_list(self, tmp_path: Path) -> None:
        """``channels list`` shows all channels with status."""
        mgr = ConfigManager(config_dir=tmp_path)
        config = mgr.load()
        config.channels.telegram.enabled = True
        mgr.save(config)

        runner = CliRunner()
        with patch("animus_bootstrap.config.ConfigManager", return_value=mgr):
            result = runner.invoke(app, ["channels", "list"])

        assert result.exit_code == 0
        assert "telegram" in result.output
        assert "Enabled" in result.output

    def test_channels_enable(self, tmp_path: Path) -> None:
        """``channels enable telegram`` sets enabled=True and persists."""
        mgr = ConfigManager(config_dir=tmp_path)
        mgr.save(mgr.load())  # create initial config

        runner = CliRunner()
        with patch("animus_bootstrap.config.ConfigManager", return_value=mgr):
            result = runner.invoke(app, ["channels", "enable", "telegram"])

        assert result.exit_code == 0
        assert "enabled" in result.output.lower()

        reloaded = mgr.load()
        assert reloaded.channels.telegram.enabled is True

    def test_channels_disable(self, tmp_path: Path) -> None:
        """``channels disable webchat`` sets enabled=False and persists."""
        mgr = ConfigManager(config_dir=tmp_path)
        config = mgr.load()
        config.channels.webchat.enabled = True
        mgr.save(config)

        runner = CliRunner()
        with patch("animus_bootstrap.config.ConfigManager", return_value=mgr):
            result = runner.invoke(app, ["channels", "disable", "webchat"])

        assert result.exit_code == 0
        assert "disabled" in result.output.lower()

        reloaded = mgr.load()
        assert reloaded.channels.webchat.enabled is False

    def test_channels_enable_unknown(self, tmp_path: Path) -> None:
        """``channels enable bogus`` exits with error."""
        mgr = ConfigManager(config_dir=tmp_path)
        mgr.save(mgr.load())

        runner = CliRunner()
        with patch("animus_bootstrap.config.ConfigManager", return_value=mgr):
            result = runner.invoke(app, ["channels", "enable", "bogus"])

        assert result.exit_code == 1
        assert "Unknown channel" in result.output

    def test_channels_disable_unknown(self, tmp_path: Path) -> None:
        """``channels disable bogus`` exits with error."""
        mgr = ConfigManager(config_dir=tmp_path)
        mgr.save(mgr.load())

        runner = CliRunner()
        with patch("animus_bootstrap.config.ConfigManager", return_value=mgr):
            result = runner.invoke(app, ["channels", "disable", "bogus"])

        assert result.exit_code == 1
        assert "Unknown channel" in result.output
