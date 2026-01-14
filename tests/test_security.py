"""Security validation tests for Animus tools."""

import pytest
from pathlib import Path

from animus.config import ToolsSecurityConfig
from animus.tools import (
    _validate_path,
    _validate_command,
    _set_security_config,
    _tool_read_file,
    _tool_list_files,
    _tool_run_command,
)


@pytest.fixture
def security_config():
    """Create a test security config."""
    return ToolsSecurityConfig(
        allowed_paths=[str(Path.home())],
        blocked_paths=[
            "/etc/shadow",
            "/etc/passwd",
            "~/.ssh/id_*",
        ],
        command_enabled=True,
        command_blocklist=[
            "rm -rf /",
            "rm -rf ~",
            ":(){:|:&};:",
        ],
        command_timeout_seconds=10,
    )


@pytest.fixture(autouse=True)
def setup_security(security_config):
    """Set up security config for all tests."""
    _set_security_config(security_config)
    yield
    _set_security_config(None)


class TestPathValidation:
    """Tests for path validation."""

    def test_allow_home_directory(self):
        """Should allow paths under home directory."""
        is_valid, error = _validate_path(str(Path.home()))
        assert is_valid
        assert error is None

    def test_allow_home_subdirectory(self):
        """Should allow subdirectories of home."""
        is_valid, error = _validate_path(str(Path.home() / "documents"))
        assert is_valid
        assert error is None

    def test_block_etc_shadow(self):
        """Should block /etc/shadow."""
        is_valid, error = _validate_path("/etc/shadow")
        assert not is_valid
        assert "blocked" in error.lower() or "denied" in error.lower()

    def test_block_etc_passwd(self):
        """Should block /etc/passwd."""
        is_valid, error = _validate_path("/etc/passwd")
        assert not is_valid
        assert "denied" in error.lower()

    def test_block_outside_allowed(self):
        """Should block paths outside allowed directories."""
        is_valid, error = _validate_path("/tmp/test.txt")
        assert not is_valid
        assert "not in allowed" in error.lower()


class TestCommandValidation:
    """Tests for command validation."""

    def test_allow_safe_command(self):
        """Should allow safe commands."""
        is_valid, error = _validate_command("ls -la")
        assert is_valid
        assert error is None

    def test_block_rm_rf_root(self):
        """Should block rm -rf /."""
        is_valid, error = _validate_command("rm -rf /")
        assert not is_valid
        assert "blocked" in error.lower()

    def test_block_rm_rf_home(self):
        """Should block rm -rf ~."""
        is_valid, error = _validate_command("rm -rf ~")
        assert not is_valid
        assert "blocked" in error.lower()

    def test_block_fork_bomb(self):
        """Should block fork bomb."""
        is_valid, error = _validate_command(":(){:|:&};:")
        assert not is_valid
        assert "blocked" in error.lower()


class TestToolSecurity:
    """Tests for tool-level security."""

    def test_read_file_blocked_path(self):
        """read_file should reject blocked paths."""
        result = _tool_read_file({"path": "/etc/shadow"})
        assert not result.success
        assert "denied" in result.error.lower()

    def test_list_files_blocked_path(self):
        """list_files should reject blocked paths."""
        result = _tool_list_files({"directory": "/etc"})
        assert not result.success
        assert "denied" in result.error.lower() or "not in allowed" in result.error.lower()

    def test_run_command_blocked(self):
        """run_command should reject blocked commands."""
        result = _tool_run_command({"command": "rm -rf /"})
        assert not result.success
        assert "blocked" in result.error.lower()


class TestDisabledCommands:
    """Tests for disabled command execution."""

    def test_disabled_commands(self):
        """Should reject all commands when disabled."""
        config = ToolsSecurityConfig(
            command_enabled=False,
            allowed_paths=[str(Path.home())],
        )
        _set_security_config(config)

        is_valid, error = _validate_command("ls")
        assert not is_valid
        assert "disabled" in error.lower()
