"""SEC-06 non-memory audit: Forge ContainerManager must not leak env values in logs.

Covers the container command-line INFO logs that historically emitted
``-e KEY=VALUE`` arguments before execution.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_forge.scheduler.containers import ContainerConfig, ContainerManager


# ---------------------------------------------------------------------------
# Adversarial secret shapes (same corpus as SEC-08)
# ---------------------------------------------------------------------------

SECRET_SHAPES = [
    "sk-ant-api03-abcdefghijklmnopqrstuvwxyz123",
    "ghp_abcdefghij1234567890ABCDEFGH",
    "Bearer abcdefghijklmnopqrstuvwxyz1234",
    "credential_value=test1234567890ABCDEF",
    "ssn_value=123-45-6789 on file",
    "ProprietaryProjectX-SECRET-SAUCE-2026",
]


class TestContainerManagerLogging:
    """ContainerManager must mask ``-e`` / ``--env`` values before logging."""

    @pytest.mark.parametrize("secret", SECRET_SHAPES)
    def test_run_task_info_masks_env_values(
        self,
        caplog: pytest.LogCaptureFixture,
        secret: str,
        tmp_path: Path,
    ) -> None:
        """INFO-level container-task log must not contain the secret value."""
        config = ContainerConfig(
            env={"API_KEY": secret, "OTHER": "safe"},
            workspace_mount=str(tmp_path),
        )
        manager = ContainerManager(config)
        manager._runtime_cmd = "docker"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"status":"ok"}'
        mock_result.stderr = ""

        with caplog.at_level(logging.INFO, logger="animus_forge.scheduler.containers"):
            with patch("subprocess.run", return_value=mock_result):
                manager.run_task("t1", "m1", "citizen", "desc", {})

        assert secret not in caplog.text
        # The redacted placeholder should appear instead of the raw value.
        assert "[REDACTED]" in caplog.text

    def test_run_task_info_preserves_command_structure(
        self,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """Operational metadata (image, runtime, volume flags) must remain visible."""
        config = ContainerConfig(
            env={"API_KEY": "secret123"},
            workspace_mount=str(tmp_path),
            image="python:3.12-slim",
        )
        manager = ContainerManager(config)
        manager._runtime_cmd = "docker"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"status":"ok"}'
        mock_result.stderr = ""

        with caplog.at_level(logging.INFO, logger="animus_forge.scheduler.containers"):
            with patch("subprocess.run", return_value=mock_result):
                manager.run_task("t1", "m1", "citizen", "desc", {})

        log_text = caplog.text
        assert "docker" in log_text
        assert "python:3.12-slim" in log_text
        assert "-e" in log_text

    @pytest.mark.parametrize("secret", SECRET_SHAPES)
    @pytest.mark.asyncio
    async def test_run_task_async_info_masks_env_values(
        self,
        caplog: pytest.LogCaptureFixture,
        secret: str,
        tmp_path: Path,
    ) -> None:
        """INFO-level async container-task log must not contain the secret value."""
        config = ContainerConfig(
            env={"API_KEY": secret, "OTHER": "safe"},
            workspace_mount=str(tmp_path),
        )
        manager = ContainerManager(config)
        manager._runtime_cmd = "docker"

        mock_process = MagicMock()
        mock_process.stdout = asyncio.StreamReader()
        mock_process.stderr = asyncio.StreamReader()

        with caplog.at_level(logging.INFO, logger="animus_forge.scheduler.containers"):
            with patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_process,
            ):
                await manager.run_task_async("t1", "m1", "citizen", "desc", {})

        assert secret not in caplog.text
        assert "[REDACTED]" in caplog.text
