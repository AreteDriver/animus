"""Tests for hardware detection and model recommendations."""

from __future__ import annotations

import subprocess
from collections import namedtuple
from unittest.mock import MagicMock, patch

from animus_forge.providers.hardware import (
    _GB_PER_BILLION_PARAMS,
    HardwareProfile,
    _detect_nvidia_gpu,
    _max_params_for_memory,
    detect_hardware,
)

# ---------------------------------------------------------------------------
# HardwareProfile dataclass
# ---------------------------------------------------------------------------


class TestHardwareProfile:
    def test_defaults(self):
        p = HardwareProfile()
        assert p.platform_id == ""
        assert p.total_memory_gb == 0.0
        assert p.available_memory_gb == 0.0
        assert p.gpu_name is None
        assert p.gpu_vram_gb is None
        assert p.unified_memory is False
        assert p.cpu_cores == 0
        assert p.max_model_params_b == 0.0
        assert p.recommended_tier == "fast"
        assert p.recommended_models == []
        assert p.warnings == []

    def test_custom_values(self):
        p = HardwareProfile(
            platform_id="arm64",
            total_memory_gb=192.0,
            gpu_name="A100",
            gpu_vram_gb=80.0,
            recommended_tier="reasoning",
            recommended_models=["qwen2.5:72b"],
            warnings=["test warning"],
        )
        assert p.platform_id == "arm64"
        assert p.gpu_name == "A100"
        assert p.gpu_vram_gb == 80.0
        assert p.warnings == ["test warning"]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestMaxParamsForMemory:
    def test_formula(self):
        # 16 GB - 2 GB overhead = 14 GB usable / 1.2 = ~11.7
        result = _max_params_for_memory(16.0)
        assert result == round(14.0 / _GB_PER_BILLION_PARAMS, 1)

    def test_low_memory_floors_at_zero(self):
        result = _max_params_for_memory(1.0)
        assert result == 0.0

    def test_zero_memory(self):
        result = _max_params_for_memory(0.0)
        assert result == 0.0


# ---------------------------------------------------------------------------
# detect_hardware() — no psutil
# ---------------------------------------------------------------------------


class TestDetectHardwareNoPsutil:
    def test_returns_degraded_profile(self):
        with patch("animus_forge.providers.hardware.HAS_PSUTIL", False):
            profile = detect_hardware()
        assert profile.recommended_tier == "fast"
        assert len(profile.recommended_models) > 0
        assert any("psutil" in w for w in profile.warnings)

    def test_no_crash_without_psutil(self):
        with patch("animus_forge.providers.hardware.HAS_PSUTIL", False):
            profile = detect_hardware()
        assert isinstance(profile, HardwareProfile)


# ---------------------------------------------------------------------------
# detect_hardware() — Apple Silicon paths
# ---------------------------------------------------------------------------


_VirtualMemory = namedtuple("svmem", ["total", "available", "percent", "used", "free"])


def _mock_psutil(total_gb: float, avail_gb: float, cores: int = 8):
    """Create mock psutil with given memory values."""
    mock = MagicMock()
    mock.virtual_memory.return_value = _VirtualMemory(
        total=int(total_gb * 1024**3),
        available=int(avail_gb * 1024**3),
        percent=round((1 - avail_gb / total_gb) * 100, 1),
        used=int((total_gb - avail_gb) * 1024**3),
        free=int(avail_gb * 1024**3),
    )
    mock.cpu_count.return_value = cores
    return mock


class TestDetectHardwareAppleSilicon:
    """Apple Silicon (Darwin + arm64) with unified memory."""

    def _detect(self, total_gb: float, avail_gb: float = None):
        if avail_gb is None:
            avail_gb = total_gb * 0.8
        mock_ps = _mock_psutil(total_gb, avail_gb)
        with (
            patch("animus_forge.providers.hardware.HAS_PSUTIL", True),
            patch("animus_forge.providers.hardware.psutil", mock_ps),
            patch("animus_forge.providers.hardware.platform") as mock_platform,
        ):
            mock_platform.system.return_value = "Darwin"
            mock_platform.machine.return_value = "arm64"
            return detect_hardware()

    def test_192gb_reasoning_tier(self):
        p = self._detect(192.0)
        assert p.recommended_tier == "reasoning"
        assert p.unified_memory is True
        assert "72b" in p.recommended_models[0] or "70b" in p.recommended_models[0]

    def test_64gb_reasoning_tier(self):
        p = self._detect(64.0)
        assert p.recommended_tier == "reasoning"
        assert any("32b" in m for m in p.recommended_models)

    def test_48gb_reasoning_tier(self):
        p = self._detect(48.0)
        assert p.recommended_tier == "reasoning"

    def test_16gb_standard_tier(self):
        p = self._detect(16.0)
        assert p.recommended_tier == "standard"

    def test_8gb_standard_tier(self):
        p = self._detect(8.0)
        assert p.recommended_tier == "standard"

    def test_4gb_fast_tier_with_warning(self):
        p = self._detect(4.0)
        assert p.recommended_tier == "fast"
        assert any("Low memory" in w for w in p.warnings)

    def test_max_model_params_set(self):
        p = self._detect(64.0)
        assert p.max_model_params_b > 0


# ---------------------------------------------------------------------------
# detect_hardware() — NVIDIA GPU paths
# ---------------------------------------------------------------------------


class TestDetectHardwareNvidia:
    """NVIDIA GPU detected via nvidia-smi."""

    def _detect(self, gpu_name: str, vram_mb: float, sys_total_gb: float = 64.0):
        mock_ps = _mock_psutil(sys_total_gb, sys_total_gb * 0.7)
        nvidia_output = f"{gpu_name}, {vram_mb}"
        with (
            patch("animus_forge.providers.hardware.HAS_PSUTIL", True),
            patch("animus_forge.providers.hardware.psutil", mock_ps),
            patch("animus_forge.providers.hardware.platform") as mock_platform,
            patch("animus_forge.providers.hardware.subprocess") as mock_sub,
        ):
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "x86_64"
            mock_sub.run.return_value = MagicMock(returncode=0, stdout=nvidia_output)
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            return detect_hardware()

    def test_a100_80gb_reasoning(self):
        p = self._detect("NVIDIA A100-SXM4-80GB", 81920)
        assert p.recommended_tier == "reasoning"
        assert p.gpu_name == "NVIDIA A100-SXM4-80GB"
        assert p.gpu_vram_gb == 80.0

    def test_rtx_4090_24gb_standard(self):
        p = self._detect("NVIDIA GeForce RTX 4090", 24576)
        assert p.recommended_tier == "standard"

    def test_rtx_3060_12gb_standard(self):
        p = self._detect("NVIDIA GeForce RTX 3060", 12288)
        assert p.recommended_tier == "standard"

    def test_rtx_3050_8gb_standard(self):
        p = self._detect("NVIDIA GeForce RTX 3050", 8192)
        assert p.recommended_tier == "standard"

    def test_low_vram_4gb_fast(self):
        p = self._detect("NVIDIA GeForce GTX 1650", 4096)
        assert p.recommended_tier == "fast"
        assert any("Low VRAM" in w for w in p.warnings)

    def test_max_model_params_based_on_vram(self):
        p = self._detect("NVIDIA A100-SXM4-80GB", 81920)
        assert p.max_model_params_b > 0


# ---------------------------------------------------------------------------
# detect_hardware() — CPU-only fallback
# ---------------------------------------------------------------------------


class TestDetectHardwareCpuOnly:
    """No GPU detected — CPU-only inference."""

    def _detect(self, total_gb: float, avail_gb: float):
        mock_ps = _mock_psutil(total_gb, avail_gb)
        with (
            patch("animus_forge.providers.hardware.HAS_PSUTIL", True),
            patch("animus_forge.providers.hardware.psutil", mock_ps),
            patch("animus_forge.providers.hardware.platform") as mock_platform,
            patch("animus_forge.providers.hardware.subprocess") as mock_sub,
        ):
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "x86_64"
            # nvidia-smi not found
            mock_sub.run.side_effect = FileNotFoundError
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            return detect_hardware()

    def test_high_ram_standard(self):
        p = self._detect(64.0, 48.0)
        assert p.recommended_tier == "standard"
        assert any("No GPU" in w for w in p.warnings)

    def test_medium_ram_fast(self):
        p = self._detect(32.0, 20.0)
        assert p.recommended_tier == "fast"

    def test_low_ram_fast_with_extra_warning(self):
        p = self._detect(16.0, 8.0)
        assert p.recommended_tier == "fast"
        assert any("Low available" in w for w in p.warnings)

    def test_cpu_cores_detected(self):
        p = self._detect(32.0, 24.0)
        assert p.cpu_cores > 0


# ---------------------------------------------------------------------------
# _detect_nvidia_gpu() edge cases
# ---------------------------------------------------------------------------


class TestDetectNvidiaGpu:
    def test_nvidia_smi_not_found(self):
        with patch("animus_forge.providers.hardware.subprocess") as mock_sub:
            mock_sub.run.side_effect = FileNotFoundError
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            assert _detect_nvidia_gpu() is None

    def test_nvidia_smi_timeout(self):
        with patch("animus_forge.providers.hardware.subprocess") as mock_sub:
            mock_sub.run.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            assert _detect_nvidia_gpu() is None

    def test_nvidia_smi_nonzero_exit(self):
        with patch("animus_forge.providers.hardware.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=1, stdout="")
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            assert _detect_nvidia_gpu() is None

    def test_nvidia_smi_bad_output(self):
        with patch("animus_forge.providers.hardware.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stdout="garbage")
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            assert _detect_nvidia_gpu() is None

    def test_nvidia_smi_valid_output(self):
        with patch("animus_forge.providers.hardware.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stdout="NVIDIA A100, 81920\n")
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            result = _detect_nvidia_gpu()
            assert result == ("NVIDIA A100", 80.0)
