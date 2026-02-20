"""Hardware detection for local LLM model recommendations."""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    HAS_PSUTIL = False

# Rule of thumb: ~1.2 GB per billion parameters at Q4 quantization
_GB_PER_BILLION_PARAMS = 1.2


@dataclass
class HardwareProfile:
    """Detected hardware capabilities and model recommendations."""

    platform_id: str = ""
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None
    unified_memory: bool = False
    cpu_cores: int = 0
    max_model_params_b: float = 0.0
    recommended_tier: str = "fast"
    recommended_models: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_hardware() -> HardwareProfile:
    """Detect system hardware and recommend local LLM configuration.

    Returns a HardwareProfile with model recommendations based on
    available memory/VRAM. Gracefully degrades if psutil is missing
    or GPU detection fails.
    """
    profile = HardwareProfile(platform_id=platform.machine())

    if not HAS_PSUTIL:
        profile.warnings.append("psutil not installed — using conservative defaults")
        profile.recommended_tier = "fast"
        profile.recommended_models = ["qwen2.5:3b", "llama3.2:1b", "phi3"]
        return profile

    # CPU info
    profile.cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1

    # System memory
    mem = psutil.virtual_memory()
    profile.total_memory_gb = round(mem.total / (1024**3), 1)
    profile.available_memory_gb = round(mem.available / (1024**3), 1)

    # Apple Silicon detection (unified memory = GPU can use all RAM)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        profile.unified_memory = True
        return _recommend_apple_silicon(profile)

    # NVIDIA GPU detection
    nvidia = _detect_nvidia_gpu()
    if nvidia is not None:
        profile.gpu_name, profile.gpu_vram_gb = nvidia
        return _recommend_nvidia(profile)

    # CPU-only fallback
    return _recommend_cpu_only(profile)


def _detect_nvidia_gpu() -> tuple[str, float] | None:
    """Detect NVIDIA GPU name and VRAM via nvidia-smi.

    Returns (gpu_name, vram_gb) or None if not available.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        line = result.stdout.strip().split("\n")[0]
        parts = line.split(",")
        if len(parts) < 2:
            return None

        name = parts[0].strip()
        vram_mb = float(parts[1].strip())
        return name, round(vram_mb / 1024, 1)
    except FileNotFoundError:
        return None
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        logger.debug("nvidia-smi detection failed", exc_info=True)
        return None


def _max_params_for_memory(memory_gb: float) -> float:
    """Estimate max model parameters (billions) for given memory."""
    # Reserve ~2 GB for OS/runtime overhead
    usable = max(0, memory_gb - 2.0)
    return round(usable / _GB_PER_BILLION_PARAMS, 1)


def _recommend_apple_silicon(profile: HardwareProfile) -> HardwareProfile:
    """Recommendations for Apple Silicon with unified memory."""
    mem = profile.total_memory_gb
    profile.max_model_params_b = _max_params_for_memory(mem)

    if mem >= 128:
        profile.recommended_tier = "reasoning"
        profile.recommended_models = [
            "qwen2.5:72b",
            "deepseek-r1:70b",
            "llama3.1:70b",
        ]
    elif mem >= 48:
        profile.recommended_tier = "reasoning"
        profile.recommended_models = [
            "qwen2.5:32b",
            "deepseek-r1:32b",
            "qwen2.5:14b",
        ]
    elif mem >= 16:
        profile.recommended_tier = "standard"
        profile.recommended_models = [
            "qwen2.5:14b",
            "qwen2.5",
            "llama3.2",
            "deepseek-r1:8b",
        ]
    elif mem >= 8:
        profile.recommended_tier = "standard"
        profile.recommended_models = ["llama3.2", "qwen2.5", "phi3"]
    else:
        profile.recommended_tier = "fast"
        profile.recommended_models = ["qwen2.5:3b", "llama3.2:1b", "phi3"]
        profile.warnings.append(f"Low memory ({mem} GB) — only small models recommended")

    return profile


def _recommend_nvidia(profile: HardwareProfile) -> HardwareProfile:
    """Recommendations based on NVIDIA GPU VRAM."""
    vram = profile.gpu_vram_gb or 0.0
    profile.max_model_params_b = _max_params_for_memory(vram)

    if vram >= 48:
        profile.recommended_tier = "reasoning"
        profile.recommended_models = [
            "qwen2.5:72b",
            "deepseek-r1:70b",
            "llama3.1:70b",
        ]
    elif vram >= 20:
        profile.recommended_tier = "standard"
        profile.recommended_models = [
            "qwen2.5:14b",
            "qwen2.5",
            "llama3.2",
            "deepseek-r1:8b",
        ]
    elif vram >= 8:
        profile.recommended_tier = "standard"
        profile.recommended_models = ["llama3.2", "qwen2.5", "phi3"]
    else:
        profile.recommended_tier = "fast"
        profile.recommended_models = ["qwen2.5:3b", "llama3.2:1b", "phi3"]
        profile.warnings.append(f"Low VRAM ({vram} GB) — only small models recommended")

    return profile


def _recommend_cpu_only(profile: HardwareProfile) -> HardwareProfile:
    """Conservative recommendations for CPU-only inference."""
    mem = profile.available_memory_gb
    profile.max_model_params_b = _max_params_for_memory(mem)
    profile.warnings.append("No GPU detected — CPU inference will be slow")

    if mem >= 32:
        profile.recommended_tier = "standard"
        profile.recommended_models = ["qwen2.5", "llama3.2", "phi3"]
    elif mem >= 16:
        profile.recommended_tier = "fast"
        profile.recommended_models = ["qwen2.5:3b", "llama3.2:1b", "phi3"]
    else:
        profile.recommended_tier = "fast"
        profile.recommended_models = ["llama3.2:1b", "phi3"]
        profile.warnings.append(f"Low available memory ({mem} GB) — expect slow inference")

    return profile
