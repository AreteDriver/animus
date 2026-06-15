"""Offline default detection for AI providers.

Detects absence of cloud API keys and defaults to Ollama.
Guarantees total probe runtime < 100 ms regardless of DNS latency.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_CLOUD_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_PROBE_TIMEOUT = 0.07  # seconds; thread join adds ~20 ms overhead


def _has_cloud_keys() -> bool:
    """Return True if any cloud API key is present in the environment."""
    return any(os.environ.get(k) for k in _CLOUD_KEYS)


def _is_reachable(host: str, port: int, timeout: float = _PROBE_TIMEOUT) -> bool:
    """Fast, bounded TCP connectivity probe using a thread with timeout.

    Guarantees total runtime < ~100 ms regardless of DNS latency.
    """
    result = [False]

    def _try_connect() -> None:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                result[0] = True
        except Exception:
            pass

    t = threading.Thread(target=_try_connect, daemon=True)
    t.start()
    t.join(timeout + 0.02)
    return result[0]


def _parse_host(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    return parsed.hostname or "localhost", parsed.port or 11434


def detect_default_provider() -> str | None:
    """Detect the appropriate default provider based on environment.

    Returns:
        ``"ollama"`` when no cloud API keys are present and no cloud
        provider is forced via ``ANIMUS_FORCE_PROVIDER``.
        ``None`` when cloud keys are present or a cloud provider is forced.
    """
    force = os.environ.get("ANIMUS_FORCE_PROVIDER", "").strip().lower()

    if force:
        if force == "ollama":
            return "ollama"
        # Any other forced provider (including unknown) skips offline defaulting
        return None

    if _has_cloud_keys():
        return None

    return "ollama"


def get_ollama_host() -> str:
    """Return the configured Ollama host URL."""
    return os.environ.get("OLLAMA_HOST", _DEFAULT_OLLAMA_HOST)


def warn_if_ollama_unreachable(host: str | None = None) -> bool:
    """Warn if Ollama host is not reachable.

    Args:
        host: Ollama host URL (defaults to ``OLLAMA_HOST`` env var).

    Returns:
        True if reachable, False otherwise.
    """
    host = host or get_ollama_host()
    hostname, port = _parse_host(host)
    reachable = _is_reachable(hostname, port)
    if not reachable:
        logger.warning("Ollama not found at %s. Install: https://ollama.ai", host)
    return reachable
