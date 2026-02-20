"""Client factories and circuit breaker setup for workflow execution."""

from __future__ import annotations

import logging

from animus_forge.utils.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# Global circuit breakers for step types
_circuit_breakers: dict[str, CircuitBreaker] = {}


def configure_circuit_breaker(
    key: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 2,
) -> CircuitBreaker:
    """Configure a circuit breaker for a step type or custom key.

    Args:
        key: Identifier for the circuit breaker (e.g., step type or custom key)
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before trying again
        success_threshold: Successes needed in half-open to close circuit

    Returns:
        The configured CircuitBreaker instance
    """
    cb = CircuitBreaker(
        name=key,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        success_threshold=success_threshold,
    )
    _circuit_breakers[key] = cb
    return cb


def get_circuit_breaker(key: str) -> CircuitBreaker | None:
    """Get circuit breaker by key."""
    return _circuit_breakers.get(key)


def reset_circuit_breakers() -> None:
    """Reset all circuit breakers (for testing)."""
    global _circuit_breakers
    for cb in _circuit_breakers.values():
        cb.reset()
    _circuit_breakers = {}


# Lazy-loaded API clients to avoid circular imports
_claude_client = None
_openai_client = None


def _get_claude_client():
    """Get or create ClaudeCodeClient instance."""
    global _claude_client
    if _claude_client is None:
        try:
            from animus_forge.api_clients.claude_code_client import ClaudeCodeClient

            _claude_client = ClaudeCodeClient()
        except Exception:
            _claude_client = False  # Mark as unavailable
    return _claude_client if _claude_client else None


def _get_openai_client():
    """Get or create OpenAIClient instance."""
    global _openai_client
    if _openai_client is None:
        try:
            from animus_forge.api_clients.openai_client import OpenAIClient

            _openai_client = OpenAIClient()
        except Exception:
            _openai_client = False  # Mark as unavailable
    return _openai_client if _openai_client else None
