"""Standalone validation functions for the onboarding wizard."""

from __future__ import annotations

import httpx


def test_anthropic_key(key: str) -> bool:
    """Validate an Anthropic API key by making a minimal API call.

    Sends a single-token request to the messages endpoint. Returns True
    if the server responds with 200, False on any error or non-200 status.

    Args:
        key: The Anthropic API key to validate.

    Returns:
        True if the key is valid, False otherwise.
    """
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=10.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def test_forge_connection(host: str, port: int) -> bool:
    """Test connectivity to a Forge orchestration engine instance.

    Makes a GET request to the ``/health`` endpoint and returns True
    if the server responds with 200.

    Args:
        host: Hostname or IP address of the Forge instance.
        port: Port number.

    Returns:
        True if Forge is reachable and healthy, False otherwise.
    """
    url = f"http://{host}:{port}/health"
    try:
        resp = httpx.get(url, timeout=2.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False
