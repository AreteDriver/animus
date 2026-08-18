"""Shared fixtures for kernel integration tests."""

from __future__ import annotations

import pytest

from animus_kernel.providers.mock_provider import MockProvider


@pytest.fixture(autouse=True)
def mock_ollama_constructor(monkeypatch):
    """Avoid live Ollama checks when tests construct HeadREPL."""

    class ConstructorProvider:
        def __init__(self, model, host="http://localhost:11434"):
            self.model = model
            self.base_url = host

        def is_configured(self):
            return True

    monkeypatch.setattr("animus_kernel.head.repl.OllamaProvider", ConstructorProvider)


@pytest.fixture
def mock_provider():
    """Return a fresh MockProvider with no lookup overrides."""
    return MockProvider()


@pytest.fixture
def mock_provider_with_responses():
    """Return a MockProvider with a lookup table for deterministic replies."""
    return MockProvider(responses={"hello": "Hello back!", "test": "Test reply"})
