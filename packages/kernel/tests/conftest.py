"""Shared fixtures for kernel integration tests."""

from __future__ import annotations

import pytest

from animus_kernel.providers.mock_provider import MockProvider


@pytest.fixture
def mock_provider():
    """Return a fresh MockProvider with no lookup overrides."""
    return MockProvider()


@pytest.fixture
def mock_provider_with_responses():
    """Return a MockProvider with a lookup table for deterministic replies."""
    return MockProvider(responses={"hello": "Hello back!", "test": "Test reply"})
