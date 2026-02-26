"""Shared test fixtures for Animus test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from animus.cognitive import CognitiveLayer, ModelConfig

# Exclude benchmark tests from normal collection (requires pytest-benchmark).
# Benchmark CI job runs them explicitly via: pytest tests/test_benchmarks.py --benchmark-only
collect_ignore = ["test_benchmarks.py"]


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temp directory for test isolation."""
    data_dir = tmp_path / "animus_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_cognitive() -> CognitiveLayer:
    """CognitiveLayer backed by a deterministic mock model."""
    return CognitiveLayer(
        ModelConfig.mock(
            default_response="Mock response.",
            response_map={},
        )
    )


@pytest.fixture
def mock_cognitive_factory():
    """Factory for creating CognitiveLayer with custom responses."""

    def _make(
        default_response: str = "Mock response.",
        response_map: dict[str, str] | None = None,
    ) -> CognitiveLayer:
        return CognitiveLayer(
            ModelConfig.mock(
                default_response=default_response,
                response_map=response_map or {},
            )
        )

    return _make
