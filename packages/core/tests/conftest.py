"""Shared test fixtures for Animus test suite."""

from __future__ import annotations

import gc
import resource
from pathlib import Path

import pytest

from animus.cognitive import CognitiveLayer, ModelConfig

# Exclude benchmark tests from normal collection (requires pytest-benchmark).
# Benchmark CI job runs them explicitly via: pytest tests/test_benchmarks.py --benchmark-only
collect_ignore = ["test_benchmarks.py"]

# --- OOM protection ---
_MEMORY_LIMIT_GB = 32
try:
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    _limit = _MEMORY_LIMIT_GB * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (_limit, hard))
except (OSError, ValueError):
    pass


@pytest.fixture(autouse=True)
def _force_gc():
    """Force garbage collection after every test to prevent memory accumulation."""
    yield
    gc.collect()


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
def fake_secret_corpus() -> dict[str, str]:
    """Inert fake secrets for SEC redaction tests.

    These values are deliberately synthetic and exercise every credential
    pattern in ``animus_types.secrets`` plus multiline and base64-encoded
    forms. They must never appear in logs, errors, or traces after redaction.
    """
    import base64

    encoded_token_input = b"sk-ant-api03-encodedfakefakefakefakefakefake"
    return {
        "anthropic_key": "sk-ant-api03-fakefakefakefakefakefakefakefakefake",
        "openai_key": "sk-abcdefghijklmnopqrstuvwxyz1234567890abcdef",
        "github_token": "ghp_fake1234567890abcdef",
        "github_pat": "github_pat_11ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcdef",
        "aws_access_key": "AKIAIOSFODNN7EXAMPLE",
        "stripe_key": "sk_test_fakefakefakefakefakefakefakefake",  # synthetic; not a real Stripe key
        "slack_token": "xoxb-fakefakefakefakefakefakefakefake",
        "bearer_token": "Bearer fakefakefakefakefakefakefakefakefakefakefakefake",
        "api_key_label": "my_secret_key_is: fakefakefakefakefakefakefake",
        "password_labeled": "password=Sup3rS3cr3tP@ssw0rd!12345",
        "encoded_token": base64.b64encode(encoded_token_input).decode(),
        "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
    }


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
