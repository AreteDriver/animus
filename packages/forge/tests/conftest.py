"""Pytest configuration and fixtures."""

import gc
import resource
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Exclude benchmark tests from normal collection (requires pytest-benchmark).
# Benchmark CI job runs them explicitly via: pytest tests/test_benchmarks.py --benchmark-only
collect_ignore = ["test_benchmarks.py", "test_self_improve_ollama_integration.py"]

# --- OOM protection ---
# Cap virtual memory at 32GB to prevent runaway tests from crashing the machine.
# Python over-allocates virtual memory so this needs headroom above actual RSS.
_MEMORY_LIMIT_GB = 32
try:
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    _limit = _MEMORY_LIMIT_GB * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (_limit, hard))
except (ValueError, resource.error):
    pass  # Some environments don't support RLIMIT_AS


@pytest.fixture(autouse=True)
def _force_gc():
    """Force garbage collection after every test to prevent memory accumulation."""
    yield
    gc.collect()
