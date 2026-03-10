"""Pytest configuration and fixtures."""

import gc
import resource

import pytest

# --- OOM protection ---
_MEMORY_LIMIT_GB = 32
try:
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    _limit = _MEMORY_LIMIT_GB * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (_limit, hard))
except (ValueError, resource.error):
    pass


@pytest.fixture(autouse=True)
def _force_gc():
    """Force garbage collection after every test to prevent memory accumulation."""
    yield
    gc.collect()
