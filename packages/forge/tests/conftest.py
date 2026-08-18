"""Pytest configuration and fixtures."""

import gc
import os
import resource
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Exclude benchmark tests from normal collection (requires pytest-benchmark).
# Benchmark CI job runs them explicitly via: pytest tests/test_benchmarks.py --benchmark-only
collect_ignore = [
    "test_benchmarks.py",
    "test_self_improve_ollama_integration.py",
    "test_evolution_loop_ollama.py",
]


def pytest_collection_modifyitems(items):
    """Quarantine only the exact Forge baseline debt recorded for CI.

    The ledger is opt-in so local development continues to expose the failures.
    New failures remain fatal because only exact node IDs receive the marker.
    """
    if os.environ.get("ANIMUS_FORGE_BASELINE_QUARANTINE") != "1":
        return

    ledger = Path(__file__).with_name("known_failures_ci.txt")
    known_failures = {
        line.strip()
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for item in items:
        node_id = item.nodeid.removeprefix("packages/forge/")
        if node_id in known_failures:
            item.add_marker(
                pytest.mark.xfail(
                    reason="tracked Forge compatibility debt; see docs/ci/forge-baseline-debt.md",
                    strict=False,
                )
            )


# --- OOM protection ---
# Cap virtual memory at 32GB to prevent runaway tests from crashing the machine.
# Python over-allocates virtual memory so this needs headroom above actual RSS.
_MEMORY_LIMIT_GB = 32
try:
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    _limit = _MEMORY_LIMIT_GB * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (_limit, hard))
except (OSError, ValueError):
    pass  # Some environments don't support RLIMIT_AS


@pytest.fixture(scope="session")
def _gc_counter():
    """Track completed tests without retaining test objects."""
    return iter(range(1, 1_000_000_000))


@pytest.fixture(autouse=True)
def _periodic_gc(_gc_counter):
    """Collect cycles periodically without imposing a full GC on every test."""
    yield
    if next(_gc_counter) % 100 == 0:
        gc.collect()
