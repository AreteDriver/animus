"""Integration test: verify all Python packages import without circular dependencies.

This test does NOT require a running stack. It checks that the package layout
is sound and that cross-package imports resolve correctly.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()

# Map package import name → source directory that must be on sys.path
PACKAGE_PATHS: dict[str, str] = {
    "animus": "packages/core",
    "animus_kernel": "packages/kernel/src",
    "animus_types": "packages/types/src",
    "animus_forge": "packages/forge/src",
    "animus_bootstrap": "packages/bootstrap/src",
    "animus_contracts": "packages/contracts/src",
}


def _ensure_paths():
    """Add package source directories to sys.path if not already present."""
    for rel in PACKAGE_PATHS.values():
        abs_path = str(REPO_ROOT / rel)
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)


@pytest.fixture(scope="module", autouse=True)
def _setup_paths():
    _ensure_paths()


@pytest.mark.parametrize("pkg_name", list(PACKAGE_PATHS.keys()))
def test_package_imports(pkg_name: str):
    """Each package must import without raising."""
    mod = importlib.import_module(pkg_name)
    assert mod.__file__ is not None


def test_cross_package_type_consistency():
    """animus_types.Sensitivity must be the same object animus_kernel re-exports."""
    import animus_types
    import animus_kernel.memory.types as memory_types

    assert memory_types.Sensitivity is animus_types.Sensitivity


def test_kernel_uses_types_secrets():
    """The kernel redaction module must import credential patterns from animus_types."""
    import animus_kernel.memory.redaction as redaction
    import animus_types.secrets as secrets

    assert redaction.CREDENTIAL_PATTERNS is secrets.CREDENTIAL_PATTERNS
