"""Test that ``pgrep`` is not used in authoritative lifecycle paths.

ADR-007 explicitly forbids ``pgrep`` in runtime state detection.
This test asserts the rule by static analysis of the lifecycle
package source — checking that ``pgrep`` and ``pkill`` are never
*called* (not merely mentioned in documentation).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "animus_bootstrap"
    / "lifecycle"
)


def _walk_sources() -> list[Path]:
    return list(PACKAGE_ROOT.glob("*.py"))


def _calls_in_module(path: Path) -> set[str]:
    """Return the set of names *called* by the module via AST.

    Walks ``ast.Call`` nodes and collects bare-name calls. Excludes
    docstrings, comments, attribute calls (e.g. ``foo.pgrep()``), and
    string literals. The set is small and exact — we use it to
    verify that ``pgrep`` / ``pkill`` are not invoked.
    """
    tree = ast.parse(path.read_text())
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    return calls


def test_no_pgrep_called_in_lifecycle_module() -> None:
    """No source file in the lifecycle package calls pgrep."""
    for path in _walk_sources():
        calls = _calls_in_module(path)
        assert "pgrep" not in calls, (
            f"pgrep() called in {path}"
        )


def test_no_kill_signal_called_in_lifecycle_module() -> None:
    """The lifecycle package must not signal PIDs directly."""
    for path in _walk_sources():
        calls = _calls_in_module(path)
        # os.kill and signal.SIGTERM/SIGKILL/SIGINT etc.
        assert "kill" not in calls, f"kill() called in {path}"


def test_no_pkill_called_in_lifecycle_module() -> None:
    """The lifecycle package must not invoke pkill."""
    for path in _walk_sources():
        calls = _calls_in_module(path)
        assert "pkill" not in calls, f"pkill() called in {path}"


def test_classification_has_no_kill_authority() -> None:
    """The ClassificationResult dataclass must not have a kill-authority field."""
    from animus_bootstrap.lifecycle.classification import ClassificationResult
    from dataclasses import fields
    field_names = {f.name for f in fields(ClassificationResult)}
    assert "allow_kill" not in field_names
    assert "kill_authority" not in field_names
