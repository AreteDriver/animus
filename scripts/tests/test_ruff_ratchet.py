"""Regression tests for the Ruff debt ratchet."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "ruff-ratchet.py"
SPEC = importlib.util.spec_from_file_location("ruff_ratchet", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ruff_ratchet = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ruff_ratchet
SPEC.loader.exec_module(ruff_ratchet)


def test_measure_counts_lint_and_format_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    results = iter(
        [
            subprocess.CompletedProcess([], 1, '[{"code":"F401"},{"code":"F841"}]', ""),
            subprocess.CompletedProcess([], 1, "2 files would be reformatted", ""),
        ]
    )
    monkeypatch.setattr(ruff_ratchet, "_run_ruff", lambda *args: next(results))

    assert ruff_ratchet.measure("packages/example") == (2, 2)


def test_main_fails_when_either_budget_increases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = tmp_path / ".ruff-baseline.json"
    baseline.write_text(
        json.dumps({"example": {"directory": "packages/example", "lint": 2, "format": 1}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ruff_ratchet, "BASELINE_FILE", baseline)
    monkeypatch.setattr(ruff_ratchet, "measure", lambda directory: (2, 2))

    assert ruff_ratchet.main() == 1


def test_main_accepts_only_equal_or_lower_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = tmp_path / ".ruff-baseline.json"
    baseline.write_text(
        json.dumps({"example": {"directory": "packages/example", "lint": 2, "format": 1}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ruff_ratchet, "BASELINE_FILE", baseline)
    monkeypatch.setattr(ruff_ratchet, "measure", lambda directory: (1, 1))

    assert ruff_ratchet.main() == 0
