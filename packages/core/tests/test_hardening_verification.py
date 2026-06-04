"""Pytest wrapper for the Stage 6 adversarial verification suite.

Imports the same scenarios that ``animus.scripts.verify_hardening`` runs
as a CLI, and exercises each as a pytest test. CI gating + report
generation use the CLI; this module proves the suite stays green inside
the regular test sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Stage 4 + 3.D scenarios call into ``animus_forge`` (self_improve / intelligence)
# and ``anthropic`` for cross-package integration assertions. When either
# package isn't installed (test-core CI matrix is core-only), the suite
# can't run — skip the whole file cleanly rather than raise at collection.
pytest.importorskip("animus_forge", reason="cross-package integration tests")
pytest.importorskip("anthropic", reason="anthropic client used in stage 3 + 4 scenarios")

from animus.scripts.verify_hardening import (  # noqa: E402
    SCENARIOS,
    run_scenarios,
    write_report,
)


@pytest.mark.parametrize(
    "name,stage,fn",
    SCENARIOS,
    ids=[f"stage{stage}-{name}" for name, stage, _ in SCENARIOS],
)
def test_adversarial_scenario(tmp_path: Path, name: str, stage: str, fn):
    """Each scenario runs against a fresh tmp_path data dir."""
    detail = fn(tmp_path)
    assert detail, "scenario returned empty detail string"


def test_full_suite_produces_report_artifact(tmp_path: Path):
    """The whole run_scenarios + write_report flow emits a JSON artifact."""
    report = run_scenarios(tmp_path)
    output_dir = tmp_path / "audit"
    path = write_report(report, output_dir)

    assert path.exists()
    assert path.name.startswith("hardening-report-")
    assert path.suffix == ".json"

    data = json.loads(path.read_text())
    assert data["total"] == len(SCENARIOS)
    assert data["total"] == data["passed"] + data["failed"]
    assert "results" in data
    assert all("stage" in r and "passed" in r for r in data["results"])


def test_full_suite_all_scenarios_pass(tmp_path: Path):
    """The closer — every scenario in the suite passes against the
    current codebase. Failing this test means a hardening gate has
    regressed and the security guarantee no longer holds."""
    report = run_scenarios(tmp_path)
    failures = [r for r in report.results if not r.passed]
    failure_summary = "\n".join(
        f"  [{r.stage}] {r.name}: {(r.error or '').splitlines()[0]}" for r in failures
    )
    assert report.failed == 0, (
        f"\n{report.failed} of {report.total} hardening scenarios FAILED:\n{failure_summary}"
    )
