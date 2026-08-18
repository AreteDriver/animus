#!/usr/bin/env python3
"""Fail when a package adds Ruff lint or formatting debt.

The baseline records debt that already exists on the PR's main-branch base.
Counts may move only downward. This keeps CI truthful without requiring an
unrelated repository-wide cleanup in every feature PR.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_FILE = Path(__file__).with_name(".ruff-baseline.json")
UNFORMATTED_RE = re.compile(r"(\d+) files? would be reformatted")


class PackageBaseline(TypedDict):
    directory: str
    lint: int
    format: int


def _run_ruff(*args: str) -> subprocess.CompletedProcess[str]:
    ruff = shutil.which("ruff")
    if ruff is None:
        raise RuntimeError("ruff is not installed")
    return subprocess.run(
        [ruff, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def measure(directory: str) -> tuple[int, int]:
    lint_result = _run_ruff("check", directory, "--output-format", "json")
    try:
        lint_count = len(json.loads(lint_result.stdout or "[]"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ruff emitted invalid JSON for {directory}") from exc

    format_result = _run_ruff("format", "--check", directory)
    format_output = format_result.stdout + format_result.stderr
    match = UNFORMATTED_RE.search(format_output)
    format_count = int(match.group(1)) if match else 0
    return lint_count, format_count


def main() -> int:
    with BASELINE_FILE.open(encoding="utf-8") as handle:
        baseline: dict[str, PackageBaseline] = json.load(handle)

    failed = False
    for package, limits in baseline.items():
        lint_count, format_count = measure(limits["directory"])
        lint_delta = lint_count - limits["lint"]
        format_delta = format_count - limits["format"]
        ok = lint_delta <= 0 and format_delta <= 0
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {package}: lint {lint_count}/{limits['lint']} "
            f"({lint_delta:+d}); format {format_count}/{limits['format']} "
            f"({format_delta:+d})"
        )
        failed |= not ok

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
