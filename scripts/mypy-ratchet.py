#!/usr/bin/env python3
"""Mypy error-count ratchet.

Fails the build if the error count for any package exceeds its recorded baseline.
Run this after ``mypy`` to enforce a downward trend on type errors.

Usage::

    python scripts/mypy-ratchet.py core kernel [forge] [...]

Each package needs a ``.mypy-baseline.json`` file in the repo root with a
mapping ``{package: {allowed: N, directory: "path/to/code"}}``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASELINE_FILE = Path(__file__).with_name(".mypy-baseline.json")


def run_mypy(directory: str) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", directory, "--ignore-missing-imports", "--no-error-summary"],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    error_count = output.count(": error:")
    if result.returncode != 0 and error_count == 0:
        raise RuntimeError(
            f"mypy failed without producing a type-error report for {directory} "
            f"(exit {result.returncode}):\n{output.strip()}"
        )
    return error_count, output


def main(argv: list[str]) -> int:
    if not BASELINE_FILE.exists():
        print(f"Ratchet baseline file not found: {BASELINE_FILE}")
        print("Create it with: python scripts/mypy-ratchet.py --init")
        return 1

    with BASELINE_FILE.open() as fh:
        baseline: dict[str, dict[str, str | int]] = json.load(fh)

    if not argv or argv[0] == "--init":
        # Re-baseline current error counts
        for pkg, cfg in baseline.items():
            try:
                count, _ = run_mypy(str(cfg["directory"]))
            except RuntimeError as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                return 2
            cfg["allowed"] = count  # type: ignore[assignment]
        with BASELINE_FILE.open("w") as fh:
            json.dump(baseline, fh, indent=2)
        print("Re-baselined mypy error counts.")
        return 0

    failed = False
    infrastructure_failed = False
    for pkg in argv:
        cfg = baseline.get(pkg)
        if cfg is None:
            print(f"[SKIP] No baseline for package '{pkg}'")
            continue
        allowed = int(cfg["allowed"])
        directory = str(cfg["directory"])
        try:
            actual, report = run_mypy(directory)
        except RuntimeError as exc:
            print(f"[ERROR] {pkg}: {exc}", file=sys.stderr)
            infrastructure_failed = True
            continue
        report_path = Path("packages") / pkg / "mypy-report.txt"
        report_path.write_text(report, encoding="utf-8")
        delta = actual - allowed
        if actual > allowed:
            print(
                f"[FAIL] {pkg}: {actual} errors (allowed {allowed}, +{delta})\n"
                f"       Fix existing errors or lower the baseline with --init"
            )
            failed = True
        else:
            print(f"[PASS] {pkg}: {actual} errors (allowed {allowed}, {delta or 'at limit'})")

    if infrastructure_failed:
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
