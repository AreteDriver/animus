#!/usr/bin/env python3
"""Traceability linter — maps ADR requirements to tests and reports gaps.

Usage:
    python scripts/traceability_linter.py [--adrs adrs/] [--tests packages/]

Exit codes:
    0 — all ADRs have at least one associated test file
    1 — one or more ADRs have zero test coverage
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _extract_adr_ids(adr_dir: Path) -> set[str]:
    """Extract ADR identifiers (e.g. ADR-001) from all markdown files in adrs/."""
    ids: set[str] = set()
    for path in adr_dir.glob("*.md"):
        text = path.read_text()
        # Capture ADR-NNN anywhere in the file
        ids.update(re.findall(r"ADR-0*\d+", text))
    # Also capture filenames like ADR-001.md
    for p in adr_dir.glob("*.md"):
        ids.update(re.findall(r"ADR-0*\d+", p.stem))
    return ids


def _normalize_adr_id(adr_id: str) -> str:
    """Normalize 'ADR-001' to 'ADR-001' (zero-padded, 3 digits)."""
    match = re.match(r"ADR-(\d+)", adr_id)
    if match:
        return f"ADR-{int(match.group(1)):03d}"
    return adr_id


def _find_test_refs(test_dirs: list[Path], adr_ids: set[str]) -> dict[str, list[str]]:
    """Map each ADR ID to the list of test files that mention it."""
    coverage: dict[str, list[str]] = {aid: [] for aid in adr_ids}
    for td in test_dirs:
        for test_file in td.rglob("test_*.py"):
            text = test_file.read_text()
            for aid in adr_ids:
                # Match ADR-001, ADR_001, adr_001 in comments or docstrings
                patterns = [
                    re.escape(aid),
                    re.escape(aid.lower()),
                    re.escape(aid.replace("-", "_")),
                    re.escape(aid.lower().replace("-", "_")),
                ]
                if any(re.search(p, text) for p in patterns):
                    rel = str(test_file.relative_to(Path.cwd()))
                    if rel not in coverage[aid]:
                        coverage[aid].append(rel)
    return coverage


def _print_report(coverage: dict[str, list[str]], threshold: int = 1) -> int:
    """Print coverage report. Returns number of ADRs below threshold."""
    covered = 0
    gaps: list[str] = []

    print("=" * 60)
    print("Traceability Report — ADR → Test Coverage")
    print("=" * 60)

    for aid in sorted(coverage):
        refs = coverage[aid]
        status = "✅" if len(refs) >= threshold else "⚠️"
        if len(refs) >= threshold:
            covered += 1
        else:
            gaps.append(aid)
        print(f"  {status} {aid}: {len(refs)} test file(s)")
        for ref in refs:
            print(f"      → {ref}")

    print("-" * 60)
    total = len(coverage)
    print(f"Summary: {covered}/{total} ADRs covered ({len(gaps)} gap(s))")
    if gaps:
        print(f"Gaps: {', '.join(gaps)}")
    print("=" * 60)
    return len(gaps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Traceability linter")
    parser.add_argument(
        "--adrs",
        type=Path,
        default=Path("adrs"),
        help="Directory containing ADR markdown files (default: adrs/)",
    )
    parser.add_argument(
        "--tests",
        type=Path,
        nargs="+",
        default=[Path("packages/core/tests"), Path("packages/forge/tests"), Path("packages/kernel/tests")],
        help="Test directories to scan (default: packages/*/tests)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="Minimum number of test files per ADR to count as covered",
    )
    args = parser.parse_args(argv)

    if not args.adrs.exists():
        print(f"ERROR: ADR directory not found: {args.adrs}")
        return 1

    adr_ids = _extract_adr_ids(args.adrs)
    if not adr_ids:
        print("WARNING: No ADR IDs found.")
        return 0

    coverage = _find_test_refs(args.tests, adr_ids)
    gaps = _print_report(coverage, threshold=args.threshold)
    return 1 if gaps > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
