#!/usr/bin/env python3
"""Validate docs/metrics/CLAIMS_EVIDENCE_MATRIX.md structure and file references."""

from __future__ import annotations

import re
from pathlib import Path

MATRIX_PATH = Path("docs/metrics/CLAIMS_EVIDENCE_MATRIX.md")


def parse_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        if re.fullmatch(r"\|[-| ]+\|", stripped):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        rows.append(cells)
    # remove header if present
    if rows and rows[0] and rows[0][0].lower() == "claim":
        rows = rows[1:]
    return rows


def extract_paths(claim_location: str) -> list[str]:
    # Support references like `README.md` or `docs/foo.md` with extra prose.
    inline = re.findall(r"`([^`]+)`", claim_location)
    candidates = inline or [claim_location]
    paths: list[str] = []
    for cand in candidates:
        for match in re.findall(r"[A-Za-z0-9_./-]+\.(?:md|yaml|yml|json|py)", cand):
            paths.append(match)
    return sorted(set(paths))


def main() -> int:
    if not MATRIX_PATH.exists():
        print(f"ERROR: missing matrix file: {MATRIX_PATH}")
        return 1

    text = MATRIX_PATH.read_text(encoding="utf-8")
    rows = parse_rows(text)

    if not rows:
        print("ERROR: no matrix rows found.")
        return 1

    errors: list[str] = []
    for idx, row in enumerate(rows, start=1):
        if len(row) < 6:
            errors.append(f"row {idx}: expected 6 columns, got {len(row)}")
            continue
        claim, claim_location, evidence_source, command, owner, cadence = row[:6]
        if not all([claim, claim_location, evidence_source, command, owner, cadence]):
            errors.append(f"row {idx}: one or more required cells are empty")
        for path in extract_paths(claim_location):
            if not Path(path).exists():
                errors.append(f"row {idx}: referenced path does not exist: {path}")

    if errors:
        print("CLAIMS-EVIDENCE VALIDATION FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print(f"Validated {len(rows)} claims-evidence rows in {MATRIX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
