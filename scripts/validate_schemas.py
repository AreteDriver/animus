#!/usr/bin/env python3
"""validate_schemas.py — CI gate for Animus JSON Schema contracts.

Checks:
  1. Every *.schema.json is valid Draft 2020-12 JSON Schema.
  2. Every schema declares a unique $id.
  3. Every $ref resolves to a known schema (no dangling refs).
  4. Every schema has a $schema declaration.
  5. Schema filenames match their $id basename.

Usage::

    python scripts/validate_schemas.py [contracts_dir]

Exit codes:
    0 — All checks passed.
    1 — One or more checks failed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

DEFAULT_CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "packages" / "contracts"


def find_schemas(contracts_dir: Path) -> list[Path]:
    """Return all *.schema.json files in the contracts directory."""
    return sorted(contracts_dir.glob("*.schema.json"))


def load_schemas(schema_paths: list[Path]) -> dict[Path, dict]:
    """Load schemas, skipping malformed JSON."""
    loaded: dict[Path, dict] = {}
    for path in schema_paths:
        try:
            loaded[path] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"  [FAIL] {path.name}: Invalid JSON — {e}")
    return loaded


def check_schema_declaration(schemas: dict[Path, dict]) -> list[str]:
    """Ensure every schema has a $schema declaration."""
    errors: list[str] = []
    for path, data in schemas.items():
        if "$schema" not in data:
            errors.append(f"{path.name}: Missing $schema declaration")
    return errors


def check_unique_ids(schemas: dict[Path, dict]) -> list[str]:
    """Ensure every schema declares a unique $id."""
    errors: list[str] = []
    seen: dict[str, Path] = {}
    for path, data in schemas.items():
        sid = data.get("$id")
        if not sid:
            errors.append(f"{path.name}: Missing $id")
            continue
        if sid in seen:
            errors.append(
                f"{path.name}: Duplicate $id '{sid}' (also in {seen[sid].name})"
            )
        else:
            seen[sid] = path
    return errors


def check_filename_matches_id(schemas: dict[Path, dict]) -> list[str]:
    """Ensure filename basename matches the $id basename."""
    errors: list[str] = []
    for path, data in schemas.items():
        sid = data.get("$id")
        if not sid:
            continue
        expected_name = sid.split("/")[-1]
        if path.name != expected_name:
            errors.append(
                f"{path.name}: Filename mismatch — $id expects '{expected_name}'"
            )
    return errors


def check_valid_draft202012(schemas: dict[Path, dict]) -> list[str]:
    """Ensure every schema validates as Draft 2020-12."""
    errors: list[str] = []
    for path, data in schemas.items():
        try:
            Draft202012Validator.check_schema(data)
        except Exception as e:
            errors.append(f"{path.name}: Invalid Draft 2020-12 — {e}")
    return errors


def collect_refs(data: dict | list, refs: set[str]) -> None:
    """Recursively collect all $ref values from a schema."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "$ref" and isinstance(value, str):
                refs.add(value)
            else:
                collect_refs(value, refs)
    elif isinstance(data, list):
        for item in data:
            collect_refs(item, refs)


def check_dangling_refs(schemas: dict[Path, dict]) -> list[str]:
    """Ensure every $ref resolves to a known schema $id or local fragment."""
    errors: list[str] = []
    ids = {data.get("$id") for data in schemas.values() if data.get("$id")}

    for path, data in schemas.items():
        refs: set[str] = set()
        collect_refs(data, refs)
        base_id = data.get("$id", "")

        for ref in refs:
            # Local fragment refs — always valid within the same schema
            if ref.startswith("#"):
                continue

            # Resolve relative refs against base $id
            if base_id and "/" in base_id:
                base = base_id.rsplit("/", 1)[0] + "/"
                resolved = ref if "://" in ref else base + ref
            else:
                resolved = ref

            if resolved not in ids:
                errors.append(
                    f"{path.name}: Dangling $ref '{ref}' (resolved: '{resolved}')"
                )

    return errors


def main() -> int:
    contracts_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONTRACTS_DIR

    print(f"Schema Validation — {contracts_dir}")
    print("=" * 50)

    schema_paths = find_schemas(contracts_dir)
    print(f"Schemas found: {len(schema_paths)}")

    schemas = load_schemas(schema_paths)
    if len(schemas) != len(schema_paths):
        print(f"  [FAIL] {len(schema_paths) - len(schemas)} file(s) could not be loaded.")

    all_errors: list[str] = []

    print("\n1. Checking $schema declarations...")
    errors = check_schema_declaration(schemas)
    all_errors.extend(errors)
    for e in errors:
        print(f"  [FAIL] {e}")
    print(f"  {'PASS' if not errors else f'{len(errors)} FAIL'}")

    print("\n2. Checking unique $id declarations...")
    errors = check_unique_ids(schemas)
    all_errors.extend(errors)
    for e in errors:
        print(f"  [FAIL] {e}")
    print(f"  {'PASS' if not errors else f'{len(errors)} FAIL'}")

    print("\n3. Checking filename ↔ $id alignment...")
    errors = check_filename_matches_id(schemas)
    all_errors.extend(errors)
    for e in errors:
        print(f"  [FAIL] {e}")
    print(f"  {'PASS' if not errors else f'{len(errors)} FAIL'}")

    print("\n4. Checking Draft 2020-12 validity...")
    errors = check_valid_draft202012(schemas)
    all_errors.extend(errors)
    for e in errors:
        print(f"  [FAIL] {e}")
    print(f"  {'PASS' if not errors else f'{len(errors)} FAIL'}")

    print("\n5. Checking for dangling $refs...")
    errors = check_dangling_refs(schemas)
    all_errors.extend(errors)
    for e in errors:
        print(f"  [FAIL] {e}")
    print(f"  {'PASS' if not errors else f'{len(errors)} FAIL'}")

    print(f"\n{'=' * 50}")
    if all_errors:
        print(f"RESULT: FAIL — {len(all_errors)} issue(s) found.")
        return 1
    print(f"RESULT: PASS — {len(schemas)} schema(s) valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
