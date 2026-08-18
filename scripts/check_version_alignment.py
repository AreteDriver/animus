#!/usr/bin/env python3
"""Monorepo version alignment checker.

Parses every ``packages/*/pyproject.toml`` and verifies that
inter-package dependency specs match the actual versions declared
in sibling packages.

Usage::

    python3 scripts/check_version_alignment.py

Exit code 0 = all aligned; 1 = drift detected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES_DIR = REPO_ROOT / "packages"


class SimpleVersion:
    """Minimal semantic version parser for local package alignment checks."""

    def __init__(self, value: str) -> None:
        self.raw = value
        parts = value.split(".")
        if not parts:
            raise ValueError(f"Unsupported version '{value}'")
        numeric_parts: list[int] = []
        for part in parts[:3]:
            digits = ""
            for character in part:
                if character.isdigit():
                    digits += character
                else:
                    break
            if digits == "":
                raise ValueError(f"Unsupported version '{value}'")
            numeric_parts.append(int(digits))
        while len(numeric_parts) < 3:
            numeric_parts.append(0)
        self.parts = tuple(numeric_parts)
        self.major = self.parts[0]

    def __str__(self) -> str:
        return self.raw

    def __lt__(self, other: "SimpleVersion") -> bool:
        return self.parts < other.parts

    def __le__(self, other: "SimpleVersion") -> bool:
        return self.parts <= other.parts

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimpleVersion):
            return False
        return self.parts == other.parts

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __gt__(self, other: "SimpleVersion") -> bool:
        return self.parts > other.parts

    def __ge__(self, other: "SimpleVersion") -> bool:
        return self.parts >= other.parts


def _specifier_contains(specifier_string: str, version: SimpleVersion) -> bool:
    """Evaluate a small subset of PEP 440 specifiers used in this monorepo."""

    for raw_specifier in specifier_string.split(","):
        specifier = raw_specifier.strip()
        if not specifier:
            continue
        for operator in ("==", "!=", "<=", ">=", "<", ">"):
            if specifier.startswith(operator):
                expected = SimpleVersion(specifier[len(operator) :].strip())
                if operator == "==" and not version == expected:
                    return False
                if operator == "!=" and not version != expected:
                    return False
                if operator == "<=" and not version <= expected:
                    return False
                if operator == ">=" and not version >= expected:
                    return False
                if operator == "<" and not version < expected:
                    return False
                if operator == ">" and not version > expected:
                    return False
                break
        else:
            raise ValueError(f"Unsupported specifier '{specifier}'")
    return True


def _extract_version_spec(dep: str) -> tuple[str, str]:
    """Parse ``pkg_name>=1.0.0,<2`` into (name, specifier_string).

    Handles extras like ``pkg[extra]>=1.0``.
    """
    # Strip extras
    if "[" in dep:
        dep = dep[: dep.index("[")]
    # Find first comparison operator
    for op in ("==", "!=", "<=", ">=", "<", ">", "~="):
        if op in dep:
            idx = dep.index(op)
            return dep[:idx].strip(), dep[idx:].strip()
    return dep.strip(), ""


def main() -> int:
    # Collect all local packages
    local_packages: dict[str, SimpleVersion] = {}
    pyproject_paths: dict[str, Path] = {}

    for pkg_dir in PACKAGES_DIR.iterdir():
        if not pkg_dir.is_dir():
            continue
        # Skip _archive and non-Python packages (e.g., pwa)
        if pkg_dir.name.startswith("_"):
            continue

        pp = pkg_dir / "pyproject.toml"
        if not pp.exists():
            continue

        data = tomllib.loads(pp.read_text())
        project = data.get("project")
        if not project:
            continue

        name = project.get("name")
        version = project.get("version")
        if not name or not version:
            continue

        local_packages[name] = SimpleVersion(version)
        pyproject_paths[name] = pp

    if not local_packages:
        print("No packages found — nothing to check.")
        return 0

    violations: list[str] = []
    warnings_: list[str] = []

    for name, version in local_packages.items():
        pp = pyproject_paths[name]
        data = tomllib.loads(pp.read_text())
        deps = data.get("project", {}).get("dependencies", [])
        opt_deps = data.get("project", {}).get("optional-dependencies", {})

        all_deps: list[str] = list(deps)
        for group in opt_deps.values():
            all_deps.extend(group)

        for dep in all_deps:
            dep_name, spec_str = _extract_version_spec(dep)
            if dep_name not in local_packages:
                continue  # external dependency — not our concern
            if dep_name == name:
                continue  # self-reference (e.g., "all" extra) — ignore

            sibling_version = local_packages[dep_name]
            if not spec_str:
                warnings_.append(
                    f"{name}: dependency '{dep_name}' has no version specifier "
                    f"(recommend pinning to >={sibling_version}, <{SiblingVersion.next_major(sibling_version)})"
                )
                continue

            try:
                contains_version = _specifier_contains(spec_str, sibling_version)
            except Exception as exc:
                violations.append(f"{name}: invalid specifier '{spec_str}' for '{dep_name}': {exc}")
                continue

            if not contains_version:
                violations.append(
                    f"{name} ({version}) depends on {dep_name} {spec_str}, "
                    f"but local {dep_name} is {sibling_version}"
                )

    if warnings_:
        print("=" * 60)
        print(f"WARNINGS: {len(warnings_)} warning(s)")
        print("=" * 60)
        for w in warnings_:
            print(f"  {w}")
        print()

    if violations:
        print("=" * 60)
        print(f"VIOLATIONS: {len(violations)} alignment issue(s)")
        print("=" * 60)
        for v in violations:
            print(f"  {v}")
        print()
        print(
            "Fix: update the dependency specifier in the consuming package's "
            "pyproject.toml to include the local version."
        )
        return 1

    print(f"OK — {len(local_packages)} packages aligned. ({len(warnings_)} warnings)")
    return 0


class SiblingVersion:
    """Helper to compute next major for warning messages."""

    @staticmethod
    def next_major(version: SimpleVersion) -> SimpleVersion:
        return SimpleVersion(f"{version.major + 1}.0.0")


if __name__ == "__main__":
    sys.exit(main())
