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
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES_DIR = REPO_ROOT / "packages"


def _extract_version_spec(dep: str) -> tuple[str, str]:
    """Parse ``pkg_name>=1.0.0,<2`` into (name, specifier_string).

    Handles extras like ``pkg[extra]>=1.0``.
    """
    # Strip extras
    if "[" in dep:
        dep = dep[:dep.index("[")]
    # Find first comparison operator
    for op in ("==", "!=", "<=", ">=", "<", ">", "~="):
        if op in dep:
            idx = dep.index(op)
            return dep[:idx].strip(), dep[idx:].strip()
    return dep.strip(), ""


def main() -> int:
    # Collect all local packages
    local_packages: dict[str, Version] = {}
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

        local_packages[name] = Version(version)
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
                spec = SpecifierSet(spec_str)
            except Exception as exc:
                violations.append(
                    f"{name}: invalid specifier '{spec_str}' for '{dep_name}': {exc}"
                )
                continue

            if not spec.contains(sibling_version):
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

    print(
        f"OK — {len(local_packages)} packages aligned. "
        f"({len(warnings_)} warnings)"
    )
    return 0


class SiblingVersion:
    """Helper to compute next major for warning messages."""

    @staticmethod
    def next_major(v: Version) -> Version:
        return Version(f"{v.major + 1}.0.0")


if __name__ == "__main__":
    sys.exit(main())
