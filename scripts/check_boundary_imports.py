#!/usr/bin/env python3
"""check_boundary_imports.py — Detect forbidden cross-package imports in Animus monorepo.

Enforces the layered architecture:
    Interface (bootstrap, pwa) →
    Cognitive (forge, quorum, contracts) →
    Memory (kernel) →
    Core (core) →
    Types (types)

Lower layers must NOT import higher layers.
Also flags specific deep-import anti-patterns that bypass stable façades.

Severity levels:
    CRITICAL  — unconditional module-level import (CI blocking)
    DEFERRED  — import inside a function/method (soft dependency, tracked)
    OPTIONAL  — import inside try/except ImportError (already defensive)
    TYPE_ONLY — import inside TYPE_CHECKING block (not runtime)

Usage:
    python scripts/check_boundary_imports.py         # CI mode: CRITICAL only
    python scripts/check_boundary_imports.py --strict   # CRITICAL + DEFERRED
    python scripts/check_boundary_imports.py --audit    # all severities

To suppress a specific line:
    from animus_forge.providers.base import ModelTier  # boundary-ok

Integrate in CI (example snippet):
    - name: Boundary Import Check
      run: python scripts/check_boundary_imports.py
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# repo root relative to this script (scripts/check_boundary_imports.py)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Map directory suffix → top-level package name.
# Order matters: more specific paths first.
PACKAGE_ROOTS: list[tuple[str, str]] = [
    ("packages/core/animus/", "animus"),
    ("packages/kernel/src/animus_kernel/", "animus_kernel"),
    ("packages/forge/src/animus_forge/", "animus_forge"),
    ("packages/bootstrap/src/animus_bootstrap/", "animus_bootstrap"),
    ("packages/quorum/python/animus_quorum/", "animus_quorum"),
    ("packages/types/src/animus_types/", "animus_types"),
    ("packages/contracts/src/animus_contracts/", "animus_contracts"),
    ("packages/pwa/", "pwa"),
]

# Layer 0 = foundation. Higher number = higher layer.
# A package may only import from packages in its own layer or lower layers.
LAYERS: dict[str, int] = {
    "animus_types": 0,  # Shared types (leaf)
    "animus": 1,  # Core
    "animus_kernel": 2,  # Memory / Kernel
    "animus_contracts": 3,  # Contracts (cognitive layer)
    "animus_forge": 3,  # Cognitive — Forge
    "animus_quorum": 3,  # Cognitive — Quorum
    "animus_bootstrap": 4,  # Interface — Bootstrap
    "pwa": 4,  # Interface — PWA
}

# Specific deep-import patterns to forbid regardless of layer or context.
# Keys are importer top-level packages; values are forbidden module prefixes.
DEEP_IMPORT_DENYLIST: dict[str, list[str]] = {
    "animus_bootstrap": [
        "animus_kernel.head.checkpoint",  # Use CheckpointFacade instead
    ],
}

# Files/directories to skip (substring match on path)
SKIP_PATHS: tuple[str, ...] = (
    "/tests/",
    "/test_",
    "conftest.py",
    "_archive/",
    "/.venv/",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    importer_pkg: str
    imported_module: str
    rule: str
    severity: str
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_path(path: Path) -> bool:
    ps = str(path)
    for pat in SKIP_PATHS:
        if pat in ps:
            return True
    return False


def _find_package_prefix(path: Path) -> tuple[str, str] | None:
    """Return (package_root_abs, top_level_module) for *path* if known."""
    for rel_dir, pkg_name in PACKAGE_ROOTS:
        root = REPO_ROOT / rel_dir
        try:
            path.relative_to(root)
            return str(root), pkg_name
        except ValueError:
            continue
    return None


def _module_path(file_path: Path, pkg_root: str, top_level: str) -> str:
    """Compute full module dotted path for a source file."""
    rel = file_path.relative_to(Path(pkg_root))
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([top_level, *parts])


def _resolve_relative(module: str, current_module: str) -> str:
    """Turn a relative import string into an absolute module path."""
    if not module.startswith("."):
        return module
    current_parts = current_module.split(".")
    dots = 0
    for ch in module:
        if ch == ".":
            dots += 1
        else:
            break
    base = current_parts[:-dots] if dots <= len(current_parts) else []
    rest = module[dots:]
    if rest:
        return ".".join(base + [rest])
    return ".".join(base)


def _top_level(module: str) -> str:
    return module.split(".")[0]


def _line_has_boundary_ok(source_line: str) -> bool:
    return "# boundary-ok" in source_line or "# noqa: boundary" in source_line


# ---------------------------------------------------------------------------
# AST context analysis
# ---------------------------------------------------------------------------


def _is_in_type_checking_block(stack: list[ast.AST]) -> bool:
    """Return True if the node stack includes a TYPE_CHECKING if-block."""
    for node in stack:
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                return True
    return False


def _is_in_function(stack: list[ast.AST]) -> bool:
    """Return True if the node stack includes a function or method definition."""
    for node in stack:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
    return False


def _is_in_try_import_error(stack: list[ast.AST]) -> bool:
    """Return True if the node stack includes a try/except ImportError block."""
    for node in stack:
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                if handler.type is None:
                    # bare except
                    continue
                if isinstance(handler.type, ast.Name) and handler.type.id == "ImportError":
                    return True
                if isinstance(handler.type, ast.Tuple):
                    for elt in handler.type.elts:
                        if isinstance(elt, ast.Name) and elt.id == "ImportError":
                            return True
    return False


def _classify_import_context(stack: list[ast.AST]) -> str:
    """Classify import context: TYPE_ONLY, OPTIONAL, DEFERRED, or CRITICAL."""
    if _is_in_type_checking_block(stack):
        return "TYPE_ONLY"
    if _is_in_try_import_error(stack):
        return "OPTIONAL"
    if _is_in_function(stack):
        return "DEFERRED"
    return "CRITICAL"


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self, current_module: str, source_lines: list[str]) -> None:
        self.current_module = current_module
        self.source_lines = source_lines
        self.imports: list[tuple[int, str, list[ast.AST]]] = []  # line, abs_module, stack

    def _visit_import(self, node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.imports.append((node.lineno, alias.name, self._stack.copy()))
        else:
            if node.module is None:
                mod = "." * node.level + (node.names[0].name if node.names else "")
            else:
                mod = "." * node.level + node.module
            abs_mod = _resolve_relative(mod, self.current_module)
            self.imports.append((node.lineno, abs_mod, self._stack.copy()))

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self._visit_import(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self._visit_import(node)
        self.generic_visit(node)

    # Override visit to maintain a node stack for context analysis
    def visit(self, node: ast.AST) -> None:  # noqa: N802
        if not hasattr(self, "_stack"):
            self._stack: list[ast.AST] = []
        self._stack.append(node)
        try:
            super().visit(node)
        finally:
            self._stack.pop()


def _check_file(path: Path, min_severity: str) -> list[Violation]:
    """Check a single file for boundary violations."""
    severity_rank = {"TYPE_ONLY": 0, "OPTIONAL": 1, "DEFERRED": 2, "CRITICAL": 3}
    min_rank = severity_rank.get(min_severity, 3)

    violations: list[Violation] = []
    result = _find_package_prefix(path)
    if result is None:
        return violations
    pkg_root, importer_pkg = result

    # Skip if not in layers (e.g., standalone utilities, archived code)
    if importer_pkg not in LAYERS:
        return violations

    current_module = _module_path(path, pkg_root, importer_pkg)
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)

    try:
        tree = ast.parse(source, str(path))
    except SyntaxError as exc:
        print(f"WARNING: syntax error in {path}: {exc}", file=sys.stderr)
        return violations

    visitor = _ImportVisitor(current_module, lines)
    visitor.visit(tree)

    for lineno, abs_module, stack in visitor.imports:
        imported_top = _top_level(abs_module)

        # Skip stdlib / third-party
        if imported_top not in LAYERS:
            continue

        # Skip self-imports within the same package
        if imported_top == importer_pkg:
            continue

        source_line = lines[lineno - 1] if lineno <= len(lines) else ""
        if _line_has_boundary_ok(source_line):
            continue

        context = _classify_import_context(stack)

        # --- Layer rule --------------------------------------------------
        importer_layer = LAYERS[importer_pkg]
        imported_layer = LAYERS.get(imported_top)
        if imported_layer is not None and importer_layer < imported_layer:
            # Deep imports are always CRITICAL regardless of context
            severity = context if context != "TYPE_ONLY" else "TYPE_ONLY"
            if severity_rank[severity] >= min_rank:
                violations.append(
                    Violation(
                        path=path,
                        line=lineno,
                        importer_pkg=importer_pkg,
                        imported_module=abs_module,
                        rule="LAYER_VIOLATION",
                        severity=severity,
                        message=(
                            f"{importer_pkg} (layer {importer_layer}) imports "
                            f"{abs_module} (layer {imported_layer}): reverse dependency "
                            f"[{severity}]"
                        ),
                    )
                )

        # --- Deep import rule --------------------------------------------
        denylist = DEEP_IMPORT_DENYLIST.get(importer_pkg, [])
        for forbidden in denylist:
            if abs_module.startswith(forbidden):
                violations.append(
                    Violation(
                        path=path,
                        line=lineno,
                        importer_pkg=importer_pkg,
                        imported_module=abs_module,
                        rule="DEEP_IMPORT",
                        severity="CRITICAL",
                        message=(
                            f"{importer_pkg} imports {abs_module}: "
                            f"use stable façade instead of {forbidden}"
                        ),
                    )
                )
                break

    return violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect forbidden cross-package imports in Animus monorepo."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Include DEFERRED imports in addition to CRITICAL.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Report all severities including TYPE_ONLY and OPTIONAL.",
    )
    args = parser.parse_args(argv)

    if args.audit:
        min_severity = "TYPE_ONLY"
    elif args.strict:
        min_severity = "DEFERRED"
    else:
        min_severity = "CRITICAL"

    packages_dir = REPO_ROOT / "packages"
    if not packages_dir.exists():
        print(f"ERROR: {packages_dir} not found", file=sys.stderr)
        return 2

    all_violations: list[Violation] = []

    for py_file in packages_dir.rglob("*.py"):
        if _skip_path(py_file):
            continue
        all_violations.extend(_check_file(py_file, min_severity))

    if not all_violations:
        print(f"OK — no boundary violations (severity ≥ {min_severity}).")
        return 0

    # Group by severity first, then rule
    by_severity: dict[str, list[Violation]] = {}
    for v in all_violations:
        by_severity.setdefault(v.severity, []).append(v)

    total = 0
    for sev in ("CRITICAL", "DEFERRED", "OPTIONAL", "TYPE_ONLY"):
        vs = by_severity.get(sev, [])
        if not vs:
            continue
        total += len(vs)
        print(f"\n{'=' * 60}")
        print(f"{sev}: {len(vs)} violation(s)")
        print(f"{'=' * 60}")
        # Sub-group by rule
        by_rule: dict[str, list[Violation]] = {}
        for v in vs:
            by_rule.setdefault(v.rule, []).append(v)
        for rule, rvs in sorted(by_rule.items()):
            for v in rvs:
                rel = v.path.relative_to(REPO_ROOT)
                print(f"  {rel}:{v.line}  {v.message}")

    print(f"\n{'=' * 60}")
    print(f"Total: {total} boundary violation(s) (severity ≥ {min_severity})")
    print(f"{'=' * 60}")
    print("\nTo suppress a line: add '# boundary-ok' at the end of the import line.")
    print("Modes: default=CRITICAL | --strict=+DEFERRED | --audit=all")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
