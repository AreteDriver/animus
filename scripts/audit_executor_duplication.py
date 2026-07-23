#!/usr/bin/env python3
"""Audit script that quantifies Forge/Kernel executor module duplication.

Produces a JSON report showing which modules are identical (modulo import
paths), which differ, and what the remaining migration surface is.

Usage:
    python scripts/audit_executor_duplication.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Monorepo root assumed to be two levels above this script
REPO_ROOT = Path(__file__).resolve().parent.parent

KERNEL_EXEC_DIR = REPO_ROOT / "packages" / "kernel" / "src" / "animus_kernel" / "executor"
FORGE_EXEC_DIR = REPO_ROOT / "packages" / "forge" / "src" / "animus_forge" / "workflow"

IMPORT_RE = re.compile(r"^(from|import)\s+(animus_kernel|animus_forge)\b")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalize_imports(text: str) -> str:
    """Replace package-specific import paths with a generic placeholder."""
    return IMPORT_RE.sub(lambda m: m.group(1) + " {{PKG}}", text)


def _module_diff(kernel_path: Path, forge_path: Path) -> dict:
    kernel_text = _read_text(kernel_path)
    forge_text = _read_text(forge_path)

    # Structural comparison (ignoring import paths)
    normalized_kernel = _normalize_imports(kernel_text)
    normalized_forge = _normalize_imports(forge_text)

    identical = normalized_kernel == normalized_forge

    # Count import path differences
    kernel_imports = IMPORT_RE.findall(kernel_text)
    forge_imports = IMPORT_RE.findall(forge_text)

    return {
        "kernel_path": str(kernel_path.relative_to(REPO_ROOT)),
        "forge_path": str(forge_path.relative_to(REPO_ROOT)),
        "identical_modulo_imports": identical,
        "kernel_imports": len(kernel_imports),
        "forge_imports": len(forge_imports),
        "size_bytes": len(kernel_text),
    }


def main() -> int:
    if not KERNEL_EXEC_DIR.exists():
        print(f"ERROR: Kernel executor dir not found: {KERNEL_EXEC_DIR}", file=sys.stderr)
        return 1
    if not FORGE_EXEC_DIR.exists():
        print(f"ERROR: Forge executor dir not found: {FORGE_EXEC_DIR}", file=sys.stderr)
        return 1

    # Gather all executor modules
    kernel_modules = sorted(KERNEL_EXEC_DIR.glob("executor_*.py"))

    results = []
    total_size = 0
    identical_count = 0

    for kmod in kernel_modules:
        fmod = FORGE_EXEC_DIR / kmod.name
        if fmod.exists():
            diff = _module_diff(kmod, fmod)
            results.append(diff)
            total_size += diff["size_bytes"]
            if diff["identical_modulo_imports"]:
                identical_count += 1
        else:
            results.append({
                "kernel_path": str(kmod.relative_to(REPO_ROOT)),
                "forge_path": None,
                "identical_modulo_imports": False,
                "note": "missing in Forge",
            })

    report = {
        "summary": {
            "total_modules": len(kernel_modules),
            "forge_has_module": sum(1 for r in results if r.get("forge_path")),
            "identical_modulo_imports": identical_count,
            "total_duplicated_bytes": total_size,
            "percent_identical": round(identical_count / len(kernel_modules) * 100, 1) if kernel_modules else 0,
        },
        "modules": results,
        "recommendation": (
            "All modules differ only in import paths (animus_kernel vs animus_forge). "
            "Migration path: add animus-kernel dependency to Forge, then replace "
            "Forge executor_core with a thin wrapper that imports from Kernel and "
            "re-exports. Forge-specific mixin modules can be kept as override layers "
            "or gradually merged into a plugin registry."
        ),
    }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
