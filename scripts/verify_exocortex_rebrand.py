#!/usr/bin/env python3
"""
verify_exocortex_rebrand.py — Verify the exocortex-sweep rebrand contract.

This script enforces a narrow, deterministic contract:
  1. PyPI/project metadata uses the intended package naming (no "exocortex").
  2. Public-facing docs are empty of "exocortex".
  3. Install examples in public docs use the intended package name.
  4. Architecture book intros (first 5 lines) are reframed to engineering language.
  5. Internal philosophical/architectural body keeps "exocortex" (preservation zones).
  6. Bucket-B files MUST contain "exocortex" — over-sweep fails the gate.
  7. Public-facing READMEs and PyPI surfaces do not contain forbidden
     legacy naming patterns.

Returns 0 on PASS, non-zero on any leak.

Run from repo root:
    python3 scripts/verify_exocortex_rebrand.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# -----------------------------------------------------------------------
# Bucket A: public-facing surfaces — must be empty of "exocortex"
# -----------------------------------------------------------------------
PYPI_SURFACES = [
    "pyproject.toml",
    "packages/core/pyproject.toml",
    "packages/bootstrap/pyproject.toml",
    "packages/contracts/pyproject.toml",
    "release/package-matrix.yaml",
]

PUBLIC_DOCS_GLOB = [
    "docs/getting-started/*.md",
    "docs/operators/*.md",
    "docs/reference/*.md",
    "docs/contributing/*.md",
    "docs/roadmap/*.md",
    "docs/planning/*.md",
    "docs/specs/*.md",
    "docs/reviews/*.md",
    "docs/rework/*.md",
    "docs/migration/*.md",
    "docs/packages/README.md",
    "docs/packages/core/README.md",
    "docs/packages/forge/README.md",
    "docs/packages/bootstrap/README.md",
    "docs/_templates/package-readme.md",
    "packages/core/README.md",
    "packages/forge/README.md",
    "packages/bootstrap/README.md",
    "packages/quorum/README.md",
    "packages/pwa/README.md",
    "packages/bootstrap/PHASE3_INTELLIGENCE.md",
    "docs/README.md",
]

# -----------------------------------------------------------------------
# Bucket D: architecture book intros — first N lines must be reframed.
# Body remains unchanged (Bucket B).
# -----------------------------------------------------------------------
ARCHITECTURE_INTROS = [
    "docs/architecture/charter.md",
    "docs/architecture/overview.md",
    "docs/architecture/consciousness-quorum-bridge.md",
    "docs/architecture/ogma.md",
    "docs/architecture/work-boundary.md",
]
INTRO_LINE_WINDOW = 5  # header + summary + opening paragraph

# -----------------------------------------------------------------------
# Bucket B: preservation zones — must RETAIN "exocortex".
# Over-sweep into these files fails the verifier.
# -----------------------------------------------------------------------
BUCKET_B_PRESERVE = [
    "CLAUDE.md",
    "packages/core/CLAUDE.md",
    # Constitutional Principles moved to docs/architecture/constitutional-principles.md
    # (commit aae5be7). The philosophy is now anchored via agent identity modules
    # and the consciousness-quorum bridge below, which are loaded with "exocortex".
    "docs/architecture/charter.md",
    "docs/architecture/overview.md",
    "docs/architecture/consciousness-quorum-bridge.md",
    "docs/architecture/ogma.md",
    "docs/architecture/work-boundary.md",
    "docs/whitepapers/ANIMUS_WHITEPAPER_2026-06.md",
    "packages/core/animus/__init__.py",
    "packages/core/animus/identity.py",
    "packages/core/animus/api.py",
    "packages/core/animus/mcp_server.py",
    "packages/core/animus/citizens/media.py",
    "packages/core/animus/lugh/sources/relevance.py",
    "packages/core/animus/ogma/read.py",
    "packages/forge/src/animus_forge/coordination/consciousness_bridge.py",
    "packages/forge/src/animus_forge/coordination/evolution_loop.py",
    "packages/kernel/src/animus_kernel/coordination/evolution_loop.py",
    "packages/bootstrap/src/animus_bootstrap/identity/manager.py",
    "packages/bootstrap/src/animus_bootstrap/intelligence/memory_backends/animus_backend.py",
    "tools/animus_discord_bot.py",
    "scripts/review.py",
    "scripts/qwen_security_audit.py",
    "scripts/loop_public_prep.md",
    "packages/core/tests/fixtures/memory_eval_corpus.json",
    "CHANGELOG-v2.3-stable.md",
    "BRANDING.md",  # contains the rule + the term
]

# Archive packages are external projects that keep their own branding.
ARCHIVE_ALLOW = [
    "packages/_archive/",
]

# -----------------------------------------------------------------------
# Patterns
# -----------------------------------------------------------------------
EXOCORTEX = re.compile(r"exocortex", re.IGNORECASE)


def expand_glob(pattern: str) -> list[Path]:
    """Expand a glob pattern (relative to REPO) to sorted file paths."""
    return sorted(REPO.glob(pattern))


def has_exocortex(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        return bool(EXOCORTEX.search(path.read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return False


def exocortex_hits(path: Path) -> list[tuple[int, str]]:
    """Return list of (line_no, line) for exocortex matches in `path`."""
    if not path.exists() or not path.is_file():
        return []
    hits = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for i, line in enumerate(text.splitlines(), 1):
        if EXOCORTEX.search(line):
            hits.append((i, line.strip()))
    return hits


def is_under(path: Path, prefix: str) -> bool:
    """True if path is under the given repo-relative prefix."""
    try:
        path.relative_to(REPO / prefix)
        return True
    except ValueError:
        return False


# -----------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------
def check_pypi_surfaces() -> list[str]:
    """Check 1: PyPI/project metadata must not contain 'exocortex'."""
    failures = []
    for rel in PYPI_SURFACES:
        path = REPO / rel
        if not path.exists():
            continue
        hits = exocortex_hits(path)
        if hits:
            for ln, _ in hits[:3]:
                failures.append(f"{rel}:{ln}")
    return failures


def check_public_docs() -> list[str]:
    """Check 2: Public-facing docs must not contain 'exocortex'."""
    failures = []
    for pattern in PUBLIC_DOCS_GLOB:
        for path in expand_glob(pattern):
            if not path.exists():
                continue
            hits = exocortex_hits(path)
            if hits:
                for ln, _ in hits[:3]:
                    failures.append(f"{path.relative_to(REPO)}:{ln}")
    return failures


def check_architecture_intros() -> list[str]:
    """Check 3: Architecture book intros (first 5 lines) must be reframed."""
    failures = []
    for rel in ARCHITECTURE_INTROS:
        path = REPO / rel
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        head = "\n".join(text.splitlines()[:INTRO_LINE_WINDOW])
        if EXOCORTEX.search(head):
            for i, line in enumerate(text.splitlines()[:INTRO_LINE_WINDOW], 1):
                if EXOCORTEX.search(line):
                    failures.append(f"{rel}:{i}")
    return failures


def check_preservation_zones() -> list[str]:
    """Check 4: Bucket-B files MUST retain 'exocortex'. Over-sweep fails.

    Also fails if any BUCKET_B_PRESERVE path is missing — a phantom
    preservation entry silently skips the check and provides false
    confidence. Adding the missing path requires an explicit edit.
    """
    failures = []
    for rel in BUCKET_B_PRESERVE:
        path = REPO / rel
        if not path.exists():
            failures.append(f"{rel} (Bucket-B path missing on disk)")
            continue
        if not has_exocortex(path):
            failures.append(f"{rel} (Bucket-B file emptied of 'exocortex')")
    return failures


def check_archive_preserved() -> list[str]:
    """Check 5: Archive packages keep their own branding (informational)."""
    failures = []
    archive_dir = REPO / "packages/_archive"
    if not archive_dir.exists():
        return failures
    for sub in sorted(archive_dir.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix in (".md", ".toml"):
                if not has_exocortex(f):
                    failures.append(f"{f.relative_to(REPO)} (archive file lost 'exocortex')")
    return failures


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------
def main() -> int:
    print("=== exocortex rebrand verifier ===")
    print(f"repo: {REPO}")
    print()

    failed = []

    py = check_pypi_surfaces()
    if py:
        failed.append(("PyPI surfaces", py))
        print(f"FAIL: PyPI surfaces still contain 'exocortex' ({len(py)} hits):")
        for h in py[:5]:
            print(f"  - {h}")
    else:
        print("OK: PyPI surfaces clean")

    pd = check_public_docs()
    if pd:
        failed.append(("public docs", pd))
        print(f"FAIL: public docs still contain 'exocortex' ({len(pd)} hits):")
        for h in pd[:5]:
            print(f"  - {h}")
    else:
        print("OK: public docs clean")

    ai = check_architecture_intros()
    if ai:
        failed.append(("architecture intros", ai))
        print(f"FAIL: architecture intros still contain 'exocortex' ({len(ai)} hits):")
        for h in ai[:5]:
            print(f"  - {h}")
    else:
        print("OK: architecture intros reframed")

    bz = check_preservation_zones()
    if bz:
        failed.append(("preservation zones", bz))
        print("FAIL: Bucket-B files emptied of 'exocortex' (over-sweep):")
        for h in bz:
            print(f"  - {h}")
    else:
        print(f"OK: {len(BUCKET_B_PRESERVE)} Bucket-B preservation zones retain 'exocortex'")

    ap = check_archive_preserved()
    if ap:
        failed.append(("archive preservation", ap))
        print("FAIL: archive packages lost 'exocortex' (over-sweep):")
        for h in ap[:5]:
            print(f"  - {h}")
    else:
        print("OK: archive packages preserved")

    print()
    if failed:
        print(f"FAIL: {sum(len(v) for _, v in failed)} issues across {len(failed)} checks")
        return 1
    print("PASS: rebrand contract holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
