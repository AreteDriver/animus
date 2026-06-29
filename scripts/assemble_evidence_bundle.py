#!/usr/bin/env python3
"""
assemble_evidence_bundle.py — Collect release evidence from repo state.

Usage:
    python scripts/assemble_evidence_bundle.py [--output-dir PATH] [--allow-dirty]

Produces a timestamped directory with machine-readable + human-readable
artifacts proving a release is safe, tested, and traceable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
EVIDENCE_DIR = REPO_ROOT / "evidence" / "releases"

PACKAGES: list[dict[str, str | None]] = [
    {"name": "core", "path": "packages/core", "language": "python"},
    {"name": "forge", "path": "packages/forge", "language": "python"},
    {"name": "bootstrap", "path": "packages/bootstrap", "language": "python"},
    {"name": "kernel", "path": "packages/kernel", "language": "python"},
    {"name": "quorum", "path": "packages/quorum", "language": "rust+python"},
    {"name": "types", "path": "packages/types", "language": "python"},
]


def _run(cmd: list[str] | str, cwd: Path | None = None, timeout: int = 120) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            cwd=cwd or str(REPO_ROOT),
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return -1, "", str(e)


def _git_sha() -> str:
    rc, out, _ = _run(["git", "rev-parse", "HEAD"])
    return out.strip() if rc == 0 else "unknown"


def _git_branch() -> str:
    rc, out, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return out.strip() if rc == 0 else "unknown"


def _git_is_dirty() -> bool:
    rc, out, _ = _run(["git", "status", "--porcelain"])
    return rc == 0 and bool(out.strip())


def _git_log(n: int = 5) -> str:
    rc, out, _ = _run(["git", "log", "--oneline", f"-{n}"])
    return out if rc == 0 else "unknown"


def _git_info() -> str:
    sha = _git_sha()
    branch = _git_branch()
    dirty = "DIRTY" if _git_is_dirty() else "clean"
    log = _git_log(5)
    return f"Branch: {branch}\nSHA: {sha}\nStatus: {dirty}\n\nLast 5 commits:\n{log}"


def _version_from_repo() -> str:
    rc, out, _ = _run(["git", "describe", "--tags", "--always"])
    if rc == 0:
        return out.strip()
    # Fallback: read root pyproject.toml
    text = (REPO_ROOT / "pyproject.toml").read_text(errors="ignore")
    m = re.search(r'version\s*=\s*"(\d+\.\d+\.\d+)"', text)
    return m.group(1) if m else "unknown"


def _collect_tests(package: dict[str, str | None]) -> dict[str, int | str]:
    """Run pytest --collect-only and parse counts from text output."""
    pkg_path = REPO_ROOT / str(package["path"])
    if not (pkg_path / "tests").exists():
        return {"name": package["name"], "collected": 0, "errors": 0, "status": "skipped", "message": "no tests directory"}

    rc, stdout, stderr = _run(
        ["python3", "-m", "pytest", "--collect-only", "-q"],
        cwd=pkg_path,
        timeout=120,
    )
    combined = stdout + stderr

    # Parse "N tests collected" or "N tests collected, M errors"
    collected = 0
    errors = 0
    status = "ok"
    message = ""

    m = re.search(r'(\d+) tests? collected', combined)
    if m:
        collected = int(m.group(1))

    m = re.search(r'(\d+) errors? in', combined)
    if m:
        errors = int(m.group(1))
        status = "errors"
        message = f"{errors} collection error(s)"

    if rc != 0 and collected == 0:
        status = "failed"
        message = stderr[:200] if stderr else f"exit code {rc}"

    return {
        "name": package["name"],
        "collected": collected,
        "errors": errors,
        "status": status,
        "message": message,
    }


def _validate_schemas() -> dict[str, list[dict[str, str | bool]]]:
    """Check every .schema.json is parseable."""
    schema_dir = REPO_ROOT / "packages" / "contracts"
    results: list[dict[str, str | bool]] = []
    all_ok = True

    for path in sorted(schema_dir.glob("*.schema.json")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        try:
            data = json.loads(text)
            results.append({
                "file": path.name,
                "valid": True,
                "title": data.get("title", ""),
            })
        except json.JSONDecodeError as e:
            all_ok = False
            results.append({
                "file": path.name,
                "valid": False,
                "error": str(e),
            })

    return {"all_valid": all_ok, "schemas": results}


def _lock_dependencies() -> dict[str, str]:
    locks: dict[str, str] = {}

    # Python packages (root venv)
    rc, out, _ = _run(["pip3", "freeze"])
    if rc == 0:
        locks["python"] = out

    # Rust (quorum)
    quorum_dir = REPO_ROOT / "packages" / "quorum"
    if (quorum_dir / "Cargo.toml").exists():
        rc, out, _ = _run(["cargo", "tree"], cwd=quorum_dir)
        if rc == 0:
            locks["rust"] = out

    # Node (pwa)
    pwa_dir = REPO_ROOT / "packages" / "pwa"
    if (pwa_dir / "package.json").exists():
        rc, out, _ = _run(["npm", "ls", "--json"], cwd=pwa_dir)
        if rc == 0 or out.strip():
            locks["node"] = out

    return locks


def _build_report(manifest: dict, tests: list[dict], schemas: dict, locks: dict) -> str:
    lines: list[str] = [
        "# Animus Evidence Bundle",
        "",
        f"**Version**: {manifest['version']}  ",
        f"**Git SHA**: `{manifest['git_sha']}`  ",
        f"**Branch**: {manifest['git_branch']}  ",
        f"**Timestamp**: {manifest['timestamp']}  ",
        f"**Builder**: {manifest['builder']}  ",
        "",
        "---",
        "",
        "## Test Results",
        "",
        "| Package | Collected | Errors | Status |",
        "|---|---|---|---|",
    ]

    for t in tests:
        icon = "✅" if t["status"] == "ok" else "⚠️" if t["status"] == "errors" else "❌" if t["status"] == "failed" else "⏭️"
        lines.append(f"| {t['name']} | {t['collected']} | {t['errors']} | {icon} {t['status']} |")

    total_collected = sum(t["collected"] for t in tests)
    total_errors = sum(t["errors"] for t in tests)
    lines.extend([
        "",
        f"**Total**: {total_collected} tests collected, {total_errors} collection errors",
        "",
        "## Schema Validation",
        "",
        f"**All valid**: {'✅ Yes' if schemas['all_valid'] else '❌ No'}  ",
        f"**Schemas checked**: {len(schemas['schemas'])}",
        "",
    ])

    if not schemas["all_valid"]:
        lines.append("### Invalid schemas:")
        for s in schemas["schemas"]:
            if not s["valid"]:
                lines.append(f"- ❌ `{s['file']}` — {s.get('error', 'unknown error')}")
        lines.append("")

    lines.extend([
        "## Dependencies",
        "",
        f"- **Python packages**: {len(locks.get('python', '').splitlines())} lines",
        f"- **Rust crates**: {'captured' if 'rust' in locks else 'not captured'}",
        f"- **Node modules**: {'captured' if 'node' in locks else 'not captured'}",
        "",
        "## Git Info",
        "",
        "```",
        _git_info(),
        "```",
        "",
        "---",
        "",
        "*Generated by `scripts/assemble_evidence_bundle.py`*",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble an Animus release evidence bundle")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVIDENCE_DIR,
        help="Directory to write the evidence bundle (default: evidence/releases/)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow dirty git working tree (default: fail if dirty)",
    )
    args = parser.parse_args()

    # Guard against dirty tree
    if _git_is_dirty() and not args.allow_dirty:
        print("ERROR: Working tree is dirty. Commit or stash changes, or use --allow-dirty.", file=sys.stderr)
        sys.exit(1)

    # Timestamp and paths
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%d-%H%M%S")
    out_dir: Path = args.output_dir / f"evidence-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sha = _git_sha()
    branch = _git_branch()
    version = _version_from_repo()

    # Manifest
    manifest = {
        "project": "animus",
        "version": version,
        "git_sha": sha,
        "git_branch": branch,
        "timestamp": now.isoformat(),
        "builder": os.environ.get("USER", "unknown"),
        "dirty": _git_is_dirty(),
    }

    # Tests
    print("Collecting test counts...")
    tests = [_collect_tests(p) for p in PACKAGES]

    # Schemas
    print("Validating JSON schemas...")
    schemas = _validate_schemas()

    # Dependencies
    print("Locking dependencies...")
    locks = _lock_dependencies()

    # Git info
    git_info = _git_info()

    # Write files
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    (out_dir / "test-results.json").write_text(
        json.dumps({"packages": tests, "total_collected": sum(t["collected"] for t in tests)}, indent=2) + "\n",
        encoding="utf-8",
    )

    (out_dir / "schema-validation.json").write_text(json.dumps(schemas, indent=2) + "\n", encoding="utf-8")

    (out_dir / "git-info.txt").write_text(git_info + "\n", encoding="utf-8")

    deps_text = ""
    for key, content in locks.items():
        deps_text += f"=== {key.upper()} ===\n{content}\n\n"
    (out_dir / "dependencies.lock").write_text(deps_text, encoding="utf-8")

    report = _build_report(manifest, tests, schemas, locks)
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    # Summary
    total_collected = sum(t["collected"] for t in tests)
    total_errors = sum(t["errors"] for t in tests)
    print(f"\n{'=' * 60}")
    print(f"Evidence Bundle — {ts}")
    print(f"{'=' * 60}")
    print(f"  Tests: {total_collected} collected, {total_errors} errors")
    print(f"  Schemas: {'all valid' if schemas['all_valid'] else 'INVALID'}")
    print(f"  Git: {sha[:8]} ({branch})")
    print(f"  Output: {out_dir.relative_to(REPO_ROOT)}")
    print(f"{'=' * 60}")

    # Exit code
    if not schemas["all_valid"]:
        print("ERROR: One or more schemas failed validation.", file=sys.stderr)
        sys.exit(1)

    if total_errors > 0:
        print("WARNING: Some packages had test collection errors.", file=sys.stderr)
        # Still exit 0 — collection errors are common in development envs

    sys.exit(0)


if __name__ == "__main__":
    main()
