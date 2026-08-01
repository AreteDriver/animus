#!/usr/bin/env python3
"""Pre-public sanitization scanner for Animus.

Scans the repository for secrets, owner-specific data, and hardcoded paths
that must be removed or generalized before making the repo public.

Usage:
    python scripts/pre_public_sanitize.py [--fix]

Exit codes:
    0 = clean (no blockers)
    1 = blockers found (do not make public)
    2 = runtime error
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class Finding:
    """A single sanitization finding."""

    severity: str  # critical | high | medium | low
    category: str  # secret | path | personal | owner | config
    file: Path
    line: int
    snippet: str
    fix: str | None = None  # suggested fix, if automated


@dataclass
class ScanResult:
    """Aggregate result of a scan pass."""

    findings: list[Finding] = field(default_factory=list)
    scanned_files: int = 0
    scanned_lines: int = 0

    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "critical"]

    def high(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "high"]

    def medium(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "medium"]

    def by_category(self) -> dict[str, list[Finding]]:
        d: dict[str, list[Finding]] = {}
        for f in self.findings:
            d.setdefault(f.category, []).append(f)
        return d


# ============================================================================
# Scan rules
# ============================================================================

SECRET_PATTERNS: list[tuple[str, str, str]] = [
    # (name, regex, severity)
    ("anthropic_api_key", r"sk-ant-[a-zA-Z0-9]{20,}", "critical"),
    ("openai_api_key", r"sk-[a-zA-Z0-9]{20,}", "critical"),
    ("github_token", r"ghp_[a-zA-Z0-9]{30,}", "critical"),
    ("github_pat", r"github_pat_[a-zA-Z0-9]{20,}", "critical"),
    ("fly_token", r"fly_[a-zA-Z0-9]{20,}", "critical"),
    ("vercel_token", r"vercel_[a-zA-Z0-9]{20,}", "critical"),
    ("discord_webhook", r"https://discord\.com/api/webhooks/[0-9]+/[a-zA-Z0-9_-]+", "high"),
    (
        "generic_secret",
        r"(?i)(secret|token|key|password)\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']",
        "medium",
    ),
]

OWNER_PATTERNS: list[tuple[str, str, str]] = [
    ("owner_username", r"AreteDriver", "high"),
    ("owner_username_lc", r"aretedriver", "medium"),
    ("personal_email", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "medium"),
]

PATH_PATTERNS: list[tuple[str, str, str]] = [
    ("home_path", r"/home/arete", "medium"),
    ("absolute_project_path", r"/home/arete/projects/", "low"),
]

ALLOWED_FALSE_POSITIVES: list[str] = [
    # npm lockfile integrity hashes (random base64 that happens to match patterns)
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    # compiled artifacts / virtual environments
    "__pycache__",
    ".mypy_cache",
    "node_modules",
    "dist",
    "build",
    ".venv/",
    "venv/",
    "env/",
    # test fixtures that intentionally contain dummy secrets
    "tests/fixtures",
    "test_*.py",  # test files may contain mock tokens
    "verify_hardening.py",  # adversarial test inputs with fake secrets
    # documentation that describes the pattern
    "docs/",
    "README.md",
    "SECURITY.md",
    # evidence bundles (generated, not source)
    "evidence/releases/",
    # generated / ephemeral directories
    "site/",
    ".claude/",
    "_archive/",
    # local env file is gitignored; scanner should not flag on-disk copy
    ".env",
    # the scanner itself contains the old owner name as a regex pattern
    "pre_public_sanitize.py",
    # vendored third-party assets
    "tailwindcss-cdn.js",
    "site-packages/",
    # generated build artifacts
    ".rustc_info.json",
    ".code_memory_manifest.json",
    # example config files with env var references
    "settings.example.yaml",
    # skill documentation with example emails
    "SKILL.md",
]


def _is_false_positive(file_path: Path) -> bool:
    """Check if a file is a known false-positive container."""
    path_str = str(file_path)
    for fp in ALLOWED_FALSE_POSITIVES:
        if fp in path_str:
            return True
        # Handle wildcard patterns like "test_*.py" with fnmatch
        if "*" in fp and fnmatch.fnmatch(file_path.name, fp):
            return True
        if file_path.name.endswith(fp.removeprefix("*")):
            return True
    return False


def _scan_text(content: str, file_path: Path) -> list[Finding]:
    """Scan file content for patterns."""
    findings: list[Finding] = []
    lines = content.splitlines()

    for line_no, line in enumerate(lines, start=1):
        # Secrets
        for name, pattern, severity in SECRET_PATTERNS:
            for match in re.finditer(pattern, line):
                # Skip lines that are clearly documentation, examples, or placeholders
                skip_keywords = (
                    r"(?i)(example|placeholder|dummy|mock|fake|your_|"
                    r"no-key-required|change-me-in-production|\\$\\{)"
                )
                if re.search(skip_keywords, line):
                    continue
                # Skip Python/JS dict keys and constants (e.g., key="model_name", KEY = "path")
                if re.search(r'(?i)key\s*[:=]\s*["\']\w+[_-]', line):
                    continue
                # Skip lockfile hashes
                if "sha512" in line or "integrity" in line:
                    continue
                findings.append(
                    Finding(
                        severity=severity,
                        category="secret",
                        file=file_path,
                        line=line_no,
                        snippet=match.group(0),
                        fix=f"Remove or replace {name}",
                    )
                )

        # Owner-specific data
        for name, pattern, severity in OWNER_PATTERNS:
            for match in re.finditer(pattern, line):
                # Skip documentation explaining the owner name
                if re.search(r"(?i)(example|placeholder|your username|repo owner)", line):
                    continue
                findings.append(
                    Finding(
                        severity=severity,
                        category="owner" if "username" in name else "personal",
                        file=file_path,
                        line=line_no,
                        snippet=match.group(0),
                        fix=f"Generalize or remove {name}",
                    )
                )

        # Hardcoded paths
        for name, pattern, severity in PATH_PATTERNS:
            for match in re.finditer(pattern, line):
                # Skip relative path examples in docs
                if re.search(r"(?i)(e\.g\.|example|your home|user home)", line):
                    continue
                findings.append(
                    Finding(
                        severity=severity,
                        category="path",
                        file=file_path,
                        line=line_no,
                        snippet=match.group(0),
                        fix=f"Use relative path or env var for {name}",
                    )
                )

    return findings


def _is_gitignored(repo_root: Path, file_path: Path) -> bool:
    """Check if a file is ignored by git."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "check-ignore", str(file_path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _scan_env_files(repo_root: Path) -> list[Finding]:
    """Scan for .env and secrets files that should not be committed."""
    findings: list[Finding] = []
    dangerous_patterns = [
        ".env",
        ".env.local",
        ".env.production",
        "secrets.env",
        "credentials.json",
    ]

    for pattern in dangerous_patterns:
        for path in repo_root.rglob(pattern):
            if ".git" in str(path):
                continue
            if _is_false_positive(path):
                continue
            # Skip if already gitignored
            if _is_gitignored(repo_root, path):
                continue
            findings.append(
                Finding(
                    severity="critical",
                    category="secret",
                    file=path,
                    line=1,
                    snippet=str(path.name),
                    fix=f"Add to .gitignore and remove from history: git rm --cached {path}",
                )
            )
    return findings


def _scan_gitignore(repo_root: Path) -> list[Finding]:
    """Check that .gitignore covers common secret files."""
    findings: list[Finding] = []
    gitignore = repo_root / ".gitignore"
    if not gitignore.exists():
        findings.append(
            Finding(
                severity="critical",
                category="config",
                file=repo_root,
                line=0,
                snippet="missing .gitignore",
                fix="Create .gitignore with secret patterns",
            )
        )
        return findings

    content = gitignore.read_text()
    required_patterns = [
        ".env",
        "*.env",
        "secrets.env",
        "credentials.json",
        "*.pem",
        "*.key",
    ]
    for req in required_patterns:
        if req not in content:
            findings.append(
                Finding(
                    severity="high",
                    category="config",
                    file=gitignore,
                    line=0,
                    snippet=f"missing .gitignore rule: {req}",
                    fix=f"Add '{req}' to .gitignore",
                )
            )
    return findings


def _auto_fix(result: ScanResult, repo_root: Path) -> int:
    """Apply automated fixes where safe. Returns count of fixes applied."""
    fixes_applied = 0
    for finding in result.findings:
        if finding.severity != "critical":
            continue
        if finding.category == "config" and finding.fix and ".gitignore" in finding.fix:
            gitignore = repo_root / ".gitignore"
            if gitignore.exists():
                content = gitignore.read_text()
                rule = finding.fix.split(":")[-1].strip().strip("'")
                if rule not in content:
                    with open(gitignore, "a") as f:
                        f.write(f"\n{rule}\n")
                    fixes_applied += 1
                    finding.fix = f"APPLIED: added {rule} to .gitignore"
    return fixes_applied


def scan(repo_root: Path, auto_fix: bool = False) -> ScanResult:
    """Run full scan."""
    result = ScanResult()

    # Textual scan — skip known heavy dirs
    skip_dirs = {
        "__pycache__",
        ".mypy_cache",
        "node_modules",
        ".git",
        "dist",
        "build",
        "evidence/releases",
        "packages/_archive",
    }
    for file_path in repo_root.rglob("*"):
        if file_path.is_dir():
            continue
        if any(part in skip_dirs for part in file_path.parts):
            continue
        if _is_false_positive(file_path):
            continue
        if file_path.stat().st_size > 500_000:  # skip large binaries
            continue
        if not _is_text_file(file_path):
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        result.scanned_files += 1
        result.scanned_lines += content.count("\n")
        result.findings.extend(_scan_text(content, file_path))

    # Specialized scans
    result.findings.extend(_scan_env_files(repo_root))
    result.findings.extend(_scan_gitignore(repo_root))

    if auto_fix:
        _auto_fix(result, repo_root)

    return result


def _is_text_file(path: Path) -> bool:
    """Heuristic: is this a text file we should scan?"""
    text_extensions = {
        ".py",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
        ".txt",
        ".rst",
        ".sh",
        ".bash",
        ".zsh",
        ".js",
        ".ts",
        ".tsx",
        ".css",
        ".html",
        ".xml",
        ".sql",
        ".dockerfile",
        ".makefile",
        ".gitignore",
        ".gitattributes",
        ".env",
        ".env.example",
    }
    return path.suffix.lower() in text_extensions or path.name in {
        "Makefile",
        "Dockerfile",
        ".gitignore",
        ".env",
        ".env.example",
    }


def report(result: ScanResult) -> str:
    """Generate human-readable report."""
    lines = [
        "=" * 60,
        "ANIMUS PRE-PUBLIC SANITIZATION REPORT",
        f"Generated: {datetime.now(UTC).isoformat()}Z",
        f"Scanned: {result.scanned_files} files, {result.scanned_lines} lines",
        "=" * 60,
        "",
    ]

    if not result.findings:
        lines.append("✅ CLEAN — no blockers found. Repo is ready for public release.")
        return "\n".join(lines)

    # Summary table
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Critical: {len(result.critical())}")
    lines.append(f"  High:     {len(result.high())}")
    lines.append(f"  Medium:   {len(result.medium())}")
    lines.append(f"  Total:    {len(result.findings)}")
    lines.append("")

    # Blockers first
    blockers = result.critical() + result.high()
    if blockers:
        lines.append("❌ BLOCKERS (must fix before public)")
        lines.append("-" * 40)
        for f in blockers:
            lines.append(f"  [{f.severity.upper()}] {f.category}: {f.file}:{f.line}")
            lines.append(f"      Snippet: {f.snippet[:60]}{'...' if len(f.snippet) > 60 else ''}")
            if f.fix:
                lines.append(f"      Fix: {f.fix}")
            lines.append("")

    # Medium / low
    others = result.medium()
    if others:
        lines.append("⚠️  WARNINGS (should fix before public)")
        lines.append("-" * 40)
        for f in others[:20]:  # cap to avoid spam
            lines.append(f"  [{f.severity.upper()}] {f.category}: {f.file}:{f.line}")
            lines.append(f"      Snippet: {f.snippet[:60]}{'...' if len(f.snippet) > 60 else ''}")
        if len(others) > 20:
            lines.append(f"  ... and {len(others) - 20} more")
        lines.append("")

    lines.append("=" * 60)
    lines.append("VERDICT: DO NOT MAKE PUBLIC until all CRITICAL and HIGH items are resolved.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Animus pre-public sanitization scanner")
    parser.add_argument("--fix", action="store_true", help="Apply safe automated fixes")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--repo", type=Path, default=Path("."), help="Repository root")
    args = parser.parse_args()

    result = scan(args.repo, auto_fix=args.fix)

    if args.json:
        data: dict[str, Any] = {
            "scanned_files": result.scanned_files,
            "scanned_lines": result.scanned_lines,
            "findings": [
                {
                    "severity": f.severity,
                    "category": f.category,
                    "file": str(f.file),
                    "line": f.line,
                    "snippet": f.snippet,
                    "fix": f.fix,
                }
                for f in result.findings
            ],
        }
        print(json.dumps(data, indent=2))
    else:
        print(report(result))

    if result.critical() or result.high():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
