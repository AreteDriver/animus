#!/usr/bin/env python3
"""
docs-validate.py — Comprehensive docs validation for CI and local use.

Checks:
  1. Internal markdown links resolve to existing files
  2. Markdown anchor references (#section) point to real headings
  3. No trailing whitespace in .md files
  4. No broken redirect stubs in repo root

Usage:
    python scripts/docs-validate.py [repo_root]

Exit codes:
    0 — All checks passed
    1 — One or more checks failed
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def _find_md_files(repo: Path) -> list[Path]:
    """Return all markdown files excluding vendored/cache/worktree dirs."""
    exclude = {".git", "node_modules", ".venv", ".pytest_cache", "__pycache__", ".claude"}
    files: list[Path] = []
    for f in repo.rglob("*.md"):
        if any(part in exclude for part in f.parts):
            continue
        files.append(f)
    return sorted(files)


def _slugify(text: str) -> str:
    """Convert a heading to a GitHub-style anchor slug."""
    text = text.lower().strip()
    # Remove markdown formatting
    text = re.sub(r"[#*`~_\[\]()|]", "", text)
    # Replace spaces and special chars with hyphens
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text).strip("-")
    return text


def _extract_headings(content: str) -> set[str]:
    """Extract GitHub-style anchor slugs from markdown headings."""
    headings: set[str] = set()
    for line in content.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if match:
            headings.add(_slugify(match.group(2)))
    return headings


def _check_links(repo: Path, files: list[Path]) -> list[str]:
    """Find broken internal links and anchors."""
    errors: list[str] = []
    # Pre-load all file contents and headings
    file_contents: dict[Path, str] = {}
    file_headings: dict[Path, set[str]] = {}
    for f in files:
        content = f.read_text(encoding="utf-8")
        file_contents[f] = content
        file_headings[f] = _extract_headings(content)

    link_re = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
    for md in files:
        content = file_contents[md]
        for match in link_re.finditer(content):
            url = match.group(2)
            # Skip external URLs, anchors-only, mailto
            if url.startswith(("http", "https", "mailto", "#")):
                continue
            # Split path and anchor
            if "#" in url:
                path_part, anchor = url.split("#", 1)
            else:
                path_part, anchor = url, ""
            if not path_part:
                # Anchor-only link within same file
                target_file = md
            else:
                target_file = (md.parent / path_part).resolve()
            if not target_file.exists():
                rel_md = md.relative_to(repo)
                errors.append(f"{rel_md} -> {url} (file not found)")
                continue
            # Validate anchor if present
            if anchor and anchor not in file_headings.get(target_file, set()):
                rel_md = md.relative_to(repo)
                errors.append(f"{rel_md} -> {url} (anchor '#{anchor}' not found)")
    return errors


def _check_trailing_whitespace(repo: Path, files: list[Path]) -> list[str]:
    """Find lines with trailing whitespace in markdown files.

    Ignores exactly 2 trailing spaces — those are valid Markdown hard
    line breaks (used in blockquotes, poetry, etc.).
    """
    errors: list[str] = []
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.rstrip()
            # Allow exactly 2 trailing spaces (Markdown hard line break)
            if len(line) - len(stripped) > 2:
                rel = f.relative_to(repo)
                errors.append(f"{rel}:{i}")
                break  # One error per file is enough for reporting
    return errors


def _check_redirect_stubs(repo: Path) -> list[str]:
    """Warn if root-level redirect stubs are missing expected text.

    README.md is the canonical project README, not a redirect.
    """
    stubs = [
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "ROADMAP.md",
        "PROJECT_CHARTER.md",
        "SECURITY.md",
    ]
    errors: list[str] = []
    for stub in stubs:
        path = repo / stub
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            if "This document has moved" not in content and "redirect" not in content.lower():
                errors.append(f"{stub} may need redirect stub")
    return errors


def main() -> int:
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    md_files = _find_md_files(repo)
    all_ok = True

    # 1. Link check
    link_errors = _check_links(repo, md_files)
    if link_errors:
        all_ok = False
        print(f"\n❌ Broken links: {len(link_errors)}")
        for err in link_errors:
            print(f"   {err}")
    else:
        print(f"✅ Internal links OK ({len(md_files)} files scanned)")

    # 2. Trailing whitespace
    ws_errors = _check_trailing_whitespace(repo, md_files)
    if ws_errors:
        all_ok = False
        print(f"\n❌ Trailing whitespace: {len(ws_errors)} files")
        for err in ws_errors[:20]:
            print(f"   {err}")
        if len(ws_errors) > 20:
            print(f"   ... and {len(ws_errors) - 20} more")
    else:
        print("✅ No trailing whitespace in markdown files")

    # 3. Redirect stubs
    stub_errors = _check_redirect_stubs(repo)
    if stub_errors:
        print(f"\n⚠️  Redirect stub warnings: {len(stub_errors)}")
        for err in stub_errors:
            print(f"   {err}")
    else:
        print("✅ Redirect stubs OK")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
